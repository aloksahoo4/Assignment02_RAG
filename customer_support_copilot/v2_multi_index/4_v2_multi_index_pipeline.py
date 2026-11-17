import json
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# V2 (multi-index + routing) pipeline:
# Route -> hybrid search in each chosen index (Dense + BM25 -> RRF)
#       -> merge candidates -> one global cross-encoder rerank -> top 5
#       -> answer with sources
#
# Run 1_build_indexes.py first: it creates the 4 Chroma collections that this
# script connects to.

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

embeddings_model = OpenAIEmbeddings(
    api_key=as_api_key,
    model="text-embedding-3-small"
)

llm_brain = ChatOpenAI(
    api_key=as_api_key,
    model="gpt-6-luna",
    max_tokens=300
)

# Answers get a larger token budget than the router. A question with several
# parts and several sources runs out of a small budget and comes back empty.
answer_llm = ChatOpenAI(
    api_key=as_api_key,
    model="gpt-6-luna",
    max_tokens=800
)

ALL_INDEXES = ["policy_index", "product_index", "order_index", "ticket_index"]

PER_INDEX_K = 5   # candidates each selected index contributes to the combined pool
TOP_K = 5         # results kept after the single global rerank

# Exact reply when the knowledge base does not contain the answer
NO_ANSWER = "I don't know based on the available information."


def tokenize(text):
    # "[a-z]+-\d+" keeps a document ID like "RR-004" as ONE token ("rr-004")
    # instead of splitting it into "rr" and "004".
    return re.findall(r"[a-z]+-\d+|[a-z0-9]+", text.lower())


#=======================================
# Load the 4 indexes (Chroma + BM25 for each one)
#=======================================

def load_index(index_name):
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings_model,
        collection_name=index_name,
    )

    all_data = vectorstore.get()
    chunk_texts = all_data["documents"]
    chunk_metadatas = all_data["metadatas"]
    if not chunk_texts:
        raise ValueError(f"Collection '{index_name}' is empty. Run 1_build_indexes.py first.")

    # BM25 is built in memory from the records of the Chroma collection.
    # It indexes the ID and title along with the body text, so a question that
    # names a document ID (e.g. "What does policy RR-004 say?") can match it.
    tokenized_corpus = [
        tokenize(f"{metadata['source_id']} {metadata['title']} {text}")
        for text, metadata in zip(chunk_texts, chunk_metadatas)
    ]

    return {
        "vectorstore": vectorstore,
        "bm25": BM25Okapi(tokenized_corpus),
        "texts": chunk_texts,
        "metadatas": chunk_metadatas,
    }


indexes = {index_name: load_index(index_name) for index_name in ALL_INDEXES}

print("Connected to indexes:")
for index_name, index in indexes.items():
    print(f"- {index_name}: {index['vectorstore']._collection.count()} records")

print("\nLoading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.")


#=======================================
# STAGE 1: Intent-Based Query Routing (same router as 3_intent_router.py)
#=======================================
# Each intent points to the index that holds that kind of knowledge.
# "out_of_scope" has no index: nothing is searched.
INTENT_TO_INDEX = {
    "policy": "policy_index",        # rules: returns, refunds, delivery, charges, COD
    "product": "product_index",      # product questions and known problems
    "order": "order_index",          # how to do something with an order
    "past_issue": "ticket_index",    # "has this happened before?" and similar past cases
    "out_of_scope": None,            # anything the store doesn't cover
}

ROUTER_PROMPT = """You route questions for the support copilot of Dandes Electronics, an
online store that sells consumer electronics only: phones, laptops, earbuds and
headphones, chargers and computer accessories.

Classify the question into one or more of these intents:
- policy: official store rules - return window, opened or damaged item rules,
  replacement vs refund, refund timelines, delivery times and charges, express
  delivery, Cash on Delivery limits, failed delivery attempts.
- product: a question about a specific product (box contents, specifications,
  compatibility, software, warranty period, EMI or exchange offers) or a known
  problem with a device and how to fix it.
- order: HOW TO do something with an order - cancel it, change the delivery
  address, track it, fix a failed payment, report a damaged, defective or wrong
  item, raise a warranty claim.
- past_issue: the user asks whether this happened before or wants a similar
  resolved case, OR describes a specific problem a customer is facing (a past
  ticket may show how it was fixed).
- out_of_scope: anything the store doesn't cover - a kind of product it does not
  sell (for example refrigerators, furniture or groceries) or a topic that is not
  about the store at all. Use it ALONE, never together with another intent.

Each intent has one index:
policy -> policy_index, product -> product_index, order -> order_index,
past_issue -> ticket_index, out_of_scope -> no index (empty list).

Pick every intent that is needed to answer the full question, and no extra ones.
Example: "Galaxy Buds3 Pro left earbud won't connect" is both product (a known
device problem) and past_issue (a similar case may have been resolved before).

Respond with ONLY a JSON object in exactly this format, and no other text:
{{"intents": ["product", "past_issue"], "indexes": ["product_index", "ticket_index"], "reason": "Question is about a product problem that may have happened before"}}

Question: "{question}"
"""


def parse_router_output(raw_text):
    """Turn the router's JSON text into a routing decision. Raises if it is not valid."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove a Markdown code fence around the JSON, if the model added one
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[len("json"):]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("router output is not a JSON object")

    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("'intents' must be a non-empty list")

    unknown = [intent for intent in intents if intent not in INTENT_TO_INDEX]
    if unknown:
        raise ValueError(f"unknown intents: {unknown}")

    # Keep each intent once, in the fixed order of INTENT_TO_INDEX.
    # out_of_scope only counts when it is the ONLY intent.
    intents = [intent for intent in INTENT_TO_INDEX if intent in intents]
    in_scope = [intent for intent in intents if intent != "out_of_scope"]
    intents = in_scope if in_scope else ["out_of_scope"]

    # The indexes are always rebuilt from the intents, so the two can never disagree.
    index_names = [INTENT_TO_INDEX[intent] for intent in intents if INTENT_TO_INDEX[intent]]

    return {"intents": intents, "indexes": index_names, "reason": str(data.get("reason", ""))}


def fallback_decision(error):
    """Used when the router fails: search all 4 indexes instead of crashing."""
    return {
        "intents": [intent for intent in INTENT_TO_INDEX if INTENT_TO_INDEX[intent]],
        "indexes": list(ALL_INDEXES),
        "reason": f"Fallback: router failed ({type(error).__name__}: {error}), searching all 4 indexes",
    }


def route_question(question):
    """Return the routing decision: {"intents": [...], "indexes": [...], "reason": "..."}.

    "indexes" is empty for an out-of-scope question. This function never raises:
    if the LLM call fails or the output is not valid JSON, it returns all 4 indexes.
    """
    try:
        response = llm_brain.invoke([HumanMessage(content=ROUTER_PROMPT.format(question=question))])
        return parse_router_output(response.content)
    except Exception as error:   # API error, timeout, invalid JSON, invalid content
        return fallback_decision(error)


#=======================================
# STAGE 2: Hybrid search in ONE index (Dense + BM25, fused with RRF)
#=======================================

def dense_search(index_name, question_vector, k=PER_INDEX_K):
    results = indexes[index_name]["vectorstore"].similarity_search_by_vector(question_vector, k=k)
    return [(doc.metadata["source_id"], doc.metadata["title"], doc.page_content) for doc in results]


def sparse_search(index_name, question, k=PER_INDEX_K):
    index = indexes[index_name]
    scores = index["bm25"].get_scores(tokenize(question))

    scored_chunks = [
        (text, metadata, score)
        for text, metadata, score in zip(index["texts"], index["metadatas"], scores)
        if score > 0   # score 0 = no keyword in common, not a real match
    ]
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    return [(metadata["source_id"], metadata["title"], text) for text, metadata, score in scored_chunks[:k]]


def reciprocal_rank_fusion(ranked_lists, k=60):
    rrf_scores = {}
    doc_info = {}
    for ranked_list in ranked_lists:
        for rank, (source_id, title, text) in enumerate(ranked_list, start=1):
            rrf_scores[source_id] = rrf_scores.get(source_id, 0) + 1 / (k + rank)
            doc_info[source_id] = (title, text)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(source_id, doc_info[source_id][0], doc_info[source_id][1]) for source_id, score in fused]


def exact_id_matches(index_name, question):
    """Documents of this index that the question names by ID, e.g. "RR-004"."""
    index = indexes[index_name]
    question_tokens = set(tokenize(question))
    return [
        (metadata["source_id"], metadata["title"], text)
        for text, metadata in zip(index["texts"], index["metadatas"])
        if metadata["source_id"].lower() in question_tokens
    ]


def search_index(index_name, question, question_vector, k=PER_INDEX_K):
    """Return the top-k hybrid candidates from ONE index. No reranking here."""
    dense_results = dense_search(index_name, question_vector, k=k)
    sparse_results = sparse_search(index_name, question, k=k)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    # A document named by its ID is always a candidate and goes first. Hybrid
    # search alone can rank it below documents that only CITE that ID.
    named = exact_id_matches(index_name, question)
    named_ids = {item[0] for item in named}
    fused = named + [item for item in fused if item[0] not in named_ids]

    dates = {metadata["source_id"]: metadata["date"] for metadata in indexes[index_name]["metadatas"]}

    return [
        {
            "index": index_name,
            "source_id": source_id,
            "title": title,
            "text": text,
            "date": dates[source_id],
            "id_match": source_id in named_ids,
        }
        for source_id, title, text in fused[:k]
    ]


#=======================================
# STAGE 3: Merge the candidates from all chosen indexes
#=======================================

def collect_candidates(question, index_names):
    # The question is embedded ONCE and the same vector is reused for every index.
    question_vector = embeddings_model.embed_query(question)

    candidates = []
    for index_name in index_names:
        candidates.extend(search_index(index_name, question, question_vector))
    return candidates


#=======================================
# STAGE 4: One global rerank of the merged candidates
#=======================================

def rerank(question, candidates):
    if not candidates:
        return []

    # The reranker also sees the ID and title, so a document that the question
    # names by its ID is not scored low.
    pairs = [(question, f"{c['source_id']} - {c['title']}. {c['text']}") for c in candidates]
    ce_scores = cross_encoder.predict(pairs)

    # Sorted by cross-encoder score, but a document the question names by ID
    # stays on top: the cross-encoder does not understand IDs and scores it low.
    reranked = sorted(
        zip(candidates, ce_scores),
        key=lambda x: (x[0]["id_match"], x[1]),
        reverse=True,
    )
    return [{**candidate, "score": float(score)} for candidate, score in reranked]


#=======================================
# STAGE 5: Answer Generation (grounded, with sources)
#=======================================

def generate_answer(question, results):
    """Grounded answer that cites the source IDs it used."""
    if not results:
        return NO_ANSWER

    context = "\n\n".join(
        f"[{r['source_id']}] {r['title']} (date: {r['date']})\n{r['text']}" for r in results
    )

    prompt = f"""You are a support assistant for Dandes Electronics. Answer the
user's question using ONLY the context below. If the question has several
parts, answer each part.

Cite every document you relied on by its ID in square brackets, like [TCK-1001].

Latest information wins. Every document in the context has a date. If two
documents disagree, use the one with the LATEST date, do not follow the older
one, and mention that date in your answer, for example: "Under the current
policy effective 1 July 2026, ...". If a document says when it took effect or
that it replaces older rules, mention that date too. Write dates like 1 July 2026.

If the context does not contain the information needed to answer the
question, respond with EXACTLY this sentence and nothing else:
"{NO_ANSWER}"
Do not use your own knowledge, do not guess, and do not conclude anything
from what the context fails to mention.

Context:
{context}

Question: "{question}"

Answer:"""

    response = answer_llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


#=======================================
# ORCHESTRATOR: Full V2 (multi-index + routing) Pipeline
#=======================================

def format_list(items):
    return ", ".join(items) if items else "none"


def multi_index_pipeline(question):
    print("\n" + "=" * 70)
    print(f"Question : {question}")

    # Stage 1: route the question to one or more indexes (never crashes)
    decision = route_question(question)
    intents, index_names = decision["intents"], decision["indexes"]
    print(f"Routing  : intents=[{', '.join(intents)}] → indexes=[{format_list(index_names)}]")
    print(f"Reason   : {decision['reason']}")

    # Out of scope: skip retrieval and reply with the fixed sentence
    if not index_names:
        results, answer, sources = [], NO_ANSWER, []
    else:
        # Stage 2 + 3: hybrid search in each chosen index, then merge the candidates
        candidates = collect_candidates(question, index_names)

        # Stage 4: one global rerank, keep the top 5
        results = rerank(question, candidates)[:TOP_K]
        print(f"Top {TOP_K}    : {format_list([r['source_id'] for r in results])} "
              f"(reranked from {len(candidates)} candidates)")

        # Stage 5: answer with sources
        answer = generate_answer(question, results)

        # Sources = the retrieved documents that the answer actually cites
        sources = sorted(r["source_id"] for r in results if r["source_id"] in answer)

    indented_answer = answer.replace("\n", "\n" + " " * 11)
    print(f"\nAnswer   : {indented_answer}")
    print(f"Sources  : {format_list(sources)}")

    return {
        "intents": intents,
        "indexes": index_names,
        "reason": decision["reason"],
        "results": results,
        "answer": answer,
        "sources": sources,
    }


#=======================================
# Test: the 7 test questions with their expected indexes and sources
#=======================================

test_cases = [
    {
        "question": "Can I return an opened laptop on day 8 if I have the original box and all accessories?",
        "expected_indexes": ["policy_index"],
        "expected_sources": ["RR-001", "RR-002"],
    },
    {
        "question": "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has "
                    "the same problem been resolved for another customer? What fixed it?",
        "expected_indexes": ["product_index", "ticket_index"],
        "expected_sources": ["KI-001", "TCK-1001"],
    },
    {
        "question": "How long does delivery take to a remote pincode, and is it free?",
        "expected_indexes": ["policy_index"],
        "expected_sources": ["SH-002"],
    },
    {
        "question": "A customer wants to change the address after the order shipped. What is the "
                    "official process, and do we have a resolved case like this?",
        "expected_indexes": ["order_index", "ticket_index"],
        "expected_sources": ["OG-002", "TCK-1004"],
    },
    {
        "question": "What does policy RR-004 say?",
        "expected_indexes": ["policy_index"],
        "expected_sources": ["RR-004"],
    },
    {
        "question": "A customer received a damaged phone and wants a refund instead of a "
                    "replacement. What is the official process, and do we have a previous "
                    "resolved case where a refund was issued?",
        "expected_indexes": ["policy_index", "order_index", "ticket_index"],
        "expected_sources": ["RR-003", "RR-004", "RR-005", "OG-005", "TCK-1002"],
    },
    {
        "question": "Do you sell refrigerators?",
        "expected_indexes": [],
        "expected_sources": [],
    },
]

def run_test_questions():
    for test in test_cases:
        output = multi_index_pipeline(test["question"])

        expected_indexes = test["expected_indexes"]
        expected_sources = test["expected_sources"]
        routing_ok = set(output["indexes"]) == set(expected_indexes)

        print(f"\nExpected : indexes=[{format_list(expected_indexes)}] | sources={format_list(expected_sources)}")

        # Out-of-scope question: no document is relevant, the right result is the fixed refusal
        if not expected_sources:
            refused = output["answer"].strip() == NO_ANSWER
            print(f"Check    : routing {'MATCH' if routing_ok else 'DIFFERENT'} | "
                  f"refusal {'PASS' if refused else 'FAIL'}")
            continue

        retrieved = [r["source_id"] for r in output["results"]]
        found = [source_id for source_id in expected_sources if source_id in retrieved]
        print(f"Check    : routing {'MATCH' if routing_ok else 'DIFFERENT'} | "
              f"{len(found)} / {len(expected_sources)} expected sources in the top {TOP_K}")


#=======================================
# Latest information wins: two retrieved documents disagree
#=======================================
# TCK-1003 (2025-09-18) says laptops can be returned within 7 days.
# RR-001 (2026-07-01) says 10 days and replaces all earlier return windows.
# The answer must follow the 2026 policy and mention its date.

def run_latest_information_check():
    print("\n" + "#" * 70)
    print("# LATEST INFORMATION WINS")
    print("#" * 70)

    conflict_question = ("How many days does a customer have to return an opened laptop, "
                         "and do we have a past case of an opened laptop return?")
    output = multi_index_pipeline(conflict_question)

    retrieved = [r["source_id"] for r in output["results"]]
    both_retrieved = "RR-001" in retrieved and "TCK-1003" in retrieved
    uses_latest = "10 days" in output["answer"]
    mentions_date = "July 2026" in output["answer"]

    print(f"\nCheck    : conflicting documents RR-001 and TCK-1003 both retrieved: {'YES' if both_retrieved else 'NO'}")
    print(f"           answer uses the latest rule (10 days): {'PASS' if uses_latest else 'FAIL'} | "
          f"mentions the date (1 July 2026): {'PASS' if mentions_date else 'FAIL'}")


# Runs only when this file is executed directly, not when the evaluation
# script (evaluation/1_run_versions.py) imports multi_index_pipeline from it.
if __name__ == "__main__":
    run_test_questions()
    run_latest_information_check()
