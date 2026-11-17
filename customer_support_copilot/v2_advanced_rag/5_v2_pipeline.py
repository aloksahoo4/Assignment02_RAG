import os
import re
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")

embeddings_model = OpenAIEmbeddings(
    api_key=as_api_key,
    model="text-embedding-3-small"
)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings_model,
    collection_name="support_tickets"
)

llm_brain = ChatOpenAI(
    api_key=as_api_key,
    model="gpt-5.6-luna",
    max_tokens=200
)

print(f"Connected to vector store. Items indexed: {vectorstore._collection.count()}")

all_data = vectorstore.get()
chunk_texts = all_data["documents"]
chunk_metadatas = all_data["metadatas"]


def tokenize(text):
    # "[a-z]+-\d+" keeps a document ID like "RR-004" as ONE token ("rr-004")
    # instead of splitting it into "rr" and "004".
    return re.findall(r"[a-z]+-\d+|[a-z0-9]+", text.lower())


# BM25 indexes the ID and title along with the body text, so a question that
# names a document ID (e.g. "What does policy RR-004 say?") can match it.
# The body text alone never contains the document's own ID.
tokenized_text = [
    tokenize(f"{metadata['source_id']} {metadata['title']} {text}")
    for text, metadata in zip(chunk_texts, chunk_metadatas)
]
bm25 = BM25Okapi(tokenized_text)
print("BM25 index built.")

ALL_SOURCE_IDS = {metadata["source_id"].lower() for metadata in chunk_metadatas}
ID_MATCH_BOOST = 100   # added to the BM25 score when the question names this exact document

print("\nLoading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.\n")

TOP_K = 5
RETRIEVE_K = 10
CONFIDENCE_THRESHOLD = -6.0   # tune against your own cross-encoder score distribution

# Exact reply when the knowledge base does not contain the answer
NO_ANSWER = "I don't know based on the available information."

#=======================================
# STAGE 1: Intent-Based Routing (ALWAYS RUNS)
#=======================================

class RouteDecision(BaseModel):
    needs_retrieval: bool = Field(
        description=(
            "True for ANY question that asks for information: products, what the store sells, "
            "prices, policies, orders, delivery, troubleshooting - even if you suspect the "
            "knowledge base does not cover it. Never answer such questions from your own knowledge. "
            "False ONLY for greetings, thanks and small talk that ask for no information."
        )
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Document category if retrieval is needed. Use EXACTLY one of these values "
            "(they are the only categories in the knowledge base): "
            "'returns' (return window, opened/damaged/wrong items, replacements), "
            "'refunds' (refund timelines, refund not received), "
            "'delivery' (delivery times, remote pincodes, express, failed delivery attempts), "
            "'tracking' (tracking an order, tracking not updating), "
            "'order_changes' (cancel an order, change delivery address), "
            "'payments' (payment failed, EMI, Cash on Delivery limits), "
            "'warranty' (warranty period, warranty claims), "
            "'exchange' (trading in an old phone, exchange value), "
            "'connectivity' (Bluetooth, Wi-Fi, earbuds or devices not connecting), "
            "'charging' (device or earbud not charging, charger wattage), "
            "'compatibility' (does product X work with device Y), "
            "'setup' (new device setup, eSIM transfer), "
            "'box_contents' (what is included in the box), "
            "'specifications' (hardware details of a product), "
            "'software' (pre-installed OS and apps). "
            "Leave empty if no single category clearly fits."
        )
    )
    query_clarity: Literal["clear", "vague", "ambiguous"] = Field(
        description=(
            "clear: specific, well-formed query - search as-is. "
            "vague: short/casual phrasing with ONE likely meaning - needs rewriting. "
            "ambiguous: could genuinely mean SEVERAL different things - needs multi-query."
        )
    )

router_llm = llm_brain.with_structured_output(RouteDecision)


def route_query(raw_question):
    prompt = f"""Classify this support chat question.

Question: "{raw_question}"
"""
    return router_llm.invoke([HumanMessage(content=prompt)])

#=======================================
# STAGE 2: Retrieval Primitives (METADATA FILTERING + HYBRID SEARCH)
#=======================================

def dense_search(question, category=None, k=RETRIEVE_K):
    where = {"category": category} if category else None
    results = vectorstore.similarity_search_with_score(question, k=k, filter=where)
    return [(doc.metadata["source_id"], doc.metadata["title"], doc.page_content) for doc, score in results]


def sparse_search(question, category=None, k=RETRIEVE_K):
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    # Document IDs the user typed in the question, e.g. {"rr-004"}
    asked_ids = {token for token in tokenized_query if token in ALL_SOURCE_IDS}

    scored_chunks = []
    for text, metadata, score in zip(chunk_texts, chunk_metadatas, scores):
        is_exact_id = metadata["source_id"].lower() in asked_ids

        # A document named by its ID is kept even if the router guessed another category
        if category and metadata["category"] != category and not is_exact_id:
            continue

        # Without the boost, documents that only CITE the ID ("as per RR-004") outrank it
        if is_exact_id:
            score += ID_MATCH_BOOST
        scored_chunks.append((text, metadata, score))

    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    top = scored_chunks[:k]
    return [(metadata["source_id"], metadata["title"], text) for text, metadata, score in top]


def reciprocal_rank_fusion(ranked_lists, k=60):
    rrf_scores = {}
    doc_info = {}
    for ranked_list in ranked_lists:
        for rank, (source_id, title, text) in enumerate(ranked_list, start=1):
            score = 1 / (k + rank)
            rrf_scores[source_id] = rrf_scores.get(source_id, 0) + score
            doc_info[source_id] = (title, text)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(source_id, doc_info[source_id][0], doc_info[source_id][1], score) for source_id, score in fused]


def rerank_fused_results(question, fused_results, top_n_to_rerank=8):
    candidates = fused_results[:top_n_to_rerank]
    # The reranker also sees the ID and title. With body text only, it scores the
    # document named in the question low and drops it from the final results.
    pairs = [(question, f"{source_id} - {title}. {text}") for source_id, title, text, rrf_score in candidates]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [(source_id, title, ce_score) for (source_id, title, text, rrf_score), ce_score in reranked]


def hybrid_search(question, category=None, k=TOP_K):
    """Baseline: metadata-filtered hybrid search + rerank, single query."""
    dense_results = dense_search(question, category=category)
    sparse_results = sparse_search(question, category=category)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    reranked = rerank_fused_results(question, fused)
    return reranked[:k]

#=======================================
# STAGE 3a: Query Transformation (CONDITIONAL - "vague")
#=======================================

def rewrite_query(raw_question):
    prompt = f"""Rewrite this short, vague support question into a clear,
complete question. Do not answer it - only rewrite it. Respond with
ONLY the rewritten question.

Original question: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


#=======================================
# STAGE 3b: Multi-Query Retrieval (CONDITIONAL - "ambiguous")
#=======================================

def generate_query_variants(raw_question, n=3):
    prompt = f"""This question is ambiguous and could mean several things.
Generate {n} DIFFERENT, specific interpretations of what the user might
mean, covering different possible angles (e.g. permissions, performance/
errors, data export, account access, billing). Respond with ONLY the
{n} rewritten questions, one per line, nothing else.

Original question: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    variants = [line.strip("- ").strip() for line in response.content.strip().split("\n") if line.strip()]
    return variants[:n]


def multi_query_search(raw_question, category=None, n_variants=3, k=TOP_K):
    variants = generate_query_variants(raw_question, n=n_variants)
    all_variants = [raw_question] + variants

    per_variant_fused_lists = []
    for variant in all_variants:
        dense_results = dense_search(variant, category=category)
        sparse_results = sparse_search(variant, category=category)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        simplified = [(sid, title, text) for sid, title, text, score in fused]
        per_variant_fused_lists.append(simplified)

    final_fused = reciprocal_rank_fusion(per_variant_fused_lists)
    final_results = rerank_fused_results(raw_question, final_fused)
    return variants, final_results[:k]


#=======================================
# STAGE 4: Answer Generation (ALWAYS RUNS - grounded or direct)
#=======================================

def generate_answer(raw_question, results, is_ambiguous=False):
    """Grounded answer using retrieved ticket context."""
    if not results:
        return NO_ANSWER

    context_blocks = []
    for source_id, title, score in results:
        for text, metadata in zip(chunk_texts, chunk_metadatas):
            if metadata["source_id"] == source_id:
                context_blocks.append(f"[{source_id}] {title} (date: {metadata['date']})\n{text}")
                break

    context = "\n\n".join(context_blocks)

    ambiguity_note = (
        "This question could have multiple valid causes or meanings. "
        "Address EACH distinct scenario found in the context separately - "
        "do not pick just one and ignore the others.\n\n"
        if is_ambiguous else ""
    )

    prompt = f"""You are a support assistant. Answer the user's question
using ONLY the context below. {ambiguity_note}Cite the document ID(s) you
used, like [TCK-1001].

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

Question: "{raw_question}"

Answer:"""

    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def generate_direct_answer(raw_question):
    """No retrieval needed - answer directly (e.g. greetings, chit-chat, opinion questions)."""
    prompt = f"""Reply to this support chat message briefly.

If it is a greeting, thanks or small talk, reply politely.
If it asks for ANY information (products, what the store sells, prices,
policies, orders), you have no documentation to answer from, so respond
with EXACTLY this sentence and nothing else:
"{NO_ANSWER}"

Message: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


#=======================================
# ORCHESTRATOR: Full Production Pipeline
#=======================================

def production_pipeline(raw_question):
    print(f"\n{'='*70}\nQUESTION: \"{raw_question}\"\n{'='*70}")

    # Stage 1 — always: route
    decision = route_query(raw_question)
    print(f"Routing -> needs_retrieval={decision.needs_retrieval}, "
          f"category={decision.category}, clarity={decision.query_clarity}")

    if not decision.needs_retrieval:
        print("No retrieval needed - answering directly.")
        answer = generate_direct_answer(raw_question)
        print(f"\nGenerated Answer:\n{answer}")
        return None, answer

    # Only trust the category filter when the query is unambiguous.
    safe_category = decision.category if decision.query_clarity == "clear" else None
    is_ambiguous = decision.query_clarity == "ambiguous"

    # Stage 2 — always: hybrid search (filtered only if safe_category is set)
    results = hybrid_search(raw_question, category=safe_category)
    best_score = results[0][2] if results else float("-inf")

    # Stage 3 — conditional escalation
    if is_ambiguous:
        print("-> Ambiguous query: escalating to Multi-Query Retrieval (no category filter)")
        variants, results = multi_query_search(raw_question, category=safe_category)
        for i, v in enumerate(variants, start=1):
            print(f"   Variant {i}: \"{v}\"")

    elif decision.query_clarity == "vague" or best_score < CONFIDENCE_THRESHOLD:
        print("-> Vague query or low confidence: rewriting and re-searching (no category filter)")
        rewritten = rewrite_query(raw_question)
        print(f"   Rewritten: \"{rewritten}\"")
        results = hybrid_search(rewritten, category=safe_category)

    else:
        print("-> Clear query: using baseline hybrid search result (category filter applied)")

    print(f"\nFinal Top-{TOP_K}:")
    for rank, (source_id, title, score) in enumerate(results, start=1):
        print(f"  Rank {rank}: {source_id} | score={score:.4f} | {title}")

    # Stage 4 — always: generate the grounded final answer
    answer = generate_answer(raw_question, results, is_ambiguous=is_ambiguous)
    print(f"\nGenerated Answer:\n{answer}")

    return results, answer

#=======================================
# Test
#=======================================

test_cases = {
    "Can I return an opened laptop on day 8 if I have the original box and all accessories?": {
        "RR-001": "Return window for all products",
        "RR-002": "Returning opened products",
    },
    "My Galaxy Buds3 Pro left earbud won't connect. Isthis a known issue, and has the same problem been resolved for another customer? What fixed it?": {
        "KI-001": "Galaxy Buds3 Pro: one earbud not connecting",
        "TCK-1001": "Left earbud of Galaxy Buds3 Pro not connecting"
    },
    "How long does delivery take to a remote pincode, and is it free?": {
        "SH-002": "Delivery to remote pincodes",
    },
    "A customer wants to change the address after the order shipped. What is the official process, and dowe have a resolved case like this?": {
        "OG-002": "How to change the delivery address",
        "TCK-1004": "Change address after order shipped"
    },
    "What does policy RR-004 say?": {
        "RR-004": "Refund timelines by payment method"
    },
    "A customer received a damaged phone and wants arefund instead of a replacement. What is the official process, and do we have a previous resolved case where a refund was issued?" : {
        "RR-003": "Damaged or defective products",
        "RR-004": "Refund timelines by payment method",
        "RR-005": "Replacement vs refund rules",
        "OG-005": "How to report a damaged, defective or wrong item",
        "TCK-1002": "iPhone 16 arrived with a cracked screen, customer wants refund",
    },
    "Do you sell refrigerators?": {
        "none": "none",
    },
}

# Runs only when this file is executed directly, not when the evaluation
# script (evaluation/1_run_versions.py) imports production_pipeline from it.
if __name__ == "__main__":
    for question, relevant_tickets in test_cases.items():
        print(f"\nRelevant tickets (any of these count as a good result): {list(relevant_tickets.keys())}")

        results, answer = production_pipeline(question)

        # Out-of-scope question: no document is relevant, the right result is a refusal
        if list(relevant_tickets.keys()) == ["none"]:
            refused = answer.strip() == NO_ANSWER
            print(f"\nExpected refusal: {'PASS' if refused else 'FAIL'}")
            continue

        found = []
        if results:
            for source_id, title, score in results:
                if source_id in relevant_tickets:
                    found.append(source_id)

        print(f"\nCoverage: {len(found)} / {len(relevant_tickets)} relevant tickets found -> {found}")