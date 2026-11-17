import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Run 1_build_indexes.py first: it creates the 4 Chroma collections that this
# script connects to.

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

embeddings_model = OpenAIEmbeddings(
    api_key=as_api_key,
    model="text-embedding-3-small"
)

INDEX_NAMES = ["policy_index", "product_index", "order_index", "ticket_index"]

PER_INDEX_K = 5   # candidates each selected index contributes to the combined pool
FINAL_K = 5       # results kept after the single rerank


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


indexes = {index_name: load_index(index_name) for index_name in INDEX_NAMES}

print("Connected to indexes:")
for index_name, index in indexes.items():
    print(f"- {index_name}: {index['vectorstore']._collection.count()} records "
          f"(Chroma), {index['bm25'].corpus_size} records (BM25)")

print("\nLoading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.")


#=======================================
# STAGE 1: Search ONE index (hybrid: vector + BM25, fused with RRF)
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

    return [
        {
            "index": index_name,
            "source_id": source_id,
            "title": title,
            "text": text,
            "id_match": source_id in named_ids,
        }
        for source_id, title, text in fused[:k]
    ]


#=======================================
# STAGE 2: Combine the candidates from all selected indexes
#=======================================

def collect_candidates(question, index_names):
    for index_name in index_names:
        if index_name not in indexes:
            raise ValueError(f"Unknown index '{index_name}'. Choose from: {INDEX_NAMES}")

    # The question is embedded ONCE and the same vector is reused for every index.
    question_vector = embeddings_model.embed_query(question)

    candidates = []
    for index_name in index_names:
        candidates.extend(search_index(index_name, question, question_vector))
    return candidates


#=======================================
# STAGE 3: Rerank the combined candidates ONCE
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
# Multi-index search: one chosen index, or several together
#=======================================

def multi_index_search(question, index_names, k=FINAL_K):
    """Search the selected indexes, combine their candidates and rerank once.

    index_names can be one index name (a string) or a list of index names.
    Returns (candidates, results): the combined pool before reranking and
    the final top-k after reranking.
    """
    if isinstance(index_names, str):
        index_names = [index_names]

    candidates = collect_candidates(question, index_names)
    results = rerank(question, candidates)[:k]
    return candidates, results


#=======================================
# Demo: all 4 indexes together vs only order_index + ticket_index
#=======================================

DEMO_QUESTION = (
    "A customer wants to change the address after the order shipped. "
    "What is the official process, and do we have a resolved case like this?"
)


def print_top_results(label, index_names):
    candidates, results = multi_index_search(DEMO_QUESTION, index_names)

    print("\n" + "=" * 70)
    print(f"Top {FINAL_K} results from {label}")
    print(f"Indexes searched: {index_names}")
    print(f"Candidates merged before the global rerank: {len(candidates)}")
    print("=" * 70)
    for rank, r in enumerate(results, start=1):
        print(f"  Rank {rank}: {r['source_id']} | {r['index']} | score={r['score']:.4f} "
              f"| {r['title']}")

    return candidates, results


print("\n" + "#" * 70)
print(f"QUESTION: {DEMO_QUESTION}")
print("#" * 70)

ROUTED_INDEXES = ["order_index", "ticket_index"]

all_candidates, all_results = print_top_results("ALL 4 indexes searched together", INDEX_NAMES)
routed_candidates, routed_results = print_top_results("ONLY order_index + ticket_index", ROUTED_INDEXES)

# One line: which result list is better and why.
# The question asks for an official process (order guides) and a resolved case
# (tickets), so policy and product documents cannot add anything to the answer.
all_ids = [r["source_id"] for r in all_results]
routed_ids = [r["source_id"] for r in routed_results]
off_topic = [r["source_id"] for r in all_results if r["index"] not in ROUTED_INDEXES]

print("\nWhich result list is better and why:")
if all_ids == routed_ids:
    print(f"order_index + ticket_index is better: it returns the same top {FINAL_K} as all 4 "
          f"indexes but reranks only {len(routed_candidates)} candidates instead of "
          f"{len(all_candidates)}, with no policy or product documents in the pool to push "
          f"out a relevant result.")
else:
    print(f"order_index + ticket_index is better: the question needs a process guide and a past "
          f"case, and the all-index list gives {len(off_topic)} of its top {FINAL_K} places to "
          f"policy/product documents {off_topic} that cannot answer it.")

