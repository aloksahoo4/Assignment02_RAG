import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
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

print(f"Connected to vector store. Items indexed: {vectorstore._collection.count()}")

all_data = vectorstore.get()
chunk_texts = all_data["documents"]
chunk_metadatas = all_data["metadatas"]

def tokenize(text):
    return re.findall(r"\w+", text.lower())

tokenized_corpus = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index built.")

print("\nLoading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.\n")

TOP_K = 3
RETRIEVE_K = 10

def dense_search(question, k=TOP_K):
    return vectorstore.similarity_search_with_score(question, k=k)


def sparse_search(question, k=TOP_K):
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(zip(chunk_texts, chunk_metadatas, scores))
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    return scored_chunks[:k]

#=======================================
# SOLUTION: Hybrid Search with Reciprocal Rank Fusion (RRF)
#=======================================
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    doc_info = {}  # source_id -> (title, text)

    for rank, (doc, score) in enumerate(dense_results, start=1):
        source_id = doc.metadata["source_id"]
        title = doc.metadata["title"]
        text = doc.page_content
        rrf_score = 1 / (k + rank)
        rrf_scores[source_id] = rrf_scores.get(source_id, 0) + rrf_score
        doc_info[source_id] = (title, text)

    for rank, (text, metadata, score) in enumerate(sparse_results, start=1):
        source_id = metadata["source_id"]
        title = metadata["title"]
        rrf_score = 1 / (k + rank)
        rrf_scores[source_id] = rrf_scores.get(source_id, 0) + rrf_score
        doc_info[source_id] = (title, text)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(source_id, doc_info[source_id][0], doc_info[source_id][1], score) for source_id, score in fused]


# =======================================
# Stage 3: Reranking the Fused Candidates
# =======================================

def rerank_fused_results(question, fused_results, top_n_to_rerank=6):
    candidates = fused_results[:top_n_to_rerank]

    pairs = [(question, text) for source_id, title, text, rrf_score in candidates]
    ce_scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, ce_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [(source_id, title, ce_score) for (source_id, title, text, rrf_score), ce_score in reranked]


# =======================================
# Test the SAME 2 Questions
# =======================================

test_questions = {
    "Any past case resolved using KI-001?": "TCK-1001",
    "My parcel status has been frozen for days and nobody knows where my notebook is": "TCK-1013",
    "Show me a case resolved as per KI-004": "TCK-1008",
    "Which case used OG-004?": "TCK-1010",
    "I sent the item back but the money for my pay-on-delivery order never came": "TCK-1006"
}

for i, (question, expected) in enumerate(test_questions.items(), start=1):
    print("\n" + "=" * 70)
    print(f"QUESTION {i}: {question}")
    print(f"Expected best source: {expected}")
    print("=" * 70)

    dense_results = dense_search(question)
    sparse_results = sparse_search(question)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)

    print("\nStage 2 - Fused Top-3 (RRF only, before reranking):")
    for rank, (source_id, title, text, score) in enumerate(fused_results[:TOP_K], start=1):
        flag = "  <-- EXPECTED" if source_id == expected else ""
        print(f"  Rank {rank}: {source_id} | rrf_score={score:.5f} | {title}{flag}")

    final_results = rerank_fused_results(question, fused_results)

    print("\nStage 3 - Reranked Top-3 (cross-encoder, final):")
    for rank, (source_id, title, ce_score) in enumerate(final_results[:TOP_K], start=1):
        flag = "  <-- EXPECTED" if source_id == expected else ""
        print(f"  Rank {rank}: {source_id} | ce_score={ce_score:.4f} | {title}{flag}")
