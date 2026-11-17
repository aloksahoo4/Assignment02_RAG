import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
    return re.findall(r"[a-z0-9]+", text.lower())

tokenized_text = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_text)

print("BM25 index built.")

print("\nLoading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.\n")

TOP_K = 3
RETRIEVE_K = 10


def dense_search(question, k=RETRIEVE_K):
    results = vectorstore.similarity_search_with_score(question, k=k)
    return [(doc.metadata["source_id"], doc.metadata["title"], doc.page_content) for doc, score in results]


def sparse_search(question, k=RETRIEVE_K):
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(zip(chunk_texts, chunk_metadatas, scores))
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    top = scored_chunks[:k]
    return [(metadata["source_id"], metadata["title"], text) for text, metadata, score in top]


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    doc_info = {}

    for rank, (source_id, title, text) in enumerate(dense_results, start=1):
        score = 1 / (k + rank)
        rrf_scores[source_id] = rrf_scores.get(source_id, 0) + score
        doc_info[source_id] = (title, text)

    for rank, (source_id, title, text) in enumerate(sparse_results, start=1):
        score = 1 / (k + rank)
        rrf_scores[source_id] = rrf_scores.get(source_id, 0) + score
        doc_info[source_id] = (title, text)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(source_id, doc_info[source_id][0], doc_info[source_id][1], score) for source_id, score in fused]


def rerank_fused_results(question, fused_results, top_n_to_rerank=6):
    candidates = fused_results[:top_n_to_rerank]
    pairs = [(question, text) for source_id, title, text, rrf_score in candidates]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [(source_id, title, ce_score) for (source_id, title, text, rrf_score), ce_score in reranked]


def full_pipeline(question, k=TOP_K):
    dense_results = dense_search(question)
    sparse_results = sparse_search(question)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
    final_results = rerank_fused_results(question, fused_results)
    return final_results[:k]


#=======================================
# SOLUTION: Query Rewriting
#=======================================

def rewrite_query(raw_question):
    prompt = f"""The following is a short, vague question a user typed
into a support chat. Rewrite it into a clear, complete question
that fully expresses what the user probably means. Do not answer
the question — only rewrite it. Respond with ONLY the rewritten
question, nothing else.

Original question: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


#=======================================
# Test the SAME 3 Vague Questions
#=======================================

test_questions = {
    "My money got cut but nothing shows up in my orders": "OG-004",
    "When should I escalate a courier delay?": "OG-003",
    "Replacement was also bad": "TCK-1011",
    "Can I trade in my old handset?": "FAQ-004",
    "Laptop says not charging, what charger wattage do I need?": "KI-004",
    "My order hasn't come yet": "OG-003"
}

for i, (question, expected) in enumerate(test_questions.items(), start=1):
    print("\n" + "=" * 70)
    print(f"QUESTION {i}: \"{question}\"")
    print(f"Expected best source: {expected}")
    print("=" * 70)

    rewritten = rewrite_query(question)
    print(f"Rewritten query: \"{rewritten}\"")

    results = full_pipeline(rewritten)

    print("\nFinal Top-3 (using REWRITTEN query):")
    for rank, (source_id, title, score) in enumerate(results, start=1):
        flag = "  <-- EXPECTED" if source_id == expected else ""
        print(f"  Rank {rank}: {source_id} | score={score:.4f} | {title}{flag}")


#=======================================
# SOLUTION 2: Multi-Query Retrieval
#=======================================
# One rewrite is a single guess at what the user meant.
# Multi-query asks the LLM for 3 different versions of the question,
# retrieves for EACH version, then merges all the results.
# A document missed by one version can still be found by another,
# and a document found by several versions rises to the top.

NUM_VERSIONS = 3


def generate_query_versions(raw_question, n=NUM_VERSIONS):
    prompt = f"""The following is a short, vague question a user typed
into a support chat. Write {n} different versions of this question.
Each version should use different words or look at the problem from
a different angle (the symptom, the policy involved, what the user
wants done). Do not answer the question. Respond with ONLY the {n}
versions, one per line, with no numbering and no extra text.

Original question: "{raw_question}"
"""
    response = llm_brain.invoke([HumanMessage(content=prompt)])

    versions = []
    for line in response.content.splitlines():
        # Strip any numbering/bullets/quotes the LLM added anyway
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"')
        if line and line not in versions:
            versions.append(line)
    versions = versions[:n]

    # Fallback: never retrieve with fewer than n queries
    while len(versions) < n:
        versions.append(raw_question)
    return versions


def merge_ranked_lists(ranked_lists, k=60):
    # Same RRF formula as reciprocal_rank_fusion, but for ANY number of lists.
    # Keying by source_id is what removes duplicates across the lists.
    rrf_scores = {}
    doc_info = {}

    for ranked_list in ranked_lists:
        for rank, (source_id, title, text) in enumerate(ranked_list, start=1):
            rrf_scores[source_id] = rrf_scores.get(source_id, 0) + 1 / (k + rank)
            doc_info[source_id] = (title, text)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(source_id, doc_info[source_id][0], doc_info[source_id][1], score) for source_id, score in merged]


def multi_query_pipeline(question, k=TOP_K):
    # Step 1: LLM writes 3 versions of the question
    versions = generate_query_versions(question)

    # Step 2: dense + sparse search for EACH version (3 x 2 = 6 ranked lists)
    ranked_lists = []
    for version in versions:
        ranked_lists.append(dense_search(version))
        ranked_lists.append(sparse_search(version))

    # Step 3: merge all lists into one de-duplicated candidate list
    merged_results = merge_ranked_lists(ranked_lists)

    # Step 4: rerank against the ORIGINAL question (what the user actually asked).
    # top_n_to_rerank is wider than the default 6 because 3 queries give a bigger pool.
    final_results = rerank_fused_results(question, merged_results, top_n_to_rerank=10)
    return versions, merged_results, final_results[:k]


print("\n\n" + "#" * 70)
print("# MULTI-QUERY RETRIEVAL (3 versions of the question, results merged)")
print("#" * 70)

for i, (question, expected) in enumerate(test_questions.items(), start=1):
    print("\n" + "=" * 70)
    print(f"QUESTION {i}: \"{question}\"")
    print(f"Expected best source: {expected}")
    print("=" * 70)

    versions, merged_results, results = multi_query_pipeline(question)

    print("Generated versions:")
    for n, version in enumerate(versions, start=1):
        print(f"  {n}. {version}")

    print(f"\nMerged candidates: {len(merged_results)} unique documents from {len(versions) * 2} ranked lists")

    print("\nFinal Top-3 (MULTI-QUERY, merged + reranked):")
    for rank, (source_id, title, score) in enumerate(results, start=1):
        flag = "  <-- EXPECTED" if source_id == expected else ""
        print(f"  Rank {rank}: {source_id} | score={score:.4f} | {title}{flag}")