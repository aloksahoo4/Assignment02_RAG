import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
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

RETRIEVE_K = 20   # Stage 1: fast, rough candidates
FINAL_K = 5       # Stage 2: final, reranked results

# Vague questions: no product names, loose wording. Vector search alone
# puts a related-but-wrong document first; the reranker fixes the order.
test_questions = [
    "My money got cut but nothing shows up in my orders",
    "When should I escalate a courier delay?",
    "Replacement was also bad",
    "Can I trade in my old handset?",
    "Laptop says not charging, what charger wattage do I need?"
]

expected_top_source = {
    test_questions[0]: "OG-004",
    test_questions[1]: "OG-003",
    test_questions[2]: "TCK-1011",
    test_questions[3]: "FAQ-004",
    test_questions[4]: "KI-004"
}

#=======================================
# SOLUTION 1: Cross-Encoder Reranking (sentence-transformers)
#=======================================
# Stage 1: fast retrieval gets a rough top-10 candidate list.
# Stage 2: a real cross-encoder model reads (question, chunk)
# pairs TOGETHER and re-scores each one for true relevance.

print("\nLoading cross-encoder model (first run downloads it)...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model loaded.")

print(os.path.expanduser("~/.cache/huggingface/hub"))

print("\n" + "#" * 70)
print("# CROSS-ENCODER RERANKING")
print("#" * 70)

for i, question in enumerate(test_questions, start=1):
    print("\n" + "=" * 70)
    print(f"QUESTION {i}: {question}")
    print(f"Expected best source: {expected_top_source[question]}")
    print("=" * 70)

    # Stage 1: fast retrieval, rough top-20
    candidates = vectorstore.similarity_search(question, k=RETRIEVE_K)

    # Stage 2: cross-encoder scores EACH (question, chunk) pair together
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = cross_encoder.predict(pairs)

    # Sort candidates by cross-encoder score, highest first
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    expected = expected_top_source[question]

    print(f"BEFORE reranking - Top-{FINAL_K} (vector similarity order):")
    for rank, doc in enumerate(candidates[:FINAL_K], start=1):
        is_expected = "  <-- EXPECTED BEST MATCH" if doc.metadata["source_id"] == expected else ""
        print(f"  Rank {rank}: {doc.metadata['source_id']} "
              f"| {doc.metadata['title']}{is_expected}")

    print(f"\nAFTER reranking - Top-{FINAL_K} (by cross-encoder score):")
    for rank, (doc, score) in enumerate(reranked[:FINAL_K], start=1):
        is_expected = "  <-- EXPECTED BEST MATCH" if doc.metadata["source_id"] == expected else ""
        print(f"  Rank {rank}: {doc.metadata['source_id']} "
              f"| score={score:.4f} | {doc.metadata['title']}{is_expected}")

    # Where did the expected source sit before vs after?
    before_ids = [doc.metadata["source_id"] for doc in candidates]
    after_ids = [doc.metadata["source_id"] for doc, _ in reranked]
    before_rank = before_ids.index(expected) + 1 if expected in before_ids else None
    after_rank = after_ids.index(expected) + 1 if expected in after_ids else None
    print(f"\nExpected source {expected}: rank {before_rank} before -> rank {after_rank} after "
          f"(None = not in the top-{RETRIEVE_K} candidates)")