import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

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

TOP_K = 5

#=======================================
# SOLUTION 1: Explicit Metadata Filtering
#=======================================

print("\n" + "#" * 70)
print("# EXPLICIT METADATA FILTERING")
print("#" * 70)

explicit_tests = [
    {
        "question": "My MacBook Air M3 isn't charging.?",
        "filter": {"category": "charging"}
    },
    {
        "question": "earbud not connecting?",
        "filter": {"category": "connectivity"}
    },
    {
        "question": "return my laptop after a week?",
        "filter": {"category": "returns"}
    }
]

for i, test in enumerate(explicit_tests, start=1):
    print("\n" + "=" * 70)
    print(f"QUESTION {i}: {test['question']}")
    print(f"Filter applied: {test['filter']}")
    print("=" * 70)

    results = vectorstore.similarity_search(
        test["question"],
        k=TOP_K,
        filter=test["filter"]
    )

    for doc in results:
        print(f"  -  {doc.metadata['source_id']} "
              f"| category={doc.metadata['category']}")
        print(f"    Title: {doc.metadata['title']}")