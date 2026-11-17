import json
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

#=======================================
# Multi-Index Design
#=======================================
# One Chroma collection per kind of knowledge. This map is the single source
# of truth for which doc_type goes into which index.
INDEX_DOC_TYPES = {
    "policy_index": ["return_policy", "shipping_policy"],   # official rules, must be exact
    "product_index": ["product_faq", "known_issue"],        # organised by product name and model
    "order_index": ["order_guide"],                         # step-by-step process guides
    "ticket_index": ["resolved_ticket"],                    # past cases, "problem -> fix" format
}

# Record counts expected per index, used to verify the build.
EXPECTED_COUNTS = {
    "policy_index": 10,
    "product_index": 20,
    "order_index": 6,
    "ticket_index": 14,
}

def tokenize(text):
    # "[a-z]+-\d+" keeps a document ID like "RR-004" as ONE token ("rr-004")
    # instead of splitting it into "rr" and "004".
    return re.findall(r"[a-z]+-\d+|[a-z0-9]+", text.lower())


# Step 1: Load raw KB records from the JSON file as a list of dicts.
def load_records(path="../data/dandes_electronics_kb.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


# Step 2: Convert raw record dicts into LangChain Document objects.
def load_as_documents(path="../data/dandes_electronics_kb.json"):
    records = load_records(path)
    documents = []
    for record in records:
        metadata = {
            "source_id": record["id"],
            "title": record["title"],
            "doc_type": record["doc_type"],
            "category": record["metadata"]["category"],
            "product": record["metadata"]["product"],
            "brand": record["metadata"]["brand"],
            "date": record["metadata"]["date"],
        }
        # Optional fields: only set when present (Chroma rejects None values)
        for key in ("policy_code", "related_policy"):
            value = record["metadata"].get(key)
            if value is not None:
                metadata[key] = value

        documents.append(Document(page_content=record["content"], metadata=metadata))

    return documents


# Step 3: Split the documents into one list per index, based on doc_type.
def split_by_index(documents):
    doc_type_to_index = {
        doc_type: index_name
        for index_name, doc_types in INDEX_DOC_TYPES.items()
        for doc_type in doc_types
    }

    index_documents = {index_name: [] for index_name in INDEX_DOC_TYPES}
    for doc in documents:
        doc_type = doc.metadata["doc_type"]
        if doc_type not in doc_type_to_index:
            # Fail loudly so no record is silently left out of every index
            raise ValueError(
                f"Record {doc.metadata['source_id']} has unknown doc_type '{doc_type}'"
            )
        index_documents[doc_type_to_index[doc_type]].append(doc)

    return index_documents


# No splitting: each KB record is short and self-contained, so one record = one chunk.
documents = load_as_documents()
print(f"Total documents loaded: {len(documents)}")

index_documents = split_by_index(documents)
print("\nDocuments per index:")
for index_name, docs in index_documents.items():
    print(f"- {index_name}: {len(docs)} ({', '.join(INDEX_DOC_TYPES[index_name])})")

# Step 4: Set up the embedding model
load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

embeddings_model = OpenAIEmbeddings(
    api_key=as_api_key,
    model="text-embedding-3-small"
)
print("\nLangChain OpenAIEmbeddings configured: text-embedding-3-small")

#=======================================
# Build the 4 indexes
#=======================================

# Step 5: Create one persistent Chroma collection per index.
# All collections share the same Chroma directory; explicit ids mean a re-run
# updates the existing records instead of adding duplicates.
vectorstores = {}
for index_name, docs in index_documents.items():
    vectorstores[index_name] = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=index_name,
        ids=[d.metadata["source_id"] for d in docs],
    )

print(f"\nChroma collections created at: {CHROMA_DB_PATH}")

# Step 6: Build one BM25 (keyword) index per collection.
# BM25 lives in memory and is built from the records stored in the Chroma
# collection, so it always covers exactly the same records as that collection.
# BM25 indexes the ID and title along with the body text, so a question that
# names a document ID (e.g. "What does policy RR-004 say?") can match it.
bm25_indexes = {}
for index_name, vectorstore in vectorstores.items():
    all_data = vectorstore.get()
    chunk_texts = all_data["documents"]
    chunk_metadatas = all_data["metadatas"]

    tokenized_corpus = [
        tokenize(f"{metadata['source_id']} {metadata['title']} {text}")
        for text, metadata in zip(chunk_texts, chunk_metadatas)
    ]
    bm25_indexes[index_name] = {
        "bm25": BM25Okapi(tokenized_corpus),
        "texts": chunk_texts,
        "metadatas": chunk_metadatas,
    }

print("BM25 indexes built: one per collection.")

# Step 7: Verify each index is actually searchable, both by meaning and by keyword
test_queries = {
    "policy_index": "what is the return window for a laptop",
    "product_index": "does the iPhone 16 come with a charger",
    "order_index": "how do I cancel my order",
    "ticket_index": "left earbud not connecting",
}

for index_name, query in test_queries.items():
    print("\n" + "=" * 70)
    print(f"{index_name} | Test query: '{query}'")
    print("=" * 70)

    print("Chroma (vector search) top-3:")
    results = vectorstores[index_name].similarity_search(query, k=3)
    for doc in results:
        print(f"- {doc.metadata['source_id']} | {doc.metadata['doc_type']} "
              f"| {doc.metadata['title']}")

    print("BM25 (keyword search) top-3:")
    bm25_index = bm25_indexes[index_name]
    scores = bm25_index["bm25"].get_scores(tokenize(query))
    ranked = sorted(zip(bm25_index["metadatas"], scores), key=lambda x: x[1], reverse=True)
    for metadata, score in ranked[:3]:
        print(f"- {metadata['source_id']} | {metadata['doc_type']} "
              f"| {metadata['title']}")

# Step 8: Print the document count of each index.
# Each record is one document, so the counts must be 10 / 20 / 6 / 14.
print("\n" + "=" * 70)
print("Document count per index")
print("=" * 70)
mismatches = []
for index_name, vectorstore in vectorstores.items():
    chroma_count = vectorstore._collection.count()
    bm25_count = bm25_indexes[index_name]["bm25"].corpus_size
    print(f"{index_name:<13} : {chroma_count:>2} documents")

    expected = EXPECTED_COUNTS[index_name]
    if chroma_count != expected or bm25_count != expected:
        mismatches.append(
            f"{index_name}: expected {expected}, Chroma has {chroma_count}, BM25 has {bm25_count}"
        )

if mismatches:
    print("\nCOUNT MISMATCH:")
    for mismatch in mismatches:
        print(f"- {mismatch}")
