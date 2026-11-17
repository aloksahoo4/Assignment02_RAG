import json
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# Step 1: Load raw support tickets from the JSON file as a list of dicts.
def load_tickets(path="../data/dandes_electronics_kb.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]

# Step 2: Convert raw ticket dicts into LangChain Document objects.
def load_as_documents(path="../data/dandes_electronics_kb.json"):
    tickets = load_tickets(path)
    documents = []
    for ticket in tickets:
        metadata = {
            "source_id": ticket["id"],
            "title": ticket["title"],
            "doc_type": ticket["doc_type"],
            "category": ticket["metadata"]["category"],
            "product": ticket["metadata"]["product"],
            "brand": ticket["metadata"]["brand"],
            "date": ticket["metadata"]["date"],
        }
        # Optional fields: only set when present (Chroma rejects None values)
        for key in ("policy_code", "related_policy"):
            value = ticket["metadata"].get(key)
            if value is not None:
                metadata[key] = value

        documents.append(Document(page_content=ticket["content"], metadata=metadata))

    return documents

# Load all tickets as LangChain Document objects (content + metadata)
documents = load_as_documents()
print(f"Total documents (tickets) loaded: {len(documents)}")

#=======================================
# Part 2 - Chunking
#=======================================
# No splitting: each KB record is short and self-contained, so one record = one chunk.
print(f"Sample metadata: {documents[0].metadata}")

# Step 3: Set up the embedding model

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")
chroma_db_path = os.getenv("CHROMA_DB_PATH")

embeddings_model = OpenAIEmbeddings(
    api_key=as_api_key,
    model="text-embedding-3-small"
)
print("\nLangChain OpenAIEmbeddings configured: text-embedding-3-small")

#=======================================
# Part 4 - Vector Store / Indexing
#=======================================

# Step 4: Create the persistent vector store
CHROMA_DB_PATH = chroma_db_path

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=CHROMA_DB_PATH,
    collection_name="support_tickets",
    ids=[d.metadata["source_id"] for d in documents],
)

print(f"\nVector store created at: {CHROMA_DB_PATH}")
print(f"Items indexed: {vectorstore._collection.count()}")

# Step 5: Verify it's actually searchable
results = vectorstore.similarity_search("what is the maximum cash on delivery limit", k=5)

print("\nTest query results for: 'what is the maximum cash on delivery limit'")
for doc in results:
    print(f"- {doc.metadata['source_id']}: {doc.page_content[:260]}...")
