import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")

llm_brain = ChatOpenAI(
    api_key=as_api_key,
    model="gpt-5.6-luna",
    max_tokens=200
)

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

#=======================================
# Shared Config + Helper Functions
#=======================================

TOP_K = 5

SYSTEM_INSTRUCTIONS = """You are a support assistant for Dandes Electronics.

Answer the user's question using ONLY the context provided below.
Each piece of context includes a [Source: ID - Title] line and [Date: date] before the
body text. Treat the Title as part of the actual content, not just a
label — any specific detail mentioned in the Title applies to the
entire chunk, even if not repeated in the body text.
also consider the date and give information based on the latest date not older information.

If, after considering both the Title and the content, the answer is
still not present, say:
"I don't know based on the available information."

Do not make up information beyond what's in the context. Be concise and helpful.
"""

def format_docs(docs):
    parts = []
    for doc in docs:
        source = doc.metadata["source_id"]
        title = doc.metadata["title"],
        date = doc.metadata["date"],
        parts.append(f"[Source: {source} - {title} - {date}]\n{doc.page_content}")
    return "\n\n".join(parts)


def run_pipeline(user_query):
    # Step 1: User Query
    print(f"\nStep 1 - User Query: {user_query}")

    # Step 2: Send Query to Embedding Model
    print("Step 2 - Query sent to Embedding Model")

    # Step 3: Create Query Vector
    query_vector = embeddings_model.embed_query(user_query)
    print(f"Step 3 - Query Vector created. Length: {len(query_vector)}")

    # Step 4: Search in Vector Store
    print("Step 4 - Query Vector sent to Vector Store for search")

    # Step 5: Retrieve Top-k Relevant Chunks
    retrieved_docs = vectorstore.similarity_search_by_vector(
        embedding=query_vector,
        k=TOP_K
    )
    print(f"Step 5 - Retrieved Top-{TOP_K} chunks:")
    for doc in retrieved_docs:
        print(f"  - {doc.metadata['source_id']} | {doc.metadata['title']}")

    # Step 6: Build Augmented Prompt
    context = format_docs(retrieved_docs)
    augmented_prompt = f"""
        Context:{context}
        Question: {user_query}
    """
    print("Step 6 - Augmented Prompt built (system instructions + context + question)")

    # Step 7: Generate Answer
    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=augmented_prompt)
    ]

    response = llm_brain.invoke(messages)
    generated_answer = response.content

    print(f"\nStep 7 - Generated Answer:")
    print(generated_answer)

    # Step 8: Return Final Answer + Sources
    sources = [f"{doc.metadata['source_id']} — {doc.metadata['title']}" for doc in retrieved_docs]

    print(f"\nStep 8 - Final Response")
    print(f"Answer: {generated_answer}")
    print("Retrieved Sources:")
    for source in sources:
        print(source)

    return {"query": user_query, "answer": generated_answer, "sources": sources}

#=======================================
# Test with a Single Question
#=======================================

# Runs only when this file is executed directly, not when the evaluation
# script (evaluation/1_run_versions.py) imports run_pipeline from it.
if __name__ == "__main__":
    user_query = "How long do UPI refunds take?"

    print("\n" + "=" * 60)
    print(f"QUESTION: {user_query}")
    print("=" * 60)
    run_pipeline(user_query)