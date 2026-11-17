import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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
    max_tokens=200,
)

TOP_K = 5

print(f"Connected to vector store. Items indexed: {vectorstore._collection.count()}")

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

TEST_SET = [
    {"question": "Can the Sony WH-1000XM5 stay connected to my laptop and phone together?",
     "expected_sources": ["FAQ-010"]},
    {"question": "Will Galaxy Buds3 Pro work with an iPhone?",
     "expected_sources": ["FAQ-011"]},
    {"question": "Can I fast-charge my iPhone 16 with the Anker 20W charger?",
     "expected_sources": ["FAQ-012"]},
     {"question": "Which cities have express delivery and what does it cost?",
     "expected_sources": ["SH-003"]},
    {"question": "What is the maximum order value for Cash on Delivery?",
     "expected_sources": ["SH-004"]},
    {"question": "What happens if all 3 delivery attempts fail?",
     "expected_sources": ["SH-005"]},
     {"question": "How do I claim warranty once the return window is over?",
     "expected_sources": ["OG-006"]},
     {"question": "My Logitech MX Master 3S cursor lags over Bluetooth.",
     "expected_sources": ["KI-007"]},
    {"question": "Why was my OnePlus 13 exchange value reduced at pickup?",
     "expected_sources": ["KI-008"]},
     {"question": "Customer got a Silver MacBook Air M3 instead of Midnight. What was done?",
     "expected_sources": ["TCK-1007", "OG-005", "RR-005"]},
    {"question": "A Galaxy S25 replacement is also defective. What does the customer get?",
     "expected_sources": ["TCK-1011", "RR-005"]},
    {"question": "Why can't a Bengaluru customer choose express delivery on a COD order?",
     "expected_sources": ["TCK-1012", "SH-003"]},
    {"question": "A laptop return was rejected because the charger was missing. Why?",
     "expected_sources": ["TCK-1014", "RR-002"]},
     {"question": "Can I change my address after shipping, and what happens to the refund if I refuse the delivery?",
     "expected_sources": ["OG-002", "SH-005", "TCK-1004"]},
    {"question": "I cancelled a prepaid order before shipping. How and when will I be refunded?",
     "expected_sources": ["OG-001", "RR-004"]},
    {"question": "My Galaxy Buds3 Pro earbud still doesn't connect after a reset. What are my options?",
     "expected_sources": ["KI-001", "RR-003", "OG-006"]},
    {"question": "Who repairs a Lenovo ThinkPad E14 under warranty and how is the warranty registered?",
     "expected_sources": ["FAQ-008", "OG-006"]},
     {"question": "Can I return an opened laptop after a week?",
     "expected_sources": ["RR-001", "RR-002"]},
     {"question": "What is the price of the Lenovo ThinkPad E14?",
     "expected_sources": []},
    {"question": "Do you sell the iPhone 17?",
     "expected_sources": []},
    {"question": "Do you offer a student discount?",
     "expected_sources": []},
]

#=======================================
# Step 2: Retrieval Evaluation — Precision@k, Recall@k, F1
#=======================================

#=======================================
# Step 2: Retrieval Evaluation — Precision@k, Recall@k, F1
#=======================================


def evaluate_retrieval(question, expected_sources, k=TOP_K):
    docs = vectorstore.similarity_search(question, k=k)
    retrieved_sources = set(doc.metadata["source_id"] for doc in docs)
    expected_set = set(expected_sources)

    if len(expected_set) == 0:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "retrieved_sources": list(retrieved_sources),
            "docs": docs
        }

    true_positives = len(retrieved_sources & expected_set)
    precision = true_positives / len(retrieved_sources) if retrieved_sources else 0
    recall = true_positives / len(expected_set) if expected_set else 0
    f1 = 0 if (precision + recall == 0) else 2 * (precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "retrieved_sources": list(retrieved_sources),
        "docs": docs
    }

#=======================================
# Step 3: Generate an Answer (reusing the online pipeline logic)
#=======================================

def generate_answer(question, docs):
    context = format_docs(docs)
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=augmented_prompt)
    ]
    response = llm_brain.invoke(messages)
    return response.content

#=======================================
# Step 4: Faithfulness Judge (LLM-as-judge)
#=======================================
# NOTE, this reuses llm_brain as the judge
# — same model that generated the answer.

def judge_faithfulness(context, generated_answer):
    if "I don't know based on the" in generated_answer.lower():
        return "N/A (refusal)"

    judge_prompt = f"""Context:
{context}

Generated answer: {generated_answer}

Is every claim in the generated answer directly supported by the context
above? Respond with ONLY one word: Faithful, Partial, or Unfaithful.
"""
    response = llm_brain.invoke([HumanMessage(content=judge_prompt)])
    return response.content.strip()

#=======================================
# Step 5: Refusal Accuracy Check
#=======================================

def check_refusal_accuracy(expected_sources, generated_answer):
    is_refusal = "I don't know based on the" in generated_answer.lower()
    should_refuse = len(expected_sources) == 0

    if should_refuse and is_refusal:
        return "Correct refusal"
    elif should_refuse and not is_refusal:
        return "Should have refused but didn't (hallucination risk)"
    elif not should_refuse and is_refusal:
        return "False refusal (info existed but model refused)"
    else:
        return "Correctly answered (no refusal needed)"


#=======================================
# Step 6: Run the Full Evaluation Suite
#=======================================

def run_evaluation():
    results = []

    for test_case in TEST_SET:
        question = test_case["question"]
        expected_sources = test_case["expected_sources"]

        retrieval_metrics = evaluate_retrieval(question, expected_sources)
        docs = retrieval_metrics["docs"]  # reuse — no second similarity_search call

        answer = generate_answer(question, docs)
        context = format_docs(docs)

        faithfulness = judge_faithfulness(context, answer)
        refusal_check = check_refusal_accuracy(expected_sources, answer)

        results.append({
            "question": question,
            "answer": answer,
            "precision": retrieval_metrics["precision"],
            "recall": retrieval_metrics["recall"],
            "f1": retrieval_metrics["f1"],
            "retrieved_sources": retrieval_metrics["retrieved_sources"],
            "faithfulness": faithfulness,
            "refusal_check": refusal_check
        })

    return results

#=======================================
# Step 7: Print Summary + Failures
#=======================================

eval_results = run_evaluation()

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

for r in eval_results:
    print(f"\nQ: {r['question']}")
    print(f"  Answer: {r['answer']}")
    print(f"  Retrieved Sources: {r['retrieved_sources']}")
    print(f"  Precision: {r['precision']} | Recall: {r['recall']} | F1: {r['f1']}")
    print(f"  Faithfulness: {r['faithfulness']}")
    print(f"  Refusal Check: {r['refusal_check']}")

valid_precisions = [r["precision"] for r in eval_results if r["precision"] is not None]
valid_recalls = [r["recall"] for r in eval_results if r["recall"] is not None]
valid_f1s = [r["f1"] for r in eval_results if r["f1"] is not None]

print("\n" + "=" * 70)
print("AGGREGATE METRICS")
print("=" * 70)
print(f"Avg Precision: {sum(valid_precisions)/len(valid_precisions):.2f}")
print(f"Avg Recall   : {sum(valid_recalls)/len(valid_recalls):.2f}")
print(f"Avg F1       : {sum(valid_f1s)/len(valid_f1s):.2f}")

print("\n" + "=" * 70)
print("FAILURES ONLY (needs human review)")
print("=" * 70)

for r in eval_results:

    is_failure = (
        r["faithfulness"] in ["Unfaithful", "Partial"] or
        "should have" in r["refusal_check"].lower() or
        "false refusal" in r["refusal_check"].lower() or
        (r["recall"] is not None and r["recall"] < 1.0)
    )

    if is_failure:
        print(f"\nFLAGGED: {r['question']}")
        print(f"  Answer: {r['answer']}")
        print(f"  Faithfulness: {r['faithfulness']} | Refusal Check: {r['refusal_check']}")