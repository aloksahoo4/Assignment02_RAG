import contextlib
import importlib.util
import io
import json
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Step 2 of the RAG evaluation: run all 20 golden questions through all 3
# versions and save, for every question: the answer, the retrieved document
# IDs, the sources and the routing decision.
#
# The 3 versions are the real pipeline files, imported from their folders:
#   V1 (basic)                  v1_basic_rag/2_online_pipeline.py
#   V2 (single index)           v2_advanced_rag/5_v2_pipeline.py
#   V2 (multi-index + routing)  v2_multi_index/4_v2_multi_index_pipeline.py

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EVAL_DIR)                     # customer_support_copilot/
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

TOP_K = 5   # K = 5 everywhere

#=======================================
# Same model for every version
#=======================================
# The pipeline files were written with different chat models and token limits.
# For a fair comparison every version answers with the SAME model and the SAME
# token budget, so a difference in the scores comes from retrieval and routing,
# not from the LLM.
CHAT_MODEL = "gpt-6-luna"
MAX_TOKENS = 800

shared_llm = ChatOpenAI(
    api_key=as_api_key,
    model=CHAT_MODEL,
    max_tokens=MAX_TOKENS
)


#=======================================
# Load the golden test set and the knowledge base
#=======================================

with open(os.path.join(EVAL_DIR, "golden_test_set.json"), "r", encoding="utf-8") as f:
    golden_test_set = json.load(f)

with open(os.path.join(BASE_DIR, "data", "dandes_electronics_kb.json"), "r", encoding="utf-8") as f:
    kb_documents = {doc["id"]: doc for doc in json.load(f)["documents"]}

print(f"Golden test set: {len(golden_test_set)} questions")


def build_contexts(retrieved_ids):
    """The text of each retrieved document, in one format for every version.

    3_generation_metrics.py checks the answers against these contexts.
    """
    contexts = []
    for source_id in retrieved_ids:
        doc = kb_documents[source_id]
        contexts.append(
            f"[{source_id}] {doc['title']} (date: {doc['metadata']['date']})\n{doc['content']}"
        )
    return contexts


def unique_top_k(source_ids, k=TOP_K):
    """Unique document IDs, in rank order, cut to the top k."""
    unique_ids = []
    for source_id in source_ids:
        if source_id not in unique_ids:
            unique_ids.append(source_id)
    return unique_ids[:k]


def cited_sources(answer, retrieved_ids):
    """The retrieved documents that the answer cites by ID."""
    return [source_id for source_id in retrieved_ids if source_id in answer]


#=======================================
# Load the 3 pipeline files
#=======================================

def load_pipeline(module_name, relative_path):
    """Import a pipeline file by its path (its name starts with a digit)."""
    path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with contextlib.redirect_stdout(io.StringIO()):   # hide the start-up prints
        spec.loader.exec_module(module)
    return module


print("\nLoading V1 (basic)...")
v1 = load_pipeline("v1_basic", os.path.join("v1_basic_rag", "2_online_pipeline.py"))
v1.llm_brain = shared_llm

print("Loading V2 (single index)...")
v2_single = load_pipeline("v2_single_index", os.path.join("v2_advanced_rag", "5_v2_pipeline.py"))
v2_single.llm_brain = shared_llm
v2_single.router_llm = shared_llm.with_structured_output(v2_single.RouteDecision)

print("Loading V2 (multi-index + routing)...")
v2_multi = load_pipeline("v2_multi_index", os.path.join("v2_multi_index", "4_v2_multi_index_pipeline.py"))
v2_multi.llm_brain = shared_llm
v2_multi.answer_llm = shared_llm


#=======================================
# One function per version: question -> answer, retrieved IDs, sources, routing
#=======================================

def run_v1(question):
    output = v1.run_pipeline(question)

    # V1 returns its sources as "ID — Title" strings, one per retrieved document
    retrieved_ids = unique_top_k([source.split(" — ")[0] for source in output["sources"]])

    return {
        "answer": output["answer"].strip(),
        "retrieved_ids": retrieved_ids,
        "sources": retrieved_ids,   # V1 lists every retrieved document as a source
        "routing": None,            # V1 has no router
    }


def run_v2_single(question):
    results, answer = v2_single.production_pipeline(question)

    # results is None when the pipeline answered without retrieval
    retrieved_ids = unique_top_k([source_id for source_id, title, score in results]) if results else []

    return {
        "answer": answer.strip(),
        "retrieved_ids": retrieved_ids,
        "sources": cited_sources(answer, retrieved_ids),
        "routing": None,            # one index only: there is no index routing
    }


def run_v2_multi(question):
    output = v2_multi.multi_index_pipeline(question)

    retrieved_ids = unique_top_k([r["source_id"] for r in output["results"]])

    return {
        "answer": output["answer"].strip(),
        "retrieved_ids": retrieved_ids,
        "sources": output["sources"],
        "routing": {
            "intents": output["intents"],
            "indexes": output["indexes"],
            "reason": output["reason"],
        },
    }


VERSIONS = [
    {"name": "V1 (basic)", "run": run_v1, "file": "v1_results.json"},
    {"name": "V2 (single index)", "run": run_v2_single, "file": "v2_single_index_results.json"},
    {"name": "V2 (multi-index + routing)", "run": run_v2_multi, "file": "v2_multi_index_results.json"},
]


#=======================================
# Run all 20 questions through all 3 versions and save the outputs
#=======================================

os.makedirs(RESULTS_DIR, exist_ok=True)

for version in VERSIONS:
    print("\n" + "=" * 70)
    print(f"Running {version['name']}")
    print("=" * 70)

    results = []
    for test in golden_test_set:
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # hide the pipeline's own prints
                output = version["run"](test["question"])
            error = None
        except Exception as e:   # one failed question must not lose the whole run
            output = {"answer": "", "retrieved_ids": [], "sources": [], "routing": None}
            error = f"{type(e).__name__}: {e}"

        results.append({
            "id": test["id"],
            "question": test["question"],
            "answer": output["answer"],
            "retrieved_ids": output["retrieved_ids"],
            "sources": output["sources"],
            "routing": output["routing"],
            "contexts": build_contexts(output["retrieved_ids"]),
            "error": error,
        })

        status = f"ERROR: {error}" if error else f"retrieved={output['retrieved_ids']}"
        print(f"{test['id']} | {status}")

    results_path = os.path.join(RESULTS_DIR, version["file"])
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"version": version["name"], "model": CHAT_MODEL, "top_k": TOP_K, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved: {os.path.join('evaluation', 'results', version['file'])}")
