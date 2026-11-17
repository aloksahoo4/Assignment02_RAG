import json
import os

# Step 3 of the RAG evaluation (retrieval side). Reads the outputs saved by
# 1_run_versions.py and calculates, for each version:
#   Precision@5, Recall@5, Coverage, Routing Accuracy, Refusal Accuracy, False Refusals

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

TOP_K = 5   # K = 5 everywhere

# The exact fixed sentence the copilot must use when it cannot answer
NO_ANSWER = "I don't know based on the available information."

VERSIONS = [
    {"name": "V1 (basic)", "file": "v1_results.json"},
    {"name": "V2 (single index)", "file": "v2_single_index_results.json"},
    {"name": "V2 (multi-index + routing)", "file": "v2_multi_index_results.json"},
]

with open(os.path.join(EVAL_DIR, "golden_test_set.json"), "r", encoding="utf-8") as f:
    golden_test_set = {test["id"]: test for test in json.load(f)}


def normalize(answer):
    # The model sometimes writes a curly apostrophe (don’t) or adds quotes.
    # That is still the fixed sentence, so only the typography is normalised.
    return answer.strip().strip('"').replace("’", "'")


def is_exact_refusal(answer):
    """True only when the whole answer is the exact fixed "I don't know" sentence."""
    return normalize(answer) == NO_ANSWER


def is_refusal(answer):
    """True when the answer refuses: it is, or starts with, the fixed sentence."""
    return normalize(answer).startswith(NO_ANSWER)


def unique_top_k(source_ids, k=TOP_K):
    """Unique document IDs, in rank order, cut to the top k."""
    unique_ids = []
    for source_id in source_ids:
        if source_id not in unique_ids:
            unique_ids.append(source_id)
    return unique_ids[:k]


def average(values):
    return sum(values) / len(values) if values else None


#=======================================
# Metrics for ONE question
#=======================================

def evaluate_question(test, result):
    expected_sources = test["expected_sources"]
    expected_indexes = test["expected_indexes"]
    in_scope = len(expected_sources) > 0   # an out-of-scope question has no expected sources

    top_5 = unique_top_k(result["retrieved_ids"])
    row = {
        "id": test["id"],
        "in_scope": in_scope,
        "multi_source": len(expected_indexes) >= 2,   # needs 2 or more indexes
        "retrieved_ids": top_5,
        "refused": is_refusal(result["answer"]),
        "exact_refusal": is_exact_refusal(result["answer"]),
        "precision": None,
        "recall": None,
        "routing_correct": None,
    }

    # Precision@5 and Recall@5 are calculated on in-scope questions only
    if in_scope:
        relevant_found = [source_id for source_id in top_5 if source_id in expected_sources]
        row["precision"] = len(relevant_found) / TOP_K                 # relevant docs in the top 5 / 5
        row["recall"] = len(relevant_found) / len(expected_sources)    # relevant docs in the top 5 / total expected

    # Routing: only for a version that has a router
    routing = result["routing"]
    if routing is not None:
        if in_scope:
            # Correct = searched ALL expected indexes and at most ONE extra index
            predicted = set(routing["indexes"])
            expected = set(expected_indexes)
            row["routing_correct"] = expected <= predicted and len(predicted - expected) <= 1
        else:
            # Out of scope: correct only if the router returns out_of_scope
            row["routing_correct"] = routing["intents"] == ["out_of_scope"]

    return row


#=======================================
# Metrics for ONE version
#=======================================

def evaluate_version(results):
    rows = [evaluate_question(golden_test_set[result["id"]], result) for result in results]

    in_scope_rows = [row for row in rows if row["in_scope"]]
    out_of_scope_rows = [row for row in rows if not row["in_scope"]]
    multi_source_rows = [row for row in in_scope_rows if row["multi_source"]]
    routed_rows = [row for row in rows if row["routing_correct"] is not None]

    summary = {
        "precision_at_5": average([row["precision"] for row in in_scope_rows]),
        "recall_at_5": average([row["recall"] for row in in_scope_rows]),
        # Coverage: multi-source questions only, expected sources found / expected sources needed
        "coverage": average([row["recall"] for row in multi_source_rows]),
        # % of questions routed correctly (None when the version has no router)
        "routing_accuracy": average([1.0 if row["routing_correct"] else 0.0 for row in routed_rows]),
        # % of out-of-scope questions answered with the exact fixed sentence
        "refusal_accuracy": average([1.0 if row["exact_refusal"] else 0.0 for row in out_of_scope_rows]),
        # Number of in-scope questions wrongly answered with "I don't know"
        # (the answer is, or starts with, the fixed sentence)
        "false_refusals": sum(1 for row in in_scope_rows if row["refused"]),
        "questions": len(rows),
        "in_scope_questions": len(in_scope_rows),
        "out_of_scope_questions": len(out_of_scope_rows),
        "multi_source_questions": len(multi_source_rows),
    }
    return summary, rows


def show(value, as_percent=False):
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%" if as_percent else f"{value:.2f}"


#=======================================
# Run for every version, print and save
#=======================================

all_metrics = {}

for version in VERSIONS:
    with open(os.path.join(RESULTS_DIR, version["file"]), "r", encoding="utf-8") as f:
        results = json.load(f)["results"]

    summary, rows = evaluate_version(results)
    all_metrics[version["name"]] = {"summary": summary, "questions": rows}

    print("\n" + "=" * 70)
    print(version["name"])
    print("=" * 70)
    print(f"{'ID':<5}{'Scope':<8}{'P@5':<7}{'R@5':<7}{'Routing':<10}{'Refused':<9}Top 5")
    for row in rows:
        scope = "in" if row["in_scope"] else "out"
        routing = "n/a" if row["routing_correct"] is None else ("ok" if row["routing_correct"] else "WRONG")
        refused = "exact" if row["exact_refusal"] else ("yes" if row["refused"] else "no")
        print(f"{row['id']:<5}{scope:<8}{show(row['precision']):<7}{show(row['recall']):<7}"
              f"{routing:<10}{refused:<9}{', '.join(row['retrieved_ids']) or 'none'}")

    print(f"\nPrecision@5      : {show(summary['precision_at_5'])}   ({summary['in_scope_questions']} in-scope questions)")
    print(f"Recall@5         : {show(summary['recall_at_5'])}")
    print(f"Coverage         : {show(summary['coverage'])}   ({summary['multi_source_questions']} multi-source questions)")
    print(f"Routing Accuracy : {show(summary['routing_accuracy'], as_percent=True)}")
    print(f"Refusal Accuracy : {show(summary['refusal_accuracy'], as_percent=True)}   "
          f"({summary['out_of_scope_questions']} out-of-scope questions)")
    print(f"False Refusals   : {summary['false_refusals']}")

metrics_path = os.path.join(RESULTS_DIR, "retrieval_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nSaved: {os.path.join('evaluation', 'results', 'retrieval_metrics.json')}")
