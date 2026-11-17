import csv
import json
import os

# Step 4 of the RAG evaluation: put the metrics of all 3 versions side by side
# and save the table as evaluation/comparison_table.csv.
#
# Run these first:
#   1_run_versions.py -> 2_retrieval_metrics.py -> 3_generation_metrics.py

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

VERSION_NAMES = ["V1 (basic)", "V2 (single index)", "V2 (multi-index + routing)"]

with open(os.path.join(RESULTS_DIR, "retrieval_metrics.json"), "r", encoding="utf-8") as f:
    retrieval_metrics = json.load(f)

with open(os.path.join(RESULTS_DIR, "generation_metrics.json"), "r", encoding="utf-8") as f:
    generation_metrics = json.load(f)


def score(value):
    return "n/a" if value is None else f"{value:.2f}"


def percent(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


#=======================================
# Build the comparison table
#=======================================
# Each row: metric name + a function that formats the value for one version.

def retrieval(version):
    return retrieval_metrics[version]["summary"]


def generation(version):
    return generation_metrics[version]["summary"]


ROWS = [
    ("Routing Accuracy", lambda v: percent(retrieval(v)["routing_accuracy"])),
    ("Precision@5", lambda v: score(retrieval(v)["precision_at_5"])),
    ("Recall@5", lambda v: score(retrieval(v)["recall_at_5"])),
    ("Coverage (multi-source)", lambda v: score(retrieval(v)["coverage"])),
    ("Faithfulness (LLM judge)", lambda v: score(generation(v)["faithfulness_judge"])),
    ("Faithfulness (RAGAS)", lambda v: score(generation(v)["faithfulness_ragas"])),
    ("Answer Relevancy", lambda v: score(generation(v)["answer_relevancy"])),
    ("Context Precision / Recall",
     lambda v: f"{score(generation(v)['context_precision'])} / {score(generation(v)['context_recall'])}"),
    ("Refusal Accuracy", lambda v: percent(retrieval(v)["refusal_accuracy"])),
    ("False Refusals", lambda v: str(retrieval(v)["false_refusals"])),
]

table = [["Metric"] + VERSION_NAMES]
for metric_name, format_value in ROWS:
    table.append([metric_name] + [format_value(version) for version in VERSION_NAMES])

#=======================================
# Print and save
#=======================================

column_widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]

print("\n" + "=" * 100)
print("COMPARISON TABLE")
print("=" * 100)
for row_number, row in enumerate(table):
    print("  ".join(cell.ljust(width) for cell, width in zip(row, column_widths)))
    if row_number == 0:
        print("  ".join("-" * width for width in column_widths))

csv_path = os.path.join(EVAL_DIR, "comparison_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(table)

print(f"\nSaved: {os.path.join('evaluation', 'comparison_table.csv')}")

#=======================================
# Minimum targets for V2 (multi-index + routing)
#=======================================

multi_retrieval = retrieval("V2 (multi-index + routing)")
multi_generation = generation("V2 (multi-index + routing)")


def at_least(value, target):
    return value is not None and value >= target


TARGETS = [
    ("Routing Accuracy", ">= 85%", percent(multi_retrieval["routing_accuracy"]),
     at_least(multi_retrieval["routing_accuracy"], 0.85)),
    ("Recall@5", ">= 0.80", score(multi_retrieval["recall_at_5"]),
     at_least(multi_retrieval["recall_at_5"], 0.80)),
    ("Faithfulness (LLM judge)", ">= 0.85", score(multi_generation["faithfulness_judge"]),
     at_least(multi_generation["faithfulness_judge"], 0.85)),
    ("Faithfulness (RAGAS)", ">= 0.85", score(multi_generation["faithfulness_ragas"]),
     at_least(multi_generation["faithfulness_ragas"], 0.85)),
    ("Refusal Accuracy", "100%", percent(multi_retrieval["refusal_accuracy"]),
     at_least(multi_retrieval["refusal_accuracy"], 1.0)),
    ("False Refusals", "0", str(multi_retrieval["false_refusals"]),
     multi_retrieval["false_refusals"] == 0),
]

print("\n" + "=" * 100)
print("MINIMUM TARGETS - V2 (multi-index + routing)")
print("=" * 100)
print(f"{'Metric':<28}{'Target':<12}{'Actual':<10}Result")
for metric_name, target, actual, met in TARGETS:
    print(f"{metric_name:<28}{target:<12}{actual:<10}{'MET' if met else 'MISSED'}")
