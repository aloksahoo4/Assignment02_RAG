import asyncio
import json
import math
import os
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# Step 3 of the RAG evaluation (generation side). Reads the outputs saved by
# 1_run_versions.py and calculates, for each version:
#   Faithfulness        - own LLM-as-Judge (gpt-6.1-sol) + RAGAS faithfulness
#   Answer Relevancy    - RAGAS answer_relevancy
#   Context Precision   - RAGAS, using the reference answers
#   Context Recall      - RAGAS, using the reference answers
#
# Installation required for this:
# python -m pip install ragas

warnings.filterwarnings("ignore")

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# The same judge model and embedding model score every version
JUDGE_MODEL = "gpt-6.1-sol"
EMBEDDING_MODEL = "text-embedding-3-small"

NO_ANSWER = "I don't know based on the available information."
MAX_PARALLEL_QUESTIONS = 4   # questions scored at the same time

VERSIONS = [
    {"name": "V1 (basic)", "file": "v1_results.json"},
    {"name": "V2 (single index)", "file": "v2_single_index_results.json"},
    {"name": "V2 (multi-index + routing)", "file": "v2_multi_index_results.json"},
]

with open(os.path.join(EVAL_DIR, "golden_test_set.json"), "r", encoding="utf-8") as f:
    golden_test_set = {test["id"]: test for test in json.load(f)}

#=======================================
# Judge models
#=======================================

# Own LLM-as-Judge
judge_llm = ChatOpenAI(
    api_key=as_api_key,
    model=JUDGE_MODEL,
    max_tokens=2000
)

# RAGAS uses the same judge model, through the OpenAI client
openai_client = AsyncOpenAI(api_key=as_api_key)
ragas_llm = llm_factory(JUDGE_MODEL, client=openai_client, max_tokens=4000)
ragas_embeddings = RagasOpenAIEmbeddings(client=openai_client, model=EMBEDDING_MODEL)

ragas_faithfulness = Faithfulness(llm=ragas_llm)
ragas_answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
ragas_context_precision = ContextPrecision(llm=ragas_llm)   # uses the reference answer
ragas_context_recall = ContextRecall(llm=ragas_llm)         # uses the reference answer


#=======================================
# Own LLM-as-Judge: Faithfulness
#=======================================
# The judge splits the answer into claims and checks each claim against the
# retrieved context. Score = supported claims / all claims (0 to 1).

JUDGE_PROMPT = """You are a strict fact-checker for a customer support RAG system.

Context (the ONLY information the answer was allowed to use):
{context}

Question: {question}

Answer to check:
{answer}

Break the answer into its individual factual claims. For each claim, decide
whether the context directly supports it. Document IDs in square brackets,
such as [RR-001], are citations, not claims: ignore them. A claim that goes
beyond the context, or contradicts it, is NOT supported.

Respond with ONLY a JSON object in exactly this format, and no other text:
{{"claims": [{{"claim": "...", "supported": true}}, {{"claim": "...", "supported": false}}]}}
"""


def parse_json(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove a Markdown code fence around the JSON, if the model added one
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[len("json"):]
    return json.loads(text)


async def judge_faithfulness(question, answer, contexts):
    """Return (score, unsupported_claims). The score is None if the judge fails."""
    prompt = JUDGE_PROMPT.format(context="\n\n".join(contexts), question=question, answer=answer)
    try:
        response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        claims = parse_json(response.content)["claims"]
        if not claims:
            return None, []
        supported = [claim for claim in claims if claim["supported"] is True]
        unsupported = [claim["claim"] for claim in claims if claim["supported"] is not True]
        return len(supported) / len(claims), unsupported
    except Exception as error:
        print(f"  Judge failed: {type(error).__name__}: {str(error)[:120]}")
        return None, []


#=======================================
# RAGAS metrics
#=======================================

async def ragas_score(label, metric_call):
    """Run one RAGAS metric. Returns None instead of crashing if it fails."""
    try:
        result = await metric_call
        value = float(result.value)
        return None if math.isnan(value) else value
    except Exception as error:
        print(f"  RAGAS {label} failed: {type(error).__name__}: {str(error)[:120]}")
        return None


#=======================================
# Score ONE question
#=======================================
# Generation metrics are calculated on in-scope questions only. Out-of-scope
# questions have no reference sources: they are measured by Refusal Accuracy
# in 2_retrieval_metrics.py.

def is_refusal(answer):
    """True when the answer refuses: it is, or starts with, the fixed sentence.

    Same rule as False Refusals in 2_retrieval_metrics.py. A curly apostrophe
    (don’t) is still the fixed sentence.
    """
    normalized = answer.strip().strip('"').replace("’", "'")
    return normalized.startswith(NO_ANSWER)


async def score_question(test, result, semaphore):
    question = test["question"]
    reference = test["reference_answer"]
    answer = result["answer"]
    contexts = result["contexts"]

    scores = {
        "id": test["id"],
        "faithfulness_judge": None,
        "faithfulness_ragas": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
        "unsupported_claims": [],
        "note": None,
    }

    async with semaphore:
        if not contexts:
            # Nothing was retrieved for an in-scope question: the needed context is missing
            scores["note"] = "no context retrieved"
            scores["answer_relevancy"] = 0.0
            scores["context_precision"] = 0.0
            scores["context_recall"] = 0.0
            return scores

        # Context metrics only need the question, the contexts and the reference answer
        context_precision_task = ragas_score(
            "context_precision",
            ragas_context_precision.ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        )
        context_recall_task = ragas_score(
            "context_recall",
            ragas_context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference),
        )

        if not answer or is_refusal(answer):
            # A false refusal (or an empty answer) makes no claims, so faithfulness
            # cannot be scored. It does not answer the question: relevancy is 0.
            scores["note"] = "refused or empty answer"
            scores["answer_relevancy"] = 0.0
            scores["context_precision"], scores["context_recall"] = await asyncio.gather(
                context_precision_task, context_recall_task
            )
            return scores

        (
            (scores["faithfulness_judge"], scores["unsupported_claims"]),
            scores["faithfulness_ragas"],
            scores["answer_relevancy"],
            scores["context_precision"],
            scores["context_recall"],
        ) = await asyncio.gather(
            judge_faithfulness(question, answer, contexts),
            ragas_score(
                "faithfulness",
                ragas_faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts),
            ),
            ragas_score(
                "answer_relevancy",
                ragas_answer_relevancy.ascore(user_input=question, response=answer),
            ),
            context_precision_task,
            context_recall_task,
        )
        return scores


#=======================================
# Score ONE version
#=======================================

METRIC_NAMES = [
    "faithfulness_judge",
    "faithfulness_ragas",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def show(value):
    return "n/a" if value is None else f"{value:.2f}"


async def score_version(results):
    semaphore = asyncio.Semaphore(MAX_PARALLEL_QUESTIONS)

    in_scope = [
        (golden_test_set[result["id"]], result)
        for result in results
        if golden_test_set[result["id"]]["expected_sources"]
    ]
    rows = await asyncio.gather(*[score_question(test, result, semaphore) for test, result in in_scope])

    summary = {name: average([row[name] for row in rows]) for name in METRIC_NAMES}
    summary["scored_questions"] = len(rows)
    return summary, list(rows)


async def main():
    all_metrics = {}

    for version in VERSIONS:
        with open(os.path.join(RESULTS_DIR, version["file"]), "r", encoding="utf-8") as f:
            results = json.load(f)["results"]

        print("\n" + "=" * 70)
        print(f"{version['name']} - scoring with {JUDGE_MODEL}...")
        print("=" * 70)

        summary, rows = await score_version(results)
        all_metrics[version["name"]] = {"summary": summary, "questions": rows}

        print(f"{'ID':<5}{'Faith(judge)':<14}{'Faith(RAGAS)':<14}{'AnsRel':<8}{'CtxPrec':<9}{'CtxRec':<8}Note")
        for row in rows:
            print(f"{row['id']:<5}{show(row['faithfulness_judge']):<14}{show(row['faithfulness_ragas']):<14}"
                  f"{show(row['answer_relevancy']):<8}{show(row['context_precision']):<9}"
                  f"{show(row['context_recall']):<8}{row['note'] or ''}")
            for claim in row["unsupported_claims"]:
                print(f"       unsupported claim: {claim}")

        print(f"\nFaithfulness (LLM judge) : {show(summary['faithfulness_judge'])}")
        print(f"Faithfulness (RAGAS)     : {show(summary['faithfulness_ragas'])}")
        print(f"Answer Relevancy         : {show(summary['answer_relevancy'])}")
        print(f"Context Precision        : {show(summary['context_precision'])}")
        print(f"Context Recall           : {show(summary['context_recall'])}")

    metrics_path = os.path.join(RESULTS_DIR, "generation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {os.path.join('evaluation', 'results', 'generation_metrics.json')}")


asyncio.run(main())
