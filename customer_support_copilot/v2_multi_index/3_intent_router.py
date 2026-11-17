import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()
as_api_key = os.getenv("AS_OPENAI_API_KEY")

llm_brain = ChatOpenAI(
    api_key=as_api_key,
    model="gpt-6-luna",
    max_tokens=300
)

#=======================================
# Intents and Their Indexes
#=======================================
# Each intent points to the index that holds that kind of knowledge.
# "out_of_scope" has no index: nothing is searched.
INTENT_TO_INDEX = {
    "policy": "policy_index",        # rules: returns, refunds, delivery, charges, COD
    "product": "product_index",      # product questions and known problems
    "order": "order_index",          # how to do something with an order
    "past_issue": "ticket_index",    # "has this happened before?" and similar past cases
    "out_of_scope": None,            # anything the store doesn't cover
}

ALL_INDEXES = ["policy_index", "product_index", "order_index", "ticket_index"]

ROUTER_PROMPT = """You route questions for the support copilot of Dandes Electronics, an
online store that sells consumer electronics only: phones, laptops, earbuds and
headphones, chargers and computer accessories.

Classify the question into one or more of these intents:
- policy: official store rules - return window, opened or damaged item rules,
  replacement vs refund, refund timelines, delivery times and charges, express
  delivery, Cash on Delivery limits, failed delivery attempts.
- product: a question about a specific product (box contents, specifications,
  compatibility, software, warranty period, EMI or exchange offers) or a known
  problem with a device and how to fix it.
- order: HOW TO do something with an order - cancel it, change the delivery
  address, track it, fix a failed payment, report a damaged, defective or wrong
  item, raise a warranty claim.
- past_issue: the user asks whether this happened before or wants a similar
  resolved case, OR describes a specific problem a customer is facing (a past
  ticket may show how it was fixed).
- out_of_scope: anything the store doesn't cover - a kind of product it does not
  sell (for example refrigerators, furniture or groceries) or a topic that is not
  about the store at all. Use it ALONE, never together with another intent.

Each intent has one index:
policy -> policy_index, product -> product_index, order -> order_index,
past_issue -> ticket_index, out_of_scope -> no index (empty list).

Pick every intent that is needed to answer the full question, and no extra ones.
Example: "Galaxy Buds3 Pro left earbud won't connect" is both product (a known
device problem) and past_issue (a similar case may have been resolved before).

Respond with ONLY a JSON object in exactly this format, and no other text:
{{"intents": ["product", "past_issue"], "indexes": ["product_index", "ticket_index"], "reason": "Question is about a product problem that may have happened before"}}

Question: "{question}"
"""


#=======================================
# Intent-Based Query Routing (JSON output + fallback)
#=======================================

def parse_router_output(raw_text):
    """Turn the router's JSON text into a routing decision. Raises if it is not valid."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove a Markdown code fence around the JSON, if the model added one
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[len("json"):]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("router output is not a JSON object")

    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("'intents' must be a non-empty list")

    unknown = [intent for intent in intents if intent not in INTENT_TO_INDEX]
    if unknown:
        raise ValueError(f"unknown intents: {unknown}")

    # Keep each intent once, in the fixed order of INTENT_TO_INDEX.
    # out_of_scope only counts when it is the ONLY intent.
    intents = [intent for intent in INTENT_TO_INDEX if intent in intents]
    in_scope = [intent for intent in intents if intent != "out_of_scope"]
    intents = in_scope if in_scope else ["out_of_scope"]

    # The indexes are always rebuilt from the intents, so the two can never disagree.
    indexes = [INTENT_TO_INDEX[intent] for intent in intents if INTENT_TO_INDEX[intent]]

    return {"intents": intents, "indexes": indexes, "reason": str(data.get("reason", ""))}


def fallback_decision(error):
    """Used when the router fails: search all 4 indexes instead of crashing."""
    return {
        "intents": [intent for intent in INTENT_TO_INDEX if INTENT_TO_INDEX[intent]],
        "indexes": list(ALL_INDEXES),
        "reason": f"Fallback: router failed ({type(error).__name__}: {error}), searching all 4 indexes",
    }


def decision_from_router_output(raw_text):
    try:
        return parse_router_output(raw_text)
    except Exception as error:   # invalid JSON or invalid content
        return fallback_decision(error)


def route_question(question):
    """Return the routing decision: {"intents": [...], "indexes": [...], "reason": "..."}.

    "indexes" is empty for an out-of-scope question. This function never raises:
    if the LLM call fails or the output is not valid JSON, it returns all 4 indexes.
    """
    try:
        response = llm_brain.invoke([HumanMessage(content=ROUTER_PROMPT.format(question=question))])
        raw_text = response.content
    except Exception as error:   # API error, timeout, ...
        return fallback_decision(error)

    return decision_from_router_output(raw_text)


#=======================================
# Router test: the 7 test questions, predicted vs expected indexes
#=======================================

test_questions = [
    {
        "question": "Can I return an opened laptop on day 8 if I have the original box and all accessories?",
        "expected_indexes": ["policy_index"],
    },
    {
        "question": "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has "
                    "the same problem been resolved for another customer? What fixed it?",
        "expected_indexes": ["product_index", "ticket_index"],
    },
    {
        "question": "How long does delivery take to a remote pincode, and is it free?",
        "expected_indexes": ["policy_index"],
    },
    {
        "question": "A customer wants to change the address after the order shipped. What is the "
                    "official process, and do we have a resolved case like this?",
        "expected_indexes": ["order_index", "ticket_index"],
    },
    {
        "question": "What does policy RR-004 say?",
        "expected_indexes": ["policy_index"],
    },
    {
        "question": "A customer received a damaged phone and wants a refund instead of a "
                    "replacement. What is the official process, and do we have a previous "
                    "resolved case where a refund was issued?",
        "expected_indexes": ["policy_index", "order_index", "ticket_index"],
    },
    {
        "question": "Do you sell refrigerators?",
        "expected_indexes": [],
    },
]


def format_indexes(index_names):
    return ", ".join(index_names) if index_names else "none"


matches = 0
for i, test in enumerate(test_questions, start=1):
    decision = route_question(test["question"])
    is_match = set(decision["indexes"]) == set(test["expected_indexes"])
    matches += is_match

    print("\n" + "=" * 70)
    print(f"QUESTION {i}: {test['question']}")
    print("=" * 70)
    print("Router output (JSON):")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Predicted indexes: {format_indexes(decision['indexes'])}")
    print(f"Expected indexes : {format_indexes(test['expected_indexes'])}")
    print(f"Result           : {'MATCH' if is_match else 'DIFFERENT'}")

print("\n" + "=" * 70)
print(f"Router test: {matches} / {len(test_questions)} questions routed to the expected indexes")
print("=" * 70)

#=======================================
# Fallback check: invalid router output must not crash, it searches all 4 indexes
#=======================================

print("\nFallback check (router output that is not valid):")
for bad_output in ["Sure! This looks like a policy question.", '{"intents": ["billing"]}']:
    decision = decision_from_router_output(bad_output)
    print(f"\nRouter returned  : {bad_output}")
    print(f"Indexes searched : {format_indexes(decision['indexes'])}")
    print(f"Reason           : {decision['reason']}")
