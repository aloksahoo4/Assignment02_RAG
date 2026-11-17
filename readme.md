# Customer Support Copilot (Dandes Electronics)

A RAG (retrieval-augmented generation) copilot that answers customer support
questions from a 50-record knowledge base: return and shipping policies, product
FAQs, known issues, order guides and resolved tickets.

The project builds the copilot in three versions and then measures them against
each other on the same 20 questions:

| Version | Folder | What it adds |
|---|---|---|
| V1 (basic) | `v1_basic_rag/` | One Chroma collection, vector search, top 5, answer |
| V2 (single index) | `v2_advanced_rag/` | Metadata filtering, hybrid search (vector + BM25), reranking, query rewriting |
| V2 (multi-index + routing) | `v2_multi_index/` | 4 separate indexes and an intent router that picks which ones to search |

## Project structure

```
Assignment02/
├── .env                              # your keys (you create this, see Setup)
├── readme.md
└── customer_support_copilot/
    ├── data/
    │   └── dandes_electronics_kb.json    # the knowledge base (50 records)
    ├── v1_basic_rag/
    ├── v2_advanced_rag/
    ├── v2_multi_index/
    └── evaluation/
        ├── golden_test_set.json          # 20 test questions with reference answers
        ├── results/                      # created by the evaluation scripts
        └── comparison_table.csv          # created by 4_comparison_table.py
```

## Setup

### 1. Create a virtual environment

From the project root (`Assignment02/`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

The project was built and tested with Python 3.14.

### 2. Install the dependencies

Install everything in one command, so pip picks versions that work together
(`ragas` needs an older `openai` package than pip would choose on its own):

```bash
python -m pip install langchain-core langchain-text-splitters langchain-openai \
    langchain-chroma chromadb python-dotenv rank-bm25 sentence-transformers ragas
```

### 3. Create the `.env` file

Create a file named `.env` in the project root (`Assignment02/.env`) with two values:

```
AS_OPENAI_API_KEY=your-openai-api-key
CHROMA_DB_PATH=/absolute/path/to/a/folder/for/chroma
```

- `AS_OPENAI_API_KEY` is your OpenAI API key. It needs access to
  `text-embedding-3-small`, `gpt-5.6-luna`, `gpt-6-luna` and `gpt-6.1-sol`.
- `CHROMA_DB_PATH` is the folder where Chroma stores the vector indexes. Use an
  absolute path. The folder is created on the first build.

## How to run

**Run every script from inside its own folder.** Some scripts read the knowledge
base through a relative path (`../data/...`), so they fail if started from
somewhere else.

Run the stages in the order below. Stage 1 and stage 3 each start with a build
script that must run before the other scripts in that stage.

### Stage 1: V1, basic RAG

```bash
cd customer_support_copilot/v1_basic_rag
python 1_offline_pipeline.py
python 2_online_pipeline.py
python 3_conversational_rag.py
python 4_rag_evaluation.py
```

| Script | What it does |
|---|---|
| `1_offline_pipeline.py` | **Build step.** Loads the knowledge base, embeds all 50 records and stores them in one Chroma collection, `support_tickets` |
| `2_online_pipeline.py` | Answers one question: embed the question, retrieve the top 5, generate the answer |
| `3_conversational_rag.py` | Multi-turn chat: rewrites a follow-up question using the chat history before retrieving |
| `4_rag_evaluation.py` | An early evaluation of V1 only (precision, recall, faithfulness, refusals) |

### Stage 2: V2, advanced RAG on the single index

Needs the `support_tickets` collection from stage 1.

```bash
cd customer_support_copilot/v2_advanced_rag
python 1_metadata_filtering.py
python 2_hybrid_search.py
python 3_reranking.py
python 4_query_transformation.py
python 5_v2_pipeline.py
```

| Script | What it does |
|---|---|
| `1_metadata_filtering.py` | Vector search restricted by a metadata filter such as category |
| `2_hybrid_search.py` | Vector search + BM25 keyword search, fused with Reciprocal Rank Fusion, then reranked |
| `3_reranking.py` | Cross-encoder reranking of the vector search results |
| `4_query_transformation.py` | Query rewriting and multi-query retrieval for vague questions |
| `5_v2_pipeline.py` | The full V2 single-index pipeline: route, filter, hybrid search, rerank, answer |

Scripts 1 to 4 each demonstrate one technique. Script 5 is the version that is
evaluated as "V2 (single index)".

### Stage 3: V2, multi-index with intent routing

```bash
cd customer_support_copilot/v2_multi_index
python 1_build_indexes.py
python 2_multi_index_search.py
python 3_intent_router.py
python 4_v2_multi_index_pipeline.py
```

| Script | What it does |
|---|---|
| `1_build_indexes.py` | **Build step.** Splits the knowledge base into 4 Chroma collections and builds a BM25 index for each one |
| `2_multi_index_search.py` | Searches one index or several: hybrid search in each, merge the candidates, rerank once |
| `3_intent_router.py` | Classifies a question into intents with `gpt-6-luna` and returns the indexes to search, as JSON |
| `4_v2_multi_index_pipeline.py` | The full pipeline: route, hybrid search in each chosen index, merge, one global rerank, top 5, answer with sources |

`1_build_indexes.py` ends by printing the document count of each index. The
counts must be:

```
policy_index  : 10 documents
product_index : 20 documents
order_index   :  6 documents
ticket_index  : 14 documents
```

The four indexes and the intents that route to them:

| Intent | Index | Contains |
|---|---|---|
| `policy` | `policy_index` | Return and shipping policies |
| `product` | `product_index` | Product FAQs and known issues |
| `order` | `order_index` | Order guides (cancel, change address, track, warranty claim) |
| `past_issue` | `ticket_index` | Resolved tickets |
| `out_of_scope` | none | Nothing is searched; the copilot replies "I don't know based on the available information." |

### Stage 4: Evaluation of all three versions

Needs the indexes from stage 1 and stage 3. The scripts depend on each other's
output, so run them in this order:

```bash
cd customer_support_copilot/evaluation
python 1_run_versions.py
python 2_retrieval_metrics.py
python 3_generation_metrics.py
python 4_comparison_table.py
```

| Script | What it does | Output |
|---|---|---|
| `1_run_versions.py` | Runs the 20 golden questions through all 3 versions and saves each answer, the retrieved document IDs, the sources and the routing decision | `results/v1_results.json`, `results/v2_single_index_results.json`, `results/v2_multi_index_results.json` |
| `2_retrieval_metrics.py` | Precision@5, Recall@5, Coverage, Routing Accuracy, Refusal Accuracy, False Refusals | `results/retrieval_metrics.json` |
| `3_generation_metrics.py` | Faithfulness (own LLM judge and RAGAS), Answer Relevancy, Context Precision and Context Recall (RAGAS) | `results/generation_metrics.json` |
| `4_comparison_table.py` | Puts all metrics for the 3 versions side by side and checks the minimum targets | `comparison_table.csv` |

`1_run_versions.py` imports the three real pipeline files
(`v1_basic_rag/2_online_pipeline.py`, `v2_advanced_rag/5_v2_pipeline.py` and
`v2_multi_index/4_v2_multi_index_pipeline.py`) and runs every version with the
same chat model (`gpt-6-luna`), so the comparison is fair.

## Models used

| Purpose | Model |
|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Answers and routing | `gpt-5.6-luna` in stages 1 and 2, `gpt-6-luna` in stage 3 and in the evaluation runs |
| Evaluation judge | `gpt-6.1-sol` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally) |

## Things to know before running

- **Every script calls the OpenAI API**, so each run costs a small amount.
  `evaluation/3_generation_metrics.py` is the most expensive and the slowest
  step: it makes several hundred judge calls.
- **The first run of any reranking script downloads the cross-encoder model**
  from Hugging Face (about 90 MB). Later runs use the cached copy. A warning
  about a missing `HF_TOKEN` is harmless.
- **The build scripts are safe to run again.** Records are stored under fixed
  IDs, so a second run updates them instead of adding duplicates.
- **`4_v2_multi_index_pipeline.py` needs `1_build_indexes.py` to have run
  first.** If a collection is empty it stops with a message that says so.
- **Router failures do not stop the copilot.** If the router call fails or
  returns invalid JSON, the multi-index pipeline searches all 4 indexes.
