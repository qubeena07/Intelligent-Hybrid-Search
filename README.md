# Hybrid Information Retrieval System (Wikipedia)

Streamlit application for **hybrid information retrieval** over a Wikipedia corpus.

It combines:

- **Sparse retrieval**: multiple BM25 variants (Okapi, BM25L, BM25Plus)
- **Dense retrieval**: SentenceTransformer models (MiniLM, MPNet)
- **Hybrid fusion**: Weighted Sum or Reciprocal Rank Fusion (RRF)
- **Lexical query expansion**: POS-aware WordNet synonyms filtered by embedding cosine similarity
- **Baselines**: BM25-only and Dense-only modes
- **Wikipedia ↔ Grokipedia** links: compare human-edited vs AI-generated articles side by side


## Application UI

![Hybrid IR UI](images/UI1.png)


---

## Project Structure

```
hybrid_ir_wikipedia/
├── app.py                  # Main Streamlit application
├── evaluate.py             # Offline evaluation script (nDCG, Recall, MRR)
├── test_queries.py         # 50 evaluation queries across 4 categories
├── requirements.txt        # Python dependencies
├── query_log.csv           # Auto-generated: query expansion log (created on first search)
├── search_log.csv          # Auto-generated: per-search results log (created on first search)
├── relevance_template.csv  # Manual relevance judgements for evaluation
├── retrieval_results.csv   # Output from evaluate.py
├── OPTIMIZATION_GUIDE.md   # Notes on performance optimizations
├── PERFORMANCE_FIXES.md    # Record of performance fixes applied
└── images/
    ├── UI1.png
    └── UI2.png
    └── U3.png
```

---

## Features

- **Configurable corpus size**
  Load between 1,000 and 20,000 English Wikipedia articles from Hugging Face
  (`wikimedia/wikipedia`, snapshot `20231101.en`).

- **Multiple BM25 variants**
  - `BM25Okapi` — standard TF-IDF-based BM25
  - `BM25L` — length-corrected variant
  - `BM25Plus` — lower-bound term frequency variant

- **Multiple dense encoders**
  - `all-MiniLM-L6-v2` — fast, small, good for interactive search
  - `all-mpnet-base-v2` — slower, higher-quality embeddings

- **Retrieval modes**
  - **Hybrid (BM25 + Dense)** with Weighted Sum (tunable α) or RRF
  - **BM25 baseline** (sparse only)
  - **Dense baseline** (semantic only)
  - **Compare All Methods** — runs all three side by side in one click

- **Query expansion**
  - NLTK tokenisation and POS tagging
  - POS-aware WordNet synonyms (nouns → noun synonyms only, etc.)
  - Max 3 synonyms per word to avoid query drift
  - Cosine similarity filter (threshold 0.5) using the active encoder

- **Score Breakdown**
  Each result includes an expandable plain-English explanation of its BM25
  score, dense score, and how they were fused.

- **Logging**
  Every search appends a row to `query_log.csv` and `search_log.csv`
  for offline evaluation and reproducibility.

- **Session Stats sidebar**
  Tracks total searches, average response time, and most-used retrieval mode
  within the current browser session.

---

## High-Level Architecture

```
User Query
    │
    ▼
Query Expansion (optional)
  └── NLTK tokenise → POS tag → lemmatise
  └── WordNet synonyms (POS-filtered, similarity-filtered)
    │
    ▼
┌─────────────────────────────────┐
│  Sparse (BM25)   Dense (ST)     │
│  bm25.get_scores  cos_sim()     │
│  normalise        normalise     │
└──────────┬──────────────┬───────┘
           │              │
           ▼              ▼
       Fusion Method
     ┌──────────────────────────┐
     │ Weighted Sum  │   RRF    │
     │ α·dense +     │ 1/(k+r)  │
     │ (1-α)·bm25    │ summed   │
     └──────────────────────────┘
           │
           ▼
      Top-K Results
  (title, snippet, scores, links)
           │
           ▼
    Logged to CSV
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd hybrid_ir_wikipedia
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install scipy nltk
```

> `scipy` and `nltk` are used by `app.py` but not listed in `requirements.txt`.
> NLTK corpora (WordNet, punkt, POS tagger) are downloaded automatically on
> first run.

---

## Running the App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Running Evaluation

### Step 1 — Verify the query set

```bash
python3 test_queries.py
```

Expected output:
```
Total queries : 50
  ambiguous   : 10
  conceptual  : 15
  factual     : 15
  technical   : 10

All assertions passed.
```

### Step 2 — Run offline evaluation

```bash
python3 evaluate.py
```

Results are written to `retrieval_results.csv`.

### Step 3 — Inspect search logs

After running searches in the app, inspect the auto-generated logs:

```bash
# Query expansion log
cat query_log.csv

# Full per-search log (config + top-1 scores + latency)
cat search_log.csv
```

---

## Configuration Reference

| Sidebar Control | Options | Effect |
|---|---|---|
| Number of Wikipedia articles | 1,000 – 20,000 | Corpus size (more = better recall, slower startup) |
| Sparse model | BM25Okapi / BM25L / BM25Plus | BM25 scoring variant |
| Dense model | MiniLM-L6-v2 / mpnet-base-v2 | Sentence encoder |
| Retrieval Mode | Hybrid / BM25 / Dense | Which signals are used for ranking |
| Hybrid Fusion Method | Weighted Sum / RRF | How sparse and dense scores are combined |
| Dense Weight (α) | 0.0 – 1.0 | Balance between BM25 (0) and Dense (1) in Weighted Sum |
| Top K Results | 3 – 15 | Number of results displayed |
| Enable Query Expansion | On / Off | WordNet synonym expansion |
| Show Expanded Query | On / Off | Display the expanded query string |

---

## Log File Schemas

**`query_log.csv`** — written once per search when query expansion is enabled:

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 datetime |
| `original_query` | Raw user query |
| `expanded_query` | Query after expansion |
| `terms_added` | Number of new terms added |

**`search_log.csv`** — written once per search:

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 datetime |
| `query` | Raw user query |
| `expanded_query` | Query submitted to index |
| `retrieval_mode` | `hybrid` / `bm25` / `dense` |
| `fusion_method` | Fusion label string |
| `bm25_variant` | BM25 variant used |
| `encoder_model` | Dense model label |
| `alpha_weight` | α value for Weighted Sum |
| `top1_title` | Title of the top-ranked result |
| `top1_hybrid_score` | Primary score of top result |
| `top1_bm25_norm` | Normalised BM25 score of top result |
| `top1_dense_norm` | Normalised dense score of top result |
| `response_time_ms` | Wall-clock search latency (ms) |

---

## Requirements

```
streamlit
datasets
rank-bm25
sentence-transformers
scikit-learn
scipy
nltk
```

> `torch` is installed automatically as a dependency of `sentence-transformers`.
