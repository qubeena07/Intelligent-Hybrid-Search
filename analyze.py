"""
analyze.py
==========
Two-part analysis script for the Hybrid IR System.

Part 1 — Metrics from existing retrieval_results.csv
    Reads the 'relevance_type' column (relevant=2, partial=1, not_relevant=0)
    and computes nDCG@10, Recall@10, MRR@10 for all 4 methods.
    Saves metrics_summary.txt and metrics.json.

Part 2 — Query expansion impact analysis
    Loads the Wikipedia corpus and models, then runs Hybrid-RRF with and
    without WordNet query expansion for all 50 queries.
    Saves query_expansion_impact.csv.

Usage:
    python analyze.py            # run both parts
    python analyze.py --part1    # metrics only (fast, no model load)
    python analyze.py --part2    # expansion analysis only
"""

import csv
import json
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_queries import TEST_QUERIES as _TYPED_QUERIES
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

QUERY_TYPES  = {e["query"]: e["type"] for e in _TYPED_QUERIES}
QUERIES      = [e["query"] for e in _TYPED_QUERIES]
TYPE_ORDER   = ["factual", "conceptual", "ambiguous", "technical"]
METHOD_ORDER = ["bm25_only", "dense_only", "hybrid_weighted_sum", "hybrid_rrf"]

TIER_SCORE = {"relevant": 2, "partial": 1, "not_relevant": 0}


# =============================================================================
# SHARED METRICS
# =============================================================================

def compute_ndcg(relevances, k=10):
    """Graded nDCG: gain = 2^rel - 1."""
    rel = relevances[:k]
    dcg  = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_recall(relevances, k=10):
    total = sum(1 for r in relevances if r >= 1)
    if total == 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r >= 1) / total

def compute_mrr(relevances, k=10):
    for i, r in enumerate(relevances[:k], 1):
        if r >= 1:
            return 1.0 / i
    return 0.0


# =============================================================================
# PART 1 — METRICS FROM EXISTING retrieval_results.csv
# =============================================================================

def part1_metrics(results_csv="retrieval_results.csv"):
    print("\n" + "=" * 80)
    print("PART 1 — METRICS FROM retrieval_results.csv")
    print("=" * 80 + "\n")

    # Load rows
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from '{results_csv}'\n")

    # Build per-query/method relevance lists
    # Structure: data[query][method] = [(rank, score), ...]
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        score = TIER_SCORE.get(row.get("relevance_type", "").strip(), 0)
        data[row["query"]][row["method"]].append((int(row["rank"]), score))

    # Sort by rank and extract relevance lists
    metrics_per_query = {}
    for query, methods in data.items():
        metrics_per_query[query] = {}
        for method, ranked in methods.items():
            ranked.sort(key=lambda x: x[0])
            rels = [s for _, s in ranked]
            metrics_per_query[query][method] = {
                "nDCG@10":   compute_ndcg(rels),
                "Recall@10": compute_recall(rels),
                "MRR@10":    compute_mrr(rels),
            }

    # Aggregate by method
    agg = defaultdict(lambda: {"nDCG@10": [], "Recall@10": [], "MRR@10": []})
    for query_metrics in metrics_per_query.values():
        for method, scores in query_metrics.items():
            for k, v in scores.items():
                agg[method][k].append(v)

    # Print summary table
    print(f"{'Method':<25} {'nDCG@10':<12} {'Recall@10':<12} {'MRR@10':<12}")
    print("-" * 80)
    summary_rows = {}
    for method in METHOD_ORDER:
        if method not in agg:
            continue
        nd = float(np.mean(agg[method]["nDCG@10"]))
        rc = float(np.mean(agg[method]["Recall@10"]))
        mr = float(np.mean(agg[method]["MRR@10"]))
        print(f"{method:<25} {nd:<12.4f} {rc:<12.4f} {mr:<12.4f}")
        summary_rows[method] = {"nDCG@10": nd, "Recall@10": rc, "MRR@10": mr}
    print("=" * 80 + "\n")

    # Per-query-type nDCG@10 breakdown
    print("nDCG@10 BY QUERY TYPE")
    print(f"{'Query Type':<14}|{'BM25':>10} {'Dense':>10} {'Hybrid-WS':>12} {'Hybrid-RRF':>12}")
    print("-" * 62)
    type_breakdown = []
    for qtype in TYPE_ORDER:
        vals = {m: [] for m in METHOD_ORDER}
        for query, qm in metrics_per_query.items():
            if QUERY_TYPES.get(query) == qtype:
                for method in METHOD_ORDER:
                    if method in qm:
                        vals[method].append(qm[method]["nDCG@10"])
        row_vals = [float(np.mean(vals[m])) if vals[m] else 0.0 for m in METHOD_ORDER]
        print(f"{qtype.capitalize():<14}|" + "".join(f"{v:>10.3f} " for v in row_vals))
        type_breakdown.append([qtype] + row_vals)
    print("=" * 80 + "\n")

    # Save outputs
    with open("metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("HYBRID IR EVALUATION — METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Method':<25} {'nDCG@10':<12} {'Recall@10':<12} {'MRR@10':<12}\n")
        f.write("-" * 80 + "\n")
        for method in METHOD_ORDER:
            if method in summary_rows:
                s = summary_rows[method]
                f.write(f"{method:<25} {s['nDCG@10']:<12.4f} {s['Recall@10']:<12.4f} {s['MRR@10']:<12.4f}\n")
        f.write("=" * 80 + "\n\n")
        f.write("nDCG@10 BY QUERY TYPE\n")
        f.write(f"{'Query Type':<14}|{'BM25':>10} {'Dense':>10} {'Hybrid-WS':>12} {'Hybrid-RRF':>12}\n")
        f.write("-" * 62 + "\n")
        for row in type_breakdown:
            f.write(f"{row[0].capitalize():<14}|" + "".join(f"{v:>10.3f} " for v in row[1:]) + "\n")
        f.write("=" * 80 + "\n")

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "query_level": {q: {m: {k: float(v) for k, v in s.items()}
                                 for m, s in qm.items()}
                            for q, qm in metrics_per_query.items()},
            "method_summary": {m: {k: float(np.mean(v)) for k, v in sc.items()}
                                for m, sc in agg.items()},
            "query_type_breakdown": {
                row[0]: dict(zip(METHOD_ORDER, row[1:])) for row in type_breakdown
            },
        }, f, indent=2)

    print("✓ Saved metrics_summary.txt and metrics.json\n")


# =============================================================================
# PART 2 — QUERY EXPANSION IMPACT ANALYSIS
# =============================================================================

def part2_expansion_impact():
    print("=" * 80)
    print("PART 2 — QUERY EXPANSION IMPACT ANALYSIS (Hybrid-RRF)")
    print("=" * 80 + "\n")

    # ---- Load corpus ----
    print("[1/4] Loading Wikipedia corpus (10,000 docs)...")
    from datasets import load_dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]")
    documents, titles, snippets = [], [], []
    for row in dataset:
        text = row["text"]
        if isinstance(text, str):
            doc = text.lower().replace("\n", " ").strip()
            documents.append(doc)
            titles.append(row.get("title", "Untitled"))
            snippets.append(doc[:200])
    print(f"      ✓ {len(documents)} documents\n")

    # ---- Init models ----
    print("[2/4] Loading BM25 and SentenceTransformer...")
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, util
    from sklearn.preprocessing import MinMaxScaler

    bm25 = BM25Okapi([d.split() for d in documents])
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("      ✓ BM25 built")
    print("      Encoding 10k docs...")
    embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    print("      ✓ Embeddings ready\n")

    # ---- Load query expansion from app.py ----
    print("[3/4] Running Hybrid-RRF with and without query expansion...")
    import importlib.util, os, sys as _sys
    # Import apply_query_expansion from app.py without running Streamlit
    spec = importlib.util.spec_from_file_location("app_module", "app.py")
    app_mod = importlib.util.module_from_spec(spec)
    # Suppress streamlit from executing at import time by mocking st
    import types
    fake_st = types.ModuleType("streamlit")
    for attr in ["cache_resource", "cache_data", "session_state", "sidebar",
                 "set_page_config", "title", "write", "spinner", "error",
                 "warning", "info", "success", "columns", "expander",
                 "markdown", "subheader", "header", "text_input", "button",
                 "selectbox", "slider", "checkbox", "radio", "metric",
                 "stop", "rerun", "empty"]:
        setattr(fake_st, attr, lambda *a, **kw: None)
    fake_st.cache_resource = lambda *a, **kw: (lambda f: f)
    fake_st.cache_data = lambda *a, **kw: (lambda f: f)
    fake_st.session_state = {}
    _sys.modules["streamlit"] = fake_st
    try:
        spec.loader.exec_module(app_mod)
        expand_query = app_mod.apply_query_expansion
    except Exception as e:
        print(f"      ⚠ Could not import app.py expansion ({e}), using inline version")
        expand_query = _inline_expand_query

    def _tiered(title, snippet, query):
        terms = [t.lower() for t in query.split() if len(t) > 3]
        if any(t in title.lower() for t in terms):
            return 2
        if any(t in snippet for t in terms):
            return 1
        return 0

    def rrf_search(query_tokens, top_k=10, k=60):
        scaler = MinMaxScaler()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()

        q_emb = model.encode(" ".join(query_tokens), convert_to_tensor=True)
        dense_scores = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
        dense_norm = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()

        sparse_top = list(np.argsort(bm25_scores)[::-1][:200])
        dense_top  = list(np.argsort(dense_scores)[::-1][:200])

        fused = defaultdict(float)
        for rank, idx in enumerate(sparse_top):
            fused[idx] += 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_top):
            fused[idx] += 1.0 / (k + rank + 1)
        return sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]

    results_no_qe, results_qe, expanded_queries = {}, {}, {}

    for entry in tqdm(QUERIES, desc="Queries", ncols=80):
        query = entry

        # Without expansion
        top_no_qe = rrf_search(query.split())
        rels_no_qe = [_tiered(titles[i], snippets[i], query) for i in top_no_qe]
        results_no_qe[query] = compute_ndcg(rels_no_qe)

        # With expansion
        try:
            expanded = expand_query(query, similarity_model=model)
        except Exception:
            expanded = query
        expanded_queries[query] = expanded
        top_qe = rrf_search(expanded.split())
        rels_qe = [_tiered(titles[i], snippets[i], query) for i in top_qe]
        results_qe[query] = compute_ndcg(rels_qe)

    print("\n[4/4] Computing and printing results...\n")

    # Per-type table
    print("nDCG@10 — Hybrid-RRF With vs Without Query Expansion")
    print(f"{'Query Type':<14} {'RRF no-QE':>12} {'RRF with-QE':>13} {'Delta':>8}")
    print("-" * 52)

    breakdown_rows = []
    all_no_qe, all_qe = [], []
    for qtype in TYPE_ORDER:
        no_qe_vals = [results_no_qe[q] for q in QUERIES if QUERY_TYPES.get(q) == qtype]
        qe_vals    = [results_qe[q]    for q in QUERIES if QUERY_TYPES.get(q) == qtype]
        mu_no = float(np.mean(no_qe_vals))
        mu_qe = float(np.mean(qe_vals))
        delta = mu_qe - mu_no
        sign  = "+" if delta >= 0 else ""
        print(f"{qtype.capitalize():<14} {mu_no:>12.3f} {mu_qe:>13.3f} {sign}{delta:>7.3f}")
        breakdown_rows.append([qtype, mu_no, mu_qe, delta])
        all_no_qe.extend(no_qe_vals)
        all_qe.extend(qe_vals)

    overall_no = float(np.mean(all_no_qe))
    overall_qe = float(np.mean(all_qe))
    overall_d  = overall_qe - overall_no
    sign = "+" if overall_d >= 0 else ""
    print(f"{'Overall':<14} {overall_no:>12.3f} {overall_qe:>13.3f} {sign}{overall_d:>7.3f}")
    breakdown_rows.append(["overall", overall_no, overall_qe, overall_d])
    print("=" * 52 + "\n")

    # Top 3 helped / top 3 hurt
    deltas = {q: results_qe[q] - results_no_qe[q] for q in QUERIES}
    sorted_by_delta = sorted(QUERIES, key=lambda q: deltas[q])
    hurt   = sorted_by_delta[:3]
    helped = sorted_by_delta[-3:][::-1]

    print("3 QUERIES WHERE EXPANSION HELPED MOST")
    print("-" * 80)
    for q in helped:
        print(f"  Query   : {q}")
        print(f"  Expanded: {expanded_queries[q]}")
        print(f"  Delta   : +{deltas[q]:.3f}\n")

    print("3 QUERIES WHERE EXPANSION HURT MOST")
    print("-" * 80)
    for q in hurt:
        print(f"  Query   : {q}")
        print(f"  Expanded: {expanded_queries[q]}")
        print(f"  Delta   : {deltas[q]:.3f}\n")

    # Save CSV
    csv_path = "query_expansion_impact.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "query_type", "expanded_query",
                         "ndcg_no_qe", "ndcg_with_qe", "delta"])
        for q in QUERIES:
            writer.writerow([q, QUERY_TYPES.get(q, ""),
                             expanded_queries.get(q, q),
                             f"{results_no_qe[q]:.4f}",
                             f"{results_qe[q]:.4f}",
                             f"{deltas[q]:.4f}"])
    print(f"✓ Saved '{csv_path}'\n")

    # Save breakdown CSV
    with open("query_expansion_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_type", "rrf_no_qe", "rrf_with_qe", "delta"])
        writer.writerows(breakdown_rows)
    print("✓ Saved 'query_expansion_breakdown.csv'\n")


def _inline_expand_query(query, similarity_model=None):
    """Fallback WordNet expansion if app.py import fails."""
    import nltk
    from nltk.corpus import wordnet
    tokens = nltk.word_tokenize(query)
    tagged = nltk.pos_tag(tokens)
    expanded = list(tokens)
    pos_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    for token, tag in tagged:
        wn_pos = pos_map.get(tag[0])
        if not wn_pos:
            continue
        added = 0
        for syn in wordnet.synsets(token, pos=wn_pos):
            for lemma in syn.lemmas():
                if added >= 3:
                    break
                s = lemma.name().lower().replace("_", " ")
                if s not in [t.lower() for t in expanded]:
                    expanded.append(s)
                    added += 1
            if added >= 3:
                break
    return " ".join(expanded)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_p1 = "--part2" not in sys.argv
    run_p2 = "--part1" not in sys.argv

    if run_p1:
        part1_metrics()
    if run_p2:
        part2_expansion_impact()
