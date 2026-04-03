"""
Hybrid IR System Evaluation Framework
======================================
Standalone evaluation script for hybrid information retrieval system.
Runs multiple retrieval methods on a test set of 50 queries and computes
standard IR evaluation metrics (nDCG@10, Recall@10, MRR@10).

Relevance is tiered:
    2 = relevant     (query keywords found in document title)
    1 = partial      (query keywords found in first 200 chars of document text, not title)
    0 = not relevant (neither)

nDCG uses graded gain: gain = 2^relevance - 1

Usage:
    python evaluate.py              # Run retrieval + auto-annotate with tiered relevance
    python evaluate.py --metrics    # Compute metrics from relevance_template.csv
"""

import csv
import json
import sys
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# ============================================================================
# TEST QUERY SET — imported from test_queries.py (50 queries with type labels)
# ============================================================================
from test_queries import TEST_QUERIES as _TYPED_QUERIES

# Flat list of query strings for retrieval (preserves order)
TEST_QUERIES = [entry["query"] for entry in _TYPED_QUERIES]

# Lookup: query string -> type label (factual / conceptual / ambiguous / technical)
QUERY_TYPES = {entry["query"]: entry["type"] for entry in _TYPED_QUERIES}


# ============================================================================
# EVALUATION METRICS (nDCG@10, Recall@10, MRR@10)
# ============================================================================
def compute_ndcg(relevances: List[int], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG@k) with graded relevance.
    Uses gain = 2^relevance - 1, supporting scores of 0, 1, or 2.
    Higher is better. Range: [0, 1].
    
    Args:
        relevances: List of graded relevance judgments (0=not relevant, 1=partial, 2=relevant)
        k: Cutoff rank (default 10)
    
    Returns:
        nDCG@k score
    """
    if len(relevances) == 0:
        return 0.0
    
    relevances = relevances[:k]
    
    # DCG: sum of (2^rel - 1) / log2(rank+1)
    dcg = 0.0
    for rank, rel in enumerate(relevances, 1):
        dcg += (2 ** rel - 1) / np.log2(rank + 1)
    
    # IDCG: DCG of ideally ranked results
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal_relevances, 1):
        idcg += (2 ** rel - 1) / np.log2(rank + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_recall(relevances: List[int], k: int = 10, num_relevant_total: int = None) -> float:
    """
    Compute Recall@k.
    Fraction of relevant documents retrieved. Range: [0, 1].
    
    Args:
        relevances: List of binary relevance judgments
        k: Cutoff rank (default 10)
        num_relevant_total: Total number of relevant documents for this query
    
    Returns:
        Recall@k score
    """
    if len(relevances) == 0:
        return 0.0
    
    if num_relevant_total is None:
        num_relevant_total = sum(relevances)
    
    if num_relevant_total == 0:
        return 0.0
    
    retrieved_relevant = sum(relevances[:k])
    return retrieved_relevant / num_relevant_total


def compute_mrr(relevances: List[int], k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank (MRR@k).
    Inverse of rank of first relevant result. Range: [0, 1].
    
    Args:
        relevances: List of binary relevance judgments
        k: Cutoff rank (default 10)
    
    Returns:
        MRR@k score
    """
    if len(relevances) == 0:
        return 0.0
    
    for rank, rel in enumerate(relevances[:k], 1):
        if rel >= 1:
            return 1.0 / rank
    
    return 0.0


# ============================================================================
# RETRIEVAL LOGIC (adapted from app.py)
# ============================================================================
def rrf_fusion(sparse_indices: List[int], dense_indices: List[int], k: int = 60) -> Tuple[List[int], Dict]:
    """
    Reciprocal Rank Fusion combines sparse and dense rankings.
    
    Args:
        sparse_indices: Top-k indices from sparse retrieval
        dense_indices: Top-k indices from dense retrieval
        k: RRF smoothing constant (typically 60)
    
    Returns:
        (reranked indices, fused scores dict)
    """
    fused_scores = defaultdict(float)
    
    for rank, idx in enumerate(sparse_indices):
        fused_scores[idx] += 1.0 / (k + rank + 1)
    
    for rank, idx in enumerate(dense_indices):
        fused_scores[idx] += 1.0 / (k + rank + 1)
    
    reranked = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return reranked, fused_scores


def hybrid_search_eval(
    query: str,
    documents: List[str],
    bm25_model,
    dense_model,
    embeddings,
    top_k: int = 10,
    alpha: float = 0.5,
    fusion_method: str = 'weighted_sum',
    retrieval_mode: str = 'hybrid'
) -> List[Dict]:
    """
    Performs hybrid/baseline search. Returns top-k results with all score components.
    
    Args:
        query: Query string
        documents: List of document texts
        bm25_model: Trained BM25Okapi model
        dense_model: SentenceTransformer model
        embeddings: Pre-encoded document embeddings
        top_k: Number of results to return
        alpha: Weight for dense in weighted sum (bm25 weight = 1-alpha)
        fusion_method: 'weighted_sum' or 'rrf'
        retrieval_mode: 'hybrid', 'bm25', or 'dense'
    
    Returns:
        List of result dicts with rank, index, and scores
    """
    if not query.strip():
        return []
    
    query_tokens = query.split()
    CANDIDATE_K = 200
    scaler = MinMaxScaler()
    
    # BM25 sparse retrieval
    bm25_scores = bm25_model.get_scores(query_tokens)
    bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    sparse_top_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]
    
    # Dense retrieval
    query_emb = dense_model.encode(query, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
    dense_norm = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()
    dense_top_indices = np.argsort(dense_scores)[::-1][:CANDIDATE_K]
    
    # Select top indices based on retrieval mode
    if retrieval_mode == 'bm25':
        # BM25 only
        top_indices = np.argsort(bm25_norm)[::-1][:top_k]
        final_scores = bm25_norm
    
    elif retrieval_mode == 'dense':
        # Dense only
        top_indices = np.argsort(dense_norm)[::-1][:top_k]
        final_scores = dense_norm
    
    else:  # hybrid
        if fusion_method == 'rrf':
            # RRF fusion
            final_indices, final_scores_dict = rrf_fusion(sparse_top_indices, dense_top_indices)
            top_indices = final_indices[:top_k]
            final_scores = final_scores_dict
        else:
            # Weighted sum fusion
            final_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
            top_indices = np.argsort(final_scores)[::-1][:top_k]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if isinstance(final_scores, dict):
            primary_score = final_scores.get(idx, 0.0)
        else:
            primary_score = float(final_scores[idx])
        
        results.append({
            'rank': rank,
            'doc_index': int(idx),
            'bm25_score': float(bm25_norm[idx]),
            'dense_score': float(dense_norm[idx]),
            'hybrid_score': primary_score,
        })
    
    return results


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================
def run_evaluation():
    """
    Main evaluation: load data, initialize models, run retrieval for all
    queries and methods, save results and annotation template.
    """
    print("\n" + "=" * 80)
    print("HYBRID IR EVALUATION FRAMEWORK")
    print("=" * 80 + "\n")
    
    # ---- STEP 1: Load Wikipedia corpus ----
    print("[1/5] Loading Wikipedia corpus (10,000 documents)...")
    
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train[:10000]"
        )
        
        documents, titles, snippets = [], [], []
        for row in dataset:
            text = row["text"]
            if isinstance(text, str):
                doc_text = text.lower().replace("\n", " ").strip()
                documents.append(doc_text)
                titles.append(row.get("title", "Untitled"))
                snippets.append(doc_text[:200])
        
        print(f"      ✓ Loaded {len(documents)} documents\n")
    
    except Exception as e:
        print(f"     Could not load Wikipedia dataset: {e}")
        print("      Using fallback corpus...\n")
        
        documents = [
            "artificial intelligence simulates human intelligence processes by machines",
            "machine learning is a subset of artificial intelligence focused on learning from data",
            "information retrieval is the process of obtaining relevant information from repositories",
        ]
        titles = ["Artificial Intelligence", "Machine Learning", "Information Retrieval"]
        snippets = [d[:200] for d in documents]
    
    # ---- STEP 2: Initialize models ----
    print("[2/5] Initializing retrieval models...")
    
    # BM25
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    print("      BM25Okapi initialized")
    
    # SentenceTransformer
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("      SentenceTransformer loaded")
    
    # Encode documents
    print("      Encoding documents to embeddings...")
    embeddings = dense_model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    print("       Embeddings computed\n")
    
    # ---- STEP 3: Run retrieval for all queries and methods ----
    print("[3/5] Running retrieval for all queries and methods...")
    print(f"      {len(TEST_QUERIES)} queries × 4 methods = {len(TEST_QUERIES) * 4} retrievals\n")
    
    retrieval_results = []
    
    # Define retrieval configurations
    methods = [
        ('bm25_only', 'bm25', None),
        ('dense_only', 'dense', None),
        ('hybrid_weighted_sum', 'hybrid', 'weighted_sum'),
        ('hybrid_rrf', 'hybrid', 'rrf'),
    ]
    
    for query in tqdm(TEST_QUERIES, desc="Queries", ncols=80):
        for method_name, retrieval_mode, fusion_method in methods:
            results = hybrid_search_eval(
                query=query,
                documents=documents,
                bm25_model=bm25,
                dense_model=dense_model,
                embeddings=embeddings,
                top_k=10,
                alpha=0.5,
                fusion_method=fusion_method,
                retrieval_mode=retrieval_mode,
            )
            
            # Store results
            for result in results:
                idx = result['doc_index']
                retrieval_results.append({
                    'query': query,
                    'method': method_name,
                    'rank': result['rank'],
                    'doc_title': titles[idx],
                    'doc_index': idx,
                    'doc_text_snippet': snippets[idx],
                    'bm25_score': result['bm25_score'],
                    'dense_score': result['dense_score'],
                    'hybrid_score': result['hybrid_score'],
                    'relevance_type': '',
                })
    
    print(f"\n      ✓ Generated {len(retrieval_results)} result rows\n")
    
    # ---- STEP 4: Save retrieval results CSV ----
    print("[4/5] Saving retrieval results...")
    
    results_csv = "retrieval_results.csv"
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['query', 'method', 'rank', 'doc_title', 'doc_index',
                     'doc_text_snippet', 'bm25_score', 'dense_score', 'hybrid_score', 'relevance_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retrieval_results)
    
    print(f"       Saved {len(retrieval_results)} rows to '{results_csv}'\n")
    
    # ---- STEP 5: Auto-annotate with tiered relevance and update results ----
    print("[5/5] Auto-annotating with tiered relevance...")

    def _tiered_relevance(query: str, doc_title: str, doc_snippet: str) -> int:
        """Return 2 (title match), 1 (snippet match), or 0 (no match)."""
        query_terms = [t.lower() for t in query.replace("?", "").split() if len(t) > 3]
        title_lower = doc_title.lower()
        snippet_lower = doc_snippet.lower()
        if any(term in title_lower for term in query_terms):
            return 2
        if any(term in snippet_lower for term in query_terms):
            return 1
        return 0

    TIER_LABEL = {2: "relevant", 1: "partial", 0: "not_relevant"}

    # Annotate every result row
    for row in retrieval_results:
        tier = _tiered_relevance(row["query"], row["doc_title"], row["doc_text_snippet"])
        row["relevance_type"] = TIER_LABEL[tier]

    # Re-save retrieval_results.csv with filled relevance_type
    results_csv = "retrieval_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query", "method", "rank", "doc_title", "doc_index",
                      "doc_text_snippet", "bm25_score", "dense_score", "hybrid_score", "relevance_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retrieval_results)
    print(f"       Updated '{results_csv}' with relevance_type column")

    # Build and save relevance_template.csv with tiered scores
    qd_map = {}  # (query, doc_title) -> max tier seen across methods
    for row in retrieval_results:
        key = (row["query"], row["doc_title"])
        tier = {"relevant": 2, "partial": 1, "not_relevant": 0}[row["relevance_type"]]
        qd_map[key] = max(qd_map.get(key, 0), tier)

    relevance_csv = "relevance_template.csv"
    with open(relevance_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "doc_title", "relevant"])
        writer.writeheader()
        for (query, doc_title) in sorted(qd_map.keys()):
            writer.writerow({"query": query, "doc_title": doc_title, "relevant": qd_map[(query, doc_title)]})

    tier_counts = {0: 0, 1: 0, 2: 0}
    for v in qd_map.values():
        tier_counts[v] += 1
    print(f"       Created '{relevance_csv}'")
    print(f"       {len(qd_map)} pairs: {tier_counts[2]} relevant (2), "
          f"{tier_counts[1]} partial (1), {tier_counts[0]} not_relevant (0)\n")
    
    # Final summary
    print("=" * 80)
    print("RETRIEVAL COMPLETE ✓")
    print("=" * 80)
    print("\n NEXT STEPS:")
    print(f"""
  1. Open '{relevance_csv}' in a spreadsheet editor (Excel/Google Sheets)
  
  2. For each row, annotate the 'relevant' column:
     - Enter '1' if the document is relevant to the query
     - Enter '0' if the document is NOT relevant to the query
     - Leave blank to skip
  
  3. Save the file and run evaluation:
     $ python evaluate.py --metrics
  
📁 INPUT/OUTPUT FILES:
  
  Input:
    - {results_csv}: Retrieval results with scores (DO NOT EDIT)
  
  For annotation:
    - {relevance_csv}: Fill in the 'relevant' column (0 or 1)
  
  Output (after metrics):
    - metrics.json: Detailed per-query and per-method metrics
    - metrics_summary.txt: Human-readable summary
""")
    print("=" * 80)


def compute_metrics_from_annotations(
    relevance_csv: str = "relevance_template.csv",
    results_csv: str = "retrieval_results.csv",
):
    """
    Load filled-in relevance annotations and compute IR evaluation metrics.
    Generates both JSON and human-readable text outputs.
    
    Expected relevance CSV format:
        query, doc_title, relevant (0 or 1)
    """
    print("\n" + "=" * 80)
    print("COMPUTING EVALUATION METRICS")
    print("=" * 80 + "\n")
    
    # ---- STEP 1: Load relevance judgments ----
    print(f"[1/3] Loading relevance annotations from '{relevance_csv}'...")
    
    relevance_dict = {}  # relevance_dict[query][doc_title] = relevance (0 or 1)
    num_annotations = 0
    
    try:
        with open(relevance_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = row['query']
                doc_title = row['doc_title']
                rel_str = row['relevant'].strip()
                
                if rel_str == '':
                    continue
                
                try:
                    rel = int(rel_str)
                    if rel not in [0, 1, 2]:
                        print(f"        Invalid value for ({query}, {doc_title}): {rel} (must be 0, 1, or 2)")
                        continue
                    
                    if query not in relevance_dict:
                        relevance_dict[query] = {}
                    
                    relevance_dict[query][doc_title] = rel
                    num_annotations += 1
                
                except ValueError:
                    print(f"        Non-integer value for ({query}, {doc_title}): '{rel_str}'")
    
    except FileNotFoundError:
        print(f"       File not found: {relevance_csv}")
        print(f"      Run 'python evaluate.py' first to generate the annotation template.")
        return
    
    if num_annotations == 0:
        print(f"       No annotations found in {relevance_csv}")
        print(f"      Please fill in the 'relevant' column (0=not relevant, 1=partial, 2=relevant) for each row.")
        return
    
    print(f"       Loaded {num_annotations} relevance judgments\n")
    
    # ---- STEP 2: Load retrieval results ----
    print(f"[2/3] Loading retrieval results from '{results_csv}'...")
    
    retrieval_data = []
    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            retrieval_data = list(reader)
    
    except FileNotFoundError:
        print(f"       File not found: {results_csv}")
        return
    
    print(f"       Loaded {len(retrieval_data)} retrieval records\n")
    
    # ---- STEP 3: Compute metrics ----
    print("[3/3] Computing metrics (nDCG@10, Recall@10, MRR@10)...")
    
    # metrics[query][method] = {nDCG@10, Recall@10, MRR@10}
    metrics = {}
    
    for query in tqdm(relevance_dict.keys(), desc="Queries", ncols=80):
        if query not in metrics:
            metrics[query] = {}
        
        # Get results for this query
        query_results = [r for r in retrieval_data if r['query'] == query]
        query_methods = set(r['method'] for r in query_results)
        
        # Total relevant documents for this query (score >= 1, for Recall)
        total_relevant = sum(1 for v in relevance_dict[query].values() if v >= 1)
        
        for method in query_methods:
            # Get results for this query+method, sorted by rank
            method_results = sorted(
                [r for r in query_results if r['method'] == method],
                key=lambda x: int(x['rank'])
            )
            
            # Build relevance list
            relevances = []
            for result in method_results:
                doc_title = result['doc_title']
                rel = relevance_dict[query].get(doc_title, 0)
                relevances.append(rel)
            
            # Compute metrics
            ndcg = compute_ndcg(relevances, k=10)
            recall = compute_recall(relevances, k=10, num_relevant_total=total_relevant)
            mrr = compute_mrr(relevances, k=10)
            
            metrics[query][method] = {
                'nDCG@10': ndcg,
                'Recall@10': recall,
                'MRR@10': mrr,
                'num_relevant': total_relevant,
            }
    
    # ---- STEP 4: Aggregate by method ----
    method_metrics = defaultdict(lambda: {
        'nDCG@10': [],
        'Recall@10': [],
        'MRR@10': [],
    })
    
    for query_metrics in metrics.values():
        for method, scores in query_metrics.items():
            method_metrics[method]['nDCG@10'].append(scores['nDCG@10'])
            method_metrics[method]['Recall@10'].append(scores['Recall@10'])
            method_metrics[method]['MRR@10'].append(scores['MRR@10'])
    
    # ---- STEP 5: Display results ----
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (averaged across queries)")
    print("=" * 80 + "\n")
    
    print(f"{'Method':<25} {'nDCG@10':<12} {'Recall@10':<12} {'MRR@10':<12}")
    print("-" * 80)
    
    for method in sorted(method_metrics.keys()):
        scores = method_metrics[method]
        ndcg_mean = np.mean(scores['nDCG@10'])
        recall_mean = np.mean(scores['Recall@10'])
        mrr_mean = np.mean(scores['MRR@10'])
        
        print(f"{method:<25} {ndcg_mean:<12.4f} {recall_mean:<12.4f} {mrr_mean:<12.4f}")
    
    print("=" * 80 + "\n")

    # ---- STEP 3b: Per-query-type nDCG@10 breakdown ----
    print("nDCG@10 BY QUERY TYPE")
    print("-" * 80)
    method_order = ["bm25_only", "dense_only", "hybrid_weighted_sum", "hybrid_rrf"]
    type_order   = ["factual", "conceptual", "ambiguous", "technical"]

    # Accumulate nDCG per (type, method)
    type_ndcg = {t: {m: [] for m in method_order} for t in type_order}
    for query, query_metrics in metrics.items():
        qtype = QUERY_TYPES.get(query)
        if qtype is None:
            continue
        for method, scores in query_metrics.items():
            if method in type_ndcg[qtype]:
                type_ndcg[qtype][method].append(scores["nDCG@10"])

    col = 10
    header = f"{'Query Type':<14}|" + "".join(f"{'BM25':>{col}} " if m == 'bm25_only'
              else f"{'Dense':>{col}} " if m == 'dense_only'
              else f"{'Hybrid-WS':>{col}} " if m == 'hybrid_weighted_sum'
              else f"{'Hybrid-RRF':>{col}} "
              for m in method_order)
    print(header)
    print("-" * len(header))
    breakdown_rows = []
    for qtype in type_order:
        row_vals = []
        row_str = f"{qtype.capitalize():<14}|"
        for method in method_order:
            vals = type_ndcg[qtype][method]
            mean = float(np.mean(vals)) if vals else 0.0
            row_str += f"{mean:{col}.3f} "
            row_vals.append(mean)
        print(row_str)
        breakdown_rows.append([qtype] + row_vals)
    print("=" * 80 + "\n")

    # Save breakdown CSV
    breakdown_csv = "query_type_breakdown.csv"
    with open(breakdown_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_type", "bm25_only", "dense_only", "hybrid_weighted_sum", "hybrid_rrf"])
        writer.writerows(breakdown_rows)
    print(f" Query-type breakdown saved to '{breakdown_csv}'\n")

    # ---- STEP 3c: Relevance-type breakdown for top-1 results ----
    print("TOP-1 RESULT RELEVANCE-TYPE BREAKDOWN (% of queries)")
    print("-" * 80)
    tier_breakdown = defaultdict(lambda: defaultdict(int))
    for row in retrieval_data:
        if int(row["rank"]) == 1 and "relevance_type" in row and row["relevance_type"]:
            tier_breakdown[row["method"]][row["relevance_type"]] += 1

    if tier_breakdown:
        tiers = ["relevant", "partial", "not_relevant"]
        print(f"{'Method':<25} {'Relevant':>10} {'Partial':>10} {'Not Relevant':>14}")
        print("-" * 80)
        for method in sorted(tier_breakdown.keys()):
            counts = tier_breakdown[method]
            total = sum(counts.values())
            r = counts.get("relevant", 0) / total * 100
            p = counts.get("partial", 0) / total * 100
            n = counts.get("not_relevant", 0) / total * 100
            print(f"{method:<25} {r:>9.1f}% {p:>9.1f}% {n:>13.1f}%")
        print("=" * 80 + "\n")
    else:
        print("  (relevance_type column not present in retrieval_results.csv — re-run evaluate.py to populate)\n")
        print("=" * 80 + "\n")

    # ---- STEP 6: Save outputs ----
    # JSON output
    metrics_json = "metrics.json"
    output_data = {
        'query_level': metrics,
        'method_level': {
            m: {k: [float(v) for v in vals] for k, vals in scores.items()}
            for m, scores in method_metrics.items()
        },
        'method_summary': {
            m: {
                'nDCG@10_mean': float(np.mean(method_metrics[m]['nDCG@10'])),
                'nDCG@10_std': float(np.std(method_metrics[m]['nDCG@10'])),
                'Recall@10_mean': float(np.mean(method_metrics[m]['Recall@10'])),
                'Recall@10_std': float(np.std(method_metrics[m]['Recall@10'])),
                'MRR@10_mean': float(np.mean(method_metrics[m]['MRR@10'])),
                'MRR@10_std': float(np.std(method_metrics[m]['MRR@10'])),
            }
            for m in method_metrics.keys()
        }
    }
    
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f" Detailed metrics saved to '{metrics_json}'")
    
    # Text summary
    summary_txt = "metrics_summary.txt"
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYBRID IR EVALUATION - METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation Set: {len(metrics)} queries\n")
        f.write(f"Total Annotations: {sum(len(v) for v in relevance_dict.values())}\n\n")
        
        f.write(f"{'Method':<25} {'nDCG@10':<12} {'Recall@10':<12} {'MRR@10':<12}\n")
        f.write("-" * 80 + "\n")
        
        for method in sorted(method_metrics.keys()):
            scores = method_metrics[method]
            ndcg_mean = np.mean(scores['nDCG@10'])
            recall_mean = np.mean(scores['Recall@10'])
            mrr_mean = np.mean(scores['MRR@10'])
            
            f.write(f"{method:<25} {ndcg_mean:<12.4f} {recall_mean:<12.4f} {mrr_mean:<12.4f}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Summary saved to '{summary_txt}'")
    print("\n" + "=" * 80)

    run_significance_tests(results_csv)

def analyze_query_expansion_impact():
    results = []
    for q in TEST_QUERIES:
        query = q["query"]
        qtype = q["type"]
        
        # Run Hybrid-RRF without expansion
        score_no_qe = run_hybrid_rrf(query, expand=False)
        
        # Run Hybrid-RRF with expansion  
        expanded = expand_query(query)
        score_with_qe = run_hybrid_rrf(expanded, expand=False)
        
        results.append({
            "query": query,
            "type": qtype,
            "expanded_query": expanded,
            "ndcg_no_qe": score_no_qe,
            "ndcg_with_qe": score_with_qe,
            "delta": score_with_qe - score_no_qe
        })
    
    df = pd.DataFrame(results)
    
    # Print summary table by query type
    print(df.groupby("type")[["ndcg_no_qe","ndcg_with_qe","delta"]].mean().round(3))
    
    # Print top 3 where QE helped most
    print("\nQE helped most:")
    print(df.nlargest(3, "delta")[["query","expanded_query","delta"]])
    
    # Print top 3 where QE hurt most  
    print("\nQE hurt most:")
    print(df.nsmallest(3, "delta")[["query","expanded_query","delta"]])
    
    df.to_csv("query_expansion_impact.csv", index=False)



# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
def run_significance_tests(results_csv: str = "retrieval_results.csv"):
    from scipy import stats as sp_stats

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (nDCG@10)")
    print("=" * 80 + "\n")

    relevance_dict = {}
    try:
        with open("relevance_template.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel_str = row["relevant"].strip()
                if rel_str == "":
                    continue
                try:
                    rel = int(rel_str)
                except ValueError:
                    continue
                query = row["query"]
                if query not in relevance_dict:
                    relevance_dict[query] = {}
                relevance_dict[query][row["doc_title"]] = rel
    except FileNotFoundError:
        print("   relevance_template.csv not found — run evaluate.py --metrics first.")
        return

    try:
        with open(results_csv, "r", encoding="utf-8") as f:
            retrieval_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"   {results_csv} not found — run evaluate.py first.")
        return

    method_names = ["bm25_only", "dense_only", "hybrid_weighted_sum", "hybrid_rrf"]
    queries = sorted(relevance_dict.keys())
    method_scores: Dict[str, Dict[str, float]] = {}

    for method in method_names:
        method_scores[method] = {}
        for query in queries:
            method_results = sorted(
                [r for r in retrieval_data if r["query"] == query and r["method"] == method],
                key=lambda x: int(x["rank"])
            )
            relevances = [relevance_dict[query].get(r["doc_title"], 0) for r in method_results]
            method_scores[method][query] = compute_ndcg(relevances, k=10)

    comparisons = [
        ("Hybrid-RRF vs BM25",  "hybrid_rrf",           "bm25_only"),
        ("Hybrid-RRF vs Dense", "hybrid_rrf",            "dense_only"),
        ("Hybrid-WS vs BM25",   "hybrid_weighted_sum",   "bm25_only"),
        ("Hybrid-WS vs Dense",  "hybrid_weighted_sum",   "dense_only"),
        ("Dense vs BM25",       "dense_only",            "bm25_only"),
    ]

    results_rows = []
    ALPHA = 0.05

    for label, m1, m2 in comparisons:
        scores1 = np.array([method_scores[m1][q] for q in queries])
        scores2 = np.array([method_scores[m2][q] for q in queries])
        diffs = scores1 - scores2
        if np.all(diffs == 0):
            w_p = 1.0
        else:
            _, w_p = sp_stats.wilcoxon(scores1, scores2, alternative="two-sided", zero_method="wilcox")
        _, t_p = sp_stats.ttest_rel(scores1, scores2)
        w_sig = w_p < ALPHA
        t_sig = t_p < ALPHA
        if w_sig and t_sig:
            verdict = "Yes*"
        elif w_sig or t_sig:
            verdict = "Borderline"
        else:
            verdict = "No"
        results_rows.append({
            "comparison":  label,
            "method_a":    m1,
            "method_b":    m2,
            "wilcoxon_p":  w_p,
            "ttest_p":     t_p,
            "significant": verdict,
        })

    col_w = [24, 12, 10, 13]
    header = "{:<{}} | {:>{}} | {:>{}} | {:>{}}".format(
        "Comparison", col_w[0], "Wilcoxon p", col_w[1],
        "t-test p",   col_w[2], "Significant?", col_w[3])
    sep = "-" * (sum(col_w) + 10)
    print(header)
    print(sep)
    for r in results_rows:
        print("{:<{}} | {:>{}.4f} | {:>{}.4f} | {:>{}}".format(
            r["comparison"],  col_w[0],
            r["wilcoxon_p"],  col_w[1],
            r["ttest_p"],     col_w[2],
            r["significant"], col_w[3]))
    print("=" * 80 + "\n")

    sig_csv = "significance_results.csv"
    with open(sig_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# Per-query nDCG@10 scores"])
        writer.writerow(["query"] + method_names)
        for q in queries:
            writer.writerow([q] + [f"{method_scores[m][q]:.6f}" for m in method_names])
        writer.writerow([])
        writer.writerow(["# Statistical test p-values"])
        writer.writerow(["comparison", "method_a", "method_b", "wilcoxon_p", "ttest_p", "significant"])
        for r in results_rows:
            writer.writerow([r["comparison"], r["method_a"], r["method_b"],
                             f"{r['wilcoxon_p']:.6f}", f"{r['ttest_p']:.6f}", r["significant"]])

    print(f" Full results saved to '{sig_csv}'")
    print("=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    if "--metrics" in sys.argv or "--compute-metrics" in sys.argv:
        compute_metrics_from_annotations()
    else:
        run_evaluation()
