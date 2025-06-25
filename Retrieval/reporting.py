import os
import json
import argparse
import sys
import math

def load_retrieval_results(root_dir: str):
    """
    Walks each subdirectory of root_dir, expecting folder names of the form:
        <category>_<discourse>_<retriever>
    For each .jsonl file inside, loads all JSON entries into a list
    and stores it in a dict keyed by (track, conv_type, retriever).
    """
    results = {}
    # Walk through all files under root_dir
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith('.jsonl'):
                continue
            # Parse filename: <category>_<discourse>_<retriever>_*.jsonl
            parts = os.path.basename(fname).split('_')
            if len(parts) < 3:
                continue  # unexpected filename
            category, discourse, retriever = parts[0], parts[1], parts[2]
            key = (category, discourse, retriever)
            # Initialize list if first time
            if key not in results:
                results[key] = []
            # Load JSONL file
            file_path = os.path.join(root, fname)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        results[key].append(obj)
                    except json.JSONDecodeError:
                        # skip malformed lines
                        continue
    return results

def evaluate_run(root_dir="./Retrieval/results"):


    # determine reports directory (sibling of Results)
    base_dir = os.path.dirname(root_dir)  # .../Experiments/Retrieval
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'retrieval_report.txt')

    data = load_retrieval_results(root_dir)
    # For now, just print out the keys and number of entries
    for (category, discourse, retriever), items in sorted(data.items()):
        print(f"{category} | {discourse} | {retriever}: {len(items)} items")



    # Compute recall@k for multiple k values
    recall_results = {}  # key: (category, discourse, retriever) -> {k: avg_recall}
    for (category, discourse, retriever), items in sorted(data.items()):
        # choose k list based on conv_type
        ks = [10] 
        rec_k = {}
        for k in ks:
            scores = []
            for entry in items:
                user_id = entry.get("user_ID", 0)
                gold_idx = entry["gold_index"]
                tuples = entry["index_score_tuple_list"]
                # sort by score descending
                top_k = [idx for idx, score in sorted(tuples, key=lambda x: x[1], reverse=True)[:k]]
                scores.append(1 if gold_idx in top_k else 0)
            rec_k[k] = sum(scores) / len(scores)
            print(f"Recall@{k}: {rec_k[k]:.4f}")
        recall_results[(category, discourse, retriever)] = rec_k

    # Compute Mean Reciprocal Rank at cutoff k (MRR@k)
    mrr_results = {}  # key: (category, discourse, retriever) -> {k: avg_mrr_at_k}
    for (category, discourse, retriever), items in sorted(data.items()):
        ks = [10] 
        mrr_k = {}
        for k in ks:
            rr_vals = []
            for entry in items:
                gold_idx = entry["gold_index"]
                tuples = entry["index_score_tuple_list"]
                sorted_tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
                rank = next((rank for rank, (idx, _) in enumerate(sorted_tuples[:k], start=1) if idx == gold_idx), None)
                rr_vals.append(1.0 / rank if rank is not None else 0.0)
            mrr_k[k] = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0
            print(f"MRR@{k}: {mrr_k[k]:.4f}")
        mrr_results[(category, discourse, retriever)] = mrr_k

    # Compute Normalized Discounted Cumulative Gain at cutoff k (nDCG@k)
    ndcg_results = {}  # key: (category, discourse, retriever) -> {k: avg_ndcg_at_k}
    for (category, discourse, retriever), items in sorted(data.items()):
        ks = [10] 
        ndcg_k = {}
        for k in ks:
            gains = []
            for entry in items:
                gold_idx = entry["gold_index"]
                tuples = entry["index_score_tuple_list"]
                sorted_tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
                rank = next((rank for rank, (idx, _) in enumerate(sorted_tuples[:k], start=1) if idx == gold_idx), None)
                if rank is not None:
                    gain = 1.0 / math.log2(rank + 1)
                else:
                    gain = 0.0
                gains.append(gain)
            ndcg_k[k] = sum(gains) / len(gains) if gains else 0.0
            print(f"nDCG@{k}: {ndcg_k[k]:.4f}")
        ndcg_results[(category, discourse, retriever)] = ndcg_k

    # Print LaTeX table
    def print_retrieval_latex(recall_results, f):
        categories = ["wknow", "arithmetic", "temporal"]
        discourses = ["unispeaker", "multispeaker"]
        # Header
        f.write("Experiment & Semantic & Arithmetic & Temporal & Average \\\n")
        f.write("K & Uni & Multi & Uni & Multi & Uni & Multi & Uni & Multi \\\n")
        f.write("\\hline\n")
        # For each retriever, group rows by retriever
        retrievers = sorted({r for (_, _, r) in recall_results})
        for retriever in retrievers:
            exp = f"RAG-{retriever}"
            # for each k (union of both conv types' ks)
            all_ks = sorted({k for (c, d, r), recs in recall_results.items() if r == retriever for k in recs})
            for i, k in enumerate(all_ks):
                row = []
                # first column: experiment name only on the first row for that retriever
                row.append(exp if i == 0 else "")
                row.append(str(k))
                # then, for each track and conv, insert recall or "-"
                for c in categories:
                    for d in discourses:
                        rec = recall_results.get((c, d, retriever), {}).get(k)
                        if rec is not None and k in recall_results.get((c, d, retriever), {}):
                            percent = rec * 100
                            row.append(f"{percent:.2f}")
                        else:
                            row.append("-")
                # compute average of all numeric cells collected in this row
                if 'avg_vals' in locals():
                    del avg_vals
                avg_vals = []
                for c in categories:
                    for d in discourses:
                        rec_val = recall_results.get((c, d, retriever), {}).get(k)
                        if rec_val is not None and k in recall_results.get((c, d, retriever), {}):
                            avg_vals.append(rec_val)
                if avg_vals:
                    avg = sum(avg_vals) / len(avg_vals)
                    row.append(f"{avg * 100:.2f}")
                else:
                    row.append("-")
                f.write(" & ".join(row) + " \\\\\n")
            
        f.write("\n")


    def print_retrieval_latex_avg(results, f):
        """
        Write a LaTeX table that averages the Uni and Multi conversational modes for each track.
        Accepts the same `results` structure as the detailed table function.
        """
        categories = ["wknow", "arithmetic", "temporal"]
        discourses = ["unispeaker", "multispeaker"]
        # Header
        f.write("Experiment & K & Semantic & Arithmetic & Temporal & Average \\\n")
        f.write("\\hline\n")
        # Group rows by retriever
        retrievers = sorted({r for (_, _, r) in results})
        for retriever in retrievers:
            exp = f"RAG-{retriever}"
            # gather union of all k values available for this retriever
            all_ks = sorted({k for (t, c, r), recs in results.items() if r == retriever for k in recs})
            for i, k in enumerate(all_ks):
                row = []
                row.append(exp if i == 0 else "")  # experiment label only on first row
                row.append(str(k))
                for c in categories:
                    # collect the metric for Uni and Multi if present
                    vals = [
                        results.get((c, d, retriever), {}).get(k)
                        for d in discourses
                        if k in results.get((c, d, retriever), {})
                    ]
                    if vals:
                        avg = sum(vals) / len(vals)
                        row.append(f"{avg * 100:.2f}")
                    else:
                        row.append("-")
                # compute average across tracks for this k
                track_avgs = []
                for c in categories:
                    for d in discourses:
                        vals_c = [
                            results.get((c, d, retriever), {}).get(k)
                            for d in discourses
                            if k in results.get((c, d, retriever), {})
                        ]
                        if vals_c:
                            track_avgs.extend(vals_c)
                if track_avgs:
                    overall_avg = sum(track_avgs) / len(track_avgs)
                    row.append(f"{overall_avg * 100:.2f}")
                else:
                    row.append("-")
                f.write(" & ".join(row) + " \\\n")
        f.write("\n")

        

    with open(report_path, 'w', encoding='utf-8') as f:
        # ----- Descriptive lines before each table -----
        f.write("Recall@k results for each reasoning track in both discourse settings.\n")
        print_retrieval_latex(recall_results, f)

        f.write("\nMRR@k results for each reasoning track in both discourse settings.\n")
        print_retrieval_latex(mrr_results, f)

        f.write("\nNDCG@k results for each reasoning track in both discourse settings.\n")
        print_retrieval_latex(ndcg_results, f)

        # Averaged Uni + Multi tables
        f.write("\nRecall@k averaged over Uni- and Multi‑audience settings.\n")
        print_retrieval_latex_avg(recall_results, f)

        f.write("\nMRR@k averaged over Uni- and Multi‑audience settings.\n")
        print_retrieval_latex_avg(mrr_results, f)

        f.write("\nNDCG@k averaged over Uni- and Multi‑audience settings.\n")
        print_retrieval_latex_avg(ndcg_results, f)

    return recall_results, mrr_results, ndcg_results

if __name__ == "__main__":
    recall_results, mrr_results, ndcg_results = evaluate_run()
