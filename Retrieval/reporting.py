import os
import json
import argparse
import sys
import math

def load_retrieval_results(root_dir: str):
    """
    Walks each subdirectory of root_dir, expecting folder names of the form:
        <track>_<conv_type>_<retriever>
    For each .jsonl file inside, loads all JSON entries into a list
    and stores it in a dict keyed by (track, conv_type, retriever).
    """
    results = {}
    # Walk through all files under root_dir
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith('.jsonl'):
                continue
            # Parse filename: <track>_<conv_type>_<retriever>_*.jsonl
            parts = os.path.basename(fname).split('_')
            if len(parts) < 3:
                continue  # unexpected filename
            track, conv_type, retriever = parts[0], parts[1], parts[2]
            key = (track, conv_type, retriever)
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

def main():
    parser = argparse.ArgumentParser(
        description="Load and organize retrieval results by track, conv_type, and retriever."
    )
    parser.add_argument(
        "--root_dir", "-r",
        default="./Retrieval/Results",
        help="Path to the folder containing retrieval-result subdirectories."
    )
    args = parser.parse_args()

    # determine reports directory (sibling of Results)
    base_dir = os.path.dirname(args.root_dir)  # .../Experiments/Retrieval
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'retrieval_report.txt')

    data = load_retrieval_results(args.root_dir)
    # For now, just print out the keys and number of entries
    for (track, conv_type, retriever), items in sorted(data.items()):
        print(f"{track} | {conv_type} | {retriever}: {len(items)} items")



    # Compute recall@k for multiple k values
    recall_results = {}  # key: (track, conv_type, retriever) -> {k: avg_recall}
    for (track, conv_type, retriever), items in sorted(data.items()):
        # choose k list based on conv_type
        ks = [1, 5, 10, ] if conv_type.lower() == "multi" else [1, 5, 10, 20]
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
            print(f"Recall@{k}: {rec_k[k]}")
        recall_results[(track, conv_type, retriever)] = rec_k

    # Compute Mean Reciprocal Rank at cutoff k (MRR@k)
    mrr_results = {}  # key: (track, conv_type, retriever) -> {k: avg_mrr_at_k}
    for (track, conv_type, retriever), items in sorted(data.items()):
        ks = [1, 5, 10] if conv_type.lower() == "multi" else [1, 5, 10, 20]
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
            print(f"MRR@{k}: {mrr_k[k]}")
        mrr_results[(track, conv_type, retriever)] = mrr_k

    # Compute Normalized Discounted Cumulative Gain at cutoff k (nDCG@k)
    ndcg_results = {}  # key: (track, conv_type, retriever) -> {k: avg_ndcg_at_k}
    for (track, conv_type, retriever), items in sorted(data.items()):
        ks = [1, 5, 10] if conv_type.lower() == "multi" else [1, 5, 10, 20]
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
            print(f"nDCG@{k}: {ndcg_k[k]}")
        ndcg_results[(track, conv_type, retriever)] = ndcg_k

    # Print LaTeX table
    def print_retrieval_latex(recall_results, f):
        tracks = ["S", "A", "T"]
        convs = ["Uni", "Multi"]
        # Header
        f.write("Experiment & Semantic & Arithmetic & Temporal & Average \\\n")
        f.write("K & Uni & Multi & Uni & Multi & Uni & Multi &  \\\n")
        f.write("\\hline\n")
        # For each retriever, group rows by retriever
        retrievers = sorted({r for (_, _, r) in recall_results})
        for retriever in retrievers:
            exp = f"RAG-{retriever}"
            # for each k (union of both conv types' ks)
            all_ks = sorted({k for (t, c, r), recs in recall_results.items() if r == retriever for k in recs})
            for i, k in enumerate(all_ks):
                row = []
                # first column: experiment name only on the first row for that retriever
                row.append(exp if i == 0 else "")
                row.append(str(k))
                # then, for each track and conv, insert recall or "-"
                for t in tracks:
                    for c in convs:
                        rec = recall_results.get((t, c, retriever), {}).get(k)
                        if rec is not None and k in recall_results.get((t, c, retriever), {}):
                            percent = rec * 100
                            row.append(f"{percent:.2f}")
                        else:
                            row.append("-")
                # compute average of all numeric cells collected in this row
                if 'avg_vals' in locals():
                    del avg_vals
                avg_vals = []
                for t in tracks:
                    for c in convs:
                        rec_val = recall_results.get((t, c, retriever), {}).get(k)
                        if rec_val is not None and k in recall_results.get((t, c, retriever), {}):
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
        tracks = ["S", "A", "T"]
        convs = ["Uni", "Multi"]
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
                for t in tracks:
                    # collect the metric for Uni and Multi if present
                    vals = [
                        results.get((t, c, retriever), {}).get(k)
                        for c in convs
                        if k in results.get((t, c, retriever), {})
                    ]
                    if vals:
                        avg = sum(vals) / len(vals)
                        row.append(f"{avg * 100:.2f}")
                    else:
                        row.append("-")
                # compute average across tracks for this k
                track_avgs = []
                for t in tracks:
                    vals_t = [
                        results.get((t, c, retriever), {}).get(k)
                        for c in convs
                        if k in results.get((t, c, retriever), {})
                    ]
                    if vals_t:
                        track_avgs.extend(vals_t)
                if track_avgs:
                    overall_avg = sum(track_avgs) / len(track_avgs)
                    row.append(f"{overall_avg * 100:.2f}")
                else:
                    row.append("-")
                f.write(" & ".join(row) + " \\\n")
        f.write("\n")

    def print_main_table(rec_results, mrr_results, ndcg_results, f):
        """Write the combined LaTeX table containing R@k, MRR@k, and NDCG@k in a single wide table."""
        # Header copied from the user‑provided template
        header = r"""%  \cellcolor{blue!15}
\begin{table*}[t]
\centering
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{ll*{16}{c}}
\toprule
\multirow{4}{*}{\textbf{Experiment}}
& \multirow{4}{*}{\(\boldsymbol{k}\)}
& \multicolumn{4}{c}{\textbf{Factual}}
& \multicolumn{4}{c}{\textbf{Arithmetic}}
& \multicolumn{4}{c}{\textbf{Temporal}}
& \multicolumn{4}{c}{\textbf{Average}}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-18}
& &
\multicolumn{2}{c}{Uni-Audience (Chat style)} &
\multicolumn{2}{c}{Multi-Audience (Forum style)} &
\multicolumn{2}{c}{Uni-Audience (Chat style)} &
\multicolumn{2}{c}{Multi-Audience (Forum style)} &
\multicolumn{2}{c}{Uni-Audience (Chat style)} &
\multicolumn{2}{c}{Multi-Audience (Forum style)} &
\multicolumn{2}{c}{Uni-Audience (Chat style)} &
\multicolumn{2}{c}{Multi-Audience (Forum style)}\\ 
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}
& &
MRR@\(k\) & NDCG@\(k\) &
MRR@\(k\) & NDCG@\(k\) &
MRR@\(k\) & NDCG@\(k\) &
MRR@\(k\) & NDCG@\(k\) &
MRR@\(k\) & NDCG@\(k\) &
MRR@\(k\) & NDCG@\(k\) &
MRR@\(k\) & NDCG@\(k\) & 
MRR@\(k\) & NDCG@\(k\) \\ 
\midrule
"""
        f.write(header)

        tracks = ["S", "A", "T"]
        convs = ["Uni", "Multi"]
        retrievers = sorted({r for (_, _, r) in rec_results})
        for retriever in retrievers:
            # Determine k list for this retriever (union over conv+track)
            ks = sorted({k for (t, c, r), vals in rec_results.items() if r == retriever for k in vals})
            multirow = len(ks)
            label = f"{retriever}"  # experiment label
            for row_idx, k in enumerate(ks):
                row_cells = []
                if row_idx == 0:
                    row_cells.append(f"\\multirow{{{multirow}}}{{*}}{{{label}}}")
                else:
                    row_cells.append("")  # empty cell under the multirow
                row_cells.append(str(k))

                # populate 18 metric cells in fixed order
                for t in tracks:
                    for c in convs:
                        m_val = mrr_results.get((t, c, retriever), {}).get(k)
                        n_val = ndcg_results.get((t, c, retriever), {}).get(k)
                        for metric_val in ( m_val, n_val):
                            if metric_val is None:
                                row_cells.append("--")
                            else:
                                row_cells.append(f"{metric_val * 100:.2f}")
                mrr_uni_vals = [mrr_results.get((t, "Uni", retriever), {}).get(k) for t in tracks]
                mrr_uni_vals = [v for v in mrr_uni_vals if v is not None]
                mrr_uni_avg = sum(mrr_uni_vals) / len(mrr_uni_vals) if mrr_uni_vals else None
                mrr_uni_str = f"{mrr_uni_avg * 100:.2f}" if mrr_uni_avg is not None else "--"

                ndcg_uni_vals = [ndcg_results.get((t, "Uni", retriever), {}).get(k) for t in tracks]
                ndcg_uni_vals = [v for v in ndcg_uni_vals if v is not None]
                ndcg_uni_avg = sum(ndcg_uni_vals) / len(ndcg_uni_vals) if ndcg_uni_vals else None
                ndcg_uni_str = f"{ndcg_uni_avg * 100:.2f}" if ndcg_uni_avg is not None else "--"
                if k == 20:
                    mrr_multi_str = '--'
                    ndcg_multi_str = '--'
                else:
                    mrr_multi_vals = [mrr_results.get((t, "Multi", retriever), {}).get(k) for t in tracks]
                    mrr_multi_vals = [v for v in mrr_multi_vals if v is not None]
                    mrr_multi_avg = sum(mrr_multi_vals) / len(mrr_multi_vals) if mrr_multi_vals else None
                    
                    ndcg_multi_vals = [ndcg_results.get((t, "Multi", retriever), {}).get(k) for t in tracks]
                    ndcg_multi_vals = [v for v in ndcg_multi_vals if v is not None]
                    ndcg_multi_avg = sum(ndcg_multi_vals) / len(ndcg_multi_vals) if ndcg_multi_vals else None
                    
                    mrr_multi_str = f"{mrr_multi_avg * 100:.2f}" if mrr_multi_avg is not None else "--"
                    ndcg_multi_str = f"{ndcg_multi_avg * 100:.2f}" if ndcg_multi_avg is not None else "--"
                # write the row
                f.write(" & ".join(row_cells) + f" & {mrr_uni_str} & {ndcg_uni_str} & {mrr_multi_str} & {ndcg_multi_str} \\\\ \n")
                # ----- Added: combined Uni+Multi averages -----
                # After the main row for this \(k\), add two additional rows:
                #   1) the combined average MRR of Uni and Multi
                #   2) the combined average nDCG of Uni and Multi
                # These averages are computed only from the values that exist.
                if (mrr_uni_avg is not None) or (mrr_multi_avg is not None):
                    # Combined MRR
                    if (mrr_uni_avg is not None) and (mrr_multi_avg is not None):
                        mrr_combined = (mrr_uni_avg + mrr_multi_avg) / 2
                    else:
                        mrr_combined = mrr_uni_avg if mrr_uni_avg is not None else mrr_multi_avg

                    # Combined nDCG
                    if (ndcg_uni_avg is not None) and (ndcg_multi_avg is not None):
                        ndcg_combined = (ndcg_uni_avg + ndcg_multi_avg) / 2
                    else:
                        ndcg_combined = ndcg_uni_avg if ndcg_uni_avg is not None else ndcg_multi_avg

                    # Build an empty row skeleton to keep column alignment
                    empty_cells = [''] * len(row_cells)

                    # Row for combined MRR (value placed in the first average column)
                    mrr_combined_str = f"{mrr_combined * 100:.2f}" if mrr_combined is not None else "--"
                    f.write(" & ".join(empty_cells) + f" & {mrr_combined_str} & -- & -- & -- \\\\ \n")

                    # Row for combined nDCG (value placed in the second average column)
                    ndcg_combined_str = f"{ndcg_combined * 100:.2f}" if ndcg_combined is not None else "--"
                    f.write(" & ".join(empty_cells) + f" & -- & {ndcg_combined_str} & -- & -- \\\\ \n")
                # ----- End added block -----
            f.write("\\specialrule{0.4pt}{\\aboverulesep}{\\belowrulesep}\n")
        # table footer
        f.write('''\\bottomrule 
\end{tabular}%
}
\caption{Ranking metrics (\textbf{MRR@\(k\)}, \textbf{NDCG@\(k\)}) for each reasoning track in both discourse settings. Dashes “--” appear when a track has fewer than \(k\) candidate passages.}
\label{tab:full_metrics}
\end{table*}''')
        

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

    # Write the full joint table requested by the user
    main_table_path = os.path.join(reports_dir, 'main_table.txt')
    with open(main_table_path, 'w', encoding='utf-8') as f_main:
        f_main.write("Comprehensive table of MRR@k and NDCG@k across all tracks and discourse settings.\n")
        print_main_table(recall_results, mrr_results, ndcg_results, f_main)

if __name__ == "__main__":
    main()
