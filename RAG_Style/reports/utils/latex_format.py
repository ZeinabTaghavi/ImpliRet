from typing import List
import os

# New function to save LaTeX tables for each metric
def save_latex_tables(entries: List[dict], metrics: List[str], report_output_folder: str):
    # Mapping of track codes to names
    track_names = {"A": "Arithmetic", "W": "World.K", "T": "Temporal"}
    # Sort all k values except -1 first, then append -1 last
    all_ks = sorted(k for k in {e["k"] for e in entries} if k != -1)
    ks = all_ks + ([-1] if -1 in {e["k"] for e in entries} else [])

    # Iterate each metric
    real_metrics = []
    for m in metrics:
        if m == "rouge-recall":
            real_metrics += ["rouge-1-recall", "rouge-2-recall", "rouge-l-recall"]
        else:
            real_metrics.append(m)
    # Add token and sentence metrics
    real_metrics += ["tokens-prompt", "tokens-completion", "tokens-total", "sentences"]


    for m in real_metrics:
        added_entries = 0
        lines = []
        # Header: Experiment and metric track names
        header1 = ["Experiment"] + [track_names[t] for t in ["A", "W", "T"]] + ['Uni Avg', 'Multi Avg']
        lines.append(" | ".join(header1) + " \\\\")
        # Second header: K and Uni/Multi
        header2 = ["K"] + ["Uni & Multi"] * 3 + ["", ""]
        lines.append(" | ".join(header2) + " \\\\")
        # Separator
        lines.append("\\hline")
        # Build a lookup by combined experiment key, k, track, conv_type

        lookup = {}
        for e in entries:
            exp_key = e["model_name"] + (f"-{e['retriever_type']}" if e["retriever_type"] else "")
            lookup[(exp_key, e["k"], e["category"], e["discourse_type"])] = e
        # For each experiment key
        for exp_key in sorted({
            e["model_name"] + (f"-{e['retriever_type']}" if e["retriever_type"] else "")
            for e in entries # LC, RAG-bm25, RAG-dragonplus, ... (other RAGs that will be added later)
        }):
            for k in ks:
                row = [exp_key, str(k) if k != -1 else "all(-1)"]
                uni_values = []
                multi_values = []
                
                for t in ["A", "W", "T"]:
                    for dis in ["Uni", "Multi"]:
                        entry = lookup.get((exp_key, k, t, dis))
                        if entry:
                            if m.startswith("tokens-"):
                                # m is "tokens-prompt", "tokens-completion", or "tokens-total"
                                stat = m.split("-")[1]
                                val = entry["report"]["tokens"][stat]["Mean"]
                            elif m == "sentences":
                                val = entry["report"]["sentences"]["Mean"]
                            else:
                                val = entry["report"][m]["Avg"]
                            row.append(f"{val:.2f}")
                            if dis == "Uni":
                                uni_values.append(val)
                            else:
                                multi_values.append(val)
                            added_entries += 1
                        else:
                            if not (("RAG" in exp_key and k == -1) or (dis=="Multi" and k==20)):
                                print(f"Warning: missing entry: {exp_key}, {k}, {t}, {dis}")
                                print(f"entry: {entry}")
                                print()
                            row.append("-")
                
                # Calculate averages
                uni_avg = sum(uni_values) / len(uni_values) if uni_values else "-"
                multi_avg = sum(multi_values) / len(multi_values) if multi_values else "-"
                
                # Add averages to row
                row.append(f"{uni_avg:.2f}" if uni_avg != "-" else "-")
                row.append(f"{multi_avg:.2f}" if multi_avg != "-" else "-")
                    
                lines.append(" & ".join(row) + " \\\\")
            # Separator after each experiment
            lines.append("\\hline")

        # ---- Additional summary: average between Uni and Multi for each k ----
        lines.append("")
        lines.append("\\bigskip")  # vertical space after the first table
        lines.append("\\textbf{Average between Uni and Multi per $k$} \\\\")
        lines.append("\\hline")
        header_combined = ["Experiment", "K", "Avg(Uni+Multi)"]
        lines.append(" & ".join(header_combined) + " \\\\")
        for exp_key in sorted({
            e["model_name"] + (f"-{e['retriever_type']}" if e["retriever_type"] else "")
            for e in entries
        }):
            for k in ks:
                combined_values = []
                for t in ["A", "W", "T"]:
                    for dis in ["Uni", "Multi"]:
                        entry = lookup.get((exp_key, k, t, dis))
                        if entry:
                            if m.startswith("tokens-"):
                                stat = m.split("-")[1]
                                val = entry["report"]["tokens"][stat]["Mean"]
                            elif m == "sentences":
                                val = entry["report"]["sentences"]["Mean"]
                            else:
                                val = entry["report"][m]["Avg"]
                            combined_values.append(val)
                if combined_values:
                    combined_avg = sum(combined_values) / len(combined_values)
                    lines.append(f"{exp_key} & {str(k) if k != -1 else 'all(-1)'} & {combined_avg:.2f} \\\\")
        lines.append("\\hline")
        lines.append("")  # blank line before any subsequent content
        # Write to file
        fname = os.path.join(report_output_folder, f"{m}_table.tex")
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Added {added_entries} entries for {m}")
        print(f"Missed {len(entries) - added_entries} entries for {m}")
        print(f"Filed saved to {fname}")
        print()