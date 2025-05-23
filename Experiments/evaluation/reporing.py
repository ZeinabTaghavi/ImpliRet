import argparse
import os
import glob
from typing import List
import json
import statistics

from reports.utils.latex_format import save_latex_tables
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# # --- Heatmap plotting function ---
# def plot_span_heatmap(entries: List[dict], experiment_type: str, track: str, conv_type: str, report_output_folder: str, filter_mode: int = 0):
#     """
#     Build and save a heatmap where each row corresponds to a K value,
#     columns to token positions, and intensity to gold-span counts.
#     Y-axis labels include avg rouge-l-recall for each K.
#     """
#     # Filter entries
#     filtered = [e for e in entries
#                 if e["experiment_type"] == experiment_type
#                 and e["track"] == track
#                 and e["conv_type"] == conv_type]
#     if not filtered:
#         return
#     # Determine max span length
#     max_len = max(max(e["all_spans"]) for e in filtered)
#     # filter_mode: 0=all items, 1=only rouge1>0, -1=only rouge1==0
#     fmap = filter_mode
#     # Build a list of (avg_r1, counts) then sort by avg_r1 descending
#     rows = []
#     for e_k in filtered:
#         avg_r1 = e_k["report"]["rouge-l-recall"]["Avg"]
#         counts = [0] * max_len
#         scores = e_k.get("rouge1_scores", [])
#         for idx, gs in enumerate(e_k.get("gold_spans", [])):
#             # apply filter
#             r1 = scores[idx] if idx < len(scores) else 0.0
#             if fmap == 1 and r1 <= 0:
#                 continue
#             if fmap == -1 and r1 != 0:
#                 continue
#             if not gs:
#                 continue
#             start, end = gs
#             start = max(0, min(max_len, start))
#             end = max(start, min(max_len, end))
#             for i in range(start, end):
#                 counts[i] += 1
#         rows.append((avg_r1, counts))
#     # sort descending
#     rows.sort(key=lambda x: x[0], reverse=True)
#     matrix = [[0 for _ in range(max_len)] for _ in range(120)]

#     for index, (avg_r1, counts) in enumerate(rows):
#         for i in range(int(avg_r1*100), int(avg_r1*100) + 10):
#             if sum(matrix[119-i]) == 0:
#                 matrix[119-i] = counts
#             elif i < int(avg_r1*100)+12:
#                 matrix[119-i] = counts

#     y_labels = [f"{round(avg_r1 * 100, 2)}%" for avg_r1, _ in rows]
#     # Plot heatmap with custom colormap: white for zero, then light green→blue
#     plt.figure(figsize=(10, len(y_labels)*0.5 + 2))
#     # Create a colormap: white for zeros, then light green to blue
#     if filter_mode == 1:
#         cmap = mcolors.LinearSegmentedColormap.from_list(
#             'white_green',
#             [
#                 (0.0, 'white'),
#                 (0.0001, '#c7e9c0'),  # light green
#                 (0.4, '#2ca25f'),
#                 (1.0, '#006d2c')  # dark green
#             ]
#         )
#     elif filter_mode == -1:
#         cmap = mcolors.LinearSegmentedColormap.from_list(
#             'white_red',
#             [
#                 (0.0, 'white'),
#                 (0.0001, '#fee0d2'),  # light red
#                 (0.4, '#de2d26'),
#                 (1.0, '#a50f15')  # dark red
#             ]
#         )
#     else:
#         cmap = mcolors.LinearSegmentedColormap.from_list(
#             'white_green_blue',
#             [
#                 (0.0, 'white'),
#                 (0.0001, '#c7e9c0'),  # light green
#                 (0.4, '#2171b5'),
#                 (1.0, '#042B6B')  # blue
#             ]
#         )

#     # Use a norm so that zero maps to the white entry
#     norm = mcolors.Normalize(vmin=0, vmax=max(max(row) for row in matrix) if matrix else 1)

#     plt.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
#     cbar = plt.colorbar(label='Span Count')
#     # Add legend-like annotation explaining colors
#     cbar.ax.set_ylabel('Counting Span of Gold Passages', rotation=270, labelpad=15)
#     # Remove y-axis inversion, show highest Rouge at top
#     # Replace x-axis tick labels: do not show token positions
#     plt.xticks([])
#     plt.xlabel("Input Prompt Length")
#     y_positions = [119 - (int(avg_r1*100) + 5) for avg_r1, _ in rows]
#     plt.yticks(y_positions, y_labels)
#     plt.ylabel(f"Average ROUGE-1")
#     plt.figtext(0.5, 0.01, 'Counting Span of Gold Passages in the Input Prompt, while model answered correctly.', 
#                 wrap=True, horizontalalignment='center', fontsize=8)
#     # Ensure output folder exists
#     figs_dir = os.path.join(report_output_folder, "figures")
#     os.makedirs(figs_dir, exist_ok=True)
#     out_path = os.path.join(figs_dir, f"{experiment_type}_{track}_{conv_type}_mode_{filter_mode}_span_heatmap.png")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()
#     print(f"saved {out_path}")

# Loader class for dataset answers
class DatasetAnswerLoader:
    """
    Loads all JSONL datasets at initialization and provides
    an attach_answers method to annotate result items.
    """
    def __init__(self, dataset_folder: str):
        self.data = {}
        for fname in os.listdir(dataset_folder):
            if not fname.endswith(".jsonl"):
                continue
            track, conv = fname[:-6].split("_", 1)
            path = os.path.join(dataset_folder, fname)
            rows = []
            with open(path, "r", encoding="utf-8") as df:
                for line in df:
                    rows.append(json.loads(line))
            self.data[(track, conv)] = rows

    def attach_answers(self, results_list: list, track: str, conv_type: str):
        """
        For each item in results_list, attach the reference answer
        from the preloaded dataset rows.
        """
        rows = self.data.get((track, conv_type), [])
        for item in results_list:
            rid = item.get("id")
            if isinstance(rid, int) and 0 <= rid < len(rows):
                row = rows[rid]
                if conv_type.lower() == "multi":
                    # item["gold_input"] = f"{row['message_date']}, {row['user']}: {row['user_response']}"
                    gold_input = row['context']
                    item_prompt = item["prompt"]
                    if (gold_input in item_prompt):
                        start_idx = item_prompt.find(gold_input)
                        end_idx = start_idx + len(gold_input)
                        item["gold_span"] = (start_idx, end_idx)
                        item["all_spans"] = len(item_prompt)
                    else:
                        item['gold_span'] = None
                else:
                    # item["gold_input"] = str([f"{conv[0]}, {conv[1]}: {conv[2]}" for conv in row['conversation']])
                    gold_input = row['context']
                    item_prompt = item["prompt"]
                    if (gold_input in item_prompt):
                        start_idx = item_prompt.find(gold_input)
                        end_idx = start_idx + len(gold_input)
                        item["gold_span"] = (start_idx, end_idx)
                        item["all_spans"] = len(item_prompt)
                    else:
                        item['gold_span'] = None

        return results_list


def reporting(result_path: str, metrics: List[str], report_output_folder: str, dataset_folder: str, span_filter: int):
    # Ensure the input and output directories exist
    if not os.path.isdir(result_path):
        raise FileNotFoundError(f"Results directory not found: {result_path}")
    os.makedirs(report_output_folder, exist_ok=True)

    # Initialize dataset loader for answers
    answer_loader = DatasetAnswerLoader(dataset_folder)

    # List all files under the result path and its subdirectories

    entries = []
    # Recursively find every JSON result file, regardless of nesting depth
    pattern = os.path.join(result_path, "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        fname = os.path.basename(path)
        parts = fname.split('_')
        track = parts[0]
        conv_type = parts[1]
        if "retrieval" in parts:
            experiment_type = "RAG"
            retriever_type = parts[-2]
            k = parts[-4]
        else:
            experiment_type = "LC"
            retriever_type = None
            k = parts[-2]
        try:
            k = int(k)
        except ValueError:
            raise ValueError(f"k is not an integer: {k}")
        # Derive model_name from filename
        model_name = fname.split(f"_{k}_")[0].split(f"_{conv_type}_")[-1].strip()
        entries.append({
            "experiment_type": experiment_type,
            "track": track,
            "conv_type": conv_type,
            "model_name": model_name,
            "retriever_type": retriever_type,
            "k": k,
            "result_file": path
        })
  

    # Compute detailed report for each metric
    for entry in entries:
        # Load JSON results
        with open(entry["result_file"], "r", encoding="utf-8") as jf:
            data_json = json.load(jf)
        results_list = data_json.get("results", [])
        # Attach reference answers from the dataset
        results_list = answer_loader.attach_answers(results_list, entry["track"], entry["conv_type"])
        # Store all per-item gold spans and full spans on the entry
        entry["gold_spans"] = [item.get("gold_span") for item in results_list]
        entry["all_spans"]  = [item.get("all_spans")  for item in results_list]
        n = len(results_list) or 1

        # store per-item rouge-1 scores for span filtering
        entry["rouge1_scores"] = [
            item.get("score")["rouge-1-recall"]
            for item in results_list
        ]

        # Initialize per-entry report dict
        entry["report"] = {}

        for m in metrics:
            if m == "EM":
                scores = [item.get("score", {}).get("EM", 0) for item in results_list]
                avg = sum(scores) / n
                num_p = sum(1 for s in scores if s == 1)
                num_n = n - num_p
                mn = min(scores)
                mx = max(scores)
                median = statistics.median(scores)
                entry["report"]["EM"] = {
                    "Avg": round(avg * 100, 2),
                    "Num_P": num_p,
                    "Num_N": num_n,
                    "min": round(mn * 100, 2),
                    "max": round(mx * 100, 2),
                    "Median": round(median * 100, 2)
                }
            elif m == "contains":
                scores = [item.get("score", {}).get("contains", 0) for item in results_list]
                avg = sum(scores) / n
                num_p = sum(1 for s in scores if s == 1)
                num_n = n - num_p
                mn = min(scores)
                mx = max(scores)
                median = statistics.median(scores)
                entry["report"]["contains"] = {
                    "Avg": round(avg * 100, 2),
                    "Num_P": num_p,
                    "Num_N": num_n,
                    "min": round(mn * 100, 2),
                    "max": round(mx * 100, 2),
                    "Median": round(median * 100, 2)
                }
            elif m == "rouge-recall":
                for sub in ["1", "2", "l"]:
                    key = f"rouge-{sub}-recall"
                    scores = [item.get("score", {}).get(key, 0.0) for item in results_list]
                    avg = sum(scores) / n
                    mn = min(scores)
                    mx = max(scores)
                    median = statistics.median(scores)
                    # population standard deviation
                    sd = statistics.pstdev(scores) if n > 1 else 0.0
                    entry["report"][key] = {
                        "Avg": round(avg * 100, 2),
                        "Std": round(sd * 100, 2),
                        "min": round(mn * 100, 2),
                        "max": round(mx * 100, 2),
                        "Median": round(median * 100, 2)
                    }

        # Compute token usage stats (flat fields)
        prompt_tokens = [item.get("prompt_tokens", 0) for item in results_list]
        completion_tokens = [item.get("completion_tokens", 0) for item in results_list]
        total_tokens = [item.get("total_tokens", 0) for item in results_list]

        entry["report"]["tokens"] = {
            "prompt": {
                "Min": round(min(prompt_tokens), 5),
                "Max": round(max(prompt_tokens), 5),
                "Mean": int(round(statistics.mean(prompt_tokens))),
                "Median": round(statistics.median(prompt_tokens), 5)
            },
            "completion": {
                "Min": round(min(completion_tokens), 5),
                "Max": round(max(completion_tokens), 5),
                "Mean": int(round(statistics.mean(completion_tokens))),
                "Median": round(statistics.median(completion_tokens), 5)
            },
            "total": {
                "Min": round(min(total_tokens), 5),
                "Max": round(max(total_tokens), 5),
                "Mean": int(round(statistics.mean(total_tokens))),
                "Median": round(statistics.median(total_tokens), 5)
            }
        }

        # Compute sentence count stats on responses
        sentence_counts = []
        for item in results_list:
            resp = item.get("response", "")
            # simple split on periods
            count = len([s for s in resp.split('.') if s.strip()])
            sentence_counts.append(count)
        entry["report"]["sentences"] = {
            "Min": round(min(sentence_counts), 5),
            "Max": round(max(sentence_counts), 5),
            "Mean": round(statistics.mean(sentence_counts), 5),
            "Median": round(statistics.median(sentence_counts), 5)
        }

    model_categories = {}
    for entry in entries:
        if entry["model_name"] not in model_categories:
            model_categories[entry["model_name"]] = []
        model_categories[entry["model_name"]].append(entry)

    # Print parsed entries
    print("Parsed entries:")
    print(f"Total entries: {len(entries)}")
    print(f"model_categories: {model_categories.keys()}")
    print(f"len(model_categories[list(model_categories.keys())[0]]): {len(model_categories[list(model_categories.keys())[0]])}")
    # For now, just print the values (further logic will come later)
    print(f"Result path: {result_path}")
    print(f"Metrics: {metrics}")
    print(f"Report output folder: {report_output_folder}")


    # # Plot span heatmaps for LC across all track/conv combinations
    # for track in ["T", "A", "S"]:
    #     for conv_type in ["Multi", "Uni"]:
    #         for filter_mode in [0, 1, -1]:
    #             plot_span_heatmap(entries, "LC", track, conv_type, report_output_folder, filter_mode)

    # Generate LaTeX tables for each metric
    save_latex_tables(entries, metrics, report_output_folder)

    # --- Generate token counting report ---
    token_report_path = os.path.join(report_output_folder, "token_counting.txt")
    with open(token_report_path, "w", encoding="utf-8") as tf:
        header = [
            "Experiment", "Track", "ConvType", "Model", "Retriever", "K",
            "PromptMean", "CompletionMean", "TotalMean"
        ]
        tf.write("\t".join(header) + "\n")
        for entry in entries:
            exp = entry["experiment_type"]
            track = entry["track"]
            conv = entry["conv_type"]
            model = entry["model_name"]
            retriever = entry["retriever_type"] or "-"
            k_val = entry["k"]
            tokens = entry.get("report", {}).get("tokens", {})
            p_mean_val = tokens.get("prompt", {}).get("Mean")
            c_mean_val = tokens.get("completion", {}).get("Mean")
            t_mean_val = tokens.get("total", {}).get("Mean")
            prompt_mean = "NA" if p_mean_val is None else str(int(round(p_mean_val)))
            completion_mean = "NA" if c_mean_val is None else str(int(round(c_mean_val)))
            total_mean = "NA" if t_mean_val is None else str(int(round(t_mean_val)))
            row = [
                str(exp), str(track), str(conv), str(model), str(retriever), str(k_val),
                str(prompt_mean), str(completion_mean), str(total_mean)
            ]
            tf.write("\t".join(row) + "\n")
    print(f"Token counting report written to {token_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate experiment reports from JSON result files"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="Experiments/evaluation/results",
        help="Path to directory containing experiment result JSON files"
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="Dataset_Generation/Data",
        help="Path to directory containing dataset files"
    )
    parser.add_argument(
        "--report_output_folder",
        type=str,
        default="Experiments/evaluation/reports",
        help="Directory to save generated reports"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="EM,contains,rouge-recall",
        help="Comma-separated list of metrics to include in the report"
    )
    parser.add_argument(
        "--span_filter",
        type=int,
        default=0,
        help="Span filter mode: 0=all items, 1=only rouge1>0, -1=only rouge1==0"
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dataset_folder = args.dataset_folder
    span_filter = args.span_filter
    reporting(args.result_path, metrics, args.report_output_folder, args.dataset_folder, span_filter)
