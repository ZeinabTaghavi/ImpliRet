import argparse
import os
import glob
from typing import List
import json
import statistics

from RAG_Style.reports.utils.latex_format import save_latex_tables
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datasets import load_dataset

# Loader class for dataset answers
class DatasetAnswerLoader:
    """
    Loads all JSONL datasets at initialization and provides
    an attach_answers method to annotate result items.
    """
    def __init__(self, dataset_folder: str = None):
        """
        Loads answer rows directly from the public HuggingFace dataset
        `zeinabTaghavi/ImpliRet` instead of local JSONL files.

        The dataset provides two configurations (`multispeaker`, `unispeaker`)
        and three task splits (`arithmetic`, `wknow`, `temporal`).  For
        backward‑compatibility with the existing reporting pipeline we map
        them to the original keys that were previously inferred from local
        filenames:

            arithmetic  -> 'A'
            wknow       -> 'W'
            temporal    -> 'T'
            multispeaker → 'Multi'
            unispeaker  → 'Uni'
        """
        self.data = {}

        category = ["arithmetic", "wknow", "temporal"]
        discourse_type = ["multispeaker", "unispeaker"]

        dataset_name = "zeinabTaghavi/ImpliRet"
        # iterate over each configuration (conversation style)
        for dis in discourse_type:
            # load all splits for the given configuration
            for cat in category:
                ds_dict = load_dataset("zeinabTaghavi/ImpliRet", name=dis, split=cat)
                # convert HuggingFace Dataset to a plain list of dicts
                rows = ds_dict.to_list() if hasattr(ds_dict, "to_list") else list(ds_dict)
                self.data[(cat, dis)] = rows

    def attach_answers(self, results_list: list, category: str, discourse_type: str):
        """
        For each item in results_list, attach the reference answer
        from the preloaded dataset rows.
        """
        rows = self.data.get((category, discourse_type), [])
        for item in results_list:
            rid = item.get("id")
            row = rows[rid]

            gold_input = row['pos_document']
            item_prompt = item["prompt"]
            if (gold_input in item_prompt):
                start_idx = item_prompt.find(gold_input)
                end_idx = start_idx + len(gold_input)
                item["gold_span"] = (start_idx, end_idx)
                item["all_spans"] = len(item_prompt)
            else:
                item['gold_span'] = None
          

        return results_list


def reporting(result_path: str, metrics: List[str], report_output_folder: str, warn: bool = True):
    # Ensure the input and output directories exist
    if not os.path.isdir(result_path):
        raise FileNotFoundError(f"Results directory not found: {result_path}")
    os.makedirs(report_output_folder, exist_ok=True)


    # List all files under the result path and its subdirectories

    entries = []
    # Recursively find every JSON result file, regardless of nesting depth
    pattern = os.path.join(result_path, "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        fname = os.path.basename(path)
        parts = fname.split('_')
        category = parts[0]
        discourse_type = parts[1]
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
        model_name = fname.split(f"_{k}_")[0].split(f"_{discourse_type}_")[-1].strip()
        entries.append({
            "experiment_type": experiment_type,
            "category": category,
            "discourse_type": discourse_type,
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


        n = len(results_list) or 1

        # Initialize per-entry report dict
        entry["report"] = {}
        
        if type(metrics) == str:
            metrics = [m.strip() for m in metrics.split(',') if m.strip()]

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
        prompt_tokens = [item.get("tokens", {}).get("prompt", 0) for item in results_list]
        completion_tokens = [item.get("tokens", {}).get("completion", 0) for item in results_list]
        total_tokens = [item.get("tokens", {}).get("total", 0) for item in results_list]

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


    # Generate LaTeX tables for each metric
    save_latex_tables(entries, metrics, report_output_folder, warn=warn)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate experiment reports from JSON result files"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="RAG_Style/results",
        help="Path to directory containing experiment result JSON files"
    )
    parser.add_argument(
        "--report_output_folder",
        type=str,
        default="RAG_Style/reports",
        help="Directory to save generated reports"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="EM,rouge-recall",
        help="Comma-separated list of metrics to include in the report"
    )
    parser.add_argument(
        "--warn",
        type=bool,
        default=True,
        help="Whether to print warnings"
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    reporting(args.result_path, metrics, args.report_output_folder)
