

"""
Simple wrapper around `ExperimentTester` that mirrors NoLiMa’s run‑script:

    python run_tests.py --config evaluation/run_config/T_Uni_llama.yaml

* The YAML config is parsed by `jsonargparse`.
* Any field can be overridden on the CLI.
* Paths for the dataset JSONL and results folder are derived exactly the
  same way the CLI block inside `async_evaluate.py` does, so you can run
  either file directly with the same YAML.
"""

from __future__ import annotations
import os
from jsonargparse import ArgumentParser, ActionConfigFile

# Import the tester (relative import works when run from repo root)
from async_evaluate import ExperimentTester


def main() -> None:
    parser = ArgumentParser(description="Metatag‑Indexing Experiment Runner")

    # Allow `--config some.yaml`
    parser.add_argument("--config", action=ActionConfigFile, help="YAML run_config")

    # All fields that might appear in the YAML
    parser.add_argument("--dataset_folder", type=str, default="./Dataset_Generation/Data/")
    parser.add_argument("--track", type=str, choices=["T", "A", "S"], default="T")
    parser.add_argument("--type", dest="data_type", type=str, choices=["Uni", "Multi"], default="Uni")
    parser.add_argument("--output_folder", type=str, default="./MetatagIndexing/Experiments/Results/")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--user_retrieval", type=bool, default=False)
    parser.add_argument("--retriever", type=str, default="BM25")
    parser.add_argument("--retriever_index_folder", type=str, default="./MetatagIndexing/Experiments/Retrieval/Results/")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_configs_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="EM")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print("[run_tests] Parsed CLI / YAML arguments")

    # ------------------------------------------------------------------ #
    # derive paths from the provided knobs                               #
    # ------------------------------------------------------------------ #
    data_path = os.path.join(
        args.dataset_folder, f"{args.track}_{args.data_type}.jsonl"
    )
    model_cfg_path = os.path.join(
        args.model_configs_dir, f"{args.model_name}.json"
    )
    results_dir = os.path.join(
        args.output_folder, f"{args.track}_{args.data_type}", args.model_name
    )

    # ------------------------------------------------------------------ #
    # sanity checks                                                      #
    # ------------------------------------------------------------------ #
    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Model‑config not found: {model_cfg_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset split not found: {data_path}")
    print("[run_tests] Found model‑config and dataset files.")

    # ------------------------------------------------------------------ #
    # run                                                                #
    # ------------------------------------------------------------------ #
    print("[run_tests] Initialising ExperimentTester …")
    tester = ExperimentTester(
        model_name=args.model_name,
        model_configs_dir=args.model_configs_dir,
        dataset_folder=args.dataset_folder,
        track=args.track,
        conv_type=args.data_type,                 # Uni or Multi
        results_dir=results_dir,
        retriever=args.retriever,
        retriever_index_folder=args.retriever_index_folder,
        metric=args.metric,
        k=args.k,
        use_retrieval=args.user_retrieval,
        seed=args.seed,
    )
    tester.evaluate()
    print("[run_tests] Experiment completed.")


if __name__ == "__main__":
    main()