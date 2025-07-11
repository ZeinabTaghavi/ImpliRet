# --------------------------------------------------------------------------- #
#                            Async Test Runner                                  #
# --------------------------------------------------------------------------- #
"""
Async test runner for long-context experiments.

This module provides the main entry point for running experiments, handling:
- Parsing YAML config and CLI arguments 
- Setting up paths and configurations
- Initializing and running ExperimentTester

The YAML config can be provided via --config flag and any field can be
overridden via CLI arguments.

Usage:
    python async_run_tests.py --config config.yaml
    python async_run_tests.py --model_name llama --category temporal ...
"""

# Standard library imports
from __future__ import annotations
import os
from jsonargparse import ArgumentParser, ActionConfigFile

# Local imports
try:
    from RAG_Style.scripts.asyncr.async_evaluate import ExperimentTester
except:
    from async_evaluate import ExperimentTester


# --------------------------------------------------------------------------- #
#                                Main Runner                                    #
# --------------------------------------------------------------------------- #

def main() -> None:
    """Main entry point for running experiments."""
    parser = ArgumentParser(description="Metatag‑Indexing Experiment Runner")

    # Allow `--config some.yaml`
    parser.add_argument("--config", action=ActionConfigFile, help="YAML run_config")

    # All fields that might appear in the YAML
    parser.add_argument("--category", type=str, choices=["temporal", "arithmetic", "wknow"], default="temporal")
    parser.add_argument("--discourse", dest="discourse_type", type=str, choices=["unispeaker", "multispeaker"], default="unispeaker")
    parser.add_argument("--output_folder", type=str, default="./RAG_Style/results/")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--use_retrieval", type=bool, default=False)
    parser.add_argument("--retriever", type=str, default="BM25")
    parser.add_argument("--retriever_index_folder", type=str, default="./Retrieval/results/")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_configs_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="EM")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print("[run_tests] Parsed CLI / YAML arguments")

    # --------------------------------------------------------------------------- #
    #                            Path Configuration                                 #
    # --------------------------------------------------------------------------- #
    model_cfg_path = os.path.join(
        args.model_configs_dir, f"{args.model_name}.json"
    )
    results_dir = os.path.join(
        args.output_folder, f"{args.category}_{args.discourse_type}", args.model_name
    )

    # Validate paths
    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Model‑config not found: {model_cfg_path}")
    print("[run_tests] Found model‑config.")

    # --------------------------------------------------------------------------- #
    #                            Run Experiment                                     #
    # --------------------------------------------------------------------------- #
    print("[run_tests] Initialising ExperimentTester …")
    tester = ExperimentTester(
        model_name=args.model_name,
        model_configs_dir=args.model_configs_dir,
        category=args.category,
        discourse_type=args.discourse_type,                 # Uni or Multi
        results_dir=results_dir,
        retriever=args.retriever,
        retriever_index_folder=args.retriever_index_folder,
        metric=args.metric,
        k=args.k,
        use_retrieval=args.use_retrieval,
        seed=args.seed,
    )
    tester.evaluate()
    print("[run_tests] Experiment completed.")


if __name__ == "__main__":
    main()