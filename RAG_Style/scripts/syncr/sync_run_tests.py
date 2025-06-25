# --------------------------------------------------------------------------- #
#                            Sync Test Runner                                 #
# --------------------------------------------------------------------------- #
"""
Sync test runner for long-context experiments.

This module provides the main entry point for running experiments, handling:
- Parsing YAML config and CLI arguments
- Setting up paths and configurations
- Initializing and running ExperimentTester

You can run it two ways:

1. **CLI** (your existing bash script)  
   python RAG_Style/scripts/sync/sync_run_tests.py \
          --config RAG_Style/experiment_configs/bm/A_Multi_llama_bm_1.yaml \
          --model_name llama \
          --model_configs_dir RAG_Style/model_configs

2. **Programmatically**  
   from RAG_Style.scripts.sync.sync_run_tests import run_experiment  
   run_experiment(["--config", "path/to/config.yaml", "--model_name", "llama",
                   "--model_configs_dir", "RAG_Style/model_configs"])
"""

from __future__ import annotations
import os
from jsonargparse import ArgumentParser, ActionConfigFile

# Local imports
try:
    from RAG_Style.scripts.syncr.sync_evaluation import ExperimentTester
except ImportError:
    from sync_evaluation import ExperimentTester

# --------------------------------------------------------------------------- #
#                               Main Runner                                   #
# --------------------------------------------------------------------------- #
def run_experiment(arg_list: list[str] | None = None) -> str:
    """
    Run the experiment either via CLI (arg_list=None) or programmatically by
    passing a list of CLI-style arguments.

    Example:
        run_experiment([
            "--config", "RAG_Style/experiment_configs/bm/A_Multi_llama_bm_1.yaml",
            "--model_name", "llama",
            "--model_configs_dir", "RAG_Style/model_configs"
        ])

    Returns
    -------
    str
        The path to the results directory (useful for downstream pipelines).
    """
    parser = ArgumentParser(description="Metatag-Indexing Experiment Runner")

    # Allow `--config some.yaml`
    parser.add_argument("--config", action=ActionConfigFile, help="YAML run_config")

    # All fields that might appear in the YAML
    parser.add_argument("--category", type=str,
                        choices=["temporal", "arithmetic", "wknowledge"],
                        default="temporal")
    parser.add_argument("--discourse", dest="discourse_type", type=str,
                        choices=["unispeaker", "multispeaker"],
                        default="unispeaker")
    parser.add_argument("--output_folder", type=str,
                        default="./RAG_Style/results/")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--use_retrieval", type=bool, default=False)
    parser.add_argument("--retriever", type=str, default="BM25")
    parser.add_argument("--retriever_index_folder", type=str,
                        default="./Retrieval/results/")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_configs_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="EM")
    parser.add_argument("--seed", type=int, default=42)

    # Parse either sys.argv (arg_list=None) or the provided list
    args = parser.parse_args(arg_list)
    print("[run_tests] Parsed CLI / YAML arguments")

    # ----------------------------------------------------------------------- #
    #                           Path Configuration                            #
    # ----------------------------------------------------------------------- #
    model_cfg_path = os.path.join(args.model_configs_dir,
                                  f"{args.model_name}.json")
    results_dir = os.path.join(args.output_folder,
                               f"{args.category}_{args.discourse_type}",
                               args.model_name)

    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Model-config not found: {model_cfg_path}")
    print("[run_tests] Found model-config.")

    # ----------------------------------------------------------------------- #
    #                           Run Experiment                               #
    # ----------------------------------------------------------------------- #
    print("[run_tests] Initialising ExperimentTester …")
    tester = ExperimentTester(
        model_name=args.model_name,
        model_configs_dir=args.model_configs_dir,
        category=args.category,
        discourse_type=args.discourse_type,
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
    return results_dir


if __name__ == "__main__":
    # Falls back to parsing sys.argv, so your bash script remains unchanged
    run_experiment()