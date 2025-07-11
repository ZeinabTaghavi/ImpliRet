# --------------------------------------------------------------------------- #
#                           Async Evaluation Harness                            #
# --------------------------------------------------------------------------- #
"""
Async evaluation harness for long-context experiments.

This module provides functionality for evaluating large language models on
long-context tasks, particularly focused on temporal reasoning, arithmetic,
and world knowledge across both single-speaker and multi-speaker discourse.

Key components:
- ExperimentTester: Main class that handles experiment setup, model inference,
  and evaluation
- Support for different retrieval methods via configurable retrievers
- Metrics including exact match, contains, and ROUGE-recall
- Flexible prompt templates for different discourse types
- Async inference via unified API connector

The evaluation pipeline:
1. Loads and preprocesses conversation/forum datasets
2. Optionally sets up retrieval indices
3. Runs async model inference with configurable parameters
4. Computes metrics and saves detailed results

Usage:
    tester = ExperimentTester(
        model_name="model",
        discourse_type="unispeaker|multispeaker", 
        category="temporal|arithmetic|wknowledge",
        use_retrieval=True|False,
        ...
    )
    await tester.run()
"""


# Standard library imports
import os
import json
import asyncio
import hashlib
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
import torch

# Local imports
try:
    from RAG_Style.scripts.asyncr.async_api_connector import APIConnector  # API connector for model inference
except:
    from async_api_connector import APIConnector  # API connector for model inference
from rouge_score import rouge_scorer
from datasets import load_dataset


# --------------------------------------------------------------------------- #
#                                  Helpers                                      #
# --------------------------------------------------------------------------- #

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _sha(obj: Any) -> str:
    """Generate deterministic SHA256 hash of a JSON-serializable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
#                            Main Tester Class                                  #
# --------------------------------------------------------------------------- #

class ExperimentTester:
    """
    Main class for running experiments and evaluating model performance.
    
    Handles:
    - Loading and configuring the model
    - Processing the dataset
    - Running inference
    - Computing evaluation metrics
    - Saving results
    
    One instance represents one (model, dataset) experimental run.
    """

    def __init__(
        self,
        model_name: str,
        model_configs_dir: str,
        category: str,
        discourse_type: str,
        results_dir: str,
        retriever: str,
        retriever_index_folder: str,
        metric: str = "EM , contains , rouge-recall", # Available metrics: "EM" (exact match) | "contains" | "rouge-recall"
        k: int = 5,
        use_retrieval: bool = False,            
        system_prompt: str = "",
        use_default_system_prompt: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize the experiment tester with model and dataset configurations.
        
        Args:
            model_name: Name of the model to use
            model_configs_dir: Directory containing model configs
            category: Data category (temporal/arithmetic/wknowledge)
            discourse_type: Type of discourse (unispeaker/multispeaker)
            results_dir: Directory to save results
            retriever: Retriever type to use
            retriever_index_folder: Folder containing retriever indices
            metric: Evaluation metric(s) to use
            k: Number of conversations to include in context
            use_retrieval: Whether to use retrieval
            system_prompt: Custom system prompt
            use_default_system_prompt: Whether to use default system prompt
            seed: Random seed for reproducibility
        """

        print("\n ----------- [STEP 1] Initialization -----------")
        print("Loading dataset configurations...")

        # Initialize model API connector
        cfg_path = os.path.join(model_configs_dir, f"{model_name}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Model-config {cfg_path} not found")

        with open(cfg_path) as f:
            model_cfg = json.load(f)
        self.api = APIConnector(**model_cfg)
        print("[Init] Loaded model configuration:", cfg_path)
        self.model_name = model_name
        self.model_cfg = model_cfg  # Store for max_tokens etc.

        # Load and validate dataset
        self.discourse_type = discourse_type
        self.category = category
        assert self.discourse_type in ['unispeaker', 'multispeaker'], f"Unknown data type: {self.discourse_type}"
        assert self.category in ['temporal', 'arithmetic', 'wknow'], f"Unknown category: {self.category}"
        self.dataset = load_dataset("zeinabTaghavi/ImpliRet", name=discourse_type, split=category)
        self.examples = self.dataset.to_list()

        print(f"[Init] Loaded dataset with {len(self.examples)} rows")
        if len(self.examples) == 0:
            raise ValueError(f"Dataset is empty.")
        
        print("\n ----------- [STEP 2] Processing Data -----------")
        print("Building conversation dictionaries...")
            
        # Build conversation and question lookup dictionaries
        self.conversations_dict = {}
        self.questions_dict = {}
        for data in self.examples:
            if data['tuple_set_id'] not in self.conversations_dict:
                self.conversations_dict[data['tuple_set_id']] = [data['pos_document']]
            else:
                self.conversations_dict[data['tuple_set_id']].append(data['pos_document'])

            if data['tuple_set_id'] not in self.questions_dict:
                self.questions_dict[data['tuple_set_id']] = [data['question']]
            else:
                self.questions_dict[data['tuple_set_id']].append(data['question'])

        # Additional forum dict for multispeaker mode
        if self.discourse_type == "multispeaker":
            self.forum_dict = {}
            for data in self.examples:
                if data['tuple_set_id'] not in self.forum_dict:
                    self.forum_dict[data['tuple_set_id']] = data['forum_question']

        # Define prompt templates
        self.prompt = { 
    'unispeaker': '''
**Task**
Answer the Question based on the context provided. 

**Input**
- The INPUT contains several conversations between two users as Context and a Question.
- Each conversation is mentioned by a number in the following format:
Conversation {{number}}:
- Each conversation contains 10 utterances that are separated by lines.
- Each utterance contains the date, speaker and the message in the following format:
<date>, <speaker>: <message>

**Output**
return the final answer in a new line after "Answer:" without any prefix or suffix.

INPUT:
Context: {context}

Question: {question}
        
Answer:
    ''',
    "multispeaker": '''
**Task**
Answer the Question based on the context provided. 

**Input**
- The INPUT contains a Forum Question and several Responses to the Forum Question. 
- Each response is mentioned by a number in the following format:
Response {{number}}:
- Each response is separated by a new line.
- Each response contains the date, speaker and the message in the following format:
<date>, <speaker>: <message>

**Output**
return the final answer in a new line after "Answer:" without any prefix or suffix.

INPUT:

{context}

Question: {question}
Answer:
    '''
    }
        # Setup retriever if enabled
        self.retriever = retriever
        if use_retrieval:
            print("\n ----------- [STEP 3] Setting up Retriever -----------")
            self.retriever_index_filename = os.path.join(retriever_index_folder, f"{category}_{discourse_type}_{retriever}_index.jsonl")
            try:
                self.retriever_index = load_jsonl(self.retriever_index_filename)
            except:
                raise ValueError(f"Retriever index file {self.retriever_index_filename} not found")
        else:   
            self.retriever_index = None

        # Set runtime parameters
        self.metric = metric
        self.k = k
        self.use_retrieval = use_retrieval
        self.system_prompt = system_prompt
        self.use_default_system_prompt = use_default_system_prompt
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Setup output path
        os.makedirs(results_dir, exist_ok=True)
        base = f"{category}_{discourse_type}"
        ts = int(time.time())
        if self.use_retrieval:
            self.out_path = os.path.join(
                results_dir, f"{base}_{model_name}_{k}_retrieval_{self.retriever}_{ts}.json"
            )
        else:
            self.out_path = os.path.join(
                results_dir, f"{base}_{model_name}_{k}_{ts}.json"
            )
        print("[Init] ExperimentTester initialisation complete. "
              f"Results will be written to {self.out_path}")

    def _compute_score(self, response: str, gold: str) -> Dict[str, float]:
        """
        Compute evaluation metrics between model response and gold answer.
        
        Supports:
        - Exact Match (EM)
        - Contains check
        - ROUGE-1/2/L recall scores
        
        Args:
            response: Model's response string
            gold: Gold/reference answer string
            
        Returns:
            Dictionary of computed metrics
        """
        if 'answer' in response.lower():
            response = response.split('Answer')[-1].strip()
        else:
            response = response.strip()
        score = {}
        if isinstance(gold, list):
            gold_text = " ".join(map(str, gold)).strip()
        else:
            gold_text = gold.strip()
        if  "EM" in self.metric:
            score["EM"] = int(response == gold_text)
        if "contains" in self.metric:
            score["contains"] = int(gold_text in response)
        if "rouge-recall" in self.metric:
            # Compute ROUGE-1 recall between gold and response
            scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
            r1_scores = scorer.score(gold_text, response)
            score["rouge-1-recall"] = r1_scores["rouge1"].recall
        
            # Compute ROUGE-2 recall between gold and response
            scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
            r2_scores = scorer.score(gold_text, response)
            score["rouge-2-recall"] = r2_scores["rouge2"].recall

            # Compute ROUGE-L recall between gold and response
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rl_scores = scorer.score(gold_text, response)
            score["rouge-l-recall"] = rl_scores["rougeL"].recall
        if len(score.keys()) == 0:
            raise ValueError(f"Unknown metric {self.metric}")
        return score

    def _get_context_convs(self, conversations: List[Dict[str, Any]], 
                           golden_index: int, 
                           k: int, 
                           retrieved_indices: List[Tuple[int, float]] = None
                            ) -> List[str]:
        """
        Select conversations to include in the context.
        
        Args:
            conversations: List of all available conversations
            golden_index: Index of the golden/correct conversation
            k: Number of conversations to select
            retrieved_indices: Optional list of (index, score) tuples from retriever
            
        Returns:
            List of selected conversations
        """
        golden_conv = conversations[golden_index]
        if self.retriever_index is not None:
            # Use top k retrieved conversations
            retrieved_indices_sorted = sorted(self.retriever_index, key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in retrieved_indices_sorted[:k]]
            context_convs = [conversations[idx] for idx in top_k_indices]
        elif self.retriever_index is None:
            if k == -1:
                context_convs = conversations
            elif k == 1:
                context_convs = [golden_conv]
            elif k > 1: 
                # Sample k-1 random conversations and add golden
                other_convs = conversations[:golden_index] + conversations[golden_index+1:]
                sampled_convs = random.sample(other_convs, min(k - 1, len(other_convs)))
                context_convs = [golden_conv] + sampled_convs
            else:
                raise ValueError(f"Unknown k value: {k}")
        else:
            raise ValueError(f"Retrieved indices is None: {self.retriever_index}")
   
        random.shuffle(context_convs)

        return context_convs
        
    def _select_context(self, idx: int) -> str:
        """
        Build the full context string for a given example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Formatted context string
        """
        tuple_set_id = self.examples[idx]['tuple_set_id']
        # Get retrieved indices if using retrieval
        retrieved_indices = None
        if self.use_retrieval:
            for entry in self.retriever_index:
                if entry['tuple_set_id'] == tuple_set_id and entry['question'] == self.examples[idx]['question']:
                    retrieved_indices = entry['index_score_tuple_list']
                    break
            assert retrieved_indices is not None, f"Retrieved indices not found for tuple_set_id={tuple_set_id}, question={self.examples[idx]['question']}"
        
        golden_index = self.conversations_dict[tuple_set_id].index(self.examples[idx]['pos_document']) 
        context_convs = self._get_context_convs(self.conversations_dict[tuple_set_id], golden_index, self.k, retrieved_indices)
        
        # Format context differently for multi vs uni speaker
        if self.discourse_type == "multispeaker":
            context = "Forum Question: " + self.forum_dict[tuple_set_id] + "\n" + ' '.join([f"Response {j+1}: {conv} \n" for j, conv in enumerate(context_convs)])
        elif self.discourse_type == "unispeaker":
            context = ' '.join([f"Conversation {j+1}: {conv} \n" for j, conv in enumerate(context_convs)])
        else:
            raise ValueError(f"Unknown data type: {self.discourse_type}")
        return context
    
    async def _run_one(self, idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example through the pipeline:
        1. Build the prompt
        2. Get model response
        3. Compute evaluation metrics
        
        Args:
            idx: Example index
            ex: Example data dictionary
            
        Returns:
            Dictionary with results for this example
        """
        # Build prompt
        if "pos_document" in ex:
            context_block = self._select_context(idx)
            prompt_text = self.prompt[self.discourse_type].format(
                context=context_block,
                question=ex["question"]
            )
            
        else:
            raise ValueError(f"Unknown data row format: {ex}")
        print(f"[RunOne] Example {idx}: prompt built")

        # Get model response
        resp = await self.api.generate_response(
            system_prompt=self.system_prompt,
            user_prompt=prompt_text,
            max_tokens=self.model_cfg.get("max_tokens", None),
            temperature=self.model_cfg.get("temperature", None),
            top_p=self.model_cfg.get("top_p", None),
            add_default_system_prompt=self.use_default_system_prompt,
        )
        print(f"[RunOne] Example {idx}: model response received")

        # Compute metrics
        score = self._compute_score(resp["response"], ex["answer"])

        # Return results
        return {
            "id": ex.get("id", _sha(ex)),
            "prompt": prompt_text,
            "gold": ex["answer"],
            "response": resp["response"],
            "score": score,
            "tokens": {
                "prompt": resp.get("prompt_tokens"),
                "completion": resp.get("completion_tokens"),
                "total": resp.get("total_tokens"),
            },
            "finish_reason": resp.get("finish_reason"),
            "cached_tokens": resp.get("cached_tokens"),
        }

    def evaluate(self) -> None:
        """
        Main evaluation loop:
        1. Create async tasks for all examples
        2. Run tasks concurrently
        3. Collect results
        4. Save to output file
        """
        print(
            f"Running {self.model_name} on {len(self.examples)} examples; "
            f"writing to {self.out_path}"
        )
        # print("[Evaluate] Launching async tasks …")
        
        # Run all examples concurrently
        loop = asyncio.get_event_loop()
        tasks = [self._run_one(idx, ex) for idx, ex in enumerate(self.examples[:40])]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        print("[Evaluate] All tasks finished, writing results …")

        # Collate results
        run_obj = {
            "model": self.model_name,
            "category": self.category,
            "discourse_type": self.discourse_type,
            "metric": self.metric,
            "system_prompt": (
                self.api.SYSTEM_PROMPT if self.use_default_system_prompt else self.system_prompt
            ),
            "results": results,
            "run_hash": _sha(
                {
                    "model": self.model_name,
                    "category": self.category,
                    "discourse_type": self.discourse_type,
                    "metric": self.metric,
                    "seed": self.seed,
                }
            ),
        }

        # Save results
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(run_obj, f, indent=2, ensure_ascii=False)
        print("Finished!  Results saved at:", self.out_path)
        print("[Evaluate] Done.")


# --------------------------------------------------------------------------- #
#                              CLI Interface                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    # Setup argument parser
    parser = ArgumentParser(description="Async experiment runner")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML run_config")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_configs_dir", type=str)
    parser.add_argument("--category", type=str, default="temporal", help="temporal, arithmetic or wknow")
    parser.add_argument("--discourse_type", type=str, default="unispeaker", help="unispeaker or multispeaker")
    parser.add_argument("--results_dir", type=str, default="./RAG_Style/results/")
    parser.add_argument("--metric", type=str, default="EM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--use_retrieval", type=bool, default=False)    
    parser.add_argument("--retriever", type=str, default="BM25")
    parser.add_argument("--retriever_index_folder", type=str, default="./Experiments/Retrieval/Results/")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--use_default_system_prompt", type=bool, default=True)

    args = parser.parse_args()


    # Create and run tester
    tester = ExperimentTester(
        model_name=args.model_name,
        model_configs_dir=args.model_configs_dir,
        category=args.category,
        discourse_type=args.discourse_type,
        results_dir=args.results_dir,
        metric=args.metric,
        seed=args.seed,
        k=args.k,
        use_retrieval=args.use_retrieval,
        retriever=args.retriever,
        retriever_index_folder=args.retriever_index_folder,
        system_prompt=args.system_prompt,
        use_default_system_prompt=args.use_default_system_prompt,
    )
    # tester.evaluate()