"""
Evaluation harness for long-context experiments.

Layout:
  - Each JSONL in ./Dataset_Generatoin/Data/ contains dicts with at least
      { "id": str, "prompt": str, "answer": str }
  - One ExperimentTester is created per (model × data split × run-config).
  - A single JSON result file is produced that bundles: run metadata,
    per-example prompt / response / score, and token usage stats.

NOTE: This is intentionally minimal. Feel free to extend:
  * add more fields from your dataset
  * support different metrics
  * log full prompts or system messages
  * plug in your own prompt-building logic
"""

import os
import json
import hashlib
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from sync_model_connector import ModelLoader  # your copy sits in evaluation/
from rouge_score import rouge_scorer
import torch

# --------------------------------------------------------------------------- #
#                                  Helpers                                    #
# --------------------------------------------------------------------------- #

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _sha(obj: Any) -> str:
    """Generate deterministic hash for an object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
#                            Main Evaluator Class                             #
# --------------------------------------------------------------------------- #

class ExperimentTester:
    """
    A lightweight analogue of NoLiMa_Tester for our custom dataset.
    One instance = one (model, dataset) run.
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

        print("\n ----------- [STEP 1] Initialization -----------")
        print("Loading dataset configurations...")

        # Setup model API connector
        cfg_path = os.path.join(model_configs_dir, f"{model_name}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Model-config {cfg_path} not found")

        with open(cfg_path) as f:
            model_cfg = json.load(f)
        self.model_name = model_name
        self.model_cfg = model_cfg  # keep for max_tokens etc.
        self.category = category
        
        # Load dataset
        self.discourse_type = discourse_type
        assert self.discourse_type in ['unispeaker', 'multispeaker'], f"Unknown data type: {self.discourse_type}"
        assert category in ['temporal', 'arithmetic', 'wknowledge'], f"Unknown category: {category}"
        self.dataset = load_dataset("zeinabTaghavi/ImpliRet", name=discourse_type, split=category)
        self.examples = self.dataset.to_list()
        

        print(f"[Init] Loaded dataset with {len(self.examples)} rows")
        if len(self.examples) == 0:
            raise ValueError(f"Dataset is empty.")
            
        print("\n ----------- [STEP 2] Processing Data -----------")
        print("Building conversation dictionaries...")
        self.conversations_list = []
        self.questions_list = []
        # Build conversation and question dictionaries
        self.conversations_dict = {}
        self.questions_dict = {}
        for data in self.examples:
            if data['tuple_set_id'] not in self.conversations_dict:
                self.conversations_dict[data['tuple_set_id']] = [data['pos_document']]
            else:
                self.conversations_dict[data['tuple_set_id']].append(data['pos_document'])
            self.conversations_list.append(data['pos_document'])
            if data['tuple_set_id'] not in self.questions_dict:
                self.questions_dict[data['tuple_set_id']] = [data['question']]
            else:
                self.questions_dict[data['tuple_set_id']].append(data['question'])
            self.questions_list.append(data['question'])
        # Additional forum dict for multispeaker mode
        if self.discourse_type == "multispeaker":
            self.forum_dict = {}
            for data in self.examples:
                if data['tuple_set_id'] not in self.forum_dict:
                    self.forum_dict[data['tuple_set_id']] = data['forum_question']

        print(len(self.conversations_list))
        print('--------------------------------')
        # Define prompt templates
        self.prompt = { 'unispeaker': '''
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

Answer the following question as precisely as possible, using the information provided in the conversation. You may rely on the conversation content, the time each conversation was sent, and who sent it.

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

Answer the following question as precisely as possible, using the information provided in the responses. You may rely on the response content, the time each response was sent, and who sent it.

Question: {question}

Answer:
    '''
    }
        # Setup retriever if enabled
        self.retriever = retriever
        self.retriever_index = None
        if use_retrieval:
            print("\n ----------- [STEP 3] Setting up Retriever -----------")
            self.retriever_index_filename = os.path.join(retriever_index_folder, f"{category}_{discourse_type}_{retriever}_index.jsonl")
            try:
                self.retriever_index = load_jsonl(self.retriever_index_filename)
            except:
                raise ValueError(f"Retriever index file {self.retriever_index_filename} not found")

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
        """Compute evaluation metrics between response and gold answer."""
        if 'answer' in response.lower():
            response = response.split('Answer')[-1].strip()
        else:
            response = response.strip()
            
        score = {}
        if isinstance(gold, list):
            gold_text = " ".join(map(str, gold)).strip()
        else:
            gold_text = gold.strip()
            
        if "EM" in self.metric:
            score["EM"] = int(response == gold_text)
        if "contains" in self.metric:
            score["contains"] = int(gold_text in response)
        if "rouge-recall" in self.metric:
            # Compute ROUGE-1 recall
            scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
            r1_scores = scorer.score(gold_text, response)
            score["rouge-1-recall"] = r1_scores["rouge1"].recall
            
            # Compute ROUGE-2 recall
            scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
            r2_scores = scorer.score(gold_text, response)
            score["rouge-2-recall"] = r2_scores["rouge2"].recall

            # Compute ROUGE-L recall
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rl_scores = scorer.score(gold_text, response)
            score["rouge-l-recall"] = rl_scores["rougeL"].recall
        if len(score.keys()) == 0:
            raise ValueError(f"Unknown metric {self.metric}")
        return score

    def _get_context_convs(self, conversations: List[Dict[str, Any]], 
                           golden_index: int, 
                           k: int, 
                           retrieved_indices: List[int] = None,
                            ) -> List[str]:
        """
        Select conversations to include in the context.
        
        Args:
            conversations: List of all available conversations
            golden_index: Index of the golden/correct conversation
            k: Number of conversations to select
            
        Returns:
            List of selected conversations
        """
        golden_conv = conversations[golden_index]
        if self.retriever_index is not None:
            # Use top k retrieved conversations
            retrieved_indices_sorted = sorted(retrieved_indices, key=lambda x: x[1], reverse=True)
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
            assert self.retriever_index[idx]['question'] == self.examples[idx]['question'], f"Question mismatch: {self.retriever_index[idx]['question']} != {self.examples[idx]['question']}"
            retrieved_indices = self.retriever_index[idx]['index_score_tuple_list']
            assert retrieved_indices is not None, f"Retrieved indices not found for tuple_set_id={tuple_set_id}, question={self.examples[idx]['question']}"
        
        golden_index = idx
        context_convs = self._get_context_convs(self.conversations_list, golden_index, self.k, retrieved_indices)
        
        # Format context differently for multi vs uni speaker
        if self.discourse_type == "multispeaker":
            context = "Forum Question: " + self.forum_dict[tuple_set_id] + "\n" + ' '.join([f"Response {j+1}: {conv} \n" for j, conv in enumerate(context_convs)])
        elif self.discourse_type == "unispeaker":
            context = ' '.join([f"Conversation {j+1}: {conv} \n" for j, conv in enumerate(context_convs)])
        else:
            raise ValueError(f"Unknown data type: {self.discourse_type}")
        return context
    
    def _get_prompt(self, idx: int, ex: Dict[str, Any]) -> str:
        """Build the full prompt for a given example."""
        if 'pos_document' in ex:
            context_block = self._select_context(idx)
            prompt_text = self.prompt[self.discourse_type].format(
                context=context_block,
                question=ex["question"]
            )
        else:
            raise ValueError(f"Unknown data row format: {ex}")
        return prompt_text
        
    def evaluate(self) -> None:
        """Run evaluation on all examples."""
        print("\n ----------- [STEP 4] Running Evaluation -----------")
        print(f"Processing {len(self.examples)} examples...")
        
        prompts = [self._get_prompt(idx, ex) for idx, ex in enumerate(self.examples)]
        
        print("\n ----------- [STEP 5] Generating Responses -----------")
        print("Loading model and passing prompts to it...")

        self.model = ModelLoader(**self.model_cfg)

        print('--------------------------------')
        print("Sending prompts to model...")
        responses = self.model.generate_response(
            system_prompt=self.system_prompt,
            user_prompt=prompts,
            max_tokens=self.model_cfg.get("max_tokens", 256),
            temperature=self.model_cfg.get("temperature", 0.0),
            top_p=self.model_cfg.get("top_p", 1.0),
            add_default_system_prompt=self.use_default_system_prompt,
        )

        print("\n ----------- [STEP 6] Computing Scores -----------")
        results = []
        for i, (resp, ex) in enumerate(zip(responses, self.examples)):
            score = self._compute_score(resp["response"], ex["answer"])
            result = {
                "id": ex.get("id", i),
                "prompt": prompts[i],
                "gold": ex["answer"],
                "response": resp["response"].strip(),
                "score": score,
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "total_tokens": resp["total_tokens"],
                "finish_reason": resp["finish_reason"],
            }
            results.append(result)

        print("\n ----------- [STEP 7] Saving Results -----------")
        run_obj = {
            "model": self.model_name,
            "category": self.category,
            "discourse_type": self.discourse_type,
            "metric": self.metric,
            "system_prompt": (
                self.model.SYSTEM_PROMPT if self.use_default_system_prompt else self.system_prompt
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

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(run_obj, f, indent=2, ensure_ascii=False)
        print(f"\n ----------- Evaluation Complete -----------")
        print(f"Results saved to: {self.out_path}")

# --------------------------------------------------------------------------- #
#                              CLI Interface                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description="Sync experiment runner")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML run_config")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_configs_dir", type=str)
    parser.add_argument("--category", type=str, default="temporal", help="(T)emporal, (A)rithmetic or (S)emantic")
    parser.add_argument("--discourse_type", type=str, default="unispeaker", help="Uni or Multi")
    parser.add_argument("--results_dir", type=str, default="./RAG_Style/results/")
    parser.add_argument("--metric", type=str, default="EM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--use_retrieval", type=bool, default=False)    
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--retriever_index_folder", type=str, default="./Experiments/Retrieval/Results/")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--use_default_system_prompt", type=bool, default=True)

    args = parser.parse_args()

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
    tester.evaluate()