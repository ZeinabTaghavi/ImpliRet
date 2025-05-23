

# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# evaluation harness for Zeinab's long-context experiments.
# ---
# Layout:
#   - Each JSONL in ./Dataset_Generatoin/Data/ contains dicts with at least
#       { "id": str, "prompt": str, "answer": str }
#   - One ExperimentTester is created per (model × data split × run-config).
#   - A single JSON result file is produced that bundles: run metadata,
#     per-example prompt / response / score, and token usage stats.
#
# NOTE: This is intentionally minimal.  Feel free to extend:
#   * add more fields from your dataset
#   * support different metrics
#   * log full prompts or system messages
#   * plug in your own prompt-building logic

import os
import json
import hashlib
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple

from sync_model_connector import ModelLoader  # your copy sits in evaluation/
from rouge_score import rouge_scorer


# --------------------------------------------------------------------------- #
#                           helper: data loading                              #
# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------- #
#                          helper: deterministic hash                         #
# --------------------------------------------------------------------------- #
def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
#                          main tester / evaluator                            #
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
        dataset_folder: str,
        track: str,
        conv_type: str,
        results_dir: str,
        retriever: str,
        retriever_index_folder: str,
        metric: str = "EM , contains , rouge-recall", # "EM" (exact match) | "contains" | "rouge-recall"
        k: int = 5,
        use_retrieval: bool = False,            
        system_prompt: str = "",
        use_default_system_prompt: bool = True,
        seed: int = 42,
    ) -> None:

        print("\n ----------- [STEP 1] Initialization -----------")
        print("Loading dataset configurations...")

        # -------- model API connector --------
        cfg_path = os.path.join(model_configs_dir, f"{model_name}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Model-config {cfg_path} not found")

        with open(cfg_path) as f:
            model_cfg = json.load(f)
        self.model_name = model_name
        self.model_cfg = model_cfg  # keep for max_tokens etc.

        # -------- dataset --------
        self.conv_type = conv_type
        assert self.conv_type in ['Uni', 'Multi'], f"Unknown data type: {self.conv_type}"

        data_path = os.path.join(dataset_folder, f"{track}_{conv_type}.jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found")
        self.data_path = data_path
        try:
            self.examples = load_jsonl(data_path)
        except:
            raise ValueError(f"Data file {data_path} is not a valid JSONL file.")
        if len(self.examples) == 0:
            raise ValueError(f"{data_path} is empty.")
            
        print("\n ----------- [STEP 2] Processing Data -----------")
        print("Building conversation dictionaries...")
        
        self.conversations_dict = {}
        self.questions_dict = {}
        for data in self.examples:
            if data['user_ID'] not in self.conversations_dict:
                self.conversations_dict[data['user_ID']] = [data['context']]
            else:
                self.conversations_dict[data['user_ID']].append(data['context'])

            if data['user_ID'] not in self.questions_dict:
                self.questions_dict[data['user_ID']] = [data['question']]
            else:
                self.questions_dict[data['user_ID']].append(data['question'])

        if self.conv_type == "Multi":
            self.forum_dict = {}
            self.topic_dict = {}
            for data in self.examples:
                if data['user_ID'] not in self.forum_dict:
                    self.forum_dict[data['user_ID']] = data['forum_question']
                if data['user_ID'] not in self.topic_dict:
                    self.topic_dict[data['user_ID']] = data['topic']

        self.prompt = { 'Uni': '''
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
                "Multi": '''
**Task**
Answer the Question based on the context provided. 

**Input**
- The INPUT contains a Topic, Forum Question and several Responses to the Forum Question. 
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
        # -------- retriever --------
        self.retriever = retriever
        if use_retrieval:
            print("\n ----------- [STEP 3] Setting up Retriever -----------")
            self.retriever_index_filename = os.path.join(retriever_index_folder, f"{track}_{conv_type}_{retriever}_index.jsonl")
            try:
                self.retriever_index = load_jsonl(self.retriever_index_filename)
            except:
                raise ValueError(f"Retriever index file {self.retriever_index_filename} not found")
        # -------- run-time params --------
        self.metric = metric
        self.k = k
        self.use_retrieval = use_retrieval
        self.system_prompt = system_prompt
        self.use_default_system_prompt = use_default_system_prompt
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        # -------- output folder --------
        os.makedirs(results_dir, exist_ok=True)
        #  `<datasetName>_<modelName>_<unix>.json`
        base = os.path.splitext(os.path.basename(data_path))[0]
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
        
    # ---------------- evaluation helpers ---------------- # fix it to return dictionary
    def _compute_score(self, response: str, gold: str) -> int:
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

    # ---------------- context selection ---------------- #
    
    def _get_context_convs(self, conversations: List[Dict[str, Any]], 
                           golden_index: int, 
                           k: int, 
                           retrieved_indices: List[Tuple[int, float]] = None
                            ) -> List[str]:
  

        golden_conv = conversations[golden_index]
        if retrieved_indices is not None:
            # Use top k retrieved conversations
            retrieved_indices_sorted = sorted(retrieved_indices, key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in retrieved_indices_sorted[:k]]
            context_convs = [conversations[idx] for idx in top_k_indices]
        elif retrieved_indices is None:
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
            raise ValueError(f"Retrieved indices is None: {retrieved_indices}")

        random.shuffle(context_convs)

        return context_convs
        
    def _select_context(self, idx: int) -> str:
        user_ID = self.examples[idx]['user_ID']
        # if we are using retrieval, we need to get the index if the conversations
        retrieved_indices = None
        if self.use_retrieval:
            for entry in self.retriever_index:
                if entry['user_ID'] == user_ID and entry['question'] == self.examples[idx]['question']:
                    retrieved_indices = entry['index_score_tuple_list']
                    break
            assert retrieved_indices is not None, f"Retrieved indices not found for user_ID={user_ID}, question={self.examples[idx]['question']}"
        
        golden_index = self.conversations_dict[user_ID].index(self.examples[idx]['context']) 
        context_convs = self._get_context_convs(self.conversations_dict[user_ID], golden_index, self.k, retrieved_indices)
        if self.conv_type == "Multi":
            context = "Topic: " + self.topic_dict[user_ID] + "\n" + "Forum Question: " + self.forum_dict[user_ID] + "\n" + '\n'.join([f"Response {j+1}: {conv} \n" for j, conv in enumerate(context_convs)])
        elif self.conv_type == "Uni":
            context = ' '.join([f"Conversation {j+1}: {conv} \n" for j, conv in enumerate(context_convs)])
        else:
            raise ValueError(f"Unknown data type: {self.conv_type}")
        return context
    
    def _run_one(self, idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
        if 'context' in ex:
            context_block = self._select_context(idx)
            prompt_text = self.prompt[self.conv_type].format(
                context=context_block,
                question=ex["question"]
            )
            
        else:
            raise ValueError(f"Unknown data row format: {ex}")
        return prompt_text
        
    def evaluate(self) -> None:
        print("\n ----------- [STEP 4] Running Evaluation -----------")
        print(f"Processing {len(self.examples)} examples...")
        
        prompts = [self._run_one(idx, ex) for idx, ex in enumerate(self.examples)]
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
            "data_path": os.path.abspath(self.data_path),
            "metric": self.metric,
            "system_prompt": (
                self.model.SYSTEM_PROMPT if self.use_default_system_prompt else self.system_prompt
            ),
            "results": results,
            "run_hash": _sha(
                {
                    "model": self.model_name,
                    "data_path": self.data_path,
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
#                         CLI wrapper (optional)                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description="Sync experiment runner")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML run_config")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_configs_dir", type=str)
    parser.add_argument("--dataset_folder", type=str, default="./Dataset_Generation/Data/", help="Dataset Filename.")
    parser.add_argument("--track", type=str, default="T", help="(T)emporal, (A)rithmetic or (S)emantic")
    parser.add_argument("--conv_type", type=str, default="Uni", help="Uni or Multi")
    parser.add_argument("--results_dir", type=str, default="./Experiments/Evaluation/Results/")
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
        dataset_folder=args.dataset_folder,
        track=args.track,
        conv_type=args.conv_type,
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