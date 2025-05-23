#!/usr/bin/env python
"""
bright_rewrite.py

Rewrite every forum / dataset question with the BRIGHT chain-of-thought
prompt using a vLLM-served model (e.g. Llama-3.3-70B-Instruct).

Example
-------
python bright_rewrite.py \
    --dataset_path ./Dataset_Generation/Data/A_Multi.jsonl \
    --output_path  ./Dataset_Generation/Data/A_Multi_bright_rewrite.jsonl \
    --model_name  meta-llama/Llama-3.3-70B-Instruct
"""
import os
import json
import argparse
from typing import List, Dict

from huggingface_hub import login
from vllm import LLM, SamplingParams
import torch
import random


# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    print(f"Loading data from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} records from {path}")
    return data


def save_jsonl(records: List[Dict], path: str) -> None:
    print(f"Saving {len(records)} records to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"Successfully saved data to {path}")


# ---------------------------------------------------------------------------
# BRIGHT “chain‑of‑thought” prompt (matches PROMPT_COT_BRIGHT in ReasonIR)
# see facebookresearch/ReasonIR/synthetic_data_generation/data_gen_prompts.py
# ---------------------------------------------------------------------------
def get_user_prompt_cot_bright(question: str, output_token_limit: int = 128) -> str:
    """Replicates ReasonIR's get_user_prompt_cot_bright() helper."""
    cur_post = question.replace("\n", " ")
    prompt = (
        f"{cur_post}\n\n"
        f"Instructions:\n"
        f"1. Identify the essential problem.\n"
        f"2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n"
        f"3. Draft an answer with as many thoughts as you have.\n"
    )
    return prompt

def build_conversation(question: str):
    # ReasonIR wraps the prompt in a single *user* message
    return [
        {
            "role": "user",
            "content": get_user_prompt_cot_bright(question)
        }
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rewrite questions with BRIGHT prompt")
    parser.add_argument("--dataset_folder", default="./Dataset_Generation/Data/",
                        help="Base directory that contains {TRACK}_{CONV_TYPE}.jsonl files.")
    parser.add_argument("--output_folder", default="./MetatagIndexing/Experiments/Retrieval/Bright_Questions/",
                        help="Folder where rewritten-question files are saved.")
    parser.add_argument("--track", default="A", choices=["A", "S", "T"],
                        help="Track identifier (A=Arithmetic, S=Semantic, T=Temporal).")
    parser.add_argument("--conv_type", default="Multi", choices=["Uni", "Multi"],
                        help="Conversation type.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    print("\n=== Starting Question Rewriting Process ===")
    print(f"Track: {args.track}")
    print(f"Conversation Type: {args.conv_type}")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}")

    # Construct dataset & output paths from track and conv_type
    dataset_filename = f"{args.track}_{args.conv_type}.jsonl"
    dataset_path = os.path.join(args.dataset_folder, dataset_filename)
    print(f"\nInput dataset path: {dataset_path}")

    os.makedirs(args.output_folder, exist_ok=True)
    output_path  = os.path.join(args.output_folder, dataset_filename)
    print(f"Output will be saved to: {output_path}")

    # Login to HF if token provided
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("\nLogging into Hugging Face...")
        login(token=hf_token)
        print("Successfully logged into Hugging Face")

    # Load model via vLLM
    print("\nInitializing model...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=max(torch.cuda.device_count(), 1),
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
    )
    print("Model successfully initialized")

    # Load dataset
    print("\nLoading and processing dataset...")
    data = load_jsonl(dataset_path)
    convs = []
    for sample in data:
        q = sample.get("forum_question") or sample.get("question") or ""
        convs.append(build_conversation(q))

    print(f"Prepared {len(convs)} conversations for processing")

    # Generation
    print("\nStarting question rewriting...")
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print("Generating responses...")
    outputs = llm.chat(messages=convs, sampling_params=sampling, use_tqdm=True)

    print("\nProcessing model outputs...")
    rewrites = []
    for samp, out in zip(data, outputs):
        rewrites.append({
            "id": samp.get("user_ID"),
            "original_question": samp['question'],
            "rewritten_question": out.outputs[0].text.strip(),
        })

    save_jsonl(rewrites, output_path)
    print("\n=== Question Rewriting Process Completed ===")
    print(f"Successfully processed {len(rewrites)} questions")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()