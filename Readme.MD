# Implicit Reasoning **Evaluation** Suite

> **Measure how well language models recover facts that are only *implied*, not stated.**
>
> ⚠️ **Scope notice**: This repository is *not* a general‑purpose NLP toolkit. It contains **just the code & scripts needed to generate our synthetic benchmark, evaluate retrieval baselines, and score language‑model outputs.**

---

## Quick start

```bash
# clone & install
$ git clone https://github.com/ZeinabTaghavi/.git
$ cd ImpliRet
$ python -m venv impliret_env && source impliret_env/bin/activate
$ pip install -r requirements.txt
```

---

## Repository map

```
├── RAG_Style/             
│   ├── model_configs
│   ├── run_configs
│   ├── results
│   └── reports
├── Retrieval/         
│   ├── Retrievals
│   ├── Results
│   └── reports
└── README.md             
```

---

## 1. Generate the benchmark dataset

We synthesise query–context pairs that hide key facts. Run:

For every track (e.g., A:Arithmetics) for every corpus style (e.g., Uni:Uni-Audience(Chat Style) or Multi:Multi-Audience(Forum Style)):
    1- Generating the structure of the dataset
    2- Generating the conversation based on the structured data
    3- Merging the conversations to reach the final dataset
```bash
python Dataset_Generation/GeneratingCodes/A_Multi/Arithmetic_Multi_1_Structure_Gen.py
bash Dataset_Generation/GeneratingCodes/A_Multi/Arithmetic_Multi_2_Conversation_Gen.sh
python Dataset_Generation/GeneratingCodes/A_Multi/Arithmetic_Multi_3_Merging.py
```

Outputs:

| file                                       | purpose              |
| ------------------------------------------ | -------------------- |
| `A_Uni.jsonl`                              |     Main datasets    |
| `A_Multi.jsonl`                            |     Main datasets    |
| `T_Uni.jsonl`                              |     Main datasets    |
| `T_Multi.jsonl`                            |     Main datasets    |
| `S_Uni.jsonl`                              |     Main datasets    |
| `S_Multi.jsonl`                            |     Main datasets    |
---

## 2. Evaluate retrieval baselines

Runing the retrievals for generating the indexes:
The retrievals: BM25s, ColBertV2, Contriever, DragonPlus, HippoRagV2, ReasonIR.

Example of running:
running the retriever and reporting.
```bash
bash Retrieval/retrieve_BM25s.bash
python Retrieval/reporting.py
```

Scores (MAP, nDCG\@10…) are written to `./Retrieval/Results/A_Multi_bm25_index.jsonl`.
Reports are storing to `Retrieval/reports/retrieval_report.txt`.

⚠️ For runing HippoRAG2 and ReasonIR, we used 4 A100 GPUS.
---

## 3. Evaluate language‑model reasoning

Here we try Long context and RAG, the setting of the experiments configs are in the `Experiments/evaluation/run_configs` folder, the  config of models are also stored in `Experiments/evaluation/model_configs`.

```bash
# example with Llama3.3 70-B Turbo
# If you want to use OpenAI models, use your API-key, else if you want to load the VLLM on server, user any thing as API key (e.g., `EMPTY`).
export OPENAI_API_KEY=...
export HF_HOME=...

bash Experiments/evaluation/s_run_tests_bm.sh
# or for using server:
python Experiments/evaluation/async_evaluate.py \
  --config Experiments/evaluation/run_configs/bm/A_Multi_llama.yaml
# or for loading the vllm locally
python Experiments/evaluation/sync_evaluate.py \
  --config Experiments/evaluation/run_configs/bm/A_Multi_llama.yaml

# reporing the resutls:
python Experiments/evaluation/reporing.py
```

The outputs will be hashed and store in `Experiments/evaluation/results`


---

## Acknowledgements

Thank you so much!

# Implicit Reasoning Evaluation Suite

> **Assess how well retrieval systems and large‑language models uncover facts that are *implied* rather than explicitly stated.**  
> This repository is a **three‑stage evaluation pipeline**—dataset generation, retrieval benchmarking, and LLM reasoning—created for research on implicit information understanding.

---

## Table of Contents
1. [Quick Start](#quick-start)  
2. [Directory Structure](#directory-structure)  
3. [Stage 1 — Dataset Generation](#stage-1--dataset-generation)  
4. [Stage 2 — Retrieval Evaluation](#stage-2--retrieval-evaluation)  
5. [Stage 3 — LLM Reasoning](#stage-3--llm-reasoning)  
6. [Acknowledgements](#acknowledgements)  
7. [License](#license)  

---

## Quick Start
```bash
# Clone the repo
git clone https://github.com/ZeinabTaghavi/MetatagIndexing_2.git
cd MetatagIndexing_2

# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```
 
---

## Directory Structure
```
Dataset_Generation/             # Stage 1 — synthetic benchmark creation
│   ├── Data/                   #   ▶ final JSONL datasets
│   ├── Dataset_Helping/        #   ▶ auxiliary data files
│   └── GeneratingCodes/        #   ▶ generation scripts (.py / .sh)
Experiments/                    # Stage 3 — LLM evaluation
│   └── evaluation/
│       ├── model_configs/      #   ▶ HF / OpenAI model params
│       ├── run_configs/        #   ▶ experiment configs (YAML)
│       ├── results/            #   ▶ raw generations (JSONL)
│       └── reports/            #   ▶ metric summaries (MD / JSON)
Retrieval/                      # Stage 2 — retrieval baselines
│   ├── Retrievals/             #   ▶ indexers & searchers
│   ├── Results/                #   ▶ TREC run files
│   └── reports/                #   ▶ IR metric summaries
README.md
```

---

## Stage 1 — Dataset Generation
We synthesise multi‑turn conversations that *hide* key facts and cover three topical tracks:

| Track | Domain          | Example implicit relation                     |
|-------|-----------------|-----------------------------------------------|
| **A** | Arithmetic      | *“She doubled the number six.” ⇒ 12*          |
| **T** | Temporal        | *“He left two hours after noon.” ⇒ 14:00 hrs* |
| **S** | Spatial         | *“The café is next to the library.”*          |

For every **track** (`A`, `T`, `S`) and **audience style** (`Uni` chat vs `Multi` forum) run:

```bash
# 1 Generate structured triples
python Dataset_Generation/GeneratingCodes/A_Multi/Arithmetic_Multi_1_Structure_Gen.py

# 2 Write conversations from the structures
bash   Dataset_Generation/GeneratingCodes/A_Multi/Arithmetic_Multi_2_Conversation_Gen.sh

# 3 Merge to the final JSONL dataset
python Dataset_Generation/GeneratingCodes/A_Multi/Arithmetic_Multi_3_Merging.py
```

Generated files:

| File name         | Description                 |
|-------------------|-----------------------------|
| `A_Uni.jsonl`     | Arithmetic · single‑speaker |
| `A_Multi.jsonl`   | Arithmetic · multi‑speaker  |
| `T_Uni.jsonl`     | Temporal · single‑speaker   |
| `T_Multi.jsonl`   | Temporal · multi‑speaker    |
| `S_Uni.jsonl`     | Spatial · single‑speaker    |
| `S_Multi.jsonl`   | Spatial · multi‑speaker     |

---

## Stage 2 — Retrieval Evaluation
We benchmark six retrievers: **BM25s, ColBERT‑v2, Contriever, Dragon‑Plus, HippoRAG‑v2, ReasonIR**.

Example (BM25):

```bash
# Build index & produce rankings
bash Retrieval/retrieve_BM25s.bash

# Score the run (MAP, nDCG@10, Recall@100…)
python Retrieval/reporting.py
```

*Outputs*  
- Rankings: `Retrieval/Results/A_Multi_bm25_index.jsonl`  
- Report:  `Retrieval/reports/retrieval_report.txt`

> **GPU demand:** HippoRAG‑v2 and ReasonIR were evaluated on 4 × A100 (80 GB) GPUs.

---

## Stage 3 — LLM Reasoning
We test long‑context and RAG variants with configs in `Experiments/evaluation/run_configs`.

### Run with OpenAI (GPT‑4o) or local VLLM (Llama‑3 70B)
```bash
# OpenAI endpoint
export OPENAI_API_KEY=<your‑key>

bash Experiments/evaluation/s_run_tests_bm.sh       # convenience wrapper

# — OR — evaluate asynchronously on a remote VLLM server
python Experiments/evaluation/async_evaluate.py \
  --config Experiments/evaluation/run_configs/bm/A_Multi_llama.yaml

# — OR — run locally in synchronous mode
python Experiments/evaluation/sync_evaluate.py \
  --config Experiments/evaluation/run_configs/bm/A_Multi_llama.yaml
```

Results are hashed and written to `Experiments/evaluation/results/`,  
with aggregate metrics in `Experiments/evaluation/reports/`.

---

## Acknowledgements
This project was made possible by the enthusiasm of the **Implicit Reasoning Reading Group** and infrastructure kindly provided by LRZ. 
LMU and MCML, Munich
---
