# 📚 ImpliRet: Benchmarking the Implicit Fact Retrieval Challenge


[![arXiv badge](https://img.shields.io/badge/arXiv-2506.14407-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.14407)
[![Hugging Face](https://img.shields.io/badge/%20Hugging%20Face-ImpliRet-ffca28?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ZeinabTaghavi/ImpliRet)


**ImpliRet** (**Impli**cit Fact **Ret**rieval) is a six-subset benchmark that shifts the reasoning burden from
the *query* to the *documents themselves*.  
A query is always simple (e.g. *“Who visited a museum on October 06 2024?”*) but the relevant document
contains the answer **only implicitly**—via:

| Reasoning category |    split   | Implicit cue inside the document | Query asks for … |
|--------------------|------------|----------------------------------|------------------|
| **Arithmetic**     |`arithmetic`|relative price or percentage (*“model 2019: 2.5 × more expensive…”*) | absolute price |
| **Temporal**       |`temporal`  | relative date (*“seven days ago”*) | explicit date |
| **World Knowledge**|`wknow`     | landmark that implies a country (*“Big Ben” ⇒ UK*) | the person / item in that country |

For each category we generate two discourse styles:

|      Discourse   |      name     | style   |
|------------------|---------------|---------|
| **Multi Speaker**|`multispeaker` | Multi-speaker **forum thread** (one post ↔︎ one response)
| **Uni Speaker**  |`unispeaker`   | Uni-speaker **chat** (one-chat ↔︎ 10-turn dialogue)      

---

---
## 📈 Results

### 🔬 Retrieval Evaluatio

The table below reports **nDCG@10** (↑ higher is better) for our baseline retrievers.

| Retriever | W. Know. | Arithmetic | Temporal | Average |
|-----------|:-------:|:----------:|:--------:|:-------:|
| ***Sparse*** |  |  |  |  |
| BM25 | 14.10 | 11.06 | 11.22 | 12.13 |
| ***Late Interaction*** |  |  |  |  |
| ColBERT v2 | 16.04 | 14.93 | 12.20 | 14.39 |
| ***Dense Encoders*** |  |  |  |  |
| Contriever | 16.15 | 13.84 | 12.59 | 14.19 |
| Dragon+ | 17.15 | **14.61** | 12.53 | 14.76 |
| ReasonIR‑8B | **19.53** | 10.74 | **14.93** | **15.07** |
| ***Knowledge‑Graph‑Augmented*** |  |  |  |  |
| HippoRAG 2 | 16.38 | 14.21 | 12.50 | 14.36 |

*Table 2. nDCG@10 retrieval performance averaged over uni‑speaker and multi‑speaker documents.*

### 🧩 RAG‑style Evaluation

The table below shows **ROUGE‑1 recall** (R‑1@k) for two long‑context LLM readers when the top‑*k* retrieved documents (oracle setting) are supplied.

| Experiment | *k* | W. Know. | Arithmetic | Temporal | Average |
|------------|:---:|:--------:|:----------:|:--------:|:-------:|
| **Llama 3.3 70B** | 1   | **73.79** | **90.13** | **81.85** | **81.92** |
|                | 10  | 27.37 | 16.98 | 25.23 | 23.19 |
|                | All | 17.43 | 4.42  | 10.29 | 10.71 |
| **GPT‑4.1**     | 1   | **93.24** | **92.12** | **84.90** | **88.05** |
|                | 10  | 62.21 | 23.86 | 15.59 | 35.06 |
|                | All | 53.91 | 9.28  |  6.93 | 22.90 |

*Table 3. ROUGE‑1 recall (R‑1@k), averaged over uni‑speaker and multi‑speaker documents.*




---

## Evaluation Instruction



```bash
# clone & install
$ git clone https://github.com/ZeinabTaghavi/ImpliRet.git
$ cd ImpliRet
$ python -m venv impliret_env && source impliret_env/bin/activate
$ pip install -r requirements.txt
```

---

## Repository map

```
├── RAG_Style/     
│   ├── experiment_configs   # Config of RAG with retrievals or Oracle retriever
│   ├── model_configs        # Config of each LLM that will be used in RAG_Style 
│   ├── script               # Codes of Asyncron and Syncron experiments
│   ├── results              
│   └── reports
├── Retrieval/         
│   ├── retrievals           # Codes of each experiment
│   ├── results
│   └── reports
└── README.md             
```

---

## 1. Loading the `ImpliRet` dataset.

You can load the dataset via Huggingface `zeinabTaghavi/ImpliRet` in the following format:
There are three reasoning categories (split): `['arithmetic', 'wknow', 'temporal']` and two discource style (name) `['multispeaker','unispeaker']`
```python
from datasets import load_dataset
# forum style, temporal reasoning
ds = load_dataset("zeinabTaghavi/ImpliRet",
                  name="multispeaker",
                  split="temporal")
print(ds[0]["question"])
print(ds[0]["pos_document"])
```

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
