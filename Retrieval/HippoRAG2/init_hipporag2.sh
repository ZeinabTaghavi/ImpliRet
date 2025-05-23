#!/usr/bin/env bash
set -e

BASE="./MetatagIndexing/HippoRAG2"

echo "Creating project structure under $BASE…"
mkdir -p "$BASE"/conf
mkdir -p "$BASE"/data

echo "Writing .env…"
cat > "$BASE/.env" << 'EOF'
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key
EOF

echo "Writing requirements.txt…"
cat > "$BASE/requirements.txt" << 'EOF'
hipporag>=0.1.0
hydra-core>=1.3
python-dotenv
vllm
transformers
EOF

echo "Writing run_experiment.py…"
cat > "$BASE/run_experiment.py" << 'EOF'
import os
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv
from hipporag import HippoRAG
import json

load_dotenv()  # load .env

@hydra.main(version_base=None, config_path="conf", config_name="experiment_config")
def main(cfg: DictConfig):
    hippo = HippoRAG(
        save_dir=cfg.output_dir,
        llm_model_name=cfg.llm_model,
        llm_base_url=cfg.llm_base_url,
        embedding_model_name=cfg.embedding_model
    )
    # load corpus and queries
    with open(cfg.data.corpus) as f:
        docs = json.load(f)
    hippo.index(docs=docs)

    with open(cfg.data.queries) as f:
        queries = json.load(f)
    retrieval = hippo.retrieve(queries=queries, num_to_retrieve=cfg.retrieval.top_k)
    answers = hippo.rag_qa(retrieval)

    print(answers)

if __name__ == "__main__":
    main()
EOF

echo "Writing run_experiment.sh…"
cat > "$BASE/run_experiment.sh" << 'EOF'
#!/usr/bin/env bash
# Usage: ./run_experiment.sh
python run_experiment.py
EOF
chmod +x "$BASE/run_experiment.sh"

echo "Writing conf/experiment_config.yaml…"
cat > "$BASE/conf/experiment_config.yaml" << 'EOF'
data:
  corpus: "../data/corpus.json"
  queries: "../data/queries.json"

llm_model: "meta-llama/Llama-3.3-70B-Instruct"
llm_base_url: "http://localhost:8000/v1"
embedding_model: "facebook/contriever"

retrieval:
  top_k: 5

output_dir: "../results"
EOF

echo "Writing data/corpus.json…"
cat > "$BASE/data/corpus.json" << 'EOF'
[
  {"idx": 0, "title": "Doc A", "text": "This is the first document."},
  {"idx": 1, "title": "Doc B", "text": "Here’s the second one."},
  {"idx": 2, "title": "Doc C", "text": "And a third."}
]
EOF

echo "Writing data/queries.json…"
cat > "$BASE/data/queries.json" << 'EOF'
[
  "What does Doc A talk about?",
  "Tell me about the second document.",
  "How many docs are here?"
]
EOF

echo "Writing Dockerfile…"
cat > "$BASE/Dockerfile" << 'EOF'
FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["bash", "run_experiment.sh"]
EOF

echo "Writing README.md…"
cat > "$BASE/README.md" << 'EOF'
# HippoRAG2

# ## Setup

# ```bash
# cd MetatagIndexing/HippoRAG2
# pip install -r requirements.txt
# # fill in .env, edit conf/experiment_config.yaml if needed
# bash run_experiment.sh

# ------------------------------------------------------------

# **Usage:**

# ```bash
# chmod +x MetatagIndexing/Experiments/HippoRAG2/init_hipporag2.sh
# ./MetatagIndexing/Experiments/HippoRAG2/init_hipporag2.sh