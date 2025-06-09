#!/usr/bin/env bash
#SBATCH -p lrz-dgx-a100-80x8 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o ./retrieve_Hippo_t_uni_25_26.out
#SBATCH -e ./retrieve_Hippo_t_uni_25_26.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --max_model_len 4096 \
    --gpu-memory-utilization 0.95 \
    --api-key "" &
# ------------------------------------------------------------------
# Wait for vLLM /health endpoint to return HTTP 200 (engine ready)
# ------------------------------------------------------------------
echo "[start_vllm] Waiting for vLLM /health…"
until curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/health" | grep -q "^200$"; do
  sleep 1
  echo "[start_vllm] Waiting for vLLM /health…"
done
echo "[start_vllm] vLLM is ready!"

echo "Retrieving for track A, type Multi, retriever_name hipporag"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track A \
    --type Multi \
    --retriever_name hipporag 


echo "Retrieving for track A, type Uni, retriever_name hipporag"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track A \
    --type Uni \
    --retriever_name hipporag 


echo "Retrieving for track S, type Multi, retriever_name hipporag"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track F \
    --type Multi \
    --retriever_name hipporag 


echo "Retrieving for track S, type Uni, retriever_name hipporag"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track F \
    --type Uni \
    --retriever_name hipporag 

echo "Retrieving for track T, type Multi, retriever_name hipporag"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track T \
    --type Multi \
    --retriever_name hipporag 

echo "Retrieving for track T, type Uni, retriever_name hipporag"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track T \
    --type Uni \
    --retriever_name hipporag
