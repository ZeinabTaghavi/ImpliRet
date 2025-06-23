#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8 
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:30:00
#SBATCH -o ./s_run_tests.out
#SBATCH -e ./s_run_tests.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'
export VLLM_WORKER_MULTIPROC_METHOD=fork
conda activate base
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting the job" 
python ./RAG_Style/scripts/sync/sync_run_tests.py \
       --config ./RAG_Style/experiment_configs/bm/A_Multi_llama_bm_1.yaml

python ./RAG_Style/scripts/sync/sync_run_tests.py \
       --config ./RAG_Style/experiment_configs/oracle_retriever/A_Multi_llama_1.yaml

