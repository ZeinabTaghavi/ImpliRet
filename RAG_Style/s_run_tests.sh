#!/usr/bin/env bash
#SBATCH -p lrz-hgx-h100-94x4  
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o ./evaluation_s_run_tests_bm_a100_lrz.out
#SBATCH -e ./evaluation_s_run_tests_bm_a100_lrz.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'
export VLLM_WORKER_MULTIPROC_METHOD=fork
conda activate base
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting the job" 
python ./RAG_Style/scripts/sync/sync_evaluation.py \
       --config ./RAG_Style/experiment_configs/bm/A_Multi_llama_bm_1.yaml

python ./RAG_Style/scripts/sync/sync_evaluation.py \
       --config ./RAG_Style/experiment_configs/oracle_retriever/A_Multi_llama_1.yaml

