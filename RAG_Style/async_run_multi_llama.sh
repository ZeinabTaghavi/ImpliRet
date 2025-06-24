#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:30:00
#SBATCH -o ./async_run_multi_llama.out
#SBATCH -e ./async_run_multi_llama.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'
export VLLM_WORKER_MULTIPROC_METHOD=fork
conda activate base
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Launch local vLLM server" 
# ------------------------------------------------------------------
# Start vLLM server via helper script (background) and wait for load
# ------------------------------------------------------------------
# run_tests.sh  (top of file)
PROJECT_ROOT=/dss/dsshome1/0B/di38wip/RAG_Style   # adjust once
source "$PROJECT_ROOT/scripts/async/start_vllm.sh"

echo "Starting the job" 
python ./RAG_Style/scripts/async/async_run_tests.py \
       --config ./RAG_Style/experiment_configs/oracle_retriever/A_Multi_llama_1.yaml


# ------------------------------------------------------------------
# Shut down the vLLM server
# ------------------------------------------------------------------
echo "Stopping vLLM server (PID=$VLLM_PID)"
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

