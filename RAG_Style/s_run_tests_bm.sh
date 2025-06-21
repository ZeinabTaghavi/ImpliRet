#!/usr/bin/env bash
#SBATCH -p lrz-hgx-a100-80x4
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
python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Multi_llama_bm_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Multi_llama_bm_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Multi_llama_bm_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Uni_llama_bm_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Uni_llama_bm_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Uni_llama_bm_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/A_Uni_llama_bm_20.yaml



python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Multi_llama_bm_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Multi_llama_bm_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Multi_llama_bm_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Uni_llama_bm_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Uni_llama_bm_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Uni_llama_bm_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/S_Uni_llama_bm_20.yaml



python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/T_Multi_llama_bm_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/T_Multi_llama_bm_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/bm/T_Multi_llama_bm_10.yaml

# python Experiments/evaluation/sync_run_tests.py \
#        --config Experiments/evaluation/run_configs/bm/T_Uni_llama_bm_1.yaml

# python Experiments/evaluation/sync_run_tests.py \
#        --config Experiments/evaluation/run_configs/bm/T_Uni_llama_bm_5.yaml

# python Experiments/evaluation/sync_run_tests.py \
#        --config Experiments/evaluation/run_configs/bm/T_Uni_llama_bm_10.yaml

# python Experiments/evaluation/sync_run_tests.py \
#        --config Experiments/evaluation/run_configs/bm/T_Uni_llama_bm_20.yaml

