#!/usr/bin/env bash
#SBATCH -p lrz-dgx-a100-80x8 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o ./evaluation_s_run_tests_dp_a100_lrz.out
#SBATCH -e ./evaluation_s_run_tests_dp_a100_lrz.err
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
       --config Experiments/evaluation/run_configs/dp/A_Multi_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Multi_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Multi_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_20.yaml



python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Multi_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Multi_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Multi_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/S_Uni_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Uni_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Uni_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Uni_llama_dp_20.yaml



python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Multi_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Multi_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Multi_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_20.yaml


#!/usr/bin/env bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o ./evaluation_s_run_tests_dp_a100_lrz.out
#SBATCH -e ./evaluation_s_run_tests_dp_a100_lrz.err
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
       --config Experiments/evaluation/run_configs/dp/A_Multi_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Multi_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Multi_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/A_Uni_llama_dp_20.yaml



python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Multi_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Multi_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Multi_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/S_Uni_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Uni_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Uni_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/S_Uni_llama_dp_20.yaml



python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Multi_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Multi_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Multi_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_1.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_5.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_10.yaml

python Experiments/evaluation/sync_run_tests.py \
       --config Experiments/evaluation/run_configs/dp/T_Uni_llama_dp_20.yaml

