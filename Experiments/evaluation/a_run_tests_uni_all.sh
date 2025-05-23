#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:30:00
#SBATCH -o ./evaluation_a_run_tests_h100_lrz.out
#SBATCH -e ./evaluation_a_run_tests_h100_lrz.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'



echo "Starting the job" 
python Experiments/evaluation/async_run_tests.py \
       --config Experiments/evaluation/run_configs/LC/T_Uni_llama_LC_1_GPT.yaml
