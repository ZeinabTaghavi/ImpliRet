#!/usr/bin/env bash
#SBATCH -p lrz-hgx-a100-80x4 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o bright_rewrite.out
#SBATCH -e bright_rewrite.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Rewriting questions for track A, conv_type Multi via BRIGHT prompt"
python ./Retrieval/query_exapantion.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Bright_Questions/ \
    --track A \
    --conv_type Multi \
    --temperature 0.0


echo "Rewriting questions for track A, conv_type Uni via BRIGHT prompt"
python ./Retrieval/query_exapantion.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Bright_Questions/ \
    --track A \
    --conv_type Uni \
    --temperature 0.0

echo "Rewriting questions for track S, conv_type Multi via BRIGHT prompt"
python ./Retrieval/query_exapantion.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Bright_Questions/ \
    --track S \
    --conv_type Multi \
    --temperature 0.0

echo "Rewriting questions for track S, conv_type Uni via BRIGHT prompt"
python ./Retrieval/query_exapantion.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Bright_Questions/ \
    --track S \
    --conv_type Uni \
    --temperature 0.0

echo "Rewriting questions for track T, conv_type Multi via BRIGHT prompt"
python ./Retrieval/query_exapantion.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Bright_Questions/ \
    --track T \
    --conv_type Multi \
    --temperature 0.0

# echo "Rewriting questions for track T, conv_type Uni via BRIGHT prompt" 
# python ./Retrieval/query_exapantion.py \
#     --dataset_folder ./Dataset_Generation/Data/ \
#     --output_folder ./Retrieval/Bright_Questions/ \
#     --track T \
#     --conv_type Uni \
#     --temperature 0.0
