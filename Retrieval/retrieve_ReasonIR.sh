#!/usr/bin/env bash
#SBATCH -p mcml-dgx-a100-40x8 
#SBATCH -q mcml
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o ./retrieve_ReasonIR.out
#SBATCH -e ./retrieve_ReasonIR.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn


echo "Retrieving for track A, type Uni, retriever_name reasonir"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track A \
    --type Uni \
    --retriever_name reasonir


echo "Retrieving for track A, type Multi, retriever_name reasonir"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track A \
    --type Multi \
    --retriever_name reasonir 


echo "Retrieving for track F, type Uni, retriever_name reasonir"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track F \
    --type Uni \
    --retriever_name reasonir


echo "Retrieving for track F, type Multi, retriever_name reasonir"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track F \
    --type Multi \
    --retriever_name reasonir

echo "Retrieving for track T, type Uni, retriever_name reasonir"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track T \
    --type Uni \
    --retriever_name reasonir


echo "Retrieving for track T, type Multi, retriever_name reasonir"
python ./Retrieval/retrieve_indexing.py \
    --dataset_folder ./Dataset_Generation/Data/ \
    --output_folder ./Retrieval/Results/ \
    --track T \
    --type Multi \
    --retriever_name reasonir

