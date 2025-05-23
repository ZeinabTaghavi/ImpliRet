#!/usr/bin/env bash
#SBATCH -p lrz-hgx-h100-94x4  
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -o ./retrieve_DragonPlus.out
#SBATCH -e ./retrieve_DragonPlus.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'

export CUDA_VISIBLE_DEVICES=0,1,2,3


echo "----------------------------  DragonPlus  ----------------------------------"
echo "Retrieving for track A, type Multi, retriever_name dragonplus"
python ./Retrieval/retrieve_indexing.py --dataset_folder ./Dataset_Generation/Data/ --output_folder ./Retrieval/Results/ --track A --type Multi --retriever_name dragonplus


echo "Retrieving for track A, type Uni, retriever_name dragonplus"
python ./Retrieval/retrieve_indexing.py --dataset_folder ./Dataset_Generation/Data/ --output_folder ./Retrieval/Results/ --track A --type Uni --retriever_name dragonplus


echo "Retrieving for track T, type Multi, retriever_name dragonplus"
python ./Retrieval/retrieve_indexing.py --dataset_folder ./Dataset_Generation/Data/ --output_folder ./Retrieval/Results/ --track T --type Multi --retriever_name dragonplus


echo "Retrieving for track T, type Uni, retriever_name dragonplus"
python ./Retrieval/retrieve_indexing.py --dataset_folder ./Dataset_Generation/Data/ --output_folder ./Retrieval/Results/ --track T --type Uni --retriever_name dragonplus


echo "Retrieving for track S, type Multi, retriever_name dragonplus"
python ./Retrieval/retrieve_indexing.py --dataset_folder ./Dataset_Generation/Data/ --output_folder ./Retrieval/Results/ --track S --type Multi --retriever_name dragonplus


echo "Retrieving for track S, type Uni, retriever_name dragonplus"
python ./Retrieval/retrieve_indexing.py --dataset_folder ./Dataset_Generation/Data/ --output_folder ./Retrieval/Results/ --track S --type Uni --retriever_name dragonplus



