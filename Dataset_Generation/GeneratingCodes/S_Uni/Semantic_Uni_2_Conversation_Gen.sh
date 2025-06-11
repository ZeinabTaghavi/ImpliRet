#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-03:00:00
#SBATCH -o ./Dataset_Generation/GeneratingCodes/S_Uni/Outputs.out
#SBATCH -e ./Dataset_Generation/GeneratingCodes/S_Uni/Errors.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zeinabtaghavi1377@gmail.com

source /dss/dsshome1/0B/di38wip/.bashrc
export HF_HOME='/dss/dssmcmlfs01/pn25pu/pn25pu-dss-0000/taghavi/HuggingFaceCache/'
export HF_TOKEN='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'
export VLLM_WORKER_MULTIPROC_METHOD=fork
conda activate base
echo "Starting the job" 

srun python3 ./Dataset_Generation/GeneratingCodes/Conversation_Gen.py --track S --conv_type Uni
echo "Conversation Generation for first step is done"
srun python3 ./Dataset_Generation/GeneratingCodes/Conversation_Gen.py --track S --conv_type Uni
echo "Conversation Generation for second step is done"

echo "Job finished"