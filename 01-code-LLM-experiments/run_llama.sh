#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=150gb
#SBATCH --gres=gpu:A40:1

echo 'Running simulation'


# Load conda
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
# Activate the conda environment
conda activate llmlink

echo "Conda environment activated:"
echo $(conda env list)
echo " "
echo "Python version:"
echo $(which python)
echo " "

# activate CUDA
module load devel/cuda/11.6

models=("meta-llama/Llama-2-13b-hf") #  "meta-llama/Llama-2-7b-hf")
for i in ${!models[*]}; do
    python3 -u llama_logprobs.py \
        --model_name="${models[$i]}" \
        --task="ref_game" \
        --computation="use_surprisal"
done
