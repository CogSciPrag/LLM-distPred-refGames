#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=2:30:00
#SBATCH --mem=237gb
#SBATCH --gres=gpu:A100:2

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

python3 -u llama_boolq_test.py \
    --model_name="meta-llama/Llama-2-70b-hf" \
    --task="ref_game" \
    --computation="use_own_scoring"
