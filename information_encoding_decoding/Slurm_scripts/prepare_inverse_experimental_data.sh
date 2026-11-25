#!/bin/bash
#SBATCH -o slurm_prepare_inverse_experimental_data_%j.out
#SBATCH -e slurm_prepare_inverse_experimental_data_%j.err
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --mail-type=ALL

source /hpc/dctrl/ks723/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction

cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python prepare_inverse_experimental_data.py
