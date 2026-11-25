#!/bin/bash
#SBATCH -o slurm_prepare_inverse_seed_targets_32x32_%A_%a.out
#SBATCH -e slurm_prepare_inverse_seed_targets_32x32_%A_%a.err
#SBATCH --array=0-11%8
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -e
source /hpc/dctrl/ks723/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -u prepare_inverse_seed_targets_32x32.py
