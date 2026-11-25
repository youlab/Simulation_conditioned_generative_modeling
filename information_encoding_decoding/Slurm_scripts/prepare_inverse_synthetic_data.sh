#!/bin/bash
#SBATCH -o slurm_prepare_inverse_synthetic_data_%A_%a.out
#SBATCH -e slurm_prepare_inverse_synthetic_data_%A_%a.err
#SBATCH --array=0-7
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --mail-type=ALL

set -e
source /hpc/dctrl/ks723/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
# Config currently regenerates only synthetic images from existing manifests.
# All eight conditions run separately; seed targets and training splits stay unchanged.
python -u prepare_inverse_synthetic_data.py
