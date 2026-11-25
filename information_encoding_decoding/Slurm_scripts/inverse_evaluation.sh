#!/bin/bash
#SBATCH -o slurm_inverse_evaluation_%j.out
#SBATCH -e slurm_inverse_evaluation_%j.err
#SBATCH -p youlab-gpu
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL

set -e
source /hpc/dctrl/ks723/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -u inverse_evaluation.py
