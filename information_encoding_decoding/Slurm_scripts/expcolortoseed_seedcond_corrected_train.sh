#!/bin/bash
#SBATCH --job-name=inverse_seedcond_N30000
#SBATCH -o slurm_expcolortoseed_seedcond_corrected_train_%j.out
#SBATCH -e slurm_expcolortoseed_seedcond_corrected_train_%j.err
#SBATCH -p youlab-gpu
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -e
source /hpc/dctrl/ks723/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -u expcolortoseed_seedcond_corrected_train.py
