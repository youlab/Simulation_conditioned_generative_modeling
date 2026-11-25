#!/bin/bash
#SBATCH --job-name=synthetic_N30000_4gpu_lr5e4
#SBATCH -o slurm_expcolortoseed_vae_Diffusion_crop3left2right_seed32_lr5e4_%j.out
#SBATCH -e slurm_expcolortoseed_vae_Diffusion_crop3left2right_seed32_lr5e4_%j.err
#SBATCH -p youlab-gpu
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --mail-type=ALL

source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export EXPCOLORTOSEED_LEARNING_RATE="5e-4"
export EXPCOLORTOSEED_RUN_SUFFIX="crop3left2right_seed32_lr5e-4_control"

python expcolortoseed_vae_train_GPUs.py
