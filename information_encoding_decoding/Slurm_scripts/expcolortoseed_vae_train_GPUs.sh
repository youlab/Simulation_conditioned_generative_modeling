#!/bin/bash 
#SBATCH -o slurm_expcolortoseed_vae_Diffusion_crop3left2right_seed32_%j.out
#SBATCH -e slurm_expcolortoseed_vae_Diffusion_crop3left2right_seed32_%j.err
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
python expcolortoseed_vae_train_GPUs.py
