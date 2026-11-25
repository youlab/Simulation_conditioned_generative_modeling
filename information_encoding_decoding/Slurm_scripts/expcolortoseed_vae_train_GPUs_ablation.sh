#!/bin/bash 
#SBATCH -o slurm_expcolortoseed_vae_Diffusion_ablation_crop3left2right_seed32_%A_%a.out
#SBATCH -e slurm_expcolortoseed_vae_Diffusion_ablation_crop3left2right_seed32_%A_%a.err
#SBATCH -p youlab-gpu
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --array=0-7

# Tasks 0-6: existing image ablations. Task 7: unmodified RGB baseline.
# Each task uses one GPU and batch size 32.

source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python expcolortoseed_vae_train_GPUs_ablation.py
