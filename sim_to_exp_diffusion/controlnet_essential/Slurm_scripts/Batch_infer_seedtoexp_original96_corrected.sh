#!/bin/bash
#SBATCH --job-name=seedtoexp_96_corrected
#SBATCH --output=slurm_oe/%x_%j.out
#SBATCH --error=slurm_oe/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
export TQDM_DISABLE=1
python batch_infer_seedtoexp_original96_corrected.py "$@"
