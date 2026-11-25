#!/bin/bash 
#SBATCH -o slurm_batch_infer_Inf_ed_MORE_3x30000_seed_rerun_20260901_%a.out
#SBATCH -e slurm_batch_infer_Inf_ed_MORE_3x30000_seed_rerun_20260901_%a.err
#SBATCH -t 1-00:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --array=5,6,7
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
export TQDM_DISABLE=1
# override so the file-slicing math matches the original 30-way split (--array=5,6,7 sets SLURM_ARRAY_TASK_COUNT=3)
export SLURM_ARRAY_TASK_COUNT=30
python batch_infer_Inf_ed_MORE_3x30000_seed.py "$@"
