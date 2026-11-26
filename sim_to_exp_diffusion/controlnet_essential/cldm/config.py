import matplotlib as mpl
from pathlib import Path
from datetime import datetime

currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year


#########
# Font for plots
FPATH = Path(mpl.get_data_path(), "/hpc/group/youlab/ks723/miniconda3/Lingchong/fonts/ARIAL.TTF")  # ARIAL.TTF
#########

# For creating prompt.json files 
BASE_FOLDER= '/hpc/group/youlab/ks723/storage/'
SPECIFIC_FOLDER_SIM='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_100AUG' # Extract SimcorrtoExp.tar, run the augmentation script and use those images here
SPECIFIC_FOLDER_EXP='/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_100AUG' # Extract Exp.tar, run the augmentation script and use those images here
SPECIFIC_FOLDER_SEED='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG' # Extract Exp_SimcorrtoExp_seed.tar, run Seed_DataAugmentation.py script and use those images here
SPECIFIC_FOLDER_SEED_CORRECTED='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG_vfixed'

# For Sim to Exp dataset, model training
EXP_FOLDER_TRAIN_NONAUG='/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed/'  # Exp.tar
EXP_FOLDER_TRAIN_NONAUG_PREPROCESSED='/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_preprocess/'  # black-background 256x256 images; apply the standard 25 px crop before analysis

# For Seed to Exp dataset, model training

SEED_FOLDER_TRAIN_NONAUG="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input/"  # Exp_SimcorrtoExp_seed.tar

# Original primary cohort: 96 matched experimental/simulation images.
# Image-metric and morphology analyses combine these with the 36 fixed images below.
SIM_FOLDER_TEST   = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_v3/"  # SimcorrtoExp_testset.tar
SEED_FOLDER_TEST  = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3/"  # Exp_SimcorrtoExp_testset_seed.tar
EXP_FOLDER_TEST   = "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3/"   # Exp_testset.tar 

# Creation of different folders that will be later used for inference outputs
OUTPUT_DIR_SEEDTOEXP = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SEEDTOEXP"
OUTPUT_DIR_SIMTOEXP = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SIMTOEXP"
OUTPUT_DIR_SIMTOEXP_AREA = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SIMTOEXP_AREA"

OUTPUT_DIR_RANDOMSEEDSWEEP = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_random_seed_sweep/"
OUTPUT_DIR_ABLATION_BASE= f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}"
OUTPUT_DIR_CONTRASTIVE_LEARNING=f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/contrastive_learning_v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}/"

# For ablation study 

MAIN_FOLDER= '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v2025926_1251_simtoexp_v3/'   # default ControlNet, OUTPUT_DIR_SIMTOEXP
ABLATION_1=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1458_no_guess/'  # Guess_mode= TRUE , Run ablation script and modify the part accordingly
ABLATION_2=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1753_no_negative/' # n_prompt=""
ABLATION_3=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1756_plus_positive/' # a_prompt="best quality, extremely detailed" # default ControlNet
ABLATION_4=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1758_low_strength_point85/' # strength=0.85
ABLATION_5=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1758_high_strength_1point25/' # strength=1.25
ABLATION_6=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1759_higher_DDIM_steps_100/'  # ddim_steps=100
ABLATION_7=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_181_lower_guidance_9point0/'  # guidance_scale=9.0 # default ControlNet


# Figure 4 and Supplementary Figure 17 image grids
PRED_FOLDER_SEEDTOEXP = '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251011_841_seedtoexp_swapped_v3/'  # OUTPUT_DIR_SEEDTOEXP
PRED_FOLDER_SEEDTOEXP_CORRECTED_96 = '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20261022_SEEDTOEXP_ORIGINAL96_CORRECTED_2SAMPLES'  # two corrected predictions per original test stem

# Checkpoint path for inference 

CKPT_PATH='/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_37312560/checkpoints/epoch=4-step=51124.ckpt'   # checkpoint_simtoexp.tar
CKPT_PATH_SEEDTOEXP="/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_37726282/checkpoints/epoch=4-step=51124.ckpt" # checkpoint_seedtoexp.tar
CKPT_PATH_SEEDTOEXP_CORRECTED="/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_58136436/checkpoints/epoch=4-step=51124.ckpt"  # corrected 3-left/2-right conditioning run, completed 2026-10-08
CKPT_PATH_AREA='/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/checkpoints_area/area-epoch=04-step=51124.ckpt'


# For testing information encoding decoding 

SIM_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3'  # folder with more seeds, upto 20
OUTPUT_DIR_SIMTOEXP_INFOENCODING = f"/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}_SIMTOEXP"

# Addressing reviewer comments 

# Testing 

# For testing information encoding decoding by using the seed to exp model 

SEED_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output' # seed folder 
OUTPUT_DIR_SEEDTOEXP_INFOENCODING= f"/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}_SEEDTOEXP"

# File locations for test set seeding information 

SEED_FOLDER_TEST_SEEDINFO= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Sim_031524_96_seed_counts_testset.json'
SEED_FOLDER_TRAIN_SEEDINFO= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Sim_031524_409_seed_counts_trainset.json'

# Fixed-seed test set (19 seed configurations, up to 2 experimental replicates each),
# extends the 96-image three-way test set above to ~132 images for the morphometric analysis.
EXP_FOLDER_TEST_FIXED = '/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_Fixedseeds_v4_preprocessed/'  # Fixed_N_M.TIF, 36 files
SIM_FOLDER_TEST_FIXED_OUTPUT = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_output'  # Output_Fixed_N.png, 19 files, shared across experimental replicates of the same config
SIM_FOLDER_TEST_FIXED_INPUT = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input'  # Input_Fixed_N.png seed configs, used only to count initial seeds
DIFFUSION_FOLDER_TEST_FIXED = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v2026416_16_SIMTOEXP_FIXEDTESTSET'  # Fixed_N_M_R.png, 3 diffusion samples (R) per experimental stem
SEED_CONDITIONED_FOLDER_TEST_FIXED = '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20261022_SEEDTOEXP_FIXEDTESTSET'  # Fixed_N_R.png, 3 stochastic samples per fixed seed configuration
SEED_FOLDER_TEST_SEEDINFO_COMBINED = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Sim_031524_combined_seed_counts_testset.json'  # merges SEED_FOLDER_TEST_SEEDINFO with counts for the Fixed_N_M stems above

SIM_FOLDER_TRAIN_NONAUG= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL'
OUTPUT_DIR_SIMTOEXP_TRAIN_NONAUG = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v2026921_SIMTOEXP_TRAIN_NONAUG'
SPECIFIC_FOLDER_EXP_DIFFUSION_MORE= '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202642_19_SIMTOEXP'
SPECIFIC_FOLDER_EXP_DIFFUSION_MORE_COLONYMETRICS= '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/colony_metrics.json'
