from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PREPARATION_ROOT = _PROJECT_ROOT / 'Misc/inverse_preparation'
import matplotlib as mpl
import os
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

#########################################################################

# INVERSE PROBLEM 

# folders from Fig 2, simulations used as the base to generate 3 replicate experimental-like patterns for Figure 5 
SIM_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3'  # Sim_050924_intermediate_Tp3.tar
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output'  # Sim_050924_seed.tar

# Data augmentation using rotation (created using SeedplusSim_rotation_3reps.py script), for training the sim to seed mapping model with same input data size as the exp to seed model(90k)
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3ROTREPS'
SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input_32x32_3ROTREPS/intermediate/Tp3' 

# generated images using ControlNet pipeline, synthetic images are named as exp in the folder variable cause they look like exps
# used as the base dataset to train the exp to seed model in Figure 5-- exp folder acts as output, seed as input. 
SPECIFIC_FOLDER_EXP_DIFFUSION_MORE = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202642_19_SIMTOEXP'  # synthetic generated patterns 30k*3 replicates # selected used in Fig 6 are in Generated_patterns_selected.tar
SPECIFIC_FOLDER_SEED_DIFFUSION_MORE= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3reps' # previous targets using the 2/3 crop
SPECIFIC_FOLDER_SEED_DIFFUSION_MORE_VFIXED = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3reps_crop3left2right'

# Model checkpoints
# Completed 4-GPU simulation-conditioned inverse run (LR 5e-4), resumed as job 60235784.
SIMULATION_CONDITIONED_LR5E4_RUN_DIR = Path(__file__).resolve().parents[1] / 'lightning_logs' / 'Res_UNet_exps_color_vae_nonrandom_vDiffusion' / 'lambda0.5_pos2.0_GPUs_run7_features[16, 32, 64]_seeds_no_augmentation_crop3left2right_seed32_lr5e-4_control'
CKPT_PATH_SIMULATION_CONDITIONED_LR5E4_BEST = str(SIMULATION_CONDITIONED_LR5E4_RUN_DIR / 'checkpoints' / 'best-epoch=142-val_loss=0.3384.ckpt')
CKPT_PATH_SIMULATION_CONDITIONED_LR5E4_LAST = str(SIMULATION_CONDITIONED_LR5E4_RUN_DIR / 'checkpoints' / 'last.ckpt')

CKPT_PATH_MORESEEDS_DIFFUSION_LEGACY = '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run7_MORE_seeds_no_augmentation/checkpoints/best-epoch=169-val_loss=0.3398.ckpt'
CKPT_PATH_MORESEEDS_DIFFUSION = CKPT_PATH_SIMULATION_CONDITIONED_LR5E4_LAST  # shared Figure 5E/F simulation-conditioned inverse model
CKPT_PATH_SEEDTOSIM_MORESEEDS_3ROTREPS= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_sim_vae_nonrandom/lambda0.5_pos2.0_GPUsv3RotReps/checkpoints/best-epoch=290-val_loss=0.0080.ckpt'  # sim to seed mapping, checkpoint_simtoseed.tar

# Exp folder test, similar as the ones used in Fig 5, input seed images first converted to 32x32, no extra cropping in Figure 5
EXP_FOLDER_TEST   = "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3/"   # Exp_testset.tar in sim_to_exp_diffusion hf datasets folder
SEED_FOLDER_EXPTOSIM_TEST_32X32= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3_32x32_coordmethod"  # Exp_testset_seed.tar

# New Exp folder test, has fixed seeding configurations, seed folder are the corresponding seeding configurations
SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input_preprocessed_v3'  # Exp_testset_fixed_seed.tar
EXP_FOLDER_EXPTOSIM_TEST_FIXED= '/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_Fixedseeds_v4_preprocessed/'  # Exp_testset_fixed.tar

# Simulation test set consisting of fixed patterns, used to test simulation-seed mappping 
SEED_FOLDER_SEEDTOSIM_TEST="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_output"  #Sim_050924_ModelTesting_seed.tar
SIM_FOLDER_SEEDTOSIM_TEST= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_input/intermediate/Tp3" #Sim_050924_ModelTesting_intermediate_Tp3.tar


# Testing metrics on the full test set 

# exp to seed mapping
SEED_FOLDER_EXPTOSIM_TEST_FIXED_NONFIXED_32X32= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input_FULLTESTSET'  # combining fixed and non-fixed test sets in the same folder 
# Shared destination for updated-model predictions and experimental evaluation.
# OUTPUT_DIR_EXPTOSEED_COLOR_SAVED = "/hpc/dctrl/ks723/Information_encoding_decoding/inference/v202648_1529_EXPTOSEED_COLOR"  # previous predictions
OUTPUT_DIR_EXPTOSEED_COLOR_SAVED = "/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding/inference/EXPTOSEED_COLOR_v2"

# sim to seed mapping
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_SAVED= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026420_1618_SIMTOSEED_MORESEEDS_3ROTREPS"  # saved prediction folder for sim to seed mapping 

# exp to seed mapping for model trained on sim to seed mapping 
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET_SAVED= "/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026420_1812_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET"

# reviewer comments addition

SPECIFIC_FOLDER_EXP_DIFFUSION_MORE_SEEDCOND= '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202691_1_SEEDTOEXP_COMBINED'
CKPT_PATH_MORESEEDS_DIFFUSION_SEEDCOND = str(_PROJECT_ROOT / 'information_encoding_decoding/lightning_logs/inverse_seedconditioned_crop3left2right_seed32/N30000_seed42/checkpoints/last.ckpt')

SPECIFIC_FOLDER_EXP_DIFFUSION_MORE= '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202642_19_SIMTOEXP'
SPECIFIC_FOLDER_EXP_DIFFUSION_MORE_COLONYMETRICS= '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/colony_metrics.json'

# Colony-metrics downstream model and experimental test-set labels
CKPT_PATH_EXPTOCOLONYMETRICS = '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/information_encoding_decoding/lightning_logs/ResNet_exps_colony_metrics_vae/MSE_features[16, 32, 64]_no_tip_log_transform/checkpoints/best-epoch=236-val_loss=0.0029.ckpt'
EXP_FOLDER_TEST_COLONYMETRICS = str(_PROJECT_ROOT / 'source_data/colony_area/exp_test_colony_metrics.json')

# Real-to-synthetic inverse-decoding experiment
SEED_FOLDER_TRAIN_NONAUG = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input'
EXP_FOLDER_TRAIN_100AUG = '/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_100AUG'
SEED_FOLDER_TRAIN_100AUG = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG_vfixed'

SEED_FILES_TRAIN_NONAUG = sorted(os.listdir(SEED_FOLDER_TRAIN_NONAUG))


# Inverse-model synthetic-data ablation: Phase 1 inventory
SEED_COUNT_JSON_TRAIN_UNAUGMENTED = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Sim_031524_409_seed_counts_trainset.json'

# Reuse the existing 100-rotation training folders. Phase 1 selects the
# zero-rotation file for each original experimental plate.
EXP_FILES_TRAIN_100AUG = sorted(os.listdir(EXP_FOLDER_TRAIN_100AUG))
SEED_FILES_TRAIN_100AUG = sorted(os.listdir(SEED_FOLDER_TRAIN_100AUG))

INVERSE_ABLATION_PHASE1_OUTPUT_DIR = str(_PREPARATION_ROOT / 'phase1_experimental_inventory')
INVERSE_ABLATION_SELECTION_SIZES = (10, 50, 100)
INVERSE_ABLATION_SELECTION_SEED = 42
INVERSE_ABLATION_SELECTION_MODE = 'balanced_seed_count'


# Inverse-model synthetic-data ablation: Phase 2 experimental learning curve
INVERSE_ABLATION_PHASE2_OUTPUT_ROOT = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inverse_model_ablation/phase2_experimental_unique_configurations'
INVERSE_ABLATION_PHASE2_CONDITIONS = ('N10', 'N50', 'N100', 'NMAX')
INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES = 90000
INVERSE_ABLATION_PHASE2_SEED_LEFT_CROP = 3
INVERSE_ABLATION_PHASE2_SEED_RIGHT_CROP = 2
INVERSE_ABLATION_PHASE2_OVERWRITE_EXISTING = False

# Phase 3: synthetic unique-configuration learning curve. Reuse the existing
# simulation-conditioned predictions and original (uncropped) seed folder.
SPECIFIC_FILES_EXP_DIFFUSION_MORE = sorted(os.listdir(SPECIFIC_FOLDER_EXP_DIFFUSION_MORE))
SEED_FILES_TEST_INFOENCODING_MORESEEDS = sorted(os.listdir(SEED_FOLDER_TEST_INFOENCODING_MORESEEDS))
INVERSE_ABLATION_PHASE3_OUTPUT_ROOT = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inverse_model_ablation/phase3_synthetic_unique_configurations'
INVERSE_ABLATION_PHASE3_SELECTION_SIZES = (10, 50, 100, 500, 1000, 5000, 10000, 30000)
INVERSE_ABLATION_PHASE3_REPLICATES = (1, 2, 3)
INVERSE_ABLATION_PHASE3_OVERWRITE_EXISTING = False
# Replace only synthetic colony images using existing manifests; leave seeds/splits untouched.
INVERSE_ABLATION_PHASE3_REGENERATE_SYNTHETIC_ONLY = True

# Native seed targets, generated from the original seeds, never enlarged.
INVERSE_ABLATION_SEED32_SUBFOLDER = 'seed_32x32'
INVERSE_ABLATION_SEED32_REPORT_ROOT = str(_PREPARATION_ROOT / 'seed32_target_reports')

# Phase 4: 90k TOTAL pairs per condition, split by configuration, not image.
INVERSE_ABLATION_PHASE4_OUTPUT_ROOT = str(_PREPARATION_ROOT / 'phase4_training_manifests_32x32')
INVERSE_ABLATION_PHASE4_CONDITIONS = tuple(
    [('experimental', condition) for condition in INVERSE_ABLATION_PHASE2_CONDITIONS]
    + [('synthetic', f'N{size}') for size in INVERSE_ABLATION_PHASE3_SELECTION_SIZES]
)
INVERSE_ABLATION_PHASE4_VALIDATION_FRACTION = 0.1
INVERSE_ABLATION_PHASE4_OVERWRITE_EXISTING = False
INVERSE_ABLATION_PHASE4_BATCH_SIZE = 32
INVERSE_ABLATION_PHASE4_NUM_WORKERS = 8
INVERSE_ABLATION_PHASE4_MAX_EPOCHS = 200
INVERSE_ABLATION_PHASE4_PATIENCE = 50
INVERSE_ABLATION_PHASE4_LEARNING_RATE = 1e-4
INVERSE_ABLATION_PHASE4_DROPOUT = 0.3
INVERSE_ABLATION_PHASE4_WEIGHT_DECAY = 1e-4
INVERSE_ABLATION_PHASE4_DICE_WEIGHT = 0.5
INVERSE_ABLATION_PHASE4_POS_WEIGHT = 2.0
INVERSE_ABLATION_PHASE4_FEATURES = (16, 32, 64)
INVERSE_ABLATION_PHASE4_LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'lightning_logs', 'inverse_ablation_phase4_seed32',
)

# Phase 5: reuse Phase 4 splits and hyperparameters for the mixture comparison.
INVERSE_ABLATION_PHASE5_EXPERIMENTAL_PERCENTAGES = (0, 25, 50, 75, 100)
INVERSE_ABLATION_PHASE5_TRAIN_PERCENTAGES = (25, 50, 75)  # Endpoints already run in Phase 4.
INVERSE_ABLATION_PHASE5_OUTPUT_ROOT = str(_PREPARATION_ROOT / 'phase5_mixture_manifests_32x32')
INVERSE_ABLATION_PHASE5_LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'lightning_logs', 'inverse_ablation_phase5_seed32',
)
INVERSE_ABLATION_PHASE5_OVERWRITE_EXISTING = False

# Seed-conditioned inverse run: reuse native 32x32 targets and Phase 4 split.
INVERSE_ABLATION_SEEDCOND_LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'lightning_logs', 'inverse_seedconditioned_crop3left2right_seed32',
)


def list_seed_conditioned_predictions():
    # Inventory only when this run needs it, not when other jobs import config.
    return sorted(os.listdir(SPECIFIC_FOLDER_EXP_DIFFUSION_MORE_SEEDCOND))

EXP_FOLDER_TEST_V0= '/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set'
