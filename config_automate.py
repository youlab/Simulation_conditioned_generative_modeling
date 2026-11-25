"""Public main-figure data paths. Run setup_data.py, then reproduce_figures.py.

Set DATA_DIR below to your data storage location, then run setup_data.py.
All datasets, weights, predictions and counts are resolved inside that directory.
The research notebooks' HPC configs are not imported by the figure generator.
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
# The only path you need to change. For example:
# DATA_DIR = Path("/your/preferred/data/location")
DATA_DIR = REPO_DIR / "data"  # Default: the data folder inside this repository.

# Optional override used by --data-dir; otherwise use the setting above.
if "PATTERN_DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["PATTERN_DATA_DIR"])
DATA_DIR = Path(DATA_DIR).expanduser().resolve()
EXTRACTED_DIR = DATA_DIR / "extracted"
SEED_TO_SIM_DIR = EXTRACTED_DIR / "seed_to_sim_deterministic"
SIM_TO_EXP_DIR = EXTRACTED_DIR / "sim_to_exp_diffusion"
INFO_ENCODING_DIR = EXTRACTED_DIR / "information_encoding_decoding"
LATENTS_DIR = EXTRACTED_DIR / "latents"
MAIN_FIGURE_RANDOM_SEED = 42


def image_directory(path):
    """Account for an enclosing directory in an existing deposited tar."""
    path = Path(path)
    if not path.is_dir() or any(p.suffix.lower() in {'.png', '.tif', '.tiff'} for p in path.iterdir()):
        return str(path)
    candidates = {p.parent for p in path.rglob('*') if p.suffix.lower() in {'.png', '.tif', '.tiff'}}
    if len(candidates) != 1:
        raise ValueError(f'Expected one image directory in {path}, found {len(candidates)}')
    return str(candidates.pop())


def model_file(name):
    matches = list((SEED_TO_SIM_DIR / 'saved_models').rglob(name))
    if len(matches) > 1:
        raise ValueError(f'Ambiguous deposited model: {name}')
    return str(matches[0] if matches else SEED_TO_SIM_DIR / 'saved_models' / name)


def data_demand_models(augmentation=False):
    pattern = 'dilRESNET_DataAugmentation_intermediatetocomplex_*v1015*best_tolerant*.pt' if augmentation else 'Pixel_32x32x4to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_*v1015*best_tolerant.pt'
    parents = {p.parent for p in (SEED_TO_SIM_DIR / 'saved_models').rglob(pattern)}
    if len(parents) != 1:
        raise FileNotFoundError(f'Run setup_data.py: expected one model family for {pattern}, found {len(parents)}')
    return str(parents.pop())


FPATH = SEED_TO_SIM_DIR / 'ARIAL.TTF'
SEED_FOLDER = image_directory(SIM_TO_EXP_DIR / 'Exp_SimcorrtoExp_seed')
EXPERIMENTAL_FOLDER = image_directory(SIM_TO_EXP_DIR / 'Exp')
SIMULATED_FOLDER = image_directory(SIM_TO_EXP_DIR / 'SimcorrtoExp')
FILES_FIG1A = ['Fixed_19_6.TIF', 'Fixed_15_3.TIF', 'Fixed_25_4.TIF', 'Fixed_6_1.TIF', 'Fixed_26_3.TIF']
FILES_FIG1B = ['5_2.TIF', 'Fixed_14_2.TIF', 'Fixed_29_3.TIF', 'Fixed_15_2.TIF', 'Fixed_22_1.TIF']
SIMULATION_TEST_INPUT_FOLDER_2 = image_directory(SEED_TO_SIM_DIR / 'Sim_050924_seed')
SIMULATION_TEST_OUTPUT_FOLDER_2 = image_directory(SEED_TO_SIM_DIR / 'Sim_050924_intermediate_Tp3')
SIMULATION_TEST_INPUT_FOLDER = image_directory(SEED_TO_SIM_DIR / 'Sim_050924_ModelTesting_seed')
SIMULATION_TEST_OUTPUT_FOLDER = image_directory(SEED_TO_SIM_DIR / 'Sim_050924_ModelTesting_intermediate_Tp3')
SIMULATION_TEST_OUTPUT_FOLDER_TDB = image_directory(SEED_TO_SIM_DIR / 'Sim_050924_ModelTesting_complex_Tp3')
SIMULATION_TEST_OUTPUT_FOLDER_TDB_2 = image_directory(SEED_TO_SIM_DIR / 'Sim_050924_complex_Tp3')
LATENT_OUTPUT_SAVED = str(LATENTS_DIR / 'latents_intermediate.pickle')
MODEL_DILRESNET_FIG2 = model_file('Pixel_32x32x3to32x32x4_dilRESNET_30k_graypatterns_seedtointermediate_v101_4-1759366230_best.pt')
MODEL_DILRESNET_FIG3 = model_file('Pixel_32x32x3to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_Model_30000_v102_9-1759430803.pt')
JSON_FILE_SEEDS_SUPPFIG7 = str(SEED_TO_SIM_DIR / 'Sim_050924_seed_counts_testset_60000_70000.json')
MODEL_SAVE_LOCATION = str(SEED_TO_SIM_DIR / 'saved_models')
LOGS_SAVE_LOCATION = str(SEED_TO_SIM_DIR / 'logs_selected')

SIM_FOLDER_TEST = image_directory(SIM_TO_EXP_DIR / 'SimcorrtoExp_testset')
SEED_FOLDER_TEST = image_directory(SIM_TO_EXP_DIR / 'Exp_SimcorrtoExp_testset_seed')
EXP_FOLDER_TEST = image_directory(SIM_TO_EXP_DIR / 'Exp_testset')
MAIN_FOLDER = str(SIM_TO_EXP_DIR / 'inference_folders' / 'v2025926_1251_simtoexp_v3')
EXP_FOLDER_TEST_FIXED = image_directory(INFO_ENCODING_DIR / 'Exp_testset_fixed')
SIM_FOLDER_TEST_FIXED_INPUT = str(SIM_TO_EXP_DIR / 'fixed_testset_simulations' / 'seeds')
SIM_FOLDER_TEST_FIXED_OUTPUT = str(SIM_TO_EXP_DIR / 'fixed_testset_simulations' / 'simulations')
DIFFUSION_FOLDER_TEST_FIXED = str(SIM_TO_EXP_DIR / 'fixed_testset_predictions')
SEED_FOLDER_TEST_SEEDINFO_COMBINED = str(SIM_TO_EXP_DIR / 'combined_seed_counts_testset_132.json')

SIM_FOLDER_TEST_INFOENCODING_MORESEEDS = SIMULATION_TEST_OUTPUT_FOLDER_2
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS = SIMULATION_TEST_INPUT_FOLDER_2
SEED_FOLDER_SEEDTOSIM_TEST = SIMULATION_TEST_INPUT_FOLDER
SPECIFIC_FOLDER_EXP_DIFFUSION_MORE = image_directory(INFO_ENCODING_DIR / 'Generated_patterns_selected')
SPECIFIC_FOLDER_SEED_DIFFUSION_MORE = SIMULATION_TEST_INPUT_FOLDER_2
SEED_FOLDER_EXPTOSIM_TEST_32X32 = image_directory(INFO_ENCODING_DIR / 'Exp_testset_seed')
SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32 = image_directory(INFO_ENCODING_DIR / 'Exp_testset_fixed_seed')
EXP_FOLDER_EXPTOSIM_TEST_FIXED = EXP_FOLDER_TEST_FIXED
CKPT_PATH_SIMULATION_CONDITIONED_LR5E4_LAST = str(INFO_ENCODING_DIR / 'checkpoint_exptoseed' / 'last.ckpt')
CKPT_PATH_MORESEEDS_DIFFUSION = CKPT_PATH_SIMULATION_CONDITIONED_LR5E4_LAST
CKPT_PATH_SEEDTOSIM_MORESEEDS_3ROTREPS = str(INFO_ENCODING_DIR / 'checkpoint_simtoseed' / 'best-epoch=290-val_loss=0.0080.ckpt')
CKPT_PATH_EXPTOCOLONYMETRICS = str(INFO_ENCODING_DIR / 'checkpoint_colonyarea' / 'best-epoch=236-val_loss=0.0029.ckpt')
INVERSE_PREDICTIONS = INFO_ENCODING_DIR / 'inverse_predictions_main'
OUTPUT_DIR_EXPTOSEED_COLOR_SAVED = str(INVERSE_PREDICTIONS / 'diffusion_to_seed_132')
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_SAVED = str(INVERSE_PREDICTIONS / 'simulation_to_seed_107')
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET_SAVED = str(INVERSE_PREDICTIONS / 'simulation_model_on_experiment_132')
SEED_FOLDER_EXPTOSIM_TEST_FIXED_NONFIXED_32X32 = str(INVERSE_PREDICTIONS / 'experiment_targets_132')


def validate_setup():
    from figure_generators.public_data import validate_downloads
    return validate_downloads(DATA_DIR)
