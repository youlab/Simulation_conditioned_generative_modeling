"""Main-figure recipes using the reviewed notebooks and shared PDF exporters.

Only the computational panels of Figures 1–5 are listed here. Schematic panels
and final page composition are assembled in the manuscript editor. Image recipes
use the notebook's selected examples/models; quantitative recipes recompute
measurements from the pinned public images and checkpoints. No supplementary cells run.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = 'seed_to_sim1_sim2_deterministic'
DIFFUSION = 'sim_to_exp_diffusion/controlnet_essential'
INVERSE = 'information_encoding_decoding'


@dataclass(frozen=True)
class MainFigureTask:
    key: str
    figure: int
    notebook: str
    description: str
    outputs: tuple
    cells: tuple
    source_tables: tuple = ()


TASKS = {
    task.key: task for task in [
        MainFigureTask('1B', 1, f'{DETERMINISTIC}/Display_seed_sim_exp_Fig1.ipynb',
            'Selected seeds, PDE simulations and experiments', ('Fig1B_seed_sim_exp.pdf',), (1, 2)),
        MainFigureTask('1C', 1, f'{DETERMINISTIC}/Display_seed_sim_exp_Fig1.ipynb',
            'SD-VAE simulation and experiment reconstructions',
            ('Fig1C_simulations_reconstruction.pdf', 'Fig1C_experiments_reconstruction.pdf'), (1, 4, 5, 6, 7)),
        MainFigureTask('2B', 2, f'{DETERMINISTIC}/Prediction_seedtointermediate_dilResNet_Fig2_SuppFig7.ipynb',
            'Seed-to-PDE prediction examples', ('Fig2B_seed_to_simulation_predictions.pdf',), (2, 3, 4, 5)),
        MainFigureTask('2CD', 2, f'{DETERMINISTIC}/Scatterplot_SSIM_seedvariation_Fig2.ipynb',
            'SSIM versus seed count and selected SSIM examples',
            ('Fig2C_SSIM_vs_seed_count.pdf', 'Fig2D_ssim_hybrid_300dpi.pdf',
             'Fig2D_ssim_only_hybrid_300dpi.pdf'), (1, 2, 3, 4, 5, 6, 7)),
        MainFigureTask('3B', 3, f'{DETERMINISTIC}/Prediction_intermediatetocomplex_dilResNet_Fig3_SuppFig10.ipynb',
            'PDE-regime mapping examples', ('Fig3B_regime_predictions.pdf',), (2, 3, 5, 6, 7)),
        MainFigureTask('3CD', 3, f'{DETERMINISTIC}/Prediction_datademand_intermediatetocomplex_Fig3.ipynb',
            'Data diversity: validation loss and prediction examples',
            ('Fig3C_validation_loss.pdf', 'Fig3D_predictions.pdf'), tuple(range(2, 10))),
        MainFigureTask('3EF', 3, f'{DETERMINISTIC}/Prediction_datademand_augmentation_intermediatetocomplex_Fig3.ipynb',
            'Data augmentation: validation loss and prediction examples',
            ('Fig3E_validation_loss.pdf', 'Fig3F_predictions.pdf'), tuple(range(2, 10))),
        MainFigureTask('4B', 4, f'{DIFFUSION}/plot_Fig4_SuppFig17.ipynb',
            'Simulation-conditioned image grid from saved predictions',
            ('Fig4B_simulation_conditioned_images_300dpi.pdf',), (2,)),
        MainFigureTask('4C', 4, f'{DIFFUSION}/morphometric_variation_analysis.ipynb',
            'Plate-width KDEs with the >26 seed-count color category',
            ('Fig4C_branch_width_distributions.pdf',), (2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14)),
        MainFigureTask('4D', 4, f'{DIFFUSION}/inference_quantmetrics_simtoexp.ipynb',
            'Full 132-image LPIPS, SSIM, ORB and original CMMD summaries',
            ('Fig4D_image_comparison_metrics.pdf', 'Fig4D_image_comparison_metrics_300dpi.png'), (14, 15)),
        MainFigureTask('5C', 5, f'{INVERSE}/Display_Fig5_SuppFig23.ipynb',
            'Seed/PDE/generated-image examples from saved predictions',
            ('Fig5C_seed_simulation_generated_images.pdf',), (2,)),
        MainFigureTask('5E', 5, f'{INVERSE}/Display_Fig5_SuppFig23.ipynb',
            'Current inverse checkpoint; examples ordered by Hungarian tolerance F1',
            ('Fig5E_inverse_prediction_hybrid_300dpi.pdf', 'Fig5E_images_300dpi.pdf'), (4,)),
        MainFigureTask('5F', 5, f'{INVERSE}/Evaluation_metrics_Fig5.ipynb',
            'Last-checkpoint inverse-decoder MSE, Dice and tolerance metrics',
            ('Fig5F_inverse_metrics.pdf', 'Fig5F_MSE.pdf', 'Fig5F_Dice.pdf', 'Fig5F_tolerance_metrics.pdf'),
            (2, 3, 4, 5)),
        MainFigureTask('5G', 5, f'{INVERSE}/Evaluation_colonyarea_Fig5G.ipynb',
            'Colony-area regression recomputed with the deposited checkpoint',
            ('Fig5G_colony_area_regression.pdf',), (2, 4, 5, 9, 10, 12, 13, 14)),
    ]
}


def source_table_path(name):
    if '/' in name:
        return ROOT / 'source_data' / name
    family = INVERSE if name.startswith('Fig5') else 'sim_to_exp_diffusion'
    return ROOT / 'source_data/figure_exports' / family / 'vector based plots' / name


def select_tasks(figures=None, keys=None):
    if keys:
        return [TASKS[key] for key in dict.fromkeys(keys)]
    selected = set(figures or range(1, 6))
    return [task for task in TASKS.values() if task.figure in selected]


def generate_tasks(output_dir, *, figures=None, keys=None, timeout=7200):
    """Run isolated workers so the pipelines' models/utils modules cannot mix."""
    import json
    import subprocess
    import sys
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for task in select_tasks(figures, keys):
        subprocess.run([sys.executable, '-m', 'figure_generators.notebook_runner',
                        '--task', task.key, '--output', str(output_dir)],
                       cwd=ROOT, check=True, timeout=timeout)
        record = json.loads((output_dir / 'provenance' / f'{task.key}.json').read_text())
        paths.extend(Path(item['path']) for item in record['outputs'])
    return paths
