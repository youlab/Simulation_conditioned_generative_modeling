"""Recompute main-panel measurements using the existing notebook functions."""
from pathlib import Path


def image_metrics(config, directory, namespace):
    from image_metric_evaluation import (build_test_manifest, load_test_images,
                                         evaluate_metrics, metric_scores)
    manifest = build_test_manifest(config, include_seed_conditioned=False)
    images = load_test_images(manifest)
    result = evaluate_metrics(manifest, images, directory, recompute=True)
    for comparison, suffix in [('exp_sim', 'sim'), ('exp_pred', 'genAI'), ('sim_pred', 'simgenAI')]:
        for metric, prefix in [('lpips', 'lpips'), ('ssim', 'ssim_similarity'), ('orb', 'orb_similarity')]:
            namespace[f'{prefix}_{suffix}'] = metric_scores(result, comparison, metric)
    namespace.update(manifest=manifest, evaluation=result)


def inverse_metrics(config, directory):
    """Measure the deposited predictions; preserve the published 107/132 cohorts."""
    import cv2
    import torch
    from utils.metrics import mse_loss, dice_loss, point_matching
    from utils.preprocess import preprocess_seed
    from utils import evaluation_results
    from .public_data import sha256

    evaluation_results.RESULTS_DIR = Path(directory)
    jobs = [
        ('Simulation model\nSimulation→Seed', config.OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_SAVED, 107, True),
        ('Simulation model\nExperiment→Seed', config.OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET_SAVED, 132, False),
        ('Diffusion model\nExperiment→Seed', config.OUTPUT_DIR_EXPTOSEED_COLOR_SAVED, 132, False),
    ]
    for comparison, prediction_dir, count, simulation in jobs:
        files = sorted(Path(prediction_dir).glob('*.png'))
        if len(files) != count:
            raise ValueError(f'{comparison}: expected {count} saved predictions, found {len(files)}')
        targets, mse, dice, precision, recall, f1 = [], [], [], [], [], []
        for path in files:
            if simulation:
                name = path.name.replace('Output', 'Input', 1)
                target = Path(config.SEED_FOLDER_TEST_INFOENCODING_MORESEEDS) / name
                if not target.is_file():
                    target = Path(config.SEED_FOLDER_SEEDTOSIM_TEST) / name
                crop = {'left_crop': 2, 'right_crop': 3}
            else:
                target = Path(config.SEED_FOLDER_EXPTOSIM_TEST_FIXED_NONFIXED_32X32) / path.name
                crop = {}
            if not target.is_file():
                raise FileNotFoundError(target)
            prediction = torch.from_numpy(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            truth = torch.from_numpy(preprocess_seed(str(target), img_length=32, img_width=32, **crop))
            p, r, f = point_matching(prediction, truth, tolerance_radius=4, algorithm='hungarian')
            mse.append(mse_loss(prediction, truth).item())
            dice.append(dice_loss(prediction, truth).item())
            precision.append(p); recall.append(r); f1.append(f)
            targets.append(target.name)
        provenance = None
        if comparison.startswith('Diffusion'):
            provenance = {'checkpoint_selection': 'last.ckpt',
                          'checkpoint_sha256': sha256(config.CKPT_PATH_SIMULATION_CONDITIONED_LR5E4_LAST),
                          'prediction_source': str(prediction_dir),
                          'evaluation': 'Existing deposited binary predictions; original notebook metrics',
                          'threshold_used_for_deposited_predictions': 0.6,
                          'tolerance_radius': 4, 'matching': 'hungarian'}
        summary = evaluation_results.save_seed_metric_results(
            comparison, mse, dice, precision, recall, f1,
            [p.name for p in files], targets, provenance=provenance)
        print(comparison.replace('\n', ': '), summary, flush=True)
