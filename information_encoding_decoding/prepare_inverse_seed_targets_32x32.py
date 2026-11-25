"""Rebuild only seed targets at 32x32, reusing existing images and rotation angles."""
import csv
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from SeedplusSim_rotation_3reps import rotate_seed_coordinates
from utils import config
from utils.preprocess import preprocess_seed


def prepare_condition(family, condition):
    if family == 'experimental':
        root = config.INVERSE_ABLATION_PHASE2_OUTPUT_ROOT
        original_folder = config.SEED_FOLDER_TRAIN_NONAUG
    else:
        root = config.INVERSE_ABLATION_PHASE3_OUTPUT_ROOT
        original_folder = config.SEED_FOLDER_TEST_INFOENCODING_MORESEEDS

    condition_root = os.path.join(root, condition)
    target_folder = os.path.join(condition_root, config.INVERSE_ABLATION_SEED32_SUBFOLDER)
    report_folder = os.path.join(config.INVERSE_ABLATION_SEED32_REPORT_ROOT, family, condition)
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)
    summary_path = os.path.join(report_folder, 'summary.json')
    # Remove only a previous completion marker before rewriting these targets.
    if os.path.isfile(summary_path):
        os.remove(summary_path)

    summary = {'family': family, 'condition': condition, 'samples': 0,
               'count_changed_samples': 0, 'zero_angle_samples': 0,
               'target_folder': target_folder}
    audit_fields = ['configuration_id', 'replicate_id', 'angle_deg',
                    'original_seed_path', 'target_path', 'original_seed_count', 'target_seed_count']
    previous_seed_path = None
    with open(os.path.join(condition_root, 'manifest.csv'), newline='') as source_handle, \
         open(os.path.join(report_folder, 'seed_count_changes.csv'), 'w', newline='') as audit_handle:
        rows = csv.DictReader(source_handle)
        audit = csv.DictWriter(audit_handle, fieldnames=audit_fields)
        audit.writeheader()
        for row in tqdm(rows, total=config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
                        desc=f'{family}/{condition}'):
            original_name = row['source_filename'] if family == 'experimental' else row['original_seed_filename']
            original_path = os.path.join(original_folder, original_name)
            if original_path != previous_seed_path:
                original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                if original is None or original.shape != (32, 37):
                    raise ValueError(f'Expected an original 37-wide x 32-high seed: {original_path}')
                seed = preprocess_seed(
                    original_path,
                    left_crop=config.INVERSE_ABLATION_PHASE2_SEED_LEFT_CROP,
                    right_crop=config.INVERSE_ABLATION_PHASE2_SEED_RIGHT_CROP,
                    img_length=32, img_width=32,
                )
                seed_count = int(np.count_nonzero(seed))
                previous_seed_path = original_path

            angle = float(row['angle_deg'])
            target = rotate_seed_coordinates(seed, angle)
            target_path = os.path.join(target_folder, os.path.basename(row['target_output_path']))
            if not cv2.imwrite(target_path, target):
                raise OSError(f'Could not save seed target: {target_path}')

            target_count = int(np.count_nonzero(target))
            if angle == 0:
                if not np.array_equal(seed, target):
                    raise ValueError(f'Zero-angle target differs from original cropped seed: {target_path}')
                summary['zero_angle_samples'] += 1
            if target_count != seed_count:
                summary['count_changed_samples'] += 1
                audit.writerow(dict(
                    configuration_id=row['configuration_id'], replicate_id=row['replicate_id'],
                    angle_deg=row['angle_deg'], original_seed_path=original_path,
                    target_path=target_path, original_seed_count=seed_count,
                    target_seed_count=target_count,
                ))
            summary['samples'] += 1

    if summary['samples'] != config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise ValueError(f'{family}/{condition}: expected 90000 targets')
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(summary, flush=True)


def main():
    conditions = config.INVERSE_ABLATION_PHASE4_CONDITIONS
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        conditions = (conditions[int(task_id)],)
    for family, condition in conditions:
        prepare_condition(family, condition)


if __name__ == '__main__':
    main()
