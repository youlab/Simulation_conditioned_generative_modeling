"""Prepare nested synthetic learning curves with three predictions/configuration.

Run after reviewing settings in utils/config.py. Sources remain unchanged.
Every condition contains 90,000 image/target pairs. Synthetic predictions are
already 256 x 256: rotate directly without the experimental crop.
With REGENERATE_SYNTHETIC_ONLY enabled in config, existing manifests are reused
and only synthetic colony images are overwritten; seeds and splits are untouched.
Under Slurm, SLURM_ARRAY_TASK_ID selects one condition. Running directly
without that variable processes all conditions sequentially.
"""

import csv
import json
import os
import random
import re
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from prepare_inverse_experimental_data import (
    crop_and_resize_seed,
    format_angle,
    output_filename,
    rotate_experimental,
    rotate_seed_image,
)
from utils.config import (
    INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
    INVERSE_ABLATION_PHASE3_OUTPUT_ROOT,
    INVERSE_ABLATION_PHASE3_OVERWRITE_EXISTING,
    INVERSE_ABLATION_PHASE3_REGENERATE_SYNTHETIC_ONLY,
    INVERSE_ABLATION_PHASE3_REPLICATES,
    INVERSE_ABLATION_PHASE3_SELECTION_SIZES,
    INVERSE_ABLATION_SELECTION_SEED,
    SEED_FILES_TEST_INFOENCODING_MORESEEDS,
    SEED_FOLDER_TEST_INFOENCODING_MORESEEDS,
    SPECIFIC_FILES_EXP_DIFFUSION_MORE,
    SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
)


def inventory_predictions():
    """Match Output_<configuration>_<replicate>.png to Input_<configuration>.png."""
    groups = defaultdict(dict)
    for filename in SPECIFIC_FILES_EXP_DIFFUSION_MORE:
        if not filename.lower().endswith('.png'):
            continue
        match = re.fullmatch(r'Output_(.+)_(\d+)\.png', filename)
        if match is None:
            raise ValueError(f'Unexpected synthetic filename: {filename}')
        configuration_id, replicate = match.group(1), int(match.group(2))
        if replicate in groups[configuration_id]:
            raise ValueError(f'Duplicate prediction: {configuration_id}, {replicate}')
        groups[configuration_id][replicate] = filename

    seed_names = set(SEED_FILES_TEST_INFOENCODING_MORESEEDS)
    expected = set(INVERSE_ABLATION_PHASE3_REPLICATES)
    for configuration_id, samples in groups.items():
        if set(samples) != expected:
            raise ValueError(f'{configuration_id}: expected replicates {sorted(expected)}, found {sorted(samples)}')
        seed_name = f'Input_{configuration_id}.png'
        if seed_name not in seed_names:
            raise FileNotFoundError(f'Missing original seed: {seed_name}')
    if len(groups) != max(INVERSE_ABLATION_PHASE3_SELECTION_SIZES):
        raise ValueError(
            f'Expected the existing 30k configuration pool, found {len(groups)}. '
            'Check the configured prediction folder before proceeding.'
        )
    return groups


def rotation_schedule(configuration_ids, predictions):
    rows = []
    for rank, configuration_id in enumerate(configuration_ids, start=1):
        for replicate in INVERSE_ABLATION_PHASE3_REPLICATES:
            rows.append({
                'configuration_rank': rank,
                'configuration_id': configuration_id,
                'replicate_id': replicate,
                'source_filename': predictions[configuration_id][replicate],
                'original_seed_filename': f'Input_{configuration_id}.png',
            })
    base, remainder = divmod(INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES, len(rows))
    if base < 1:
        raise ValueError('The total size must allow all three samples per configuration')
    for index, row in enumerate(rows):
        row['rotation_count'] = base + int(index < remainder)
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_image(path, image, overwrite=False):
    if os.path.exists(path) and not (overwrite or INVERSE_ABLATION_PHASE3_OVERWRITE_EXISTING):
        return
    if not cv2.imwrite(path, image):
        raise OSError(f'Failed to write {path}')


def load_synthetic(path):
    """Read the final diffusion prediction without cropping or resizing."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != (256, 256):
        raise ValueError(f'Expected a 256 x 256 prediction: {path}, {image.shape}')
    return image


def regenerate_synthetic_rotations(condition):
    """Replace only colony images, retaining the exact existing pair schedule."""
    condition_root = os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, condition)
    summary_path = os.path.join(condition_root, 'synthetic_rotation_summary.json')
    if os.path.exists(summary_path):
        os.remove(summary_path)
    previous_source = None
    count = 0
    with open(os.path.join(condition_root, 'manifest.csv'), newline='') as handle:
        for row in tqdm(csv.DictReader(handle), desc=f'{condition}: synthetic only'):
            if row['source_filename'] != previous_source:
                source_image = load_synthetic(os.path.join(
                    SPECIFIC_FOLDER_EXP_DIFFUSION_MORE, row['source_filename']
                ))
                previous_source = row['source_filename']
            write_image(
                row['synthetic_output_path'],
                rotate_experimental(source_image, float(row['angle_deg'])),
                overwrite=True,
            )
            count += 1
    if count != INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise ValueError(f'{condition}: expected {INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES} manifest rows, found {count}')
    summary = {
        'condition': condition,
        'synthetic_files_regenerated': count,
        'preprocessing': 'rotate original 256 x 256 diffusion prediction; no crop or resize',
        'seed_targets_and_manifests_unchanged': True,
    }
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    return summary


def materialize_condition(condition, schedule):
    condition_root = os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, condition)
    source_folder = os.path.join(condition_root, 'synthetic')
    target_folder = os.path.join(condition_root, 'seed')
    os.makedirs(source_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)
    write_csv(os.path.join(condition_root, 'plate_rotation_schedule.csv'), schedule)
    fields = [
        'condition', 'dataset_index', 'configuration_id', 'replicate_id',
        'source_filename', 'original_seed_filename', 'angle_deg',
        'rotation_count_for_sample', 'synthetic_output_path', 'target_output_path',
    ]
    expected_names = set()
    current_configuration = None
    resized_seed = None
    dataset_index = 0
    manifest_path = os.path.join(condition_root, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(schedule, desc=condition):
            configuration_id = row['configuration_id']
            if configuration_id != current_configuration:
                seed_path = os.path.join(
                    SEED_FOLDER_TEST_INFOENCODING_MORESEEDS,
                    row['original_seed_filename'],
                )
                seed_image = cv2.imread(seed_path, cv2.IMREAD_COLOR)
                if seed_image is None:
                    raise FileNotFoundError(seed_path)
                resized_seed = crop_and_resize_seed(seed_image)
                current_configuration = configuration_id
            source_path = os.path.join(
                SPECIFIC_FOLDER_EXP_DIFFUSION_MORE, row['source_filename']
            )
            source_image = load_synthetic(source_path)
            angles = np.linspace(0, 360, row['rotation_count'], endpoint=False)
            for angle in angles:
                filename = output_filename(row['source_filename'], angle)
                if filename in expected_names:
                    raise ValueError(f'Duplicate output: {filename}')
                expected_names.add(filename)
                synthetic_output_path = os.path.join(source_folder, filename)
                target_output_path = os.path.join(target_folder, filename)
                write_image(synthetic_output_path, rotate_experimental(source_image, angle))
                write_image(target_output_path, rotate_seed_image(resized_seed, angle))
                writer.writerow({
                    'condition': condition,
                    'dataset_index': dataset_index,
                    'configuration_id': configuration_id,
                    'replicate_id': row['replicate_id'],
                    'source_filename': row['source_filename'],
                    'original_seed_filename': row['original_seed_filename'],
                    'angle_deg': format_angle(angle),
                    'rotation_count_for_sample': row['rotation_count'],
                    'synthetic_output_path': synthetic_output_path,
                    'target_output_path': target_output_path,
                })
                dataset_index += 1
    sources = set(os.listdir(source_folder))
    targets = set(os.listdir(target_folder))
    complete = (
        dataset_index == INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES
        and sources == targets == expected_names
    )
    return {
        'condition': condition,
        'unique_configurations': len({row['configuration_id'] for row in schedule}),
        'generated_samples': len(schedule),
        'minimum_rotations_per_sample': min(row['rotation_count'] for row in schedule),
        'maximum_rotations_per_sample': max(row['rotation_count'] for row in schedule),
        'manifest_rows': dataset_index,
        'synthetic_files': len(sources),
        'target_files': len(targets),
        'complete': complete,
    }


def selected_sizes(sizes):
    """Map the zero-based Slurm array index to exactly one configured size."""
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is None:
        return sizes
    try:
        index = int(task_id)
    except ValueError as error:
        raise ValueError(f'Invalid SLURM_ARRAY_TASK_ID: {task_id}') from error
    if not 0 <= index < len(sizes):
        raise ValueError(f'Array task {index} is outside 0..{len(sizes) - 1}')
    return (sizes[index],)


def main():
    sizes = INVERSE_ABLATION_PHASE3_SELECTION_SIZES
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError('Configuration sizes must be distinct positive integers')
    run_sizes = selected_sizes(sizes)
    array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    print(f'Array task: {array_task_id}; conditions: {run_sizes}', flush=True)
    if INVERSE_ABLATION_PHASE3_REGENERATE_SYNTHETIC_ONLY:
        for size in run_sizes:
            print(json.dumps(regenerate_synthetic_rotations(f'N{size}'), indent=2))
        return
    predictions = inventory_predictions()
    configuration_ids = sorted(predictions)
    random.Random(INVERSE_ABLATION_SELECTION_SEED).shuffle(configuration_ids)
    os.makedirs(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, exist_ok=True)
    # All jobs compute the identical shuffled order; only task 0 writes the
    # shared copy. Every condition also records ranks in its own schedule.
    if array_task_id is None or int(array_task_id) == 0:
        write_csv(
            os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, 'configuration_order.csv'),
            [{'rank': rank, 'configuration_id': value}
             for rank, value in enumerate(configuration_ids, start=1)],
        )
    audit_rows = []
    for size in run_sizes:
        schedule = rotation_schedule(configuration_ids[:size], predictions)
        audit = materialize_condition(f'N{size}', schedule)
        audit_rows.append(audit)
        write_csv(
            os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, f'N{size}', 'phase3_dataset_audit.csv'),
            [audit],
        )
        if not audit['complete']:
            raise RuntimeError(f'Dataset audit failed: {audit}')
    summary = {
        'prediction_source': SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
        'original_seed_source': SEED_FOLDER_TEST_INFOENCODING_MORESEEDS,
        'selection_seed': INVERSE_ABLATION_SELECTION_SEED,
        'selection_mode': 'nested deterministic shuffle of the existing 30k pool',
        'samples_per_configuration': len(INVERSE_ABLATION_PHASE3_REPLICATES),
        'total_examples_per_condition': INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
        'preprocessing': 'synthetic predictions: direct rotation, no crop; legacy seed crop left 3/right 2 before enlargement',
        'outputs_are_preprocessed': True,
        'conditions': audit_rows,
    }
    for size, audit in zip(run_sizes, audit_rows):
        condition_summary = dict(summary, conditions=[audit])
        summary_path = os.path.join(
            INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, f'N{size}', 'phase3_summary.json'
        )
        with open(summary_path, 'w') as handle:
            json.dump(condition_summary, handle, indent=2)
    # A sequential invocation owns all conditions and may write combined
    # reports. Array workers only write their own condition reports.
    if array_task_id is None:
        write_csv(
            os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, 'phase3_dataset_audit.csv'),
            audit_rows,
        )
        with open(os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, 'phase3_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
