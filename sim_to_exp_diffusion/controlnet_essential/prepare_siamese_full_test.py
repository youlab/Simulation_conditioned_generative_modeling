"""Prepare the 132-image Siamese cohort with the EXISTING preprocessing only.

No thresholds, crops, contrast settings or preprocessing implementations are
changed. Each named function below is called with its existing default arguments.
Prepared PNGs are read without preprocessing again in the Siamese notebook.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MANIFEST = BASE.parents[1] / 'source_data/image_metrics/test_manifest.csv'
DEFAULT_OUTPUT_DIR = BASE / 'siamese_full_test_132'
# None means the original notebook's Image.open(...).convert('L') operation.
PREPROCESSING = {
    'experiment_grayscale': ('experiment_path', 'preprocess_experimental_backgroundwhite'),
    'experiment_binary_white': ('experiment_path', 'preprocess_experimental_backgroundwhite_blackandwhite'),
    'experiment_binary_black': ('experiment_path', 'preprocess_experimental_backgroundblack_blackandwhite'),
    'simulation_grayscale': ('simulation_path', 'preprocess_simulation_graybackground'),
    'simulation_binary_black': ('simulation_path', 'preprocess_simulation'),
    'prediction_grayscale': ('simulation_conditioned_path', None),
    'prediction_binary_black': ('simulation_conditioned_path', 'preprocess_experimental_backgroundblack_blackandwhite_nocropping'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configuration_id(path):
    """196_1 -> 196; Fixed_10_2 -> Fixed_10 (never just 'Fixed')."""
    stem = Path(path).stem
    head, separator, replicate = stem.rpartition('_')
    if not separator or not replicate.isdigit():
        raise ValueError(f'Expected configuration_replicate filename: {path}')
    return head


def generate_pairs(paths, seed=42):
    """Original balanced pair sampling, extended separately within each cohort.

    Preserving input order and seed preserves the original 96-image selections.
    Singletons have no replicate pair and are excluded from these histograms;
    they remain in the separate per-image cross-source evaluation.
    """
    primary = [str(p) for p in paths if not Path(p).stem.startswith('Fixed_')]
    fixed = [str(p) for p in paths if Path(p).stem.startswith('Fixed_')]
    all_pairs, all_labels = [], []
    for cohort in (primary, fixed):
        if not cohort:
            continue
        rng = random.Random(seed)
        groups = defaultdict(list)
        for path in cohort:
            groups[configuration_id(path)].append(path)
        positive, valid = [], []
        for key, files in groups.items():
            if len(files) == 2:
                positive.append(tuple(files))
                valid.append(key)
            elif len(files) != 1:
                raise ValueError(f'Expected one or two replicates for {key}, found {len(files)}')
        if positive and len(valid) < 2:
            raise ValueError('At least two replicated configurations are needed for negative pairs')
        negative = set()
        while len(negative) < len(positive):
            first, second = rng.sample(valid, 2)
            pair = tuple(sorted((rng.choice(groups[first]), rng.choice(groups[second]))))
            negative.add(pair)
        all_pairs.extend(positive + sorted(negative))
        all_labels.extend([1] * len(positive) + [0] * len(negative))
    return all_pairs, all_labels


def ordered_paths(manifest, prepared_column, source_column):
    """Retain primary os.listdir ordering used by the historical notebook.

    This also preserves the primary reference-pool order used for S21's seeded
    within-configuration reference selection. Fixed items follow the manifest.
    """
    primary = {Path(row[source_column]).name: row for row in manifest if row['subset'] == 'primary'}
    folders = {str(Path(row[source_column]).parent) for row in manifest if row['subset'] == 'primary'}
    if len(folders) != 1:
        raise ValueError(f'Expected one original primary folder for {source_column}')
    folder = folders.pop()
    rows = [primary[name] for name in os.listdir(folder) if name in primary]
    if len(rows) != len(primary):
        raise ValueError('Primary source files no longer match the prepared manifest')
    rows.extend(row for row in manifest if row['subset'] == 'fixed')
    return [str(row[prepared_column]) for row in rows]


def prepare_full_test(source_manifest=DEFAULT_SOURCE_MANIFEST, output_dir=DEFAULT_OUTPUT_DIR, *, rebuild=False):
    from PIL import Image
    import numpy as np
    from cldm import preprocess

    source_manifest, output_dir = Path(source_manifest), Path(output_dir)
    source_rows = read_csv(source_manifest)
    ids = [row['image_id'] for row in source_rows]
    counts = {s: sum(row['subset'] == s for row in source_rows) for s in ('primary', 'fixed')}
    if len(ids) != len(set(ids)) or counts != {'primary': 96, 'fixed': 36}:
        raise ValueError(f'Expected 96 primary + 36 fixed unique image IDs, got {counts}')
    source_columns = sorted({column for column, _ in PREPROCESSING.values()})
    source_paths = sorted({row[column] for row in source_rows for column in source_columns})
    source_hashes = {path: sha256(path) for path in source_paths}
    preprocess_path = Path(preprocess.__file__)
    preprocess_hash = sha256(preprocess_path)
    signature = hashlib.sha256(json.dumps({
        'manifest': source_rows, 'input_sha256': source_hashes,
        'preprocess_sha256': preprocess_hash, 'preprocessing': PREPROCESSING,
        'grayscale_conversion': "PIL.Image.convert('L'), as in the existing notebook",
    }, sort_keys=True).encode()).hexdigest()
    settings_path = output_dir / 'preparation_settings.json'
    if settings_path.exists() and not rebuild:
        settings = json.loads(settings_path.read_text())
        if settings['input_signature'] != signature:
            raise ValueError('Prepared inputs differ from current sources; use rebuild=True or a new directory')
        for path, digest in settings['prepared_image_sha256'].items():
            if not Path(path).is_file() or sha256(path) != digest:
                raise ValueError(f'Prepared image missing or modified: {path}')
        print(f'Using verified prepared folders: {output_dir}', flush=True)
        return read_csv(output_dir / 'prepared_manifest.csv'), settings
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in PREPROCESSING:
        (output_dir / 'preprocessed' / variant).mkdir(parents=True, exist_ok=True)
    prepared, output_hashes, processed_cache = [], {}, {}
    for index, row in enumerate(source_rows):
        result = dict(row)
        for variant, (source_column, function_name) in PREPROCESSING.items():
            source = row[source_column]
            key = (source, function_name)
            if key not in processed_cache:
                if function_name is None:
                    with Image.open(source) as image:
                        array = np.array(image.convert('L'))
                else:
                    # Existing function, existing defaults; no new preprocessing.
                    array = np.array(Image.fromarray(getattr(preprocess, function_name)(source)).convert('L'))
                if array.shape != (256, 256) or array.dtype != np.uint8:
                    raise ValueError(f'Unexpected prepared image format for {source}: {array.shape}, {array.dtype}')
                processed_cache[key] = array
            array = processed_cache[key]
            path = (output_dir / 'preprocessed' / variant / f"{row['image_id']}.png").resolve()
            Image.fromarray(array).save(path)
            with Image.open(path) as image:
                if not np.array_equal(np.array(image), array):
                    raise AssertionError(f'PNG save changed pixels: {path}')
            result[variant + '_path'] = str(path)
            output_hashes[str(path)] = sha256(path)
        prepared.append(result)
        if (index + 1) % 24 == 0 or index + 1 == len(source_rows):
            print(f'Prepared {index + 1}/{len(source_rows)} image triplets', flush=True)
    if sha256(preprocess_path) != preprocess_hash:
        raise AssertionError('Existing preprocessing source changed during preparation')
    groups = defaultdict(list)
    for row in source_rows:
        groups[row['configuration_id']].append(row['image_id'])
    settings = {
        'created_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'source_manifest': str(source_manifest.resolve()), 'n_images': 132,
        'n_configurations': len(groups), 'subset_counts': counts,
        'replicated_configurations': sum(len(v) == 2 for v in groups.values()),
        'singleton_images': [v[0] for v in groups.values() if len(v) == 1],
        'preprocessing': PREPROCESSING, 'preprocess_module': str(preprocess_path),
        'preprocess_source_sha256_before': preprocess_hash,
        'preprocess_source_sha256_after': sha256(preprocess_path),
        'all_existing_function_defaults_preserved': True,
        'lossless_png_pixel_checks': len(output_hashes),
        'input_signature': signature, 'source_image_sha256': source_hashes,
        'prepared_image_sha256': output_hashes,
        'note': 'Prepared PNGs must not be passed through preprocessing again; the original 224x224 Siamese transform remains in the notebook.',
    }
    write_csv(output_dir / 'prepared_manifest.csv', prepared)
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    return prepared, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    _, settings = prepare_full_test(args.source_manifest, args.output_dir, rebuild=args.rebuild)
    print(json.dumps({k: settings[k] for k in ('n_images', 'n_configurations', 'replicated_configurations', 'singleton_images')}, indent=2))


if __name__ == '__main__':
    main()
