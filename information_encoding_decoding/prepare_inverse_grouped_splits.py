"""Create configuration-grouped split CSVs only; never copy or change images."""
import csv
import hashlib
import json
import os

from utils import config


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validation_configurations(order, fraction, seed):
    """One held-out ID per ten ranked IDs, stable across nested conditions.

    The final partial block also has one holdout. The small learning-curve
    points are multiples of ten; the 195-ID experimental pool has 20 holdouts.
    Hash ordering avoids always choosing a particular seed-count rank.
    """
    if fraction != 0.1:
        raise ValueError('This nested block split implements validation_fraction=0.1 only')
    validation_ids = set()
    for start in range(0, len(order), 10):
        block = order[start:start + 10]
        # Keep the same selection rule so previously prepared splits do not change.
        selected_id = min(
            block,
            key=lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest(),
        )
        validation_ids.add(selected_id)
    return validation_ids


def configuration_order(family):
    if family == 'synthetic':
        rows = read_csv(os.path.join(
            config.INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, 'configuration_order.csv'))
        rows.sort(key=lambda row: int(row['rank']))
        return [row['configuration_id'] for row in rows]
    rows = read_csv(os.path.join(config.INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
                                'experimental_configuration_inventory.csv'))
    ranked = [row for row in rows if row['selection_rank'] not in ('', 'nan')]
    ranked.sort(key=lambda row: float(row['selection_rank']))
    order = [row['configuration_id'] for row in ranked]
    included = set(order)
    order.extend(sorted(row['configuration_id'] for row in rows
                        if row['configuration_id'] not in included))
    return order


def split_rows(rows, family, condition, validation_ids):
    """Pair images from the existing manifest and assign whole configurations."""
    if len(rows) != config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise ValueError(f'{family}/{condition}: expected 90k manifest rows')

    if family == 'experimental':
        source_field = 'experimental_output_path'
    else:
        source_field = 'synthetic_output_path'

    output = {'train': [], 'validation': []}
    for row in rows:
        source = row[source_field]
        condition_root = os.path.dirname(os.path.dirname(row['target_output_path']))
        target = os.path.join(condition_root, config.INVERSE_ABLATION_SEED32_SUBFOLDER,
                              os.path.basename(row['target_output_path']))
        if os.path.basename(source) != os.path.basename(target):
            raise ValueError(f'Unpaired filenames: {source}, {target}')

        split = 'validation' if row['configuration_id'] in validation_ids else 'train'
        output[split].append({
            'family': family,
            'condition': condition,
            'split': split,
            'configuration_id': row['configuration_id'],
            'replicate_id': row['replicate_id'],
            'angle_deg': row['angle_deg'],
            'source_path': source,
            'target_path': target,
        })

    if not output['train'] or not output['validation']:
        raise ValueError(f'{family}/{condition}: training and validation must both be nonempty')
    return output


def configuration_summary(rows, validation_ids):
    """Count pairs and biological/generated replicates for each configuration."""
    groups = {}
    for row in rows:
        configuration_id = row['configuration_id']
        if configuration_id not in groups:
            groups[configuration_id] = {'pairs': 0, 'replicates': set()}
        groups[configuration_id]['pairs'] += 1
        groups[configuration_id]['replicates'].add(row['replicate_id'])

    summary = []
    for configuration_id, group in sorted(groups.items()):
        summary.append({
            'configuration_id': configuration_id,
            'split': 'validation' if configuration_id in validation_ids else 'train',
            'pair_count': group['pairs'],
            'replicate_count': len(group['replicates']),
        })
    return summary


def prepare_condition(family, condition, order, validation_ids):
    if family == 'experimental':
        source_root = config.INVERSE_ABLATION_PHASE2_OUTPUT_ROOT
    else:
        source_root = config.INVERSE_ABLATION_PHASE3_OUTPUT_ROOT
    rows = read_csv(os.path.join(source_root, condition, 'manifest.csv'))

    expected_ids = order if condition == 'NMAX' else order[:int(condition[1:])]
    actual_ids = {row['configuration_id'] for row in rows}
    if actual_ids != set(expected_ids):
        raise ValueError(f'{family}/{condition}: configurations do not match the nested selection')

    split = split_rows(rows, family, condition, validation_ids)
    counts = configuration_summary(rows, validation_ids)
    destination = os.path.join(config.INVERSE_ABLATION_PHASE4_OUTPUT_ROOT, family, condition)
    os.makedirs(destination, exist_ok=True)
    write_csv(os.path.join(destination, 'train.csv'), split['train'])
    write_csv(os.path.join(destination, 'validation.csv'), split['validation'])
    write_csv(os.path.join(destination, 'configuration_split.csv'), counts)

    validation_count = len(actual_ids & validation_ids)
    return {
        'family': family,
        'condition': condition,
        'total_pairs': len(rows),
        'total_configurations': len(actual_ids),
        'train_configurations': len(actual_ids) - validation_count,
        'validation_configurations': validation_count,
        'train_pairs': len(split['train']),
        'validation_pairs': len(split['validation']),
        'configuration_overlap': 0,
    }


def main():
    root = config.INVERSE_ABLATION_PHASE4_OUTPUT_ROOT
    conditions = config.INVERSE_ABLATION_PHASE4_CONDITIONS

    # Native targets must be prepared before creating the new training manifests.
    for family, condition in conditions:
        report = os.path.join(config.INVERSE_ABLATION_SEED32_REPORT_ROOT,
                              family, condition, 'summary.json')
        if not os.path.isfile(report):
            raise FileNotFoundError(f'Run prepare_inverse_seed_targets_32x32.py first: {report}')

    # Check all destinations first to avoid overwriting an existing split.
    if not config.INVERSE_ABLATION_PHASE4_OVERWRITE_EXISTING:
        for family, condition in conditions:
            destination = os.path.join(root, family, condition)
            if os.path.exists(destination):
                raise FileExistsError(f'Review existing split before overwriting: {destination}')

    orders, held_out = {}, {}
    families = sorted({family for family, _ in conditions})
    for family in families:
        orders[family] = configuration_order(family)
        held_out[family] = validation_configurations(
            orders[family],
            config.INVERSE_ABLATION_PHASE4_VALIDATION_FRACTION,
            config.INVERSE_ABLATION_SELECTION_SEED,
        )

    summaries = []
    for family, condition in conditions:
        summary = prepare_condition(family, condition, orders[family], held_out[family])
        summaries.append(summary)
        print(summary, flush=True)

    write_csv(os.path.join(root, 'split_summary.csv'), summaries)
    with open(os.path.join(root, 'split_settings.json'), 'w') as handle:
        json.dump(dict(seed=config.INVERSE_ABLATION_SELECTION_SEED,
                       validation_fraction=config.INVERSE_ABLATION_PHASE4_VALIDATION_FRACTION,
                       method='one hash-selected holdout per ranked block of ten',
                       budget='90000 total train plus validation pairs',
                       image_contents_checked=False), handle, indent=2)


if __name__ == '__main__':
    main()
