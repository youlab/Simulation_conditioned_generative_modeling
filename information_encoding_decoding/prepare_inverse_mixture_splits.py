"""Prepare fixed-90k experimental/synthetic mixtures without copying images."""
import json
import os

from prepare_inverse_grouped_splits import read_csv, write_csv
from utils import config


def mixture_name(experimental_percentage):
    return f'E{experimental_percentage:03d}_S{100 - experimental_percentage:03d}'


def load_pools():
    """Reuse full-data Phase 4 manifests, including their held-out configurations."""
    pools = {}
    for family, condition in (('experimental', 'NMAX'), ('synthetic', 'N30000')):
        pools[family] = {}
        for split in ('train', 'validation'):
            path = os.path.join(config.INVERSE_ABLATION_PHASE4_OUTPUT_ROOT,
                                family, condition, f'{split}.csv')
            pools[family][split] = read_csv(path)
    return pools


def select_experimental(rows, count):
    """Thin the rotation-ordered manifest evenly, keeping coverage of all plates.

    The NMAX manifest has consecutive, evenly spaced rotations for each plate.
    Selecting evenly spaced row indices spreads the retained angles over each
    plate's rotation range instead of taking only its first few rotations.
    """
    if count == 0:
        return []
    return [rows[index * len(rows) // count] for index in range(count)]


def select_synthetic(rows, configuration_count):
    """Take a nested prefix of configurations, retaining all three predictions."""
    if configuration_count == 0:
        return []
    groups = {}
    for row in rows:
        groups.setdefault(row['configuration_id'], []).append(row)

    selected_ids = list(groups)[:configuration_count]
    selected = []
    for configuration_id in selected_ids:
        predictions = groups[configuration_id]
        if {row['replicate_id'] for row in predictions} != {'1', '2', '3'} or len(predictions) != 3:
            raise ValueError(f'{configuration_id}: expected exactly three predictions')
        selected.extend(predictions)
    if len(selected) != configuration_count * 3:
        raise ValueError('Not enough synthetic configurations in the source split')
    return selected


def build_mixture(pools, experimental_percentage):
    total = config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES
    experimental_count = total * experimental_percentage // 100
    synthetic_count = total - experimental_count

    # Preserve the experimental pool's existing train/validation proportion.
    experimental_train_pool = pools['experimental']['train']
    experimental_validation_pool = pools['experimental']['validation']
    pool_size = len(experimental_train_pool) + len(experimental_validation_pool)
    experimental_train_count = (
        experimental_count * len(experimental_train_pool) + pool_size // 2
    ) // pool_size
    experimental_validation_count = experimental_count - experimental_train_count

    # These five mixture points give whole triplets and a 90/10 configuration split.
    synthetic_configurations = synthetic_count // 3
    synthetic_validation_configurations = synthetic_configurations // 10
    synthetic_train_configurations = synthetic_configurations - synthetic_validation_configurations

    selected = {
        'train': select_experimental(experimental_train_pool, experimental_train_count)
                 + select_synthetic(pools['synthetic']['train'], synthetic_train_configurations),
        'validation': select_experimental(experimental_validation_pool, experimental_validation_count)
                      + select_synthetic(pools['synthetic']['validation'], synthetic_validation_configurations),
    }

    # Namespace IDs so experimental and synthetic configurations cannot collide.
    mixture = mixture_name(experimental_percentage)
    output = {}
    for split, rows in selected.items():
        output[split] = []
        for row in rows:
            output[split].append({
                **row,
                'condition': mixture,
                'configuration_id': f"{row['family']}:{row['configuration_id']}",
                'original_configuration_id': row['configuration_id'],
                'source_condition': row['condition'],
            })

    train_ids = {row['configuration_id'] for row in output['train']}
    validation_ids = {row['configuration_id'] for row in output['validation']}
    if train_ids & validation_ids:
        raise ValueError(f'{mixture}: configuration leakage')
    if len(output['train']) + len(output['validation']) != total:
        raise ValueError(f'{mixture}: combined train/validation budget must be {total}')
    return output


def mixture_summary(mixture, experimental_percentage):
    summary = {
        'condition': mixture_name(experimental_percentage),
        'experimental_percentage': experimental_percentage,
    }
    for split, rows in mixture.items():
        summary[f'{split}_pairs'] = len(rows)
        for family in ('experimental', 'synthetic'):
            family_rows = [row for row in rows if row['family'] == family]
            summary[f'{split}_{family}_pairs'] = len(family_rows)
            summary[f'{split}_{family}_configurations'] = len({
                row['configuration_id'] for row in family_rows
            })
    summary['total_pairs'] = len(mixture['train']) + len(mixture['validation'])
    summary['experimental_pairs'] = (summary['train_experimental_pairs']
                                     + summary['validation_experimental_pairs'])
    summary['synthetic_pairs'] = summary['total_pairs'] - summary['experimental_pairs']

    if experimental_percentage == 0:
        summary['training_run'] = os.path.join(
            config.INVERSE_ABLATION_PHASE4_LOG_ROOT,
            f'synthetic_N30000_seed{config.INVERSE_ABLATION_SELECTION_SEED}')
    elif experimental_percentage == 100:
        summary['training_run'] = os.path.join(
            config.INVERSE_ABLATION_PHASE4_LOG_ROOT,
            f'experimental_NMAX_seed{config.INVERSE_ABLATION_SELECTION_SEED}')
    else:
        summary['training_run'] = os.path.join(
            config.INVERSE_ABLATION_PHASE5_LOG_ROOT,
            f"{summary['condition']}_seed{config.INVERSE_ABLATION_SELECTION_SEED}")
    return summary


def main():
    root = config.INVERSE_ABLATION_PHASE5_OUTPUT_ROOT
    percentages = config.INVERSE_ABLATION_PHASE5_EXPERIMENTAL_PERCENTAGES
    if not config.INVERSE_ABLATION_PHASE5_OVERWRITE_EXISTING:
        for percentage in percentages:
            destination = os.path.join(root, mixture_name(percentage))
            if os.path.exists(destination):
                raise FileExistsError(f'Review existing mixture before overwriting: {destination}')

    pools = load_pools()
    summaries = []
    for percentage in percentages:
        mixture = build_mixture(pools, percentage)
        destination = os.path.join(root, mixture_name(percentage))
        os.makedirs(destination, exist_ok=True)
        write_csv(os.path.join(destination, 'train.csv'), mixture['train'])
        write_csv(os.path.join(destination, 'validation.csv'), mixture['validation'])
        summary = mixture_summary(mixture, percentage)
        summaries.append(summary)
        print(summary, flush=True)
    write_csv(os.path.join(root, 'mixture_summary.csv'), summaries)
    with open(os.path.join(root, 'mixture_settings.json'), 'w') as handle:
        json.dump({
            'percentages': percentages,
            'total_train_plus_validation_pairs': config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
            'experimental_pool': 'Phase 4 experimental NMAX',
            'synthetic_pool': 'Phase 4 synthetic N30000',
            'experimental_sampling': 'evenly spaced existing rotation indices',
            'synthetic_sampling': 'nested configurations, three predictions each',
            'validation': 'inherited Phase 4 configuration assignments',
            'endpoint_training': 'reuse existing Phase 4 runs',
        }, handle, indent=2)


if __name__ == '__main__':
    main()
