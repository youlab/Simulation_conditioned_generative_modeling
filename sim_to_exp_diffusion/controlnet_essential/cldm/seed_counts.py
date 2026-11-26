"""Native seed-pixel counts and the manuscript's >26 display category."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SEED_COUNT_LIMIT = 26
SEED_OVERFLOW_POSITION = SEED_COUNT_LIMIT + 1


def count_seed_pixels(path):
    """Each white native-grid pixel is one seed, including touching pixels."""
    with Image.open(path) as image:
        raw = np.asarray(image.convert('L'))
    if raw.shape not in ((32, 32), (32, 37)):
        raise ValueError(f'Expected native seed image, got {raw.shape}: {path}')
    if not np.isin(raw, [0, 255]).all():
        raise ValueError(f'Expected binary 0/255 seed pixels: {path}')
    native = raw[:, 3:-2] if raw.shape == (32, 37) else raw
    count = int(np.count_nonzero(native > 127))
    if count != int(np.count_nonzero(raw > 127)):
        raise ValueError(f'Seed pixels found in the padding: {path}')
    return count


def load_seed_counts(recorded_path, fixed_seed_folder):
    """Copy primary metadata and recount fixed configurations from native pixels.

    Historical combined JSONs used components for fixed images. Recounting here
    also handles those JSONs without modifying them or any exported source table.
    Returned values are actual counts (37/57), never overflow positions (27).
    """
    with open(recorded_path) as file:
        counts = json.load(file)
    fixed_counts = {}
    for name in counts:
        stem = Path(name).stem
        if stem.startswith('Fixed_'):
            configuration = stem.split('_')[1]
            if configuration not in fixed_counts:
                fixed_counts[configuration] = count_seed_pixels(
                    Path(fixed_seed_folder) / f'Input_Fixed_{configuration}.png'
                )
            counts[name] = fixed_counts[configuration]
    return counts


def seed_count_positions(counts):
    """Keep 1..26 numeric; encode every >26 count at the display position 27."""
    return np.minimum(np.asarray(counts), SEED_OVERFLOW_POSITION)


def seed_count_label(position):
    return '>26' if position == SEED_OVERFLOW_POSITION else str(int(position))


def add_seed_count_plot_columns(frame):
    """Add explicit display columns without overwriting actual seed_count."""
    frame['seed_count_plot'] = seed_count_positions(frame['seed_count'])
    frame['seed_count_group'] = frame['seed_count_plot'].map(seed_count_label)
    return frame


def summarize_seed_count_groups(frame, column, source_order, interval):
    """Pool plates within display categories; retain actual count ranges."""
    records = []
    for source in source_order:
        plates = frame.loc[frame['source'] == source].copy()
        add_seed_count_plot_columns(plates)
        for position, group in plates.groupby('seed_count_plot'):
            valid = group.loc[group[column].notna()]
            values = valid[column].to_numpy(float)
            if not len(values):
                continue
            lower, upper = interval(values)
            records.append({
                'source': source, 'metric': column,
                'seed_count_plot': position,
                'seed_count_group': seed_count_label(position),
                'actual_seed_count_min': int(valid['seed_count'].min()),
                'actual_seed_count_max': int(valid['seed_count'].max()),
                'plate_count': len(values), 'mean': values.mean(),
                'sd': values.std(ddof=1) if len(values) > 1 else np.nan,
                'ci_lower': lower, 'ci_upper': upper,
            })
    return pd.DataFrame(records)


def plot_seed_count_summary(ax, summary, *, connect_overflow=False, **kwargs):
    """Plot grouped means, optionally connecting the labelled overflow endpoint."""
    summary = summary.sort_values('seed_count_plot')
    if connect_overflow:
        return ax.errorbar(
            summary['seed_count_plot'], summary['mean'],
            yerr=np.maximum(0, np.vstack([
                summary['mean'] - summary['ci_lower'],
                summary['ci_upper'] - summary['mean'],
            ])), **kwargs,
        )
    for overflow in (False, True):
        part = summary.loc[summary['seed_count_plot'].gt(SEED_COUNT_LIMIT) == overflow]
        if part.empty:
            continue
        options = dict(kwargs)
        if overflow:
            options['linestyle'] = 'none'
            options['label'] = '_nolegend_'
        ax.errorbar(
            part['seed_count_plot'], part['mean'],
            yerr=np.maximum(0, np.vstack([
                part['mean'] - part['ci_lower'],
                part['ci_upper'] - part['mean'],
            ])), **options,
        )


def configure_seed_count_axis(ax):
    """Label the terminal category explicitly and avoid overlapping 26/>26."""
    ticks = [1, 5, 10, 15, 20, 26, SEED_OVERFLOW_POSITION]
    ax.set_xticks(ticks, [seed_count_label(value) for value in ticks])
    ax.set_xlim(0, 30)
    labels = ax.get_xticklabels()
    labels[-2].set_horizontalalignment('right')
    labels[-1].set_horizontalalignment('left')


def configure_seed_count_colorbar(colorbar):
    # Keep the overflow label at the bar's endpoint and omit the crowded 26 tick.
    # This changes labels only: count 26 still has its own grayscale value.
    ticks = [1, 5, 10, 15, 20, SEED_OVERFLOW_POSITION]
    colorbar.set_ticks(ticks, labels=[seed_count_label(value) for value in ticks])
