"""Save and load computed Figure 5F metrics; no manually entered plot values."""
import csv
import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'source_data/inverse_metrics_fig5'


def save_seed_metric_results(comparison, mse, dice, precision, recall, f1,
                             prediction_files, target_files, *, provenance=None):
    """Persist the existing per-image calculations and their population summary."""
    arrays = {name: np.asarray(values, dtype=float) for name, values in
              [('mse', mse), ('dice', dice), ('precision', precision),
               ('recall', recall), ('f1', f1)]}
    count = len(arrays['mse'])
    if not count or any(len(a) != count or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Expected complete finite metric arrays of equal length')
    if min(len(prediction_files), len(target_files)) < count:
        raise ValueError('Metric rows have no corresponding source filenames')
    per_image = [dict(comparison=comparison, image_index=i,
                      prediction_file=prediction_files[i], target_file=target_files[i],
                      **{name: float(values[i]) for name, values in arrays.items()})
                 for i in range(count)]
    summary = dict(comparison=comparison, n_images=count,
                   **{name+'_'+stat: float(getattr(np, stat)(values))
                      for name, values in arrays.items() for stat in ['mean', 'std']})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, new_rows in [('per_image_metrics.csv', per_image),
                                ('summary_metrics.csv', [summary])]:
        path = RESULTS_DIR / filename
        if path.exists():
            with path.open(newline='') as handle:
                previous = list(csv.DictReader(handle))
        else:
            previous = []
        rows = [r for r in previous if r['comparison'] != comparison] + new_rows
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(new_rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if provenance is not None:
        path = RESULTS_DIR / 'evaluation_provenance.json'
        records = json.loads(path.read_text()) if path.is_file() else {}
        records[comparison] = provenance
        path.write_text(json.dumps(records, indent=2) + '\n')
    return summary


def load_seed_metric_summary():
    """Load the three measured comparisons in the established figure order."""
    import pandas as pd
    path = RESULTS_DIR / 'summary_metrics.csv'
    if not path.is_file():
        raise FileNotFoundError('Run the experiment/simulation-to-seed evaluation notebooks first: '+str(path))
    results = pd.read_csv(path).set_index('comparison')
    provenance_path = RESULTS_DIR / 'evaluation_provenance.json'
    if not provenance_path.is_file():
        raise ValueError('Figure 5F checkpoint provenance is missing; rerun Evaluation_metrics_exptoseed.ipynb with last.ckpt.')
    provenance = json.loads(provenance_path.read_text())
    current = provenance.get('Diffusion model\nExperiment→Seed', {})
    if current.get('checkpoint_selection') != 'last.ckpt':
        raise ValueError('Figure 5F must use last.ckpt consistently with Figure 5E; rerun Evaluation_metrics_exptoseed.ipynb.')
    order = ['Diffusion model\nExperiment→Seed', 'Simulation model\nExperiment→Seed',
             'Simulation model\nSimulation→Seed']
    if results.index.has_duplicates or any(name not in results.index for name in order):
        raise ValueError('Expected exactly one saved summary for each Figure 5F comparison')
    return results.loc[order].reset_index()
