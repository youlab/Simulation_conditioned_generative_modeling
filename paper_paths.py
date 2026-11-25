"""Locations for compact paper evidence and ignored local analysis artifacts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATA = PROJECT_ROOT / 'source_data'
MISC = PROJECT_ROOT / 'Misc'


def inverse_results(name):
    return SOURCE_DATA / 'inverse_evaluation' / name


def inverse_preparation(name):
    return MISC / 'inverse_preparation' / name


def figure_source_path(path):
    """Keep exact figure arrays/provenance together, separate from PDF exports."""
    path = Path(path).resolve()
    if SOURCE_DATA in path.parents:
        return path
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path
    if 'vector based plots' not in relative.parts:
        return path
    if path.suffix == '.npz':
        return MISC / 'figure_arrays' / relative
    return SOURCE_DATA / 'figure_exports' / relative
