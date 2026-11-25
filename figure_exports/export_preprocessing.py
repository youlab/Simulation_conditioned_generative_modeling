"""Run the original Supplementary Figure 11 preprocessing/export cells."""
from pathlib import Path
import json
import os
import sys


def export_preprocessing():
    notebook_dir = Path(__file__).resolve().parents[1] / 'seed_to_sim1_sim2_deterministic'
    notebook = json.loads((notebook_dir / 'Image_preprocessing_Expimages.ipynb').read_text())
    previous = Path.cwd()
    namespace = {'__name__': '__main__'}
    sys.path.insert(0, str(notebook_dir))
    try:
        os.chdir(notebook_dir)
        for index in (0, 1, 3, 4, 5, 6):
            source = ''.join(notebook['cells'][index].get('source', []))
            exec(compile(source, f'Supplementary Figure 11 cell {index + 1}', 'exec'), namespace)
    finally:
        os.chdir(previous)
        sys.path.remove(str(notebook_dir))
