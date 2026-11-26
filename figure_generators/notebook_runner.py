"""Execute a declared main-panel recipe without changing notebook/source files."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import types

from .main_figures import ROOT, TASKS, source_table_path


def clean_source(source):
    # Inline display settings have no role in a headless export process.
    return '\n'.join(line for line in source.splitlines()
                     if not line.lstrip().startswith(('%config InlineBackend.', '%matplotlib ')))


def notebook_cells(task):
    path = ROOT / task.notebook
    notebook = json.loads(path.read_text())
    result = {}
    for index in task.cells:
        cell = notebook['cells'][index - 1]
        if cell['cell_type'] != 'code':
            raise ValueError(f'{task.notebook}: expected code at cell {index}')
        source = cell['source']
        result[index] = ''.join(source) if isinstance(source, list) else source
    return result


def main_only_groups(source):
    """Keep the notebook's main 4D group; do not create supplementary plots."""
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'comparison_groups' for t in node.targets)]
    if len(matches) != 1 or not isinstance(matches[0].value, ast.List):
        raise ValueError('Figure 4D comparison_groups changed; review the recipe')
    node = matches[0]
    first = node.value.elts[0]
    if ast.literal_eval(first)[1] != [0, 1, 3]:
        raise ValueError('Unexpected Figure 4D comparison selection')
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + 'comparison_groups = ' + repr([ast.literal_eval(first)]) + '\n' + ''.join(lines[node.end_lineno:])


def main_metric_plot(source):
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(isinstance(target, ast.Tuple) for target in node.targets):
                if 'LPIPS_SEED_' in ast.unparse(node):
                    continue
            if 'comparison_data' in names:
                node.value.elts[2] = ast.Constant(None)  # supplementary-only row, never selected
            if 'CMMD_VALUES' in names:
                node.value.elt = ast.Call(func=ast.Attribute(value=ast.Name(id='cmmd_by_comparison', ctx=ast.Load()), attr='get', ctx=ast.Load()), args=[ast.Name(id='key', ctx=ast.Load())], keywords=[])
        nodes.append(node)
    tree.body = nodes
    return ast.unparse(ast.fix_missing_locations(tree))


def build_steps(task):
    """Return notebook cells and small adapters, leaving originals untouched."""
    cells = notebook_cells(task)
    steps = []
    for index in task.cells:
        source = cells[index]
        if task.key == '1B' and index == 2:
            source = source.replace('seed_data = preprocess_simulation_input_data(',
                                    'seed_files = list(FILES_FIG1A)  # deposited TIFFs contain the same native grids\nseed_data = preprocess_simulation_input_data(')
        if task.key == '4D' and index == 14:
            steps.append((None, '_load_image_metrics()'))
            source = main_metric_plot(main_only_groups(source))
        if task.key == '4D' and index == 15:
            start = source.index('# Preserve the previous active results')
            stop = source.index('# Replot only;', start)
            source = source[:start] + source[stop:]
        if task.key in ('3CD', '3EF') and index == 5:
            source = 'from models.vae import encode_img\nwith torch.no_grad():\n    yprime_in = encode_img(y3.to(device)).cpu()\n'
        if task.key == '5F' and index == 2:
            steps.append((None, '_measure_inverse_metrics()'))
        if task.key == '5G' and index == 12:
            # Only area regression is in the paper. Keep the exact metric
            # function and omit the notebook's optional tip diagnostics.
            source = source[:source.index('\nmetric_summary =')]
        steps.append((index, clean_source(source)))
        if task.key in ('1B', '2B', '3B', '5C', '5E') and index == task.cells[-1]:
            steps.append((None, '_export_current_grid()'))
        if task.key in ('3CD', '3EF') and index == 7:
            filename = 'Fig3D_predictions.pdf' if task.key == '3CD' else 'Fig3F_predictions.pdf'
            steps.append((None, f"_export_figure(plt.gcf(), {filename!r}, mode='images_only')"))
    return steps


def check_task(task):
    steps = build_steps(task)
    for index, source in steps:
        compile(source, f'{task.notebook}:cell{index}', 'exec')
    missing = [str(source_table_path(name)) for name in task.source_tables
               if not source_table_path(name).is_file()]
    if missing:
        raise FileNotFoundError('Missing measured source tables: ' + ', '.join(missing))
    if any(not re.match(rf'^Fig{task.figure}[A-Z]', name) for name in task.outputs):
        raise ValueError('Only current main-figure outputs may be declared')
    return steps


def install_local_results_guard(data_dir=None):
    forbidden = ['/hpc/group/youlab/ks723/storage/',
                 '/hpc/group/youlab/ks723/miniconda3/Lingchong/',
                 '/hpc/group/youlab/ks723/miniconda3/saved_models/',
                 '/hpc/dctrl/ks723/Information_encoding_decoding/',
                 str(ROOT / 'source_data') + '/']
    for family in ('information_encoding_decoding', 'sim_to_exp_diffusion/controlnet_essential'):
        forbidden.extend(str(ROOT / family / part) + '/' for part in ('inference', 'lightning_logs'))

    allowed = str(Path(data_dir).resolve()) if data_dir is not None else None

    def guard(event, args):
        if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0]))
            if allowed is not None and (path == allowed or path.startswith(allowed + os.sep)):
                return
            if any(path.startswith(prefix) for prefix in forbidden):
                raise PermissionError('Public reproduction cannot read local research results: ' + path)
    sys.addaudithook(guard)


def run_task(task, output_dir):
    steps = check_task(task)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = output_dir / 'provenance'
    provenance_dir.mkdir(exist_ok=True)

    from .public_data import manifest as public_manifest
    release = public_manifest()
    os.environ['PATTERN_VAE_REVISION'] = release['vae_revision']
    os.environ['PATTERN_CLIP_REVISION'] = release['clip_revision']
    if os.environ.get('PATTERN_DENY_HPC_RESULTS') == '1':
        from config_automate import DATA_DIR
        install_local_results_guard(DATA_DIR)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import numpy as np
    import pandas as pd
    import figure_exports
    import config_automate

    # Preserve a user-selected cache. No HuggingFace/model import happens in
    # the CLI or dry-run path. All measurements use public downloaded inputs.
    os.environ.setdefault('HF_HOME', str(Path(config_automate.DATA_DIR) / '.huggingface_cache'))
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
    os.environ.setdefault('TORCH_HOME', str(Path(config_automate.DATA_DIR) / '.torch_cache'))
    family = Path(task.notebook).parent
    sys.path.insert(0, str(ROOT / family))
    os.chdir(ROOT / family)
    overrides = getattr(config_automate, 'MAIN_FIGURE_PATH_OVERRIDES', {}).get(str(family), {})
    config_name = 'cldm.config' if 'controlnet_essential' in str(family) else 'utils.config'
    # Install a public-only config BEFORE importing notebook/model modules.
    # This avoids executing the research configs and their training-folder scans.
    config = types.ModuleType(config_name)
    config.__dict__.update({k: v for k, v in vars(config_automate).items() if k.isupper()})
    if task.key in ('3CD', '3EF'):
        config.MODEL_SAVE_LOCATION = config_automate.data_demand_models(task.key == '3EF')
    data_output = output_dir / 'data' / task.key
    data_output.mkdir(parents=True, exist_ok=True)
    config.EXP_FOLDER_TEST_COLONYMETRICS = str(data_output / 'experimental_colony_targets.json')
    sys.modules[config_name] = config
    parent = importlib.import_module(config_name.rsplit('.', 1)[0])
    setattr(parent, 'config', config)
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f'Unknown {config_name} setting: {key}')
        setattr(config, key, value)
    font_path = Path(config.FPATH)
    if not font_path.is_file():
        raise FileNotFoundError(f'Public figure font missing: {font_path}; run setup_data.py')

    seed = getattr(config_automate, 'MAIN_FIGURE_RANDOM_SEED', 42)
    random.seed(seed)
    np.random.seed(seed)
    # Seed model sampling when a recipe actually imports torch.
    if task.key in ('1B', '1C', '2B', '2CD', '3B', '3CD', '3EF', '5E'):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    receipt_path = Path(config_automate.DATA_DIR) / 'public_data_receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    original_orders = {str((config_automate.EXTRACTED_DIR / folder).resolve()): names
                       for folder, names in receipt.get('file_orders', {}).items()}

    namespace = {'__name__': '__main__', 'np': np, 'pd': pd, 'plt': plt,
                 'Path': Path, 'display': lambda *args, **kwargs: None}
    produced = set()
    used_tables = []
    allowed = set(task.outputs)
    original_savefig = Figure.savefig
    original_export = figure_exports.export_figure
    original_show = plt.show
    export_names = ('export_figure', 'save_vector_pdf', 'save_hybrid_pdf', 'save_images_only_pdf')
    original_aliases = {name: getattr(figure_exports, name) for name in export_names}

    def destination(path):
        name = Path(path).name
        if name not in allowed:
            raise ValueError(f'Task {task.key} attempted an undeclared/supplementary export: {name}')
        return output_dir / name

    def savefig(fig, path, *args, **kwargs):
        if hasattr(path, 'write'):
            return original_savefig(fig, path, *args, **kwargs)
        target = destination(path)
        result = original_savefig(fig, target, *args, **kwargs)
        produced.add(target.name)
        return result

    def export(fig, path, *, mode='vector', **kwargs):
        return original_export(fig, destination(path), mode=mode, **kwargs)

    Figure.savefig = savefig
    figure_exports.export_figure = export
    figure_exports.save_vector_pdf = export
    figure_exports.save_hybrid_pdf = lambda fig, path, **kw: export(fig, path, mode='hybrid', **kw)
    figure_exports.save_images_only_pdf = lambda fig, path, **kw: export(fig, path, mode='images_only', **kw)
    plt.show = lambda *args, **kwargs: None  # retain the actual notebook artists
    namespace['_export_figure'] = export
    namespace['_original_order'] = lambda folder: original_orders.get(str(Path(folder).resolve()))

    def load_image_metrics():
        from .public_results import image_metrics
        image_metrics(config, data_output, namespace)

    def measure_inverse_metrics():
        from .public_results import inverse_metrics
        inverse_metrics(config, data_output)

    def export_current_grid():
        if task.key == '5E':
            export(namespace['fig'], task.outputs[0], mode='hybrid')
            export(namespace['fig'], task.outputs[1], mode='images_only')
        else:
            figure = namespace.get('fig') if task.key in ('1B', '5C') else plt.gcf()
            export(figure, task.outputs[0], mode='images_only')

    namespace.update(_load_image_metrics=load_image_metrics,
                     _measure_inverse_metrics=measure_inverse_metrics,
                     _export_current_grid=export_current_grid)
    executed = []
    try:
        for index, source in steps:
            # Two notebooks have a historical literal Arial path; use the
            # configured font without changing plot/model selection.
            source = source.replace("'/hpc/group/youlab/ks723/miniconda3/Lingchong/fonts/ARIAL.TTF'", repr(str(font_path)))
            source = source.replace('"/hpc/group/youlab/ks723/miniconda3/Lingchong/fonts/ARIAL.TTF"', repr(str(font_path)))
            source = re.sub(r"_vector_runpy\.run_path\(str\(.*?config\.py'\s*\)\)\['FPATH'\]", repr(str(font_path)), source, flags=re.S)
            print(f'[{task.key}] {task.notebook}: cell {index or "adapter"}', flush=True)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in ('preprocess_simulation_input_data', 'preprocess_simulation_output_data')
                        and len(node.args) == 3 and not any(k.arg == 'img_filenames' for k in node.keywords)):
                    node.keywords.append(ast.keyword(arg='img_filenames', value=ast.Call(
                        func=ast.Name(id='_original_order', ctx=ast.Load()), args=[node.args[0]], keywords=[])))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and (node.func.value.id, node.func.attr) == ('torch', 'load') and not any(k.arg == 'map_location' for k in node.keywords):
                        node.keywords.append(ast.keyword(arg='map_location', value=ast.Constant('cpu')))
                    elif node.func.attr == 'load_from_checkpoint' and not any(k.arg == 'map_location' for k in node.keywords):
                        node.keywords.append(ast.keyword(arg='map_location', value=ast.Constant('cpu')))
            exec(compile(ast.fix_missing_locations(tree), f'{task.notebook}:cell{index}', 'exec'), namespace)
            if index is not None:
                executed.append({'cell': index, 'sha256': hashlib.sha256(source.encode()).hexdigest()})
        if produced != allowed:
            raise RuntimeError(f'Task {task.key} did not generate all expected outputs: {sorted(allowed - produced)}')
        outputs = []
        for name in task.outputs:
            path = output_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f'Empty or missing output: {path}')
            if path.suffix == '.pdf' and not path.read_bytes().startswith(b'%PDF-'):
                raise RuntimeError(f'Invalid PDF: {path}')
            outputs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if task.key == '4C':
            namespace['plate_df'].to_csv(data_output / 'plate_metrics.csv', index=False)
            namespace['terminal_width_df'].to_csv(data_output / 'terminal_widths.csv', index=False)
        if task.key == '5G':
            namespace['results'].to_csv(data_output / 'colony_area_predictions.csv', index=False)
        record = {'task': task.key, 'figure': task.figure, 'notebook': task.notebook,
                  'cells': executed, 'outputs': outputs, 'random_seed': seed,
                  'data_directory': str(config_automate.DATA_DIR),
                  'public_dataset': __import__('figure_generators.public_data', fromlist=['manifest']).manifest(),
                  'path_overrides': overrides,
                  'source_tables': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                    for p in dict.fromkeys(used_tables)],
                  'source_mode': 'public images/checkpoints; existing notebook processing',
                  'schematic_panels': 'assembled separately in the manuscript editor'}
        (provenance_dir / f'{task.key}.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
        return outputs
    finally:
        Figure.savefig = original_savefig
        plt.show = original_show
        for name, function in original_aliases.items():
            setattr(figure_exports, name, function)
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_task(TASKS[args.task], args.output)


if __name__ == '__main__':
    main()
