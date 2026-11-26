"""Download and extract the pinned public files used by main Figures 1–5."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.parse import quote
from urllib.request import urlopen

MANIFEST_PATH = Path(__file__).with_name('public_data_manifest.json')


def manifest():
    return json.loads(MANIFEST_PATH.read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_downloads(data_dir, *, hashes=False):
    missing = []
    for name, expected in manifest()['files'].items():
        path = Path(data_dir) / name
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            missing.append(str(path))
        elif hashes and sha256(path) != expected['sha256']:
            missing.append(str(path) + ' (checksum mismatch)')
        extracted = Path(data_dir) / 'extracted' / (name[:-4] if name.endswith('.tar') else name)
        if not extracted.exists():
            missing.append(str(extracted))
    if not (Path(data_dir) / 'public_data_receipt.json').is_file():
        missing.append(str(Path(data_dir) / 'public_data_receipt.json'))
    return not missing, missing


def download(data_dir):
    release = manifest()
    for name, expected in release['files'].items():
        path = Path(data_dir) / name
        if path.is_file() and path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256']:
            print('Already verified:', name, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.partial')
        url = f"https://huggingface.co/datasets/{release['repo_id']}/resolve/{release['revision']}/{quote(name)}"
        print('Downloading:', name, flush=True)
        with urlopen(url, timeout=120) as response, partial.open('wb') as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if partial.stat().st_size != expected['bytes'] or sha256(partial) != expected['sha256']:
            raise ValueError('Public download failed checksum verification: ' + name)
        partial.replace(path)


def main_selection(data_dir):
    """Select original archive members needed by the current main notebooks."""
    import ast
    import re
    from .main_figures import ROOT, TASKS

    def assigned_literal(path, name, cell=None):
        text = Path(path).read_text()
        if cell is not None:
            text = ''.join(json.loads(text)['cells'][cell - 1]['source'])
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        return ast.literal_eval(node.value.args[0] if isinstance(node.value, ast.Call) else node.value)

    prefix = 'seed_to_sim_deterministic/'
    orders, member_maps = {}, {}
    for label in ('seed', 'intermediate_Tp3', 'complex_Tp3'):
        relative = prefix + 'Sim_050924_' + label
        with tarfile.open(Path(data_dir) / (relative + '.tar')) as archive:
            members = [m.name for m in archive if m.isfile()]
        parents = {str(Path(m).parent) for m in members}
        if len(parents) != 1:
            raise ValueError('Expected one image directory in ' + relative)
        folder = str(Path(relative) / parents.pop())
        orders[folder] = sorted(Path(m).name for m in members)
        member_maps[label] = (relative + '.tar', {Path(m).name: m for m in members}, folder)

    count_path = Path(data_dir) / prefix / 'Sim_050924_seed_counts_testset_60000_70000.json'
    score_seeds = list(json.loads(count_path.read_text()))[:1000]
    examples = assigned_literal(ROOT / TASKS['5C'].notebook, 'input_selected_files', cell=2)
    with tarfile.open(Path(data_dir) / 'information_encoding_decoding/inverse_predictions_main.tar') as archive:
        inverse_seeds = {Path(m.name).name.replace('Output', 'Input', 1) for m in archive
                         if m.isfile() and 'simulation_to_seed_107/' in m.name}
    seed_order = orders[member_maps['seed'][2]]
    intermediate_order = orders[member_maps['intermediate_Tp3'][2]]
    complex_order = orders[member_maps['complex_Tp3'][2]]
    # These are the same numerical slices used by the existing notebook loaders.
    selected = {
        'seed': set(score_seeds + examples + seed_order[70000:70025]) | (inverse_seeds & set(seed_order)),
        'intermediate_Tp3': {name.replace('Input', 'Output', 1) for name in score_seeds + examples}
                            | set(intermediate_order[70000:70100]),
        'complex_Tp3': set(complex_order[70000:70100]),
    }
    selection = {}
    for label, filenames in selected.items():
        relative, names, _ = member_maps[label]
        missing = filenames - names.keys()
        if missing:
            raise FileNotFoundError(f'Main-figure images missing from {relative}: {sorted(missing)}')
        selection[relative] = {names[name] for name in filenames}

    config_path = ROOT / 'config_automate.py'
    examples_a = assigned_literal(config_path, 'FILES_FIG1A')
    examples_b = assigned_literal(config_path, 'FILES_FIG1B')
    for archive_name, wanted in [('Exp', set(examples_a + examples_b)),
                                  ('SimcorrtoExp', set(examples_a + examples_b)),
                                  ('Exp_SimcorrtoExp_seed', set(examples_a))]:
        relative = 'sim_to_exp_diffusion/' + archive_name + '.tar'
        with tarfile.open(Path(data_dir) / relative) as archive:
            selection[relative] = {m.name for m in archive if m.isfile() and Path(m.name).name in wanted}
        assert len(selection[relative]) == len(wanted), relative

    relative = 'sim_to_exp_diffusion/inference_folders.tar'
    with tarfile.open(Path(data_dir) / relative) as archive:
        selection[relative] = {m.name for m in archive if m.isfile()
                               and Path(m.name).parent.name == 'v2025926_1251_simtoexp_v3'}
    assert len(selection[relative]) == 96

    models = {assigned_literal(config_path, name) for name in ('MODEL_DILRESNET_FIG2', 'MODEL_DILRESNET_FIG3')}
    patterns = [assigned_literal(ROOT / TASKS[key].notebook, 'pattern', cell=6) for key in ('3CD', '3EF')]
    relative = prefix + 'saved_models.tar'
    with tarfile.open(Path(data_dir) / relative) as archive:
        selection[relative] = {m.name for m in archive if m.isfile() and
                               (Path(m.name).name in models or any(re.match(p, Path(m.name).name) for p in patterns))}
    return selection, orders


def extract(data_dir):
    """Keep the deposited archive layout and original image bytes unchanged."""
    data_dir = Path(data_dir)
    selection, orders = main_selection(data_dir)
    extracted_counts = {}
    for name, expected in manifest()['files'].items():
        source = data_dir / name
        if not source.is_file() or sha256(source) != expected['sha256']:
            raise ValueError('Run setup_data.py: missing or changed public file ' + str(source))
        destination = data_dir / 'extracted' / name
        if name.endswith('.tar'):
            destination = destination.with_suffix('')
            destination.mkdir(parents=True, exist_ok=True)
            wanted = selection.get(name)
            print('Extracting main-figure files:', name, 'all' if wanted is None else len(wanted), flush=True)
            extracted_counts[name] = 0
            resolved_destination = destination.resolve()
            with tarfile.open(source) as archive:
                for member in archive:
                    if not member.isfile() or (wanted is not None and member.name not in wanted):
                        continue
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(resolved_destination) or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsafe archive member: ' + member.name)
                    archive.extract(member, destination)
                    extracted_counts[name] += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    receipt = {'repo_id': manifest()['repo_id'], 'revision': manifest()['revision'],
               'files': manifest()['files'], 'image_processing': 'none; original archive bytes extracted',
               'extraction_scope': 'main_figures_only', 'extracted_file_counts': extracted_counts,
               'file_orders': orders}
    (data_dir / 'public_data_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
