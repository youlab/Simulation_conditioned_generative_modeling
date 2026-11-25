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


def extract(data_dir):
    """Keep the deposited archive layout and original image bytes unchanged."""
    data_dir = Path(data_dir)
    for name, expected in manifest()['files'].items():
        source = data_dir / name
        if not source.is_file() or sha256(source) != expected['sha256']:
            raise ValueError('Run setup_data.py: missing or changed public file ' + str(source))
        destination = data_dir / 'extracted' / name
        if name.endswith('.tar'):
            destination = destination.with_suffix('')
            destination.mkdir(parents=True, exist_ok=True)
            print('Extracting:', name, flush=True)
            with tarfile.open(source) as archive:
                for member in archive:
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsafe archive member: ' + member.name)
                    archive.extract(member, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    receipt = {'repo_id': manifest()['repo_id'], 'revision': manifest()['revision'],
               'files': manifest()['files'], 'image_processing': 'none; original archive bytes extracted'}
    (data_dir / 'public_data_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
