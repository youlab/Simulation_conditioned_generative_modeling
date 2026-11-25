#!/usr/bin/env python3
"""Download and extract public assets for the main figures.

    Set DATA_DIR in config_automate.py, then:
    python setup_data.py
    python reproduce_figures.py

No research-machine datasets, training runs, or local result tables are used.
The figure recipes run the existing notebook preprocessing and inference.
"""
import argparse
from pathlib import Path

from figure_generators.public_data import download, extract, manifest, validate_downloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Optional override of DATA_DIR in config_automate.py')
    parser.add_argument('--skip-download', action='store_true', help='Extract previously downloaded, checksum-verified archives')
    args = parser.parse_args(argv)
    if args.data_dir is None:
        from config_automate import DATA_DIR
        data_dir = Path(DATA_DIR).expanduser().resolve()
    else:
        data_dir = args.data_dir.expanduser().resolve()
    print('Public dataset:', manifest()['repo_id'], flush=True)
    print('Pinned revision:', manifest()['revision'], flush=True)
    print('Data directory:', data_dir, flush=True)
    if not args.skip_download:
        download(data_dir)
    extract(data_dir)
    valid, missing = validate_downloads(data_dir)
    if not valid:
        raise FileNotFoundError('\n'.join(missing))
    if args.data_dir is None:
        print('Setup complete. Run: python reproduce_figures.py')
    else:
        print('Setup complete. Run reproduce_figures.py --data-dir', data_dir)
    print('Pretrained VAE, CLIP and LPIPS weights download when their recipes first run.')
    print('Figure 3 uses the existing VAE encoder on its selected test images; no full latent-cache upload is needed.')


if __name__ == '__main__':
    main()
