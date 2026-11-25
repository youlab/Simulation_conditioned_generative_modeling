#!/usr/bin/env python3
"""Generate the current computational main-figure panels (Figures 1–5 only).

Examples:
  python reproduce_figures.py --list
  python reproduce_figures.py --dry-run
  python reproduce_figures.py --figures 4 5 --output ./figure_outputs
  python reproduce_figures.py --tasks 4C 4D 5F 5G --output ./figure_outputs

All inputs come from the pinned HF dataset downloaded by setup_data.py.
Quantitative panels recompute the existing notebook measurements from those inputs. Schematics and final
assembled figure pages are prepared in the manuscript editor.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys

from figure_generators.main_figures import ROOT, TASKS, select_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--figures', type=int, nargs='+', choices=range(1, 6), help='Main figures to generate (default: 1–5)')
    selection.add_argument('--tasks', nargs='+', choices=TASKS, help='Selected main-panel recipes; paired panels share one inference run')
    parser.add_argument('--data-dir', type=Path, help='Optional override of DATA_DIR in config_automate.py')
    parser.add_argument('--output', type=Path, help='Output directory (default: figure_outputs/<UTC timestamp>)')
    parser.add_argument('--list', action='store_true', help='List main panels and source notebooks')
    parser.add_argument('--dry-run', action='store_true', help='Check notebook recipe syntax; do not load models or write files')
    parser.add_argument('--timeout', type=int, default=7200, help='Maximum seconds per recipe (default: 7200)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip early preflight; workers still validate outputs')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.data_dir is not None:
        os.environ['PATTERN_DATA_DIR'] = str(args.data_dir.expanduser().resolve())
    tasks = select_tasks(args.figures, args.tasks)
    for task in tasks:
        print(f'{task.key}: {task.description}\n  {task.notebook}\n  Outputs: {", ".join(task.outputs)}')
    if args.list:
        print('\nMain computational panels only; supplementary cells and manuscript schematics are excluded.')
        return 0
    if not args.skip_validation or args.dry_run:
        from figure_generators.notebook_runner import check_task
        try:
            for task in tasks:
                check_task(task)
        except (OSError, ValueError, SyntaxError, IndexError) as error:
            print(f'Preflight failed: {error}', file=sys.stderr)
            return 1
    if args.dry_run:
        print(f'\nDry run passed: {len(tasks)} recipes, {sum(len(t.outputs) for t in tasks)} main-panel files. No models loaded or files written.')
        return 0
    from config_automate import validate_setup
    valid, missing = validate_setup()
    if not valid:
        print('Run setup_data.py --data-dir <directory> first. Missing public assets:\n' + '\n'.join(missing[:8]), file=sys.stderr)
        return 1
    output = (args.output or ROOT / 'figure_outputs' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        print(f'\nGenerating {task.key} into {output}', flush=True)
        try:
            subprocess.run([sys.executable, '-m', 'figure_generators.notebook_runner',
                            '--task', task.key, '--output', str(output)],
                           cwd=ROOT, check=True, timeout=args.timeout)
            record = json.loads((output / 'provenance' / f'{task.key}.json').read_text())
            records.append({'task': task.key, 'status': 'success', **record})
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as error:
            print(f'FAILED {task.key}: {error}', file=sys.stderr)
            records.append({'task': task.key, 'status': 'failed', 'error': str(error)})
    manifest = {'figures': sorted({task.figure for task in tasks}), 'main_figures_only': True,
                'tasks': records, 'output_directory': str(output),
                'schematics': 'Assembled separately in the manuscript editor.'}
    (output / 'reproduction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    failed = [record['task'] for record in records if record['status'] != 'success']
    print(f'\nCompleted {len(records) - len(failed)}/{len(tasks)} recipes. Outputs: {output}')
    if failed:
        print('Failed recipes: ' + ', '.join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
