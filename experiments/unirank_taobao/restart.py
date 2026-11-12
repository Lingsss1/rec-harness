"""Start an independent, explicitly budgeted macro-AUC run on cached Taobao data."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import resource
import shutil
import subprocess
import sys
import time

from common import read, write, sha

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def restart(source, root, hours=12, gpus=(0, 1, 2, 3)):
    source = source.resolve()
    root = root.resolve()
    if root.exists():
        raise ValueError('Use a new campaign directory; completed or partial runs remain preserved')
    if type(hours) is not int or hours <= 0:
        raise ValueError('Positive integer hours required')
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError('The current task has exactly four initial topic lanes')
    prior = read(source / 'setup.json')
    acquisition = read(source / 'acquisition-status.json')
    manifest = read(source / 'dataset-source-manifest.json')
    if acquisition['phase'] != 'downloaded_and_verified':
        raise ValueError('The source dataset must be fully verified')
    root.mkdir(parents=True)
    created = time.time()
    write(root / 'status.json', {'phase': 'checking_cached_dataset', 'updated_unix': created})
    data = root / 'data/Taobao_Action'
    data.mkdir(parents=True)
    cached = source / 'data/Taobao_Action'
    # Independent mutable feature metadata, shared immutable parquet blocks.
    for path in cached.iterdir():
        if path.is_file():
            shutil.copy2(path, data / path.name)
    for split in ['train', 'valid', 'test']:
        (data / split).symlink_to(cached / split, target_is_directory=True)
    checked_bytes = 0
    check_started = time.time()
    for row in manifest['files']:
        path = data / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['local_sha256']:
            raise ValueError('Cached released file identity mismatch: ' + row['path'])
        checked_bytes += row['bytes']
    # The publication verification receipt is copied unchanged, never dated anew.
    write(root / 'dataset-source-manifest.json', manifest)
    write(root / 'acquisition-status.json', acquisition)
    write(root / 'cached-data-check.json', {'status': 'passed', 'source_campaign': str(source),
          'original_verified_unix': manifest['verified_unix'], 'checked_files': len(manifest['files']),
          'checked_bytes': checked_bytes, 'check_started_unix': check_started, 'check_finished_unix': time.time(),
          'test_access': 'file digests only; no test examples, labels or predictions parsed'})
    shutil.copytree(source / 'upstream', root / 'upstream',
                    ignore=shutil.ignore_patterns('.git', '__pycache__'))
    if (source / 'reference-cache').exists():
        shutil.copytree(source / 'reference-cache', root / 'reference-cache')
    setup = {**prior, 'created_unix': created, 'previous_campaign': str(source),
             'new_run_authority': 'user: repeat Taobao research for12h, select by average AUC',
             'research_startpoint': 'fresh DIN reference; no historical trial imported as a search incumbent',
             'initial_knowledge': 'external method knowledge only; previous experimental trajectories and test outcomes not supplied',
             'prior_test_use': 'The fixed official test split was evaluated in the previous campaign; any later reuse is supplementary.'}
    write(root / 'setup.json', setup)
    anchor = time.time()
    write(root / 'budget-policy.json', {'duration_seconds': hours * 3600,
          'start_event': 'new_campaign_started_with_verified_dataset', 'campaign_start_unix': anchor,
          'authority': 'user: start an independent12h Taobao research run using average AUC',
          'pre_start_costs': 'cached data integrity check retained separately; all preparation after this fixed start counts'})
    write(root / 'budget.json', {'gpus': list(gpus), 'model': 'gpt-6.1-sol', 'reasoning_effort': 'high'})
    write(root / 'user-protocol-choice.json', {'history_policy': 'official_offline',
          'origin': 'inherited explicit user choice: follow the paper', 'at_unix': anchor})
    write(root / 'auto-validation-only.json', {'enabled': True,
          'test_policy': 'closed to research; future reuse of the previously assessed official split is supplementary'})
    from build import build
    build(root)
    from selection import POLICY
    if read(root / 'budget.json')['seconds'] != hours * 3600:
        raise ValueError('Build changed the user-requested budget')
    if read(root / 'setup.json')['selection_policy'] != POLICY:
        raise ValueError('Build changed the macro-AUC rule')
    original_instructions = PROJECT / 'task_designs/autonomous_research_v1/agent/instructions.txt'
    if sha(original_instructions) != sha(root / 'source/task-design/agent/instructions.txt'):
        raise ValueError('Generic agent instructions changed')
    budget = read(root / 'budget.json')
    write(root / 'restart-receipt.json', {'source_campaign': str(source), 'root': str(root),
          'budget': budget, 'selection_policy': POLICY, 'previous_campaign_modified': False,
          'generic_instructions_unchanged': True, 'historical_trials_imported': 0,
          'gpus': list(gpus), 'expected_end_utc': datetime.fromtimestamp(budget['deadline_unix'], timezone.utc).isoformat()})
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cmd = [sys.executable, '-u', str(root / 'source/campaign.py'), '--root', str(root),
           '--history-policy', 'official_offline']
    environment = {**os.environ, 'REC_CTR_PROJECT': str(PROJECT), 'OMP_NUM_THREADS': '4',
                   'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    with (root / 'research.log').open('ab') as log:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=environment)
    write(root / 'launch-process.json', {'pid': child.pid, 'started_unix': time.time(), 'argv': cmd,
          'start_ticks': Path('/proc/' + str(child.pid) + '/stat').read_text().split()[21]})
    print(str(root))
    print('PID', child.pid)
    print('DEADLINE_UTC', datetime.fromtimestamp(budget['deadline_unix'], timezone.utc).isoformat())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hours', type=int, default=12)
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    restart(args.source_campaign, args.root, args.hours, tuple(int(x) for x in args.gpus.split(',')))
