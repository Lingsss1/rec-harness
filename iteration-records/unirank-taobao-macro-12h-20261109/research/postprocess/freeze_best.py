"""Package the final validation-selected checkpoint after research closes; no test access."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

from common import read, write, sha


def freeze(root):
    sys.path.insert(0, str(root / 'source'))
    from selection import best_trial, validate_policy, task_deltas
    protocol = read(root / 'protocol.json')
    validate_policy(protocol['selection_policy'])
    states = [read(root / 'topics' / f'gpu-{gpu}' / 'state.json') for gpu in read(root / 'budget.json')['gpus']]
    if any(s['phase'] not in ['research_stopped', 'cancelled', 'failed_before_assessment'] for s in states):
        raise ValueError('Research lanes must all be closed before selection')
    references = read(root / 'references.json')['trials']
    trials = [trial for state in states for trial in state['trials']]
    chosen = best_trial({'trials': references + trials})
    din = next(row for row in references if row.get('baseline_kind') == 'din')
    output = root / 'best-model'
    if output.exists():
        raise ValueError('Final artifacts already exist; do not overwrite')
    source = Path(chosen['model_path'])
    checkpoint = Path(chosen['output']) / 'checkpoint.pt'
    if sha(source) != chosen['source_sha256'] or sha(checkpoint) != chosen['metrics']['checkpoint_sha256']:
        raise ValueError('Selected source/checkpoint digest mismatch')
    output.mkdir()
    shutil.copy2(source, output / 'model.py')
    shutil.copy2(checkpoint, output / 'checkpoint.pt')
    write(output / 'model-config.json', chosen['config'])
    receipt = {'status': 'complete', 'frozen_unix': time.time(), 'trial_id': chosen['id'],
               'role': chosen['role'], 'protocol_hash': protocol['identity'],
               'selection_policy': protocol['selection_policy'], 'validation': chosen['metrics']['validation'],
               'din_validation': din['metrics']['validation'], 'per_task_deltas_vs_din': task_deltas(chosen, din),
               'parameters': chosen['metrics']['parameters'], 'source_sha256': sha(output / 'model.py'),
               'checkpoint_sha256': sha(output / 'checkpoint.pt'), 'trial_output': chosen['output'],
               'seed': chosen['seed'], 'train_seconds': chosen['metrics']['train_seconds'],
               'training_job_wall_seconds': chosen['wall_seconds'], 'study_ref': chosen.get('study_ref'),
               'effective_config': chosen['metrics']['effective_config'], 'test_accessed': False,
               'budget': read(root / 'budget.json'),
               'search_summary': {'completed_rounds': sum(s['completed_research_rounds'] for s in states),
                                  'training_attempts': len(trials),
                                  'successful_attempts': sum(t['status'] == 'success' for t in trials),
                                  'failed_attempts': sum(t['status'] != 'success' for t in trials)},
               'limitations': ['single training seed; validation selection',
                              'the official test split was assessed in an earlier campaign; later reuse is supplementary']}
    write(output / 'manifest.json', receipt)
    write(root / 'final-validation-summary.json', receipt)


def run(root, watch):
    root = root.resolve()
    with (root / 'finalization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'finalizer-process.json', {'pid': os.getpid(), 'started_unix': time.time()})
        while True:
            status = read(root / 'status.json')
            if status['phase'] == 'setup_failed':
                raise ValueError('Research setup failed; inspect research.log')
            if status['phase'] in ['topic_research_stopped', 'topic_research_with_failures', 'terminated_by_user']:
                freeze(root)
                return
            if not watch:
                raise ValueError('Research has not closed')
            time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        run(args.root, args.watch)
    except BaseException as error:
        write(args.root / 'finalization-failure.json', {'error': repr(error), 'at_unix': time.time()})
        raise
