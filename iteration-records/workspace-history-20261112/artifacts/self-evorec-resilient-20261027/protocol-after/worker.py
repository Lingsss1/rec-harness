"""Train candidate scripts; use only the pinned external RecHarness grader."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import save
from .prepare import DATASETS, REPO, sha


def evaluate(trial, data, mode, output):
    sys.path.insert(0, str(REPO))
    from gagc.benchmarks.amazon_reviews.harness import evaluate as grade
    start = time.monotonic()
    result = grade(predict_script=str(trial / 'working/predict.py'),
                   data_dir=str(data / 'trainval'),
                   test_dir=str(data / 'test') if mode == 'test' else None,
                   mode='test' if mode == 'test' else 'val', datasets=DATASETS,
                   max_eval_users=10000 if mode == 'test' else 200, seed=42, verbose=False)
    record = {'score': result.primary_metric, 'aggregate': result.aggregate,
              'per_dataset': {name: {'metrics': entry.metrics, 'n_users': entry.n_users}
                              for name, entry in result.per_dataset.items()},
              'evaluation_seconds': time.monotonic() - start,
              'predictor_sha256': sha(trial / 'working/predict.py')}
    if set(record['per_dataset']) != set(DATASETS):
        raise RuntimeError('A candidate must predict every dataset')
    save(output / 'result.json', record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--single-model', action='store_true')
    args = parser.parse_args()
    trial, data = args.trial.resolve(), args.data.resolve()
    output = args.output.resolve() if args.output else trial
    output.mkdir(parents=True, exist_ok=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; never silently run CPU experiments')
    torch.set_num_threads(4)
    # A dedicated 16-core slot is selected by its position in the four-card allocation.
    visible = int(os.environ['CUDA_VISIBLE_DEVICES'])
    slot = visible - 4
    cpus = set(range(64 + 16 * slot, 80 + 16 * slot))
    if not cpus <= os.sched_getaffinity(0):
        raise RuntimeError('Requested CPU allocation is unavailable')
    os.sched_setaffinity(0, cpus)
    save(output / 'resources.json', {'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                                   'cpu_affinity': sorted(cpus), 'threads': torch.get_num_threads()})
    if args.mode == 'train':
        env = {**os.environ, 'GAGC_DATA_DIR': str(data / 'trainval'), 'GAGC_DATASETS': ','.join(DATASETS)}
        if args.smoke:
            env.update(SASREC_EPOCHS='5', SASREC_BATCHES='2', SASREC_BATCH='32')
        # Training runs in a child so candidate imports cannot replace the host grader.
        with (trial / 'training.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(trial / 'train.py')], cwd=trial,
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        if args.single_model:
            # Isolate audit imports and GPU caches from the pinned external grader.
            # Its elapsed time is inside the trial's original GPU allocation.
            target = trial / 'single-model-audit.json'
            with (trial / 'single-model-audit.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'amazon_aligned.single_model',
                                '--audit', str(trial), '--data', str(data), '--output', str(target)],
                               cwd=Path(__file__).resolve().parents[1],
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                               stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
            from .core import read
            audit = read(target)
            if audit.get('eligible') is not True:
                raise ValueError('Single-model constraint rejected candidate: ' + audit['reason'])
    evaluate(trial, data, args.mode, output)
    if args.mode == 'train' and (trial / 'diagnosis_tools.py').exists():
        from .diagnostics import collect
        collect(trial, data)


if __name__ == '__main__':
    main()
