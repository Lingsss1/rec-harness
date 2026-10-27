"""Run the three methods sequentially on the same four otherwise idle GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .bootstrap import bootstrap
bootstrap()
from .core import read, save
from .prepare import ROOT, WORKSPACE, REPO, sha


def manifest(config):
    roots = [WORKSPACE / 'amazon_aligned', REPO / 'gagc/benchmarks/amazon_reviews',
             WORKSPACE / 'artifacts/MLEvolve/engine', WORKSPACE / 'artifacts/MLEvolve/agents',
             WORKSPACE / 'artifacts/MLEvolve/llm', WORKSPACE / 'artifacts/MLEvolve/utils',
             WORKSPACE / 'artifacts/Self-EvolveRec-code',
             WORKSPACE / 'mlevolve_codex', WORKSPACE / 'mle_star_codex']
    paths = []
    for root in roots:
        paths += [p for p in root.rglob('*.py') if '.git' not in p.parts and '__pycache__' not in p.parts]
    paths += [REPO / 'gagc/templates/perdataset/train.py', ROOT / 'data/prepared.json',
              ROOT / 'data/trainval/dataset_stats.json', ROOT / 'config.json',
              ROOT / 'environment.json', REPO / 'run2.sh', REPO / 'gagc/tools.py',
              REPO / 'gagc/data_preprocess.py', REPO / 'gagc/compute_dataset_stats.py',
              WORKSPACE / 'staged-autoresearch/staged_research/codex.py',
              WORKSPACE / 'artifacts/MLEvolve/config/config.yaml',
              WORKSPACE / 'artifacts/Self-EvolveRec-code/configs/config_baseline.yaml']
    for directory in ['artifacts/mlevolve-movies-2024/embedding-model',
                      'artifacts/amazon-aligned-20261022/runtime-packages']:
        paths += [p for p in (WORKSPACE / directory).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for category in config['alignment']['datasets']:
        paths += [ROOT / 'data/trainval' / (category + '_train.txt'),
                  ROOT / 'data/trainval' / (category + '_valid.txt'),
                  ROOT / 'data/trainval' / (category + '_text_name_dict.json.gz'),
                  ROOT / 'data/test' / (category + '_test.txt')]
    return {str(p.relative_to(WORKSPACE)): sha(p) for p in sorted(set(paths))}


def queue(config_path):
    config = read(config_path)
    root = Path(config['root'])
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(config['cpu_cores']))
    frozen = read(root / 'frozen.json')
    if manifest(config) != frozen['files']:
        raise RuntimeError('Registered code/config/data changed before queue execution')
    results = []
    for method in config['methods']:
        destination = root / method
        destination.mkdir(exist_ok=True)
        if (destination / 'summary.json').exists():
            results.append({'method': method, 'status': 'already_complete'})
            continue
        if (destination / 'events.jsonl').exists() or (destination / 'failure.json').exists():
            results.append({'method': method, 'status': 'existing_run_requires_review'})
            continue
        if manifest(config) != frozen['files']:
            raise RuntimeError('Registered inputs changed before ' + method)
        while True:
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                  '--format=csv,noheader,nounits'], text=True)
            devices = {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
                       for row in inventory.strip().splitlines()}
            if all(devices[gpu][0] < 512 and devices[gpu][1] < 5 for gpu in config['gpus']):
                break
            save(root / 'queue-status.json', {'status': 'waiting_for_idle_gpus', 'method': method,
                 'completed': results, 'devices': devices, 'updated_at': time.time()})
            time.sleep(15)
        save(root / 'queue-status.json', {'status': 'running', 'method': method, 'completed': results,
                                          'gpu_ids': config['gpus'], 'updated_at': time.time()})
        command = [config['controller_python'], '-u', '-m', 'amazon_aligned.run',
                   '--config', str(config_path), '--method', method]
        with (destination / 'controller.log').open('a') as log:
            proc = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}, start_new_session=True)
            save(destination / 'launcher.json', {'pid': proc.pid, 'command': command, 'started_at': time.time()})
            code = proc.wait()
        results.append({'method': method, 'status': 'complete' if code == 0 else 'failed', 'exit_code': code})
    save(root / 'queue-status.json', {'status': 'complete' if all(r['status'] in ['complete', 'already_complete'] for r in results)
                                    else 'finished_with_failures', 'completed': results, 'updated_at': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--queue', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve()
    if args.queue:
        return queue(path)
    config = read(path)
    root = Path(config['root'])
    if (root / 'launcher.json').exists():
        raise RuntimeError('Existing queue registration; inspect state before restarting')
    acceptance = read(root / 'preflight/acceptance.json')
    if not acceptance['passed'] or acceptance['config_sha256'] != sha(path):
        raise RuntimeError('Preflight must pass before formal launch')
    save(root / 'frozen.json', {'files': manifest(config), 'registered_at': time.time(),
                              'config_sha256': sha(path)})
    command = [sys.executable, '-u', '-m', 'amazon_aligned.launch', '--config', str(path), '--queue']
    with (root / 'queue.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    save(root / 'launcher.json', {'pid': proc.pid, 'command': command, 'started_at': time.time()})
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Queue exited immediately; inspect queue.log')
    print(json.dumps({'pid': proc.pid, 'root': str(root), 'log': str(root / 'queue.log')}))


if __name__ == '__main__':
    main()
