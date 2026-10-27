"""Two additional fresh repetitions per method on disjoint four-GPU queues."""
import argparse
from copy import deepcopy
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .bootstrap import bootstrap
bootstrap()
from .core import read, save
from .launch import manifest
from .prepare import ROOT, WORKSPACE, sha

DESTINATION = WORKSPACE / 'artifacts/amazon-repeats-20261025'


def registered_files(plan):
    config = read(plan['queues'][0]['jobs'][0]['config'])
    result = manifest(config)
    result[str((DESTINATION / 'plan.json').relative_to(WORKSPACE))] = sha(DESTINATION / 'plan.json')
    for queue in plan['queues']:
        for job in queue['jobs']:
            path = Path(job['config'])
            result[str(path.relative_to(WORKSPACE))] = sha(path)
    return result


def prepare():
    if (DESTINATION / 'plan.json').exists():
        raise RuntimeError('Existing repetitions plan; inspect instead of overwriting')
    original = read(ROOT / 'config.json')
    for method in ['self_evolverec', 'mlestar']:
        previous = read(ROOT / method / 'summary.json')
        if previous['status'] != 'complete':
            raise RuntimeError('The first run must be complete: ' + method)
    queues = []
    for method, gpus in [('self_evolverec', [0, 1, 2, 3]), ('mlestar', [4, 5, 6, 7])]:
        jobs = []
        for repetition in [2, 3]:
            config = deepcopy(original)
            run_root = DESTINATION / f'{method}-run-{repetition:02d}'
            config['root'] = str(run_root)
            config['methods'] = [method]
            config['gpus'] = gpus
            config['cpu_cores'] = list(range(16 * gpus[0], 16 * (gpus[-1] + 1)))
            config['replication'] = {'method': method, 'repetition': repetition,
                 'first_run': str(ROOT / method), 'independent_gpu_seconds': 43200,
                 'fresh_llm_calls_and_baseline': True,
                 'seeds': 'Same preprocessing=0, training/evaluation=42 and native database seeds; independent LLM generations',
                 'reuse_policy': 'Reuse only verified prepared inputs, pinned source and dependency packages; no prior trained weights, proposals, metrics or experience memory'}
            if method == 'mlestar':
                config['replication']['protocol_fix'] = (
                    'Use the already validated fixed candidate-ID enum and one-trajectory code transport '
                    'from the first-run recovery; fresh run has no cached calls or receipts.')
            path = run_root / 'config.json'
            save(path, config)
            jobs.append({'method': method, 'repetition': repetition, 'root': str(run_root), 'config': str(path)})
        queues.append({'id': 'gpu-' + '-'.join(map(str, gpus)), 'gpus': gpus,
                       'cpu_cores': list(range(16 * gpus[0], 16 * (gpus[-1] + 1))), 'jobs': jobs})
    plan = {'root': str(DESTINATION), 'created_at': time.time(), 'queues': queues,
            'llm_model': original['llm_model'], 'llm_effort': original['llm_effort'],
            'gpu_seconds_per_job': 43200, 'total_new_gpu_seconds': 172800,
            'completed_runs_per_method_after_queue': 3}
    save(DESTINATION / 'plan.json', plan)
    save(DESTINATION / 'frozen.json', {'files': registered_files(plan), 'registered_at': time.time(),
         'baseline_preflight': str(ROOT / 'preflight/acceptance.json'),
         'native_preflight': str(ROOT / 'preflight/native-self.json'),
         'mlestar_transport_acceptance': str(ROOT / 'mlestar/recovery/status.json')})
    print(json.dumps({'prepared': True, 'root': str(DESTINATION), 'queues': queues}, ensure_ascii=False))


def check_files(plan):
    if registered_files(plan) != read(DESTINATION / 'frozen.json')['files']:
        raise RuntimeError('Registered repetition source/config/data changed')


def queue_status(queue, value):
    save(DESTINATION / 'queues' / queue['id'] / 'status.json', {**value, 'updated_at': time.time()})


def run_queue(plan, queue):
    directory = DESTINATION / 'queues' / queue['id']
    lock = (directory / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(queue['cpu_cores']))
    completed = []
    try:
        for job in queue['jobs']:
            check_files(plan)
            config = read(job['config'])
            folder = Path(job['root']) / job['method']
            folder.mkdir(exist_ok=True)
            if any((folder / name).exists() for name in ['events.jsonl', 'summary.json', 'failure.json']):
                raise RuntimeError('Fresh repetition already contains execution evidence: ' + str(folder))
            while True:
                inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                     '--format=csv,noheader,nounits'], text=True)
                devices = {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
                           for row in inventory.strip().splitlines()}
                if all(devices[g][0] < 512 and devices[g][1] < 5 for g in queue['gpus']):
                    break
                queue_status(queue, {'status': 'waiting_for_idle_gpus', 'active': job,
                                     'completed': completed, 'devices': devices})
                time.sleep(15)
            queue_status(queue, {'status': 'running', 'active': job, 'completed': completed,
                                 'gpus': queue['gpus'], 'initial_gpu_inventory': devices})
            command = [config['controller_python'], '-u', '-m', 'amazon_aligned.repeats',
                       '--worker', job['config']]
            with (folder / 'controller.log').open('a') as log:
                process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                          'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '42'})
                save(folder / 'launcher.json', {'pid': process.pid, 'command': command, 'time': time.time()})
                code = process.wait()
            completed.append({**job, 'status': 'complete' if code == 0 else 'failed', 'exit_code': code})
        queue_status(queue, {'status': 'complete' if all(j['exit_code'] == 0 for j in completed)
                             else 'finished_with_failures', 'completed': completed})
    except BaseException:
        queue_status(queue, {'status': 'failed', 'completed': completed, 'error': traceback.format_exc()})
        raise


def run_worker(plan, config_path):
    check_files(plan)
    jobs = [j for q in plan['queues'] for j in q['jobs']]
    job = next(j for j in jobs if Path(j['config']) == config_path.resolve())
    config = read(config_path)
    method = job['method']
    folder = Path(job['root']) / method
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((folder / name).exists() for name in ['events.jsonl', 'summary.json', 'failure.json']):
        raise RuntimeError('Fresh repetition cannot reuse an existing run')
    save(folder / 'status.json', {'status': 'running', 'method': method,
         'repetition': job['repetition'], 'gpus': config['gpus'], 'started_at': time.time()})
    try:
        if method == 'mlestar':
            from . import mlestar
            from .mlestar_resume import ResumeController
            # Fresh directory: the validated repair transport makes new calls;
            # none of the earlier experiment's responses/receipts are reachable.
            mlestar.Controller = ResumeController
            result = mlestar.run(config)
        else:
            from .self_evolverec import run
            result = run(config)
    except BaseException:
        save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(folder / 'status.json', {'status': 'failed', 'time': time.time()})
        raise
    save(folder / 'status.json', {'status': 'complete', 'test_hr10': result['test']['score'],
                                 'finished_at': time.time()})


def launch(plan):
    check_files(plan)
    if (DESTINATION / 'launcher.json').exists():
        raise RuntimeError('Repetitions already launched; inspect their durable status')
    registrations = []
    for queue in plan['queues']:
        directory = DESTINATION / 'queues' / queue['id']
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'amazon_aligned.repeats', '--queue', queue['id']]
        with (directory / 'queue.log').open('a') as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        registrations.append({'queue': queue['id'], 'pid': process.pid, 'command': command})
    save(DESTINATION / 'launcher.json', {'queues': registrations, 'started_at': time.time()})
    time.sleep(2)
    print(json.dumps({'root': str(DESTINATION), 'launched': registrations}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--queue')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.prepare:
        return prepare()
    plan = read(DESTINATION / 'plan.json')
    if args.queue:
        return run_queue(plan, next(q for q in plan['queues'] if q['id'] == args.queue))
    if args.worker:
        return run_worker(plan, args.worker)
    return launch(plan)


if __name__ == '__main__':
    main()
