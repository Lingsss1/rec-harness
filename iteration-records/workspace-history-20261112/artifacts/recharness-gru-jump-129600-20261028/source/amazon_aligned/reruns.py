"""Three MLEvolve single-model searches and two repaired Self-EvolveRec searches."""
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

DESTINATION = WORKSPACE / 'artifacts/amazon-reruns-20261026'


def prepare():
    if (DESTINATION / 'plan.json').exists():
        raise RuntimeError('Rerun plan already exists; never overwrite a registered experiment')
    original = read(ROOT / 'config.json')
    queues = []
    for method, count, gpus, label in [
            ('self_evolverec', 2, [0, 1, 2, 3], 'self_evolverec-fixed'),
            ('mlevolve', 3, [4, 5, 6, 7], 'mlevolve-single')]:
        jobs = []
        for repetition in range(1, count + 1):
            config = deepcopy(original)
            run_root = DESTINATION / f'{label}-run-{repetition:02d}'
            config.update(root=str(run_root), methods=[method], gpus=gpus,
                          cpu_cores=list(range(gpus[0] * 16, (gpus[-1] + 1) * 16)),
                          single_model_search=method == 'mlevolve',
                          repair_diagnosis_contract=method == 'self_evolverec')
            config['replication'] = {
                'method': method, 'repetition': repetition, 'independent_gpu_seconds': 43200,
                'fresh_llm_calls_and_baseline': True,
                'reuse_policy': 'only pinned data, source and dependencies; no prior models, proposals, feedback or memories',
                'seeds': 'preprocessing=0; training/evaluation=42; native seeds unchanged; independent LLM generations',
                'single_model_final_selection': 'highest validation HR@10 among verified single-model predictors before held-out evaluation'}
            adaptations = config['alignment']['method_adaptations'][method]
            if method == 'mlevolve':
                adaptations.update(use_fusion=False, use_aggregation=False, max_fusion_drafts=0,
                    single_model_constraint='all generation/review calls and charged per-trial predictor audits; one independently trained scorer/category')
            else:
                adaptations.update(diagnosis_contract_fix='restore missing upstream diagnosis_interpreter_prompt before training; preserve evolved probe measurements',
                    diagnostic_payload_fix='retain headline measurements/definitions, bound nested raw examples before rendering prompts; archive original diagnostics',
                    conversation_fix='drop complete old exchanges above 700000 chars; preserve current code/diff',
                    candidate_feedback_errors='native debugging within the same ledger; no resetting budget',
                    single_model_scope='preserve native search; filter final candidates under the previously requested single-model policy')
            path = run_root / 'config.json'
            save(path, config)
            jobs.append({'id': run_root.name, 'method': method, 'repetition': repetition,
                         'root': str(run_root), 'config': str(path)})
        queues.append({'id': 'gpu-' + '-'.join(map(str, gpus)), 'gpus': gpus,
                       'cpu_cores': config['cpu_cores'], 'jobs': jobs})
    plan = {'root': str(DESTINATION), 'created_at': time.time(), 'queues': queues,
            'llm_model': original['llm_model'], 'llm_effort': original['llm_effort'],
            'gpu_seconds_per_job': 43200, 'total_new_gpu_seconds': 216000,
            'authorization': 'three fresh MLEvolve no-ensemble searches and two fresh repaired Self-EvolveRec searches',
            'historical_evidence': 'previous runs and their results remain intact; protocol-before archives the adapters before these changes'}
    save(DESTINATION / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'root': str(DESTINATION), 'jobs': 5}))


def registered_files(plan):
    files = manifest(read(plan['queues'][0]['jobs'][0]['config']))
    for path in [DESTINATION / 'plan.json', *[Path(j['config']) for q in plan['queues'] for j in q['jobs']]]:
        files[str(path.relative_to(WORKSPACE))] = sha(path)
    return files


def check_files(plan):
    if registered_files(plan) != read(DESTINATION / 'frozen.json')['files']:
        raise RuntimeError('Registered source/config/data changed during rerun queue')


def queue_status(queue, **value):
    save(DESTINATION / 'queues' / queue['id'] / 'status.json', {**value, 'updated_at': time.time()})


def run_worker(plan, config_path):
    check_files(plan)
    job = next(j for q in plan['queues'] for j in q['jobs'] if Path(j['config']) == config_path.resolve())
    config = read(config_path)
    method = job['method']
    folder = Path(job['root']) / method
    folder.mkdir(exist_ok=True)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((folder / name).exists() for name in ['events.jsonl', 'summary.json', 'failure.json']):
        raise RuntimeError('Fresh run cannot reuse an executed directory')
    save(folder / 'status.json', {'status': 'running', **job, 'gpus': config['gpus'], 'started_at': time.time()})
    try:
        from .rerun_controller import SingleSelectionMixin, FreshSelfController
        if method == 'mlevolve':
            from . import mlevolve
            class FreshMLEController(SingleSelectionMixin, mlevolve.NativeController):
                pass
            mlevolve.NativeController = FreshMLEController
            result = mlevolve.run(config)
        else:
            from . import self_evolverec
            self_evolverec.Controller = FreshSelfController
            result = self_evolverec.run(config)
    except BaseException:
        save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(folder / 'status.json', {'status': 'failed', 'time': time.time()})
        raise
    save(folder / 'status.json', {'status': 'complete', 'test_hr10': result['test']['score'],
                                 'finished_at': time.time()})


def run_queue(plan, queue):
    directory = DESTINATION / 'queues' / queue['id']
    lock = (directory / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(queue['cpu_cores']))
    completed = []
    try:
        for job in queue['jobs']:
            check_files(plan)
            folder = Path(job['root']) / job['method']
            folder.mkdir(exist_ok=True)
            if any((folder / name).exists() for name in ['events.jsonl', 'summary.json', 'failure.json']):
                raise RuntimeError('Fresh job already has execution evidence: ' + str(folder))
            while True:
                inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                      '--format=csv,noheader,nounits'], text=True)
                devices = {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
                           for row in inventory.strip().splitlines()}
                if all(devices[g][0] < 512 and devices[g][1] < 5 for g in queue['gpus']):
                    break
                queue_status(queue, status='waiting_for_idle_gpus', active=job, completed=completed, devices=devices)
                time.sleep(15)
            queue_status(queue, status='running', active=job, completed=completed, initial_gpu_inventory=devices)
            command = [read(job['config'])['controller_python'], '-u', '-m', 'amazon_aligned.reruns', '--worker', job['config']]
            with (folder / 'controller.log').open('a') as log:
                process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                         'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '42'})
                save(folder / 'launcher.json', {'pid': process.pid, 'command': command, 'started_at': time.time()})
                code = process.wait()
            completed.append({**job, 'status': 'complete' if code == 0 else 'failed', 'exit_code': code})
        queue_status(queue, status='complete' if all(j['exit_code'] == 0 for j in completed)
                     else 'finished_with_failures', completed=completed)
    except BaseException:
        queue_status(queue, status='failed', completed=completed, error=traceback.format_exc())
        raise


def launch(plan):
    if (DESTINATION / 'launcher.json').exists():
        raise RuntimeError('Rerun queues already launched')
    preflight = read(DESTINATION / 'preflight/acceptance.json')
    if not preflight['passed'] or preflight['plan_sha256'] != sha(DESTINATION / 'plan.json'):
        raise RuntimeError('Fresh protocol preflight required')
    current = registered_files(plan)
    if current != preflight['files']:
        raise RuntimeError('Source changed after preflight')
    save(DESTINATION / 'frozen.json', {'files': current, 'registered_at': time.time(),
                                      'acceptance_sha256': sha(DESTINATION / 'preflight/acceptance.json')})
    registrations = []
    for queue in plan['queues']:
        directory = DESTINATION / 'queues' / queue['id']
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'amazon_aligned.reruns', '--queue', queue['id']]
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
