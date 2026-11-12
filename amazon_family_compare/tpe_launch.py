"""Preregister, verify and launch compute-matched traditional TPE experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import importlib.metadata
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
from .prepare import ROOT, WORKSPACE, REPO, sha
from .tpe import DESTINATION, RUNTIME, SPACE, SAMPLER, TEMPLATES, BASELINE, TPEController, source, optuna_module


def prepare(models, repetitions):
    if (DESTINATION / 'plan.json').exists():
        raise RuntimeError('Existing TPE plan; inspect instead of overwriting')
    optuna = optuna_module()
    queues = []
    original = read(ROOT / 'config.json')
    for group, gpus in enumerate([[0, 1, 2, 3], [4, 5, 6, 7]]):
        jobs = []
        for i in range(group, repetitions, 2):
            folder = DESTINATION / f'run-{i + 1:02d}'
            config = deepcopy(original)
            config.update(root=str(folder), methods=['tpe'], gpus=gpus,
                cpu_cores=list(range(gpus[0] * 16, (gpus[-1] + 1) * 16)),
                llm_model=None, llm_effort=None, single_model_search=False,
                hpo_models=models, sampler_seed=42 + i)
            config['alignment']['llm_reference_difference'] = 'No LLM: Optuna TPE proposes all configurations.'
            config['alignment']['llm_configuration'] = None
            config['alignment']['search_scope'] = 'Fixed SASRec hyperparameter optimization; shared settings across four datasets; no model-family search or code evolution.' if len(models) == 1 else 'Predefined model-family and hyperparameter optimization.'
            config['alignment']['method_adaptations']['tpe'] = {
                'implementation': 'Optuna TPESampler 4.6.0 with asynchronous ask/tell',
                'sampler': SAMPLER, 'sampler_seed': config['sampler_seed'],
                'common_baseline': 'Fresh, exact pinned sasrec_perdataset baseline; charged once and registered as trial 0.',
                'scope': 'fixed SASRec family, shared hyperparameters across four categories; one independent model per category' if len(models) == 1 else 'predefined model-family and hyperparameter search',
                'initialization': 'five completed initial observations including the common baseline before model-based TPE sampling',
                'pruning': 'NopPruner; preserve template early stopping and 200-epoch/200-batch ceilings',
                'failed_trials': 'charge actual GPU duration and tell Optuna FAIL; never fabricate objective values',
                'persistence': 'SQLite study, frozen proposals and checksummed sampler RNG state',
                'comparison_limit': 'fixed SASRec HPO with matched compute/data/evaluation; LLM methods can modify training/model code, so search spaces differ'}
            config['alignment']['method_adaptations'] = {'tpe': config['alignment']['method_adaptations']['tpe']}
            config['replication'] = {'repetition': i + 1, 'fresh_baseline_and_search': True,
                'training_seed': 42, 'sampler_seed': config['sampler_seed'],
                'search_budget_gpu_seconds': 43200,
                'reuse_policy': 'reuse verified input data and pinned software only; no previous models, search results or LLM feedback'}
            path = folder / 'config.json'
            save(path, config)
            jobs.append({'id': f'tpe-{i + 1:02d}', 'root': str(folder), 'config': str(path)})
        if jobs:
            queues.append({'id': 'gpu-' + '-'.join(map(str, gpus)), 'gpus': gpus,
                           'cpu_cores': list(range(gpus[0] * 16, (gpus[-1] + 1) * 16)), 'jobs': jobs})
    effective_space = {k: v for k, v in SPACE.items() if k != 'model' and
        (not any(k.startswith(name + '_') for name in TEMPLATES) or any(k.startswith(name + '_') for name in models))}
    if len(models) > 1:
        effective_space['model'] = models
    save(DESTINATION / 'plan.json', {'queues': queues, 'models': models, 'fixed_model': models[0] if len(models) == 1 else None, 'repetitions': repetitions,
        'algorithm': 'TPE', 'implementation': 'Optuna 4.6.0', 'sampler': SAMPLER,
        'search_space': effective_space, 'gpu_seconds_per_job': 43200,
        'total_new_gpu_seconds': repetitions * 43200, 'uses_llm': False,
        'selection': 'highest host macro validation HR@10, verified single model; freeze before held-out testing',
        'test_policy': 'same final test protocol, separate inference overhead; test results never feed back to TPE',
        'references': ['https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization',
                       'https://optuna.readthedocs.io/en/v4.6.0/reference/samplers/generated/optuna.samplers.TPESampler.html'],
        'created_at': time.time()})
    save(DESTINATION / 'environment.json', {'optuna': optuna.__version__,
        'isolated_dependencies': sorted((d.metadata['Name'], d.version)
                                       for d in importlib.metadata.distributions(path=[str(RUNTIME)]))})
    print(json.dumps({'prepared': True, 'models': models, 'repetitions': repetitions, 'root': str(DESTINATION)}))


def registered_files(plan):
    config = read(plan['queues'][0]['jobs'][0]['config'])
    result = manifest(config)
    paths = [DESTINATION / 'plan.json', DESTINATION / 'environment.json']
    paths += [Path(j['config']) for q in plan['queues'] for j in q['jobs']]
    for name in plan['models']:
        paths += list((REPO / 'gagc/templates' / TEMPLATES[name][0]).glob('*.py'))
    paths += [p for p in RUNTIME.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    result.update({str(p.relative_to(WORKSPACE)): sha(p) for p in paths})
    return result


def smoke_params(model):
    p = {**BASELINE, 'model': model, 'hidden': 32, 'maxlen': 32, 'batch': 64}
    for name in list(p):
        if name.startswith('sasrec_') and model != 'sasrec':
            p.pop(name)
    for key in ['layers'] if model == 'gru4rec' else ['blocks', 'kernel'] if model == 'nextitnet' else ['blocks', 'heads']:
        name = model + '_' + key
        p[name] = SPACE[name][0]
    return p


def preflight(phase, plan):
    directory = DESTINATION / 'preflight'
    directory.mkdir(exist_ok=True)
    if phase == 'tests':
        with (directory / 'tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'amazon_family_compare.test_tpe',
                'amazon_family_compare.test_accounting', '-v'], cwd=WORKSPACE,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        save(directory / 'tests.json', {'passed': True})
    elif phase == 'gpu':
        config = read(plan['queues'][0]['jobs'][0]['config'])
        c = TPEController({**config, 'root': str(directory)})
        results = []
        gpus = config['gpus']
        probes = [(plan['models'][0], g) for g in gpus] if len(plan['models']) == 1 else [
            (model, gpus[i % len(gpus)]) for i, model in enumerate(plan['models'])]
        for offset in range(0, len(probes), len(gpus)):
            batch = probes[offset:offset + len(gpus)]
            def one(pair):
                model, gpu = pair
                return c.trial('model-' + model + '-gpu' + str(gpu), source(smoke_params(model)), gpu,
                               requested=600, smoke=True)
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                results += list(pool.map(one, batch))
        if any(r['status'] != 'success' for r in results):
            save(directory / 'gpu-failure.json', {'receipts': results})
            raise RuntimeError('Template training/predictor preflight failed')
        audits = []
        for receipt in results:
            trial = Path(receipt['output'])
            output = trial / 'single-model-audit.json'
            with output.with_suffix('.log').open('w') as log:
                subprocess.run([config['worker_python'], '-m', 'amazon_family_compare.single_model',
                    '--audit', str(trial), '--data', config['data_dir'], '--output', str(output)],
                    cwd=WORKSPACE, env=c.environment(''), stdout=log, stderr=subprocess.STDOUT,
                    timeout=300, check=True)
            audit = read(output)
            if audit['eligible'] is not True:
                raise RuntimeError('A template failed single-model verification: ' + trial.name)
            audits.append({'id': receipt['id'], 'models': audit.get('model_classes')})
        save(directory / 'gpu.json', {'passed': True, 'receipts': results, 'single_model_audits': audits,
            'gpu_seconds_outside_search_budget': sum(r['allocated_gpu_seconds'] for r in results)})
    else:
        checks = {name: read(directory / (name + '.json'))['passed'] for name in ['tests', 'gpu']}
        if not all(checks.values()):
            raise RuntimeError('TPE preflight not passed')
        save(directory / 'acceptance.json', {'passed': True, 'checks': checks,
            'files': registered_files(plan), 'registered_at': time.time()})
    print('TPE_PREFLIGHT_' + phase.upper() + '_PASSED', flush=True)


def check(plan):
    if registered_files(plan) != read(DESTINATION / 'frozen.json')['files']:
        raise RuntimeError('Registered TPE source/config/data changed')


def queue_status(queue, **value):
    save(DESTINATION / 'queues' / queue['id'] / 'status.json', {**value, 'updated_at': time.time()})


def run_queue(queue, plan):
    directory = DESTINATION / 'queues' / queue['id']
    lock = (directory / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(queue['cpu_cores']))
    completed = []
    try:
        for job in queue['jobs']:
            check(plan)
            folder = Path(job['root']) / 'tpe'
            folder.mkdir(exist_ok=True)
            if (folder / 'events.jsonl').exists():
                raise RuntimeError('Fresh TPE job already contains execution evidence')
            while True:
                inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                devices = {int(row.split(',')[0]): [int(v.strip()) for v in row.split(',')[1:]]
                           for row in inventory.strip().splitlines()}
                if all(devices[g][0] < 512 and devices[g][1] < 5 for g in queue['gpus']):
                    break
                queue_status(queue, status='waiting_for_idle_gpus', active=job, devices=devices, completed=completed)
                time.sleep(15)
            queue_status(queue, status='running', active=job, completed=completed)
            command = [sys.executable, '-u', '-m', 'amazon_family_compare.tpe_launch', '--worker', job['config']]
            with (folder / 'controller.log').open('a') as log:
                process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                         'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '42'})
                save(folder / 'launcher.json', {'pid': process.pid, 'command': command, 'started_at': time.time()})
                code = process.wait()
            completed.append({**job, 'exit_code': code, 'status': 'complete' if code == 0 else 'failed'})
        queue_status(queue, status='complete' if all(j['exit_code'] == 0 for j in completed) else 'finished_with_failures', completed=completed)
    except BaseException:
        queue_status(queue, status='failed', error=traceback.format_exc(), completed=completed)
        raise


def worker(config_path, plan):
    check(plan)
    config = read(config_path)
    folder = Path(config['root']) / 'tpe'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(folder / 'status.json', {'status': 'running', 'gpus': config['gpus'], 'uses_llm': False,
                                'started_at': time.time()})
    try:
        from .tpe import run
        result = run(config)
    except BaseException:
        save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(folder / 'status.json', {'status': 'failed', 'time': time.time()})
        raise
    save(folder / 'status.json', {'status': 'complete', 'validation_hr10': result['selected']['score'],
                                  'test_hr10': result['test']['score'], 'finished_at': time.time()})


def launch(plan):
    if (DESTINATION / 'launcher.json').exists():
        raise RuntimeError('TPE already launched; inspect durable status')
    accepted = read(DESTINATION / 'preflight/acceptance.json')
    current = registered_files(plan)
    if not accepted['passed'] or current != accepted['files']:
        raise RuntimeError('Accepted code/config/data changed')
    save(DESTINATION / 'frozen.json', {'files': current, 'registered_at': time.time()})
    rows = []
    for queue in plan['queues']:
        directory = DESTINATION / 'queues' / queue['id']
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'amazon_family_compare.tpe_launch', '--queue', queue['id']]
        with (directory / 'queue.log').open('a') as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rows.append({'queue': queue['id'], 'pid': process.pid, 'command': command})
    save(DESTINATION / 'launcher.json', {'queues': rows, 'started_at': time.time()})
    time.sleep(2)
    print(json.dumps({'root': str(DESTINATION), 'launched': rows}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--models', choices=['multi', 'sasrec'], default='sasrec')
    parser.add_argument('--repetitions', type=int, choices=[1, 3], default=1)
    parser.add_argument('--preflight', choices=['tests', 'gpu', 'accept'])
    parser.add_argument('--queue')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.prepare:
        return prepare(list(TEMPLATES) if args.models == 'multi' else ['sasrec'], args.repetitions)
    plan = read(DESTINATION / 'plan.json')
    if args.preflight:
        return preflight(args.preflight, plan)
    if args.queue:
        return run_queue(next(q for q in plan['queues'] if q['id'] == args.queue), plan)
    if args.worker:
        return worker(args.worker, plan)
    return launch(plan)


if __name__ == '__main__':
    main()
