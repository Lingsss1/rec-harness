"""Freeze, preflight and detach the native RecHarness matched control."""
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
from .prepare import ROOT, REPO, WORKSPACE, sha
from .launch import manifest
from .recharness import RecHarnessController, native

CONFIG = ROOT / 'recharness/config.json'


def configure(path):
    if path.exists():
        return read(path)
    config = deepcopy(read(ROOT / 'config.json'))
    config['methods'] = ['recharness']
    config['alignment']['llm_reference_difference'] = (
        'RecHarness native GLM orchestrator and Claude Code implementation transport both replaced '
        'with the identical Codex gpt-6.1-sol/medium transport used by the other three methods.')
    config['alignment']['method_adaptations'] = {'recharness': {
        'implementation': 'native propose_action_group / execute_trial_group / promote_winner / update_thompson_state',
        'orchestration': 'fixed native documented workflow; LLM writes hypotheses and implementations; native tools choose all routes, promotions and memory updates',
        'execution': 'same frozen worker and validation/test grader as the three completed methods',
        'mutation_transport': 'native implementation prompt converted to find/replace edits for train.py including its predictor writer; three attempts per candidate',
        'budget_transport': 'exact summed worker durations; code generation excluded; shared remaining GPU seconds prorated before each group',
        'selection': 'native canonical incumbent after provisional jumps and rollbacks; held-out test evaluated once',
        'routing': 'Thompson enabled; native mutex, ceiling, stagnation gate, retune/backfill and SkillOpt memory',
        'basin_transfer_rho': 0.25,
        'search_space': 'native selected low-level edit scopes, preserving model family as required by native orchestrator',
        'seed': 42}}
    save(path, config)
    return config


def files(config, path):
    registered = manifest(config)
    registered[str(path.relative_to(WORKSPACE))] = sha(path)
    for p in (REPO / 'gagc').rglob('*.py'):
        if '__pycache__' not in p.parts:
            registered[str(p.relative_to(WORKSPACE))] = sha(p)
    return registered


def preflight(config, path):
    root = Path(config['root']) / 'recharness'
    test = deepcopy(config)
    test['root'] = str(root / 'preflight')
    c = RecHarnessController(test, smoke=True)
    c.setup()
    baseline_specs = json.loads(native.propose_action_group(c.store.state().to_json()))
    baseline = c.group(baseline_specs)
    if len(baseline) != 1 or not native._is_valid_trial(baseline[0]):
        raise RuntimeError('Real GPU baseline preflight failed')
    specs = json.loads(native.propose_action_group(c.store.state().to_json(), json.dumps(baseline[0])))
    if len(specs) != 4:
        raise RuntimeError('Fresh native search must supply four Thompson trials')
    results = c.group(specs, baseline[0])
    if not all(native._is_valid_trial(r) for r in results):
        raise RuntimeError('Native Codex / four-GPU / promotion / memory preflight failed')
    state = c.store.state()
    elapsed = sum(r['wall_time_secs'] for r in baseline + results)
    if abs(state.global_budget - (config['search_budget_gpu_seconds'] - elapsed)) > 1e-5:
        raise RuntimeError('Native summed-GPU accounting preflight failed')
    if state.round_idx != 2 or not state.baseline_done or not state.experiment_memory:
        raise RuntimeError('Native promotion/memory transition did not complete')
    save(root / 'preflight/acceptance.json', {'passed': True, 'config_sha256': sha(path),
         'real_baseline': baseline, 'real_mutations': results,
         'model': config['llm_model'], 'effort': config['llm_effort'],
         'gpu_seconds_outside_formal_budget': elapsed, 'state_round': state.round_idx,
         'native_experiment_memory_records': len(state.experiment_memory), 'time': time.time()})
    print('RECHARNESS_PREFLIGHT_PASSED', flush=True)


def worker(config, path):
    root = Path(config['root']) / 'recharness'
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registration = read(root / 'frozen.json')
    if files(config, path) != registration['files']:
        raise RuntimeError('Registered RecHarness code/config/data changed')
    if (root / 'events.jsonl').exists() or (root / 'summary.json').exists():
        raise RuntimeError('Existing formal RecHarness run; inspect before resuming')
    os.sched_setaffinity(0, set(config['cpu_cores']))
    while True:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True)
        devices = {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
                   for row in inventory.strip().splitlines()}
        if config.get('allow_gpu_sharing', False) or all(devices[g][0] < 512 and devices[g][1] < 5 for g in config['gpus']):
            break
        save(root / 'status.json', {'status': 'waiting_for_idle_gpus', 'devices': devices, 'time': time.time()})
        time.sleep(15)
    save(root / 'status.json', {'status': 'running', 'gpus': config['gpus'], 'started_at': time.time(),
                              'llm_model': config['llm_model'], 'llm_effort': config['llm_effort'],
                              'shared_gpus': config.get('allow_gpu_sharing', False),
                              'initial_gpu_inventory': devices})
    try:
        result = RecHarnessController(config).run()
    except BaseException:
        save(root / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(root / 'status.json', {'status': 'failed', 'time': time.time()})
        raise
    save(root / 'status.json', {'status': 'complete', 'test_score': result['test']['score'], 'time': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve()
    config = configure(path)
    logging.basicConfig(level=logging.INFO)
    if args.preflight:
        os.sched_setaffinity(0, set(config['cpu_cores']))
        return preflight(config, path)
    if args.worker:
        return worker(config, path)
    root = Path(config['root']) / 'recharness'
    if (root / 'launcher.json').exists():
        raise RuntimeError('Existing registered RecHarness launch')
    acceptance = read(root / 'preflight/acceptance.json')
    if not acceptance['passed'] or acceptance['config_sha256'] != sha(path):
        raise RuntimeError('The matched native preflight must pass first')
    save(root / 'frozen.json', {'files': files(config, path), 'registered_at': time.time(),
         'config_sha256': sha(path), 'preflight_sha256': sha(root / 'preflight/acceptance.json')})
    command = [sys.executable, '-u', '-m', 'amazon_aligned.recharness_launch', '--config', str(path), '--worker']
    with (root / 'controller.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=log,
             stderr=subprocess.STDOUT, start_new_session=True,
             env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    save(root / 'launcher.json', {'pid': proc.pid, 'command': command, 'started_at': time.time()})
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('RecHarness exited immediately; inspect controller.log')
    print(json.dumps({'pid': proc.pid, 'root': str(root)}))


if __name__ == '__main__':
    main()
