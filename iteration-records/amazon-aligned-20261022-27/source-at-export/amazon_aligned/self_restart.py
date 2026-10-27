"""Fresh two-run Self-EvolveRec plan with retrying, checkpointed controllers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from .resilient_self import ResilientSelfController, TransportUnavailable

DESTINATION = WORKSPACE / 'artifacts/self-evorec-resilient-20261027'


def prepare():
    if (DESTINATION / 'plan.json').exists():
        raise RuntimeError('Existing plan must not be overwritten')
    original = read(WORKSPACE / 'artifacts/amazon-reruns-20261026/self_evolverec-fixed-run-01/config.json')
    jobs = []
    for i, gpus in enumerate([[0, 1, 2, 3], [4, 5, 6, 7]], 1):
        config = deepcopy(original)
        folder = DESTINATION / f'run-{i:02d}'
        config.update(root=str(folder), gpus=gpus, cpu_cores=list(range(gpus[0] * 16, (gpus[-1] + 1) * 16)),
            resilient_transport=True, repair_diagnosis_contract=True, single_model_search=False,
            llm_slot_directory=str(DESTINATION / 'llm-slots'), global_llm_concurrency=4,
            llm_attempts=3, retry_backoff_seconds=[5, 15], transport_cooldown_seconds=60)
        config['replication'] = {'repetition': i, 'fresh_baseline_and_llm_calls': True,
            'reuse_policy': 'only pinned data, source and dependencies; no earlier search proposals, feedback or trained weights',
            'restart_policy': 'reuse exact requests, training receipts and native database checkpoints only within this new run',
            'search_gpu_seconds': 43200, 'llm_time_and_backoff': 'excluded, as in previous aligned experiments',
            'selection': 'highest validation HR@10 verified single model; freeze selection before held-out test'}
        config['alignment']['method_adaptations']['self_evolverec']['transport_repair'] = {
            'attempts': 3, 'attempt_timeout_seconds': 1200, 'backoff_seconds': [5, 15],
            'all_attempts_failed': 'cooldown 60 seconds, resume same pending native step',
            'concurrency': 'four active LLM subprocesses shared across both independent runs',
            'cache': 'successful exact requests within native trajectory/iteration or candidate feedback; no cross-run reuse',
            'checkpoints': 'commit native population/islands and iteration after each iteration; persist pending parent and inspirations',
            'process_cleanup': 'kill entire owned CLI process group on completion or timeout'}
        path = folder / 'config.json'
        save(path, config)
        jobs.append({'id': f'self-evolverec-resilient-{i:02d}', 'config': str(path),
                     'root': str(folder), 'gpus': gpus, 'cpu_cores': config['cpu_cores']})
    save(DESTINATION / 'plan.json', {'jobs': jobs, 'created_at': time.time(), 'gpu_seconds_per_job': 43200,
        'total_gpu_seconds': 86400, 'model': original['llm_model'], 'effort': original['llm_effort'],
        'scope': 'two fresh independent Self-EvolveRec reruns; historical failures retained'})
    print(json.dumps({'prepared': True, 'jobs': 2, 'root': str(DESTINATION)}))


def files(plan):
    result = manifest(read(plan['jobs'][0]['config']))
    for path in [DESTINATION / 'plan.json', *[Path(j['config']) for j in plan['jobs']]]:
        result[str(path.relative_to(WORKSPACE))] = sha(path)
    return result


def status(job, **value):
    save(Path(job['root']) / 'status.json', {**value, 'updated_at': time.time()})


def worker(job, plan):
    if files(plan) != read(DESTINATION / 'frozen.json')['files']:
        raise RuntimeError('Registered source/config/data changed')
    config = read(job['config'])
    folder = Path(job['root']) / 'self_evolverec'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(job['cpu_cores']))
    if (folder / 'summary.json').exists():
        status(job, status='complete', test_hr10=read(folder / 'summary.json')['test']['score'])
        return
    while True:
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
        inventory = {int(row.split(',')[0]): [int(v.strip()) for v in row.split(',')[1:]]
                     for row in devices.strip().splitlines()}
        if all(inventory[g][0] < 512 and inventory[g][1] < 5 for g in job['gpus']):
            break
        status(job, status='waiting_for_idle_gpus', inventory=inventory)
        time.sleep(15)
    from . import self_evolverec
    self_evolverec.Controller = ResilientSelfController
    while True:
        status(job, status='running', gpus=job['gpus'], model=config['llm_model'], effort=config['llm_effort'])
        try:
            result = self_evolverec.run(config)
        except TransportUnavailable as exc:
            status(job, status='waiting_for_llm', reason=str(exc), next_retry_seconds=60)
            time.sleep(60)
            continue
        except BaseException:
            save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
            status(job, status='failed', reason='non-transient exception; retained for review')
            raise
        status(job, status='complete', test_hr10=result['test']['score'], validation_hr10=result['selected']['score'])
        return


def preflight(phase, plan):
    directory = DESTINATION / 'preflight'
    directory.mkdir(exist_ok=True)
    if phase == 'tests':
        with (directory / 'tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'amazon_aligned.test_resilient_self',
                'amazon_aligned.test_protocol_fixes', 'amazon_aligned.test_accounting', '-v'],
                cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, check=True)
        save(directory / 'tests.json', {'passed': True, 'checks': 'fault injection, retry/cache, training reuse, native checkpoint/parent restore, slot cap and accounting'})
    elif phase == 'gpu':
        from .self_evolverec import UPSTREAM
        from .core import Controller
        diagnosis = (UPSTREAM / 'examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text()
        controllers = [Controller({**read(j['config']), 'root': str(directory / j['id'])}, 'gpu') for j in plan['jobs']]
        pairs = [(c, g) for c in controllers for g in c.c['gpus']]
        def one(pair):
            c, g = pair
            return c.trial(f'gpu-{g}', c.baseline(), g, requested=600,
                           files={'diagnosis_tools.py': diagnosis}, smoke=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = list(pool.map(one, pairs))
        if any(r['status'] != 'success' for r in receipts):
            raise RuntimeError('Eight-GPU training/diagnostic preflight failed')
        save(directory / 'gpu.json', {'passed': True, 'receipts': receipts,
            'gpu_seconds_outside_search': sum(r['allocated_gpu_seconds'] for r in receipts)})
    elif phase == 'llm':
        # Test each independent cache/transport, then a full 20-user baseline
        # feedback chain. A short probe timeout prevents a silent long wait.
        from .self_evolverec import UPSTREAM, feedback
        sys.path.insert(0, str(UPSTREAM))
        from .core import schema, TEXT
        rows = []
        for job in plan['jobs']:
            c = ResilientSelfController({**read(job['config']), 'root': str(directory / job['id']),
                'llm_timeout': 120, 'llm_attempts': 2}, 'transport')
            value = c.llm('Transport health check: return status=ready.',
                         schema({'status': {'type': 'string', 'enum': ['ready']}}))
            assert value['status'] == 'ready'
            rows.append({'id': job['id'], 'passed': True})
        job = plan['jobs'][0]
        c = ResilientSelfController({**read(job['config']), 'root': str(directory / 'feedback-chain'),
            'llm_timeout': 120, 'llm_attempts': 2}, 'self_evolverec')
        receipt = read(directory / job['id'] / 'gpu/smoke/gpu-0/receipt.json')
        result = feedback(c, receipt)
        assert 'simulator_comment' in result and 'diagnosis_comment' in result
        # Reusing a complete feedback artifact must require no further requests.
        count = c.call_index
        assert feedback(c, receipt) == result and c.call_index == count
        save(directory / 'llm.json', {'passed': True, 'health': rows, 'full_20_user_feedback_chain': True,
            'feedback_calls': count, 'cached_feedback_reused': True,
            'model': c.c['llm_model'], 'effort': c.c['llm_effort'], 'probe_timeout': 120, 'formal_timeout': 1200})
    else:
        checks = {name: read(directory / (name + '.json'))['passed'] for name in ['tests', 'gpu', 'llm']}
        if not all(checks.values()):
            raise RuntimeError('Preflight not passed')
        save(directory / 'acceptance.json', {'passed': True, 'checks': checks, 'files': files(plan),
                                            'plan_sha256': sha(DESTINATION / 'plan.json')})
    print('SELF_RESTART_PREFLIGHT_' + phase.upper() + '_PASSED', flush=True)


def launch(plan):
    if (DESTINATION / 'launcher.json').exists():
        raise RuntimeError('Jobs already registered; inspect status instead of duplicating')
    accepted = read(DESTINATION / 'preflight/acceptance.json')
    current = files(plan)
    if not accepted['passed'] or accepted['files'] != current:
        raise RuntimeError('Accepted code/config/data changed')
    save(DESTINATION / 'frozen.json', {'files': current, 'registered_at': time.time()})
    processes = []
    for job in plan['jobs']:
        command = [sys.executable, '-u', '-m', 'amazon_aligned.self_restart', '--worker', job['id']]
        with (Path(job['root']) / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '42'})
        processes.append({'id': job['id'], 'pid': process.pid, 'command': command})
    save(DESTINATION / 'launcher.json', {'processes': processes, 'started_at': time.time()})
    time.sleep(2)
    print(json.dumps({'root': str(DESTINATION), 'launched': processes}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', choices=['tests', 'gpu', 'llm', 'accept'])
    parser.add_argument('--worker')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.prepare:
        return prepare()
    plan = read(DESTINATION / 'plan.json')
    if args.preflight:
        return preflight(args.preflight, plan)
    if args.worker:
        return worker(next(j for j in plan['jobs'] if j['id'] == args.worker), plan)
    return launch(plan)


if __name__ == '__main__':
    main()
