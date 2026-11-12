"""Three fresh GRU4Rec jump-v2 runs, using the pinned aligned executor.

Only this fork and a new results root are written. The existing TPE experiment
and original RecHarness stay unchanged. Queues own four GPUs per active run.
"""
import argparse
from contextlib import contextmanager
import contextvars
from copy import deepcopy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from matched_runner import FORK, WORKSPACE, JumpController, native, read, save
from amazon_aligned import resilient_self as transport
from amazon_aligned.protocol_fixes import SINGLE_MODEL_CONTRACT
from amazon_aligned.prepare import sha

DESTINATION = WORKSPACE / 'artifacts/recharness-gru-jump-20261028'
TPE = WORKSPACE / 'artifacts/amazon-tpe-sasrec-20261028/run-01/tpe'

# This binding is process-local. The shared transport module on disk is frozen
# by the running TPE experiment and must not be edited.
transport.TASK = transport.TASK.replace(
    'Start from RecHarness sasrec_perdataset.',
    'Start from RecHarness gru4rec_perdataset. Preserve the GRU recurrent model family.')


class GRUJumpController(JumpController):
    llm_slot = transport.ResilientSelfController.llm_slot
    run_attempt = transport.ResilientSelfController.run_attempt

    def __init__(self, config, smoke=False):
        super().__init__(config, smoke=smoke)
        self.scope = contextvars.ContextVar('gru_request_scope', default='baseline')

    @contextmanager
    def request_scope(self, value):
        token = self.scope.set(value)
        try:
            yield
        finally:
            self.scope.reset(token)

    def llm(self, instruction, output_schema=None, search=False, key=None):
        instruction = ('COLD START AND SEARCH FAMILY: GRU4Rec. Keep the GRU recurrent '
            'backbone; any architecture jump edits within that family.\n' +
            SINGLE_MODEL_CONTRACT + '\n' + instruction)
        with self.request_scope(key or 'unspecified'):
            while True:
                try:
                    return transport.ResilientSelfController.llm(
                        self, instruction, output_schema, search, key)
                except transport.TransportUnavailable as exc:
                    # Retry the same pending request, in this still-running native
                    # step. No budget reset, new selection or retraining occurs.
                    self.event('llm_cooldown', key=key, error=str(exc), seconds=60)
                    if self.smoke:
                        raise
                    time.sleep(60)

    def baseline(self):
        path = Path(self.c['gru_baseline']['source'])
        if sha(path) != self.c['gru_baseline']['sha256']:
            raise RuntimeError('Registered GRU baseline changed')
        return path.read_text()

    def setup(self):
        super().setup()
        os.environ['GAGC_COLD_START'] = 'gru4rec_perdataset'
        (self.workspace / 'predict.py').write_text(
            (FORK / 'gagc/templates/gru4rec_perdataset/predict.py').read_text())

    def environment(self, gpu):
        result = super().environment(gpu)
        if self.smoke:
            result.update(GRU4REC_EPOCHS='5', GRU4REC_BATCHES='2', GRU4REC_BATCH='32')
        return result

    def finish(self):
        # Preserve native jump promotion/rollback selection. Every admitted
        # candidate was already audited inside its charged GPU trial.
        trial = Path(self.best['output'])
        audit = read(trial / 'single-model-audit.json')
        if audit.get('eligible') is not True:
            raise RuntimeError('Native incumbent did not pass the single-model audit')
        if sha(trial / 'working/predict.py') != audit['predictor_sha256']:
            raise RuntimeError('Selected predictor changed after its audit')
        if any(sha(Path(p)) != expected for p, expected in audit['checkpoint_sha256'].items()):
            raise RuntimeError('Selected checkpoint changed after its audit')
        save(self.root / 'single-model-selection.json', {
            'policy': 'native incumbent after jump resolution; verified single model',
            'selected': self.brief(self.best), 'audit': audit})
        return super().finish()


def inventory():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return {int(row.split(',')[0]): [int(v.strip()) for v in row.split(',')[1:]]
            for row in output.strip().splitlines()}


def idle(devices, gpus):
    return all(devices[g][0] < 512 and devices[g][1] < 5 for g in gpus)


def prepare():
    if (DESTINATION / 'plan.json').exists():
        raise RuntimeError('Existing plan must not be overwritten')
    original = FORK / 'gagc/templates/gru4rec_perdataset/train.py'
    source = original.read_text()
    before = ('        lengths = (seq != 0).sum(dim=1).clamp(min=1) - 1\n'
              '        return out[torch.arange(seq.size(0), device=seq.device), lengths]')
    after = ('        # Histories are LEFT padded; count-1 indexes padding on short histories.\n'
             '        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)\n'
             '        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)\n'
             '        return out[torch.arange(seq.size(0), device=seq.device), last]')
    if source.count(before) != 1:
        raise RuntimeError('GRU padding correction no longer matches its original template')
    baseline = DESTINATION / 'baseline/train.py'
    baseline.parent.mkdir(parents=True)
    baseline.write_text(source.replace(before, after))
    baseline_info = {'source': str(baseline), 'sha256': sha(baseline),
        'original_source': str(original), 'original_sha256': sha(original),
        'correction': 'Select last nonpadding recurrent state for the left-padded GRU histories',
        'origin': 'upstream GRU4Rec template; no historical evolved candidate or trained weights',
        'defaults': {'maxlen': 128, 'hidden': 128, 'layers': 1, 'dropout': .1,
                     'batch': 128, 'batches': 200, 'epochs': 200, 'patience': 20}}
    save(DESTINATION / 'baseline/provenance.json', baseline_info)
    original_config = read(WORKSPACE / 'artifacts/amazon-aligned-20261022/recharness/config.json')
    jobs = []
    for number, gpus in [(1, [4, 5, 6, 7]), (2, [0, 1, 2, 3]), (3, [4, 5, 6, 7])]:
        folder = DESTINATION / f'run-{number:02d}'
        config = deepcopy(original_config)
        config.update(root=str(folder / 'experiment'), gpus=gpus,
            cpu_cores=list(range(16*gpus[0], 16*(gpus[-1]+1))),
            methods=['recharness'], search_budget_gpu_seconds=43200,
            single_model_search=True, gru_baseline=baseline_info,
            llm_model='gpt-6.1-sol', llm_effort='medium', llm_timeout=1200,
            llm_attempts=3, retry_backoff_seconds=[5, 15],
            llm_slot_directory=str(DESTINATION / 'llm-slots'), global_llm_concurrency=4)
        config['alignment']['cold_start'] = 'gru4rec_perdataset'
        config['alignment']['search_scope'] = 'GRU4Rec model-family code evolution with jump-v2; single-model candidates only'
        config['alignment']['method_adaptations'] = {'recharness': {
            'implementation': 'isolated native RecHarness jump-v2 fork with aligned GPU executor',
            'jump_policy': read(FORK / 'JUMP_POLICY.json'), 'jump_enabled': True,
            'baseline': baseline_info,
            'transport': 'same gpt-6.1-sol/medium; three retries then cooldown at the same pending request; process-group cleanup',
            'selection': 'native incumbent after safe jump finalization; validation only; final held-out test once',
            'single_model': 'independent prediction fusion disallowed; audit is charged within each GPU trial'}}
        config['replication'] = {'repetition': number, 'independent_search': True,
            'fresh_baseline_weights': True, 'cross_run_response_or_memory_reuse': False,
            'training_seed': 42, 'routing_seed': 42, 'llm_sampling': 'independent calls per run'}
        config['jump_fork'] = {'source': str(FORK), 'version': 'jump-v2', 'enabled': True}
        save(folder / 'config.json', config)
        jobs.append({'id': number, 'config': str(folder / 'config.json'), 'root': str(folder), 'gpus': gpus})
        save(folder / 'status.json', {'status': 'queued', 'gpus': gpus, 'budget_gpu_seconds': 43200})
    save(DESTINATION / 'plan.json', {'jobs': jobs, 'created_at': time.time(),
        'gpu_seconds_per_run': 43200, 'total_gpu_seconds': 129600,
        'queues': [{'gpus': [4, 5, 6, 7], 'jobs': [1, 3]},
                   {'gpus': [0, 1, 2, 3], 'jobs': [2], 'wait_for': str(TPE)}],
        'llm_model': 'gpt-6.1-sol', 'llm_effort': 'medium',
        'preflight': 'short GPU and LLM integration check; separate overhead, not search evidence',
        'comparison': 'new GRU cold start; older SASRec cold-start runs are not an identical-start ablation'})


def files():
    paths = list(FORK.rglob('*.py')) + list((WORKSPACE / 'amazon_aligned').glob('*.py'))
    paths += [FORK / 'JUMP_POLICY.json', DESTINATION / 'plan.json',
              DESTINATION / 'baseline/train.py', DESTINATION / 'baseline/provenance.json']
    paths += list(DESTINATION.glob('run-*/config.json'))
    config = read(DESTINATION / 'run-01/config.json')
    paths += [Path(config['data_dir']) / 'prepared.json',
              WORKSPACE / 'artifacts/RecHarness-6lyc/gagc/benchmarks/amazon_reviews/harness.py',
              WORKSPACE / 'artifacts/RecHarness-6lyc/gagc/benchmarks/amazon_reviews/grader.py']
    return {str(p): sha(p) for p in paths if '__pycache__' not in p.parts}


def preflight():
    folder = DESTINATION / 'preflight'
    folder.mkdir(exist_ok=False)
    save(DESTINATION / 'status.json', {'status': 'preflight', 'started_at': time.time()})
    # CPU regression checks cover local climbing, success/rollback, budget caps,
    # incomplete windows and the corrected baseline's sequence indexing.
    with (folder / 'tests.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pytest', str(FORK / 'tests'), '-q',
                        '--junitxml', str(folder / 'tests.xml')],
            cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, check=True,
            env={**os.environ, 'PYTHONPATH': str(FORK) + os.pathsep + str(WORKSPACE)})
    config = deepcopy(read(DESTINATION / 'run-01/config.json'))
    os.sched_setaffinity(0, set(config['cpu_cores']))
    if not idle(inventory(), config['gpus']):
        raise RuntimeError('Preflight requires GPUs 4-7 to be idle')
    config.update(root=str(folder), llm_timeout=180)
    controller = GRUJumpController(config, smoke=True)
    controller.setup()
    baseline_specs = json.loads(native.propose_action_group(controller.store.state().to_json()))
    baseline = controller.group(baseline_specs)
    if len(baseline) != 1 or not native._is_valid_trial(baseline[0]):
        raise RuntimeError('GRU GPU baseline preflight failed')
    specs = json.loads(native.propose_action_group(controller.store.state().to_json(), json.dumps(baseline[0])))
    results = controller.group(specs, baseline[0])
    if len(results) != 4 or not all(native._is_valid_trial(r) for r in results):
        raise RuntimeError('GRU four-GPU / native LLM / audit / promotion preflight failed')
    total = sum(r['wall_time_secs'] for r in baseline + results)
    state = controller.store.state()
    if abs(state.global_budget - (43200 - total)) > 1e-5:
        raise RuntimeError('Native and charged-duration preflight accounting diverged')
    finalization = json.loads(native.finalize_jump_search('preflight'))
    if not finalization['ok']:
        raise RuntimeError('Native jump finalization preflight failed')
    save(folder / 'acceptance.json', {'passed': True, 'baseline': baseline, 'mutations': results,
        'gpu_seconds_outside_formal_budget': total, 'time': time.time(),
        'llm_model': config['llm_model'], 'effort': config['llm_effort'],
        'checks': ['fork regression tests', 'GRU left-padding index', 'five real GPU trials',
                   'four independent LLM edits', 'single-model audits', 'native accounting/promotion/memory']})


def validate_frozen():
    if files() != read(DESTINATION / 'frozen.json')['files']:
        raise RuntimeError('Registered GRU experiment source/config/data manifest changed')


def job_worker(number):
    validate_frozen()
    folder = DESTINATION / f'run-{number:02d}'
    config = read(folder / 'config.json')
    if Path(config['root']).exists():
        raise RuntimeError('Formal output already exists; never restart with a fresh budget')
    os.sched_setaffinity(0, set(config['cpu_cores']))
    if not idle(inventory(), config['gpus']):
        raise RuntimeError('Assigned GPUs are no longer idle')
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['GAGC_JUMP_ENABLED'] = 'true'
    save(folder / 'status.json', {'status': 'running', 'gpus': config['gpus'],
        'started_at': time.time(), 'pid': os.getpid(), 'budget_gpu_seconds': 43200,
        'llm_model': config['llm_model'], 'effort': config['llm_effort']})
    try:
        result = GRUJumpController(config).run()
    except BaseException:
        save(folder / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(folder / 'status.json', {'status': 'failed', 'updated_at': time.time()})
        raise
    save(folder / 'status.json', {'status': 'complete', 'time': time.time(),
        'test_hr10': result['test']['score'], 'validation_hr10': result['selected']['score'],
        'spent_gpu_seconds': result['budget']['spent_gpu_seconds']})


def queue_worker(index):
    plan = read(DESTINATION / 'plan.json')
    queue = plan['queues'][index]
    folder = DESTINATION / 'queues' / f'queue-{index}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = []
    for number in queue['jobs']:
        while True:
            # Do not capture idle gaps in another experiment's LLM phase.
            dependency_done = ('wait_for' not in queue or
                (Path(queue['wait_for']) / 'summary.json').exists() or
                (Path(queue['wait_for']) / 'failure.json').exists())
            devices = inventory()
            if dependency_done and idle(devices, queue['gpus']):
                break
            value = {'status': 'waiting_for_tpe' if not dependency_done else 'waiting_for_idle_gpus',
                     'next_run': number, 'completed': completed, 'updated_at': time.time()}
            save(folder / 'status.json', value)
            save(DESTINATION / f'run-{number:02d}/status.json', {**value, 'gpus': queue['gpus']})
            time.sleep(15)
        validate_frozen()
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--job', str(number)]
        with (DESTINATION / f'run-{number:02d}/controller.log').open('a') as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(folder / 'status.json', {'status': 'running', 'run': number, 'pid': process.pid,
                'completed': completed, 'updated_at': time.time()})
            code = process.wait()
        completed.append({'run': number, 'exit_code': code})
        if code:
            # Keep failed evidence, continue the other independent repetition.
            save(folder / f'run-{number:02d}-failure.json', {'exit_code': code, 'time': time.time()})
    save(folder / 'status.json', {'status': 'complete', 'completed': completed, 'time': time.time()})


def orchestrate():
    try:
        preflight()
        save(DESTINATION / 'frozen.json', {'files': files(), 'registered_at': time.time(),
            'preflight_acceptance_sha256': sha(DESTINATION / 'preflight/acceptance.json')})
        children = []
        for index in range(2):
            folder = DESTINATION / 'queues' / f'queue-{index}'
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--queue', str(index)]
            with (folder / 'supervisor.log').open('a') as log:
                process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append({'queue': index, 'pid': process.pid, 'command': command})
        save(DESTINATION / 'launcher.json', {'processes': children, 'started_at': time.time()})
        save(DESTINATION / 'status.json', {'status': 'launched', 'time': time.time(), 'processes': children})
    except BaseException:
        save(DESTINATION / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(DESTINATION / 'status.json', {'status': 'failed', 'time': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--orchestrate', action='store_true')
    parser.add_argument('--queue', type=int)
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.prepare:
        prepare()
    elif args.orchestrate:
        orchestrate()
    elif args.queue is not None:
        queue_worker(args.queue)
    elif args.job is not None:
        job_worker(args.job)
    elif args.start:
        if (DESTINATION / 'supervisor.json').exists():
            raise RuntimeError('Already registered; inspect existing processes')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--orchestrate']
        with (DESTINATION / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                     'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '42'})
        save(DESTINATION / 'supervisor.json', {'pid': process.pid, 'command': command, 'time': time.time()})
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError('Experiment supervisor exited; inspect supervisor.log')
        print(json.dumps({'root': str(DESTINATION), 'pid': process.pid, 'status': 'preflight_then_launch'}))
    else:
        parser.error('Use --prepare or --start')


if __name__ == '__main__':
    main()
