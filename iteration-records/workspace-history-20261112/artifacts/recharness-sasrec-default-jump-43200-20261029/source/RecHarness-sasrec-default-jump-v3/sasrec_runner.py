"""One fresh 43,200 GPU-second SASRec eight-direction jump search.

Only this fork and a new results root are written. The existing TPE experiment
and original RecHarness stay unchanged. Queues own four GPUs per active run.
"""
import argparse
import ast
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
from amazon_aligned.core import schema, TEXT
from gagc import semantic_arms

DESTINATION = WORKSPACE / 'artifacts/recharness-sasrec-default-jump-43200-20261029'
TPE = WORKSPACE / 'artifacts/amazon-tpe-sasrec-20261028/run-01/tpe'

# This binding is process-local. The shared transport module on disk is frozen
# by the running TPE experiment and must not be edited.
transport.TASK = transport.TASK.replace(
    'Start from RecHarness sasrec_perdataset.',
    'Start from the original RecHarness SASRec default cold-start program and hyperparameters. Preserve the SASRec model family.')


class SASRecJumpController(JumpController):
    llm_slot = transport.ResilientSelfController.llm_slot
    run_attempt = transport.ResilientSelfController.run_attempt

    def __init__(self, config, smoke=False):
        super().__init__(config, smoke=smoke)
        self.scope = contextvars.ContextVar('sasrec_request_scope', default='baseline')
        prior = self.c.get('prior_protocol_charge', {})
        if prior.get('gpu_seconds', 0) and not smoke:
            if sha(Path(prior['ledger'])) != prior['ledger_sha256']:
                raise RuntimeError('The prior protocol ledger changed')
            key = 'prior-protocol-charge'
            with self.ledger.transaction() as state:
                if key not in state['settled']:
                    if state['spent_gpu_seconds'] or state['reservations'] or state['settled']:
                        raise RuntimeError('Prior charge must be imported before new GPU work')
                    state['spent_gpu_seconds'] = prior['gpu_seconds']
                    state['settled'][key] = {'allocated_gpu_seconds': prior['gpu_seconds'],
                        'reservation': prior['gpu_seconds'], 'overrun_seconds': 0.,
                        'source_ledger': prior['ledger'], 'reason': 'user changed arm design during baseline'}

    @contextmanager
    def request_scope(self, value):
        token = self.scope.set(value)
        try:
            yield
        finally:
            self.scope.reset(token)

    def llm(self, instruction, output_schema=None, search=False, key=None):
        instruction = ('COLD START AND SEARCH FAMILY: SASRec. Keep the SASRec Transformer '
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
        path = Path(self.c['base_model']['source'])
        if sha(path) != self.c['base_model']['sha256']:
            raise RuntimeError('Registered SASRec baseline changed')
        return path.read_text()

    def setup(self):
        super().setup()
        os.environ['GAGC_ARM_DESIGN'] = 'semantic_v3'
        os.environ['GAGC_COLD_START'] = 'perdataset'
        state = self.store.state()
        state.global_budget = self.ledger.available()
        state.initial_budget_secs = self.c['search_budget_gpu_seconds']
        state.experiment_skill = ('Use only the eight registered edit directions under four parent categories. '
            'Regular candidates change one direction. A jump jointly changes exactly two directions from different parent categories '
            'inside one SASRec model, with four local retune groups and safe rollback. '
            'No additional regular arm, independent model fusion, external data, or held-out feedback.')
        self.store.put(('gagc', 'thompson'), '/state.json', {'content': state.to_json()})
        (self.workspace / 'predict.py').write_text(
            (FORK / 'gagc/templates/perdataset/predict.py').read_text())

    def hypotheses(self, specs, diagnostics):
        reply = self.llm('Write one concise hypothesis for EACH supplied slot, under 800 characters. '
            'The selected category scopes are mandatory. Slots assigned the same category need '
            'different concrete edits. A jump must explain why its two changes work together. '
            'Return hypotheses only; routing and code execution belong to the host.\n' +
            json.dumps({'selected': specs, 'diagnostics': diagnostics,
                        'state': json.loads(self.store.state().to_json())}),
            schema({str(i): TEXT for i in range(len(specs))}),
            key=f'hypotheses-{self.store.state().round_idx:04d}')
        if any(len(value) > 800 for value in reply.values()):
            raise ValueError('Native hypothesis limit exceeded')
        return {spec.get('hypothesis_key', f'slot-{i}'): reply[str(i)] for i, spec in enumerate(specs)}

    def implement(self, source, spec, feedback, attempt, key):
        categories = semantic_arms.categories(spec['arm'], spec.get('selected_categories'))
        if not categories or categories != spec.get('selected_categories'):
            raise ValueError('Unregistered semantic mutation scope')
        category_enum = {'type': 'string', 'enum': list(semantic_arms.REGULAR)}
        grouped = schema({'category': category_enum, 'rationale': TEXT,
            'edits': {'type': 'array', 'items': schema({'find': TEXT, 'replace': TEXT})}})
        instruction = ('Implement the supplied semantic-arm mutation using exact find/replace edits '
            'to train.py, including its embedded predictor writer when necessary. Return one '
            'category_edits entry per selected category, and no others. Each entry must make a '
            'substantive change, not a comment or an empty edit. For a jump, BOTH categories '
            'must change coherently in this one jointly trained SASRec model. Shared shape/checkpoint '
            'plumbing may be included under its originating category. Preserve the supplied data '
            'splits, causal histories, seed 42, evaluator, per-dataset routing and SASRec Transformer backbone. '
            'Keep epoch cap 200. Do not inspect files or use shell tools. Do not introduce independent '
            'scorers, ensembles, imported weights, or extra datasets.\nSCOPE:\n' +
            semantic_arms.hint(spec['arm'], categories) + '\nSPEC:\n' + json.dumps(spec) +
            '\nREPAIR FEEDBACK:\n' + feedback + '\nATTEMPT: ' + str(attempt) +
            '\nCURRENT TRAIN.PY:\n' + source)
        reply = self.llm(instruction, schema({'category_edits': {'type': 'array', 'items': grouped},
                                            'change_summary': TEXT}), key=key)
        entries = reply['category_edits']
        actual = [entry['category'] for entry in entries]
        if len(actual) != len(categories) or set(actual) != set(categories):
            raise ValueError('Edits must cover exactly the selected category set')
        changed = source
        flattened = []
        for entry in entries:
            if not entry['edits'] or not entry['rationale'].strip():
                raise ValueError('Each selected category needs real edits and a rationale')
            before = changed
            changed, error = native._apply_edits(changed, entry['edits'])
            if error:
                raise ValueError(error)
            if changed == before:
                raise ValueError('No-op category edit: ' + entry['category'])
            # Require executable syntax changes for each declared category.
            # This does not claim to prove semantic scope: the source and
            # category rationale are retained for inspection.
            if ast.dump(ast.parse(changed)) == ast.dump(ast.parse(before)):
                raise ValueError('Comment-only category edit: ' + entry['category'])
            flattened.extend(entry['edits'])
        return changed, {**reply, 'edits': flattened, 'scope_validation': 'exact categories and nontrivial AST changes; semantic labels prompt-enforced'}

    def environment(self, gpu):
        result = super().environment(gpu)
        if self.smoke:
            result.update(SASREC_EPOCHS='5', SASREC_BATCHES='2', SASREC_BATCH='32')
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
    original = WORKSPACE / 'artifacts/RecHarness-6lyc/gagc/templates/perdataset/train.py'
    baseline = DESTINATION / 'baseline/train.py'
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(original.read_bytes())
    baseline_info = {'source': str(baseline), 'sha256': sha(baseline),
        'original_source': str(original), 'original_sha256': sha(original),
        'origin': 'original RecHarness default SASRec cold start, without TPE parameter bindings',
        'weights': 'fresh initialization, no checkpoint reuse',
        'defaults': {'hidden': 64, 'blocks': 2, 'heads': 1, 'maxlen': 128,
            'batch': 128, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 0.0,
            'epochs': 200, 'batches': 200, 'patience': 20},
        'epochs_note': 'matched host sets SASREC_EPOCHS=200, overriding the standalone template default 300'}
    save(DESTINATION / 'baseline/provenance.json', baseline_info)
    original_config = read(WORKSPACE / 'artifacts/recharness-gru-jump-129600-20261028/run-01/config.json')
    jobs = []
    for number, gpus in [(1, [0, 1, 2, 3])]:
        folder = DESTINATION / f'run-{number:02d}'
        config = deepcopy(original_config)
        config.update(root=str(folder / 'experiment'), gpus=gpus,
            cpu_cores=list(range(16*gpus[0], 16*(gpus[-1]+1))),
            methods=['recharness'], search_budget_gpu_seconds=43200,
            single_model_search=True, base_model=baseline_info,
            llm_model='gpt-6.1-sol', llm_effort='medium', llm_timeout=1200,
            llm_attempts=3, retry_backoff_seconds=[5, 15],
            llm_slot_directory=str(DESTINATION / 'llm-slots'), global_llm_concurrency=4)
        config['prior_protocol_charge'] = {'gpu_seconds': 0.,
            'reason': 'Fresh default-cold-start SASRec search requested with 43200 GPU seconds; historical experiments archived separately'}
        config['arm_policy'] = semantic_arms.policy()
        config.pop('gru_baseline', None)
        config['alignment']['cold_start'] = 'sasrec_perdataset'
        config['llm_slot_directory'] = original_config['llm_slot_directory']
        config['alignment']['search_scope'] = 'eight SASRec edit directions under four parent categories; cross-category two-direction combination jumps; single-model candidates only'
        config['alignment']['method_adaptations'] = {'recharness': {
            'implementation': 'isolated native RecHarness jump-v3 semantic-arm fork with aligned GPU executor',
            'jump_policy': read(FORK / 'JUMP_POLICY.json'), 'jump_enabled': True,
            'baseline': baseline_info,
            'transport': 'same gpt-6.1-sol/medium; three retries then cooldown at the same pending request; process-group cleanup',
            'selection': 'native incumbent after safe jump finalization; validation only; final held-out test once',
            'single_model': 'independent prediction fusion disallowed; audit is charged within each GPU trial'}}
        config['replication'] = {'repetition': number, 'independent_search': True,
            'fresh_baseline_weights': True, 'cross_run_response_or_memory_reuse': False,
            'training_seed': 42, 'routing_seed': 42, 'llm_sampling': 'independent calls per run'}
        config['jump_fork'] = {'source': str(FORK), 'version': 'jump-v3-semantic', 'enabled': True}
        save(folder / 'config.json', config)
        jobs.append({'id': number, 'config': str(folder / 'config.json'), 'root': str(folder), 'gpus': gpus})
        save(folder / 'status.json', {'status': 'queued', 'gpus': gpus, 'budget_gpu_seconds': 43200})
    save(DESTINATION / 'plan.json', {'jobs': jobs, 'created_at': time.time(),
        'run_count': 1, 'gpu_seconds_per_run': 43200, 'total_gpu_seconds': 43200,
        'queues': [{'gpus': [0, 1, 2, 3], 'jobs': [1]}],
        'llm_model': 'gpt-6.1-sol', 'llm_effort': 'medium',
        'preflight': 'short GPU and LLM integration check; separate overhead, not search evidence',
        'comparison': 'Original default SASRec cold start; same eight-direction jump protocol and 43200 GPU seconds as the completed SASRec TPE-start follow-up'})
    save(DESTINATION / 'arm-policy.json', semantic_arms.policy())


def files():
    paths = list(FORK.rglob('*.py')) + list((WORKSPACE / 'amazon_aligned').glob('*.py'))
    paths += [FORK / 'JUMP_POLICY.json', DESTINATION / 'arm-policy.json', DESTINATION / 'plan.json',
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
        raise RuntimeError('Preflight requires GPUs 0-3 to be idle')
    config.update(root=str(folder), llm_timeout=180)
    controller = SASRecJumpController(config, smoke=True)
    controller.setup()
    baseline_specs = json.loads(native.propose_action_group(controller.store.state().to_json()))
    baseline = controller.group(baseline_specs)
    if len(baseline) != 1 or not native._is_valid_trial(baseline[0]):
        raise RuntimeError('SASRec GPU baseline preflight failed')
    specs = json.loads(native.propose_action_group(controller.store.state().to_json(), json.dumps(baseline[0])))
    results = controller.group(specs, baseline[0])
    if len(results) != 4 or not all(native._is_valid_trial(r) for r in results):
        raise RuntimeError('SASRec four-GPU / native LLM / audit / promotion preflight failed')
    # A synthetic plateau opens the real gate for integration testing only.
    # This smoke state, proposals and scores are never reused by formal search.
    state = controller.store.state()
    state.basin_score_history[state.current_basin_id] = [state.current_basin_score]*3
    controller.store.put(('gagc', 'thompson'), '/state.json', {'content': state.to_json()})
    jump_specs = json.loads(native.propose_action_group(state.to_json()))
    if len(jump_specs) != 1 or not jump_specs[0].get('is_jumping'):
        raise RuntimeError('Synthetic plateau did not open the budget-safe integration jump')
    jump_results = controller.group(jump_specs, max(results, key=lambda r: r['val_score']))
    if len(jump_results) != 1 or not native._is_valid_trial(jump_results[0]):
        raise RuntimeError('Two-direction SASRec combination-jump integration failed')
    total = sum(r['wall_time_secs'] for r in baseline + results + jump_results)
    state = controller.store.state()
    if abs(state.global_budget - (config['search_budget_gpu_seconds'] - total)) > 1e-5:
        raise RuntimeError('Native and charged-duration preflight accounting diverged')
    finalization = json.loads(native.finalize_jump_search('preflight'))
    if not finalization['ok']:
        raise RuntimeError('Native jump finalization preflight failed')
    save(folder / 'acceptance.json', {'passed': True, 'baseline': baseline, 'mutations': results,
        'combination_jump': jump_results, 'synthetic_plateau_for_integration_only': True,
        'jump_finalization': finalization,
        'gpu_seconds_outside_formal_budget': total, 'time': time.time(),
        'llm_model': config['llm_model'], 'effort': config['llm_effort'],
        'checks': ['fork regression tests', 'original default SASRec source and cold-start defaults', 'six real GPU trials',
                   'four regular-direction edits and a real two-direction jump', 'single-model audits',
                   'native accounting/promotion/memory', 'safe jump finalization']})


def validate_frozen():
    if files() != read(DESTINATION / 'frozen.json')['files']:
        raise RuntimeError('Registered SASRec experiment source/config/data manifest changed')


def job_worker(number):
    validate_frozen()
    folder = DESTINATION / f'run-{number:02d}'
    config = read(folder / 'config.json')
    if Path(config['root']).exists():
        raise RuntimeError('Formal output already exists; never restart with a fresh budget')
    os.sched_setaffinity(0, set(config['cpu_cores']))
    stable = 0
    while stable < 3:
        stable = stable + 1 if idle(inventory(), config['gpus']) else 0
        if stable < 3:
            save(folder / 'status.json', {'status': 'waiting_for_stable_idle_gpus', 'time': time.time()})
            time.sleep(3)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['GAGC_JUMP_ENABLED'] = 'true'
    os.environ['GAGC_ARM_DESIGN'] = 'semantic_v3'
    save(folder / 'status.json', {'status': 'running', 'gpus': config['gpus'],
        'started_at': time.time(), 'pid': os.getpid(), 'budget_gpu_seconds': config['search_budget_gpu_seconds'],
        'llm_model': config['llm_model'], 'effort': config['llm_effort']})
    try:
        result = SASRecJumpController(config).run()
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
        for index in range(len(read(DESTINATION / 'plan.json')['queues'])):
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
