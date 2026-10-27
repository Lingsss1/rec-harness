"""Post-search single-model eligibility audit and validation-only reselection."""
import argparse
import ast
from collections import defaultdict
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from .core import Controller, read, save
from .prepare import DATASETS, ROOT, WORKSPACE, sha

DESTINATION = WORKSPACE / 'artifacts/amazon-single-model-20261026'
REPEATS = WORKSPACE / 'artifacts/amazon-repeats-20261025'


def model_audit(trial, data):
    """Inspect the actual predictor, metadata and CPU checkpoint-loading path."""
    predictor = trial / 'working/predict.py'
    source = predictor.read_text()
    tree = ast.parse(source)
    names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    banned = {'rank_fusion', 'load_ensemble', 'fused_scores', 'score_members',
              'rankensemble', 'ensemblepredictionmodel'}
    multi = sorted(names & banned)
    if multi:
        return {'eligible': False, 'reason': 'multiple independently trained model scores are combined',
                'evidence_names': multi}
    if not (trial / 'train.py').exists():
        return {'eligible': False, 'reason': 'score ensemble of other trial predictors'}
    # Fixed transition/count scorers combined at the score level are excluded.
    # Jointly learned features/branches of one trained network are allowed.
    for path in sorted((trial / 'working').glob('*meta.pkl')):
        meta = pickle.loads(path.read_bytes())
        for key in ['graph_weight', 'transition_weight', 'transition_alpha']:
            value = meta.get(key, 0.)
            if isinstance(value, (int, float)) and value != 0:
                return {'eligible': False, 'reason': 'neural score fused with an independently fitted transition scorer',
                        'evidence': {'metadata': path.name, key: value}}
        if meta.get('residual_mode') in {'transition', 'popularity'} and 'transition_table' in meta:
            return {'eligible': False, 'reason': 'neural score fused with a separately fitted transition/popularity scorer',
                    'evidence': {'metadata': path.name, 'residual_mode': meta['residual_mode']}}
    import torch
    torch.set_num_threads(4)
    loaded = defaultdict(set)
    models = defaultdict(list)
    last_file = [None]
    original_load = torch.load
    original_state = torch.nn.Module.load_state_dict
    def dataset_of(path):
        matches = [name for name in DATASETS if name in str(path)]
        return matches[0] if len(matches) == 1 else 'unrouted'
    def load(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        last_file[0] = str(path)
        loaded[dataset_of(path)].add(str(Path(path).resolve()))
        return value
    def state(model, *args, **kwargs):
        models[dataset_of(last_file[0])].append({'class': type(model).__name__, 'identity': id(model)})
        return original_state(model, *args, **kwargs)
    torch.load = load
    torch.nn.Module.load_state_dict = state
    try:
        spec = importlib.util.spec_from_file_location('_single_model_predictor', predictor)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for dataset in DATASETS:
            with (data / 'trainval' / (dataset + '_train.txt')).open() as stream:
                first = stream.readline().split()
                user = int(first[0])
                history = [int(first[1])]
                for line in stream:
                    uid, item = map(int, line.split())
                    if uid != user:
                        break
                    history.append(item)
            values = module.predict(user, history, list(range(1, 11)), dataset_name=dataset)
            if len(values) != 10 or not torch.isfinite(torch.as_tensor(values)).all():
                raise ValueError('Predictor audit returned invalid scores')
    finally:
        torch.load = original_load
        torch.nn.Module.load_state_dict = original_state
    cached = defaultdict(list)
    visited = set()
    independent_fusions = []
    def inspect_cache(value, dataset='unrouted'):
        key = (id(value), dataset)
        if key in visited:
            return
        visited.add(key)
        if isinstance(value, torch.nn.Module):
            cached[dataset].append({'class': type(value).__name__, 'identity': id(value)})
            if getattr(value, 'transition_graph', None) is not None and getattr(value, 'graph_weight', 0.) != 0:
                independent_fusions.append({'dataset': dataset, 'kind': 'independent transition graph score'})
            if getattr(value, 'residual_mode', None) in {'transition', 'popularity'}:
                independent_fusions.append({'dataset': dataset, 'kind': 'independent count-based residual score'})
        elif isinstance(value, dict):
            for name, child in value.items():
                inspect_cache(child, name if name in DATASETS else dataset)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect_cache(child, dataset)
        elif hasattr(value, 'models') and isinstance(value.models, dict):
            inspect_cache(value.models, dataset)
    for name in ['_models', '_MODELS', 'models', '_cache', '_predictor']:
        if name in vars(module):
            inspect_cache(vars(module)[name])
    counts = {name: len(loaded[name]) for name in DATASETS}
    model_counts = {name: max(len({m['identity'] for m in models[name]}),
                              len({m['identity'] for m in cached[name]})) for name in DATASETS}
    result = {'checkpoint_files_per_dataset': counts, 'loaded_models_per_dataset': model_counts,
              'checkpoint_files': {name: sorted(loaded[name]) for name in DATASETS},
              'model_classes': {name: sorted({m['class'] for m in models[name] + cached[name]}) for name in DATASETS}}
    if independent_fusions:
        return {**result, 'eligible': False, 'reason': 'loaded neural model adds an independent fitted scorer',
                'evidence': independent_fusions}
    if loaded['unrouted'] or any(counts[name] == 0 for name in DATASETS):
        return {**result, 'eligible': None, 'reason': 'unrecognized checkpoint routing requires manual review'}
    if any(counts[name] > 1 or model_counts[name] > 1 for name in DATASETS):
        return {**result, 'eligible': False, 'reason': 'inference loads multiple trained checkpoints/models for a category'}
    if any('ensemble' in c.lower() for classes in result['model_classes'].values() for c in classes):
        return {**result, 'eligible': False, 'reason': 'loaded model class explicitly combines independent experts'}
    return {**result, 'eligible': True, 'reason': 'one trained model/checkpoint per dataset; no independent score fusion',
            'predictor_sha256': sha(predictor), 'training_source_sha256': sha(trial / 'train.py')}


def prepare():
    if (DESTINATION / 'policy.json').exists():
        raise RuntimeError('Existing single-model registration; inspect instead of overwriting')
    jobs = [{'id': method + '-01', 'method': method, 'run': str(ROOT / method),
             'config': str(ROOT / 'config.json')} for method in ['mlevolve', 'mlestar', 'self_evolverec']]
    jobs += [{'id': 'recharness-01', 'method': 'recharness', 'run': str(ROOT / 'recharness'),
              'config': str(ROOT / 'recharness/config.json')},
             {'id': 'recharness-02', 'method': 'recharness',
              'run': str(ROOT / 'recharness-replica-gpu0-3/recharness'),
              'config': str(ROOT / 'recharness-replica-gpu0-3/recharness/config.json')}]
    for method in ['mlestar', 'self_evolverec']:
        for repetition in [2, 3]:
            parent = REPEATS / f'{method}-run-{repetition:02d}'
            jobs.append({'id': f'{method}-{repetition:02d}', 'method': method,
                         'run': str(parent / method), 'config': str(parent / 'config.json')})
    policy = {'created_at': time.time(), 'jobs': jobs,
         'selection': 'highest existing host validation HR@10 among successful whole-program single-model candidates, including trained ablations; ties prefer the original selected candidate, then candidate id',
         'eligibility': 'one independent trained scorer per category; exclude model/seed/checkpoint score or rank ensembles and independently fitted graph/popularity score fusion; jointly trained internal features/layers remain eligible',
         'budget': 'preserve all consumed search GPU seconds including excluded ensembles; no retraining or additional search budget; final tests are separately recorded inference overhead',
         'scope': 'post-hoc eligibility reselection of existing searches, not fresh ensemble-forbidden searches',
         'test_policy': 'freeze selected candidate and audit before held-out testing; reuse an identical already-evaluated selection when available'}
    save(DESTINATION / 'policy.json', policy)
    print(json.dumps({'prepared': True, 'root': str(DESTINATION), 'jobs': len(jobs)}))


def finished(job):
    root = Path(job['run'])
    if not ((root / 'summary.json').exists() or (root / 'failure.json').exists()):
        return False
    budget = read(root / 'budget.json')
    if budget['reservations']:
        return False
    with (root / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def select(job):
    directory = DESTINATION / job['id']
    directory.mkdir(exist_ok=True)
    config = read(job['config'])
    records = []
    for path in Path(job['run']).glob('trials/*/receipt.json'):
        record = read(path)
        if record['status'] == 'success':
            records.append(record)
    original = Path(job['run']) / 'summary.json'
    selected_id = read(original)['selected']['id'] if original.exists() else None
    records.sort(key=lambda r: (-r['score'], r['id'] != selected_id, r['id']))
    audits = []
    for receipt in records:
        target = directory / 'audits' / (receipt['id'] + '.json')
        if not target.exists():
            command = [config['worker_python'], '-m', 'amazon_aligned.single_model', '--audit',
                       receipt['output'], '--data', config['data_dir'], '--output', str(target)]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.with_suffix('.log').open('w') as log:
                result = subprocess.run(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                    env=Controller.environment(SimpleNamespace(c=config), ''), timeout=300)
            if result.returncode:
                raise RuntimeError('Candidate audit failed; manual inspection required: ' + str(target))
        audit = read(target)
        current_predictor = Path(receipt['output']) / 'working/predict.py'
        if audit.get('predictor_sha256') != sha(current_predictor):
            raise RuntimeError('Audited predictor changed: ' + receipt['id'])
        audits.append({'id': receipt['id'], 'validation_hr10': receipt['score'], 'audit': audit})
        if audit['eligible'] is None:
            raise RuntimeError('Candidate needs manual eligibility review: ' + receipt['id'])
        if audit['eligible']:
            selection = {'selected_at': time.time(), 'selected': receipt, 'audit': audit,
                         'higher_or_tied_candidates_reviewed': audits,
                         'search_budget': read(Path(job['run']) / 'budget.json'),
                         'original_search_status': 'complete' if (Path(job['run']) / 'summary.json').exists() else 'interrupted'}
            save(directory / 'selection.json', selection)
            return selection
    raise RuntimeError('No verified single-model candidate exists')


def idle_gpu():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = {int(line.split(',')[0]): [int(v.strip()) for v in line.split(',')[1:]]
               for line in text.strip().splitlines()}
    for group in [[4, 5, 6, 7], [0, 1, 2, 3]]:
        queue = REPEATS / 'queues' / ('gpu-' + '-'.join(map(str, group))) / 'status.json'
        if queue.exists() and read(queue)['status'] not in {'complete', 'finished_with_failures', 'failed'}:
            continue
        if all(devices[g][0] < 512 and devices[g][1] < 5 for g in group):
            return group[0]
    return None


def evaluate(job, selection, gpu):
    root = Path(job['run'])
    directory = DESTINATION / job['id']
    config = read(job['config'])
    selected = selection['selected']
    for relative, checksum in selection['audit'].get('source_files', {}).items():
        if sha(Path(selected['output']) / relative) != checksum:
            raise RuntimeError('Selected inference source changed after selection')
    for name, checksum in selection['audit'].get('checkpoint_sha256', {}).items():
        if sha(Path(name)) != checksum:
            raise RuntimeError('Selected checkpoint changed after selection')
    summary = read(root / 'summary.json') if (root / 'summary.json').exists() else None
    reuse = bool(summary and summary['selected']['output'] == selected['output']
                 and summary['test']['predictor_sha256'] == selection['audit']['predictor_sha256'])
    if reuse:
        test, overhead = summary['test'], {'reused_existing_test': True, 'new_gpu_seconds': 0.}
    else:
        if gpu is None:
            return False
        output = directory / 'final-test'
        output.mkdir(exist_ok=False)
        command = [config['worker_python'], '-u', '-m', 'amazon_aligned.worker', '--trial', selected['output'],
                   '--data', config['data_dir'], '--mode', 'test', '--output', str(output)]
        start = time.monotonic()
        with (output / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                  env=Controller.environment(SimpleNamespace(c=config), gpu), start_new_session=True)
            try:
                code = process.wait(timeout=config['final_timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = -9
        overhead = {'reused_existing_test': False, 'new_gpu_seconds': time.monotonic() - start,
                    'gpu': gpu, 'separate_final_inference_overhead': True}
        save(output / 'receipt.json', {**overhead, 'exit_code': code})
        if code:
            raise RuntimeError('Single-model final evaluation failed; retained for inspection')
        test = read(output / 'result.json')
    save(directory / 'summary.json', {'status': 'complete', 'job': job, 'selection': selection,
         'test': test, 'inference': overhead, 'policy': str(DESTINATION / 'policy.json')})
    return True


def coordinator():
    lock = (DESTINATION / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registration = read(DESTINATION / 'registration.json')
    if (registration['policy_sha256'] != sha(DESTINATION / 'policy.json')
            or registration['auditor_sha256'] != sha(Path(__file__))
            or registration['worker_sha256'] != sha(WORKSPACE / 'amazon_aligned/worker.py')
            or registration['prepared_data_sha256'] != sha(ROOT / 'data/prepared.json')):
        raise RuntimeError('Registered single-model policy/code/data changed')
    policy = read(DESTINATION / 'policy.json')
    while True:
        states = []
        for job in policy['jobs']:
            directory = DESTINATION / job['id']
            directory.mkdir(exist_ok=True)
            if (directory / 'summary.json').exists():
                states.append({'id': job['id'], 'status': 'complete', 'test_hr10': read(directory / 'summary.json')['test']['score']})
                continue
            if (directory / 'failure.json').exists():
                states.append({'id': job['id'], 'status': 'needs_review'})
                continue
            if not finished(job):
                states.append({'id': job['id'], 'status': 'waiting_for_search_completion'})
                continue
            try:
                selection = read(directory / 'selection.json') if (directory / 'selection.json').exists() else select(job)
                completed = evaluate(job, selection, idle_gpu())
                states.append({'id': job['id'], 'status': 'complete' if completed else 'waiting_for_idle_gpu_group',
                               'selected': selection['selected']['id'], 'validation_hr10': selection['selected']['score']})
            except Exception:
                save(directory / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
                states.append({'id': job['id'], 'status': 'needs_review'})
            save(DESTINATION / 'status.json', {'jobs': states, 'updated_at': time.time()})
        done = all(item['status'] == 'complete' for item in states)
        save(DESTINATION / 'status.json', {'status': 'complete' if done else 'running', 'jobs': states, 'updated_at': time.time()})
        if done:
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--coordinator', action='store_true')
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.audit:
        result = model_audit(args.audit, args.data)
        result['predictor_sha256'] = sha(args.audit / 'working/predict.py')
        result['source_files'] = {str(p.relative_to(args.audit)): sha(p) for p in args.audit.rglob('*.py')
                                  if '__pycache__' not in p.parts}
        result['checkpoint_sha256'] = {name: sha(Path(name)) for paths in result.get('checkpoint_files', {}).values()
                                       for name in paths}
        result['auditor_sha256'] = sha(Path(__file__))
        result['audited_at'] = time.time()
        return save(args.output, result)
    if args.prepare:
        return prepare()
    if args.coordinator:
        return coordinator()
    if (DESTINATION / 'launcher.json').exists():
        raise RuntimeError('Single-model coordinator is already registered')
    command = [sys.executable, '-u', '-m', 'amazon_aligned.single_model', '--coordinator']
    save(DESTINATION / 'registration.json', {'policy_sha256': sha(DESTINATION / 'policy.json'),
         'auditor_sha256': sha(Path(__file__)), 'worker_sha256': sha(WORKSPACE / 'amazon_aligned/worker.py'),
         'prepared_data_sha256': sha(ROOT / 'data/prepared.json'), 'registered_at': time.time()})
    with (DESTINATION / 'coordinator.log').open('a') as log:
        process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(DESTINATION / 'launcher.json', {'pid': process.pid, 'command': command, 'started_at': time.time()})
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError('Single-model coordinator exited immediately')
    print(json.dumps({'pid': process.pid, 'root': str(DESTINATION)}))


if __name__ == '__main__':
    main()
