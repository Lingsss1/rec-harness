"""Version-2 checkpoint reevaluation, with no optimizer or training path.

Validation selects among each method's original winner and recovered failed
checkpoints. Test predictions are requested only after selection is persisted.
The original MLE-STAR ensemble is reevaluated with its existing fixed weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from mle_star_codex.run import blend, read, save, sha, source_parts
from mle_star_codex.worker import CHECK_VERSION, metrics, read as read_split, rows_for

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / 'artifacts/evaluator-fix-v2/reevaluation'
DATA = WORKSPACE / 'artifacts/mle-star-movies-2024/data'


def collect():
    entries, missing = [], []
    def add(family, key, source, checkpoint, kind, old_metric=None):
        identifier = family.lower().replace(' ', '-') + '__' + key
        if any(e['id'] == identifier for e in entries):
            return identifier
        source, checkpoint = Path(source), Path(checkpoint)
        if not checkpoint.exists():
            missing.append({'family': family, 'id': identifier, 'reason': 'no trained checkpoint',
                            'original_source': str(source)})
            return None
        source_parts(source.read_text())
        entries.append({'family': family, 'id': identifier, 'key': key, 'kind': kind,
            'source': str(source), 'checkpoint': str(checkpoint),
            'source_sha256': sha(source), 'checkpoint_sha256': sha(checkpoint),
            'original_validation_hr10': old_metric})
        return identifier

    mle = WORKSPACE / 'artifacts/mle-star-movies-2024'
    selection = read(mle / 'selection.json')
    best = max(selection['members'], key=lambda m: m['metrics']['HR@10'])
    add('Baseline', 'baseline', mle / 'models/baseline.py', mle / 'trials/baseline/checkpoint.pt', 'original', .395)
    add('MLE-STAR', best['id'], best['source'], mle / 'trials' / best['id'] / 'checkpoint.pt', 'original', best['metrics']['HR@10'])
    ensemble = []
    for member, weight in zip(selection['members'], selection['plan']['weights']):
        if weight:
            key = add('MLE-STAR', member['id'], member['source'], mle / 'trials' / member['id'] / 'checkpoint.pt',
                      'ensemble_auxiliary', member['metrics']['HR@10'])
            ensemble.append({'id': key, 'weight': weight})

    auto = WORKSPACE / 'artifacts/autoresearch-movies-2024'
    winner = read(auto / 'selection.json')['selected']
    add('Autoresearch', winner['id'], winner['source'], auto / 'trials' / winner['id'] / 'checkpoint.pt',
        'original', winner['metrics']['HR@10'])
    staged = WORKSPACE / 'artifacts/staged-movies-2024-aligned44/study/trajectories/movies_nextitnet'
    for arm in 'ABCD':
        parent = staged / arm / '1'
        state = read(parent / 'state.json')
        key = 't%03d' % state['best_trial']
        add('Staged-' + arm, key, parent / 'artifacts' / key / 'model.py',
            parent / 'validation' / key / 'checkpoint/checkpoint.pt', 'original', state['best_metric'])
        for record in state['records']:
            if not record['valid']:
                key = 't%03d' % record['trial']
                add('Staged-' + arm, key, parent / 'artifacts' / key / 'model.py',
                    parent / 'validation' / key / 'adapter_scratch/training/checkpoint.pt', 'recovered')

    evolve = WORKSPACE / 'artifacts/mlevolve-movies-2024'
    summary = read(evolve / 'summary.json')
    key = summary['selected_id']
    add('MLEvolve', key, evolve / 'models' / (key + '.py'), evolve / 'trials' / key / 'checkpoint.pt',
        'original', summary['selected_validation_hr10'])
    for file in sorted((evolve / 'trials').glob('*/receipt.json')):
        receipt = read(file)
        request = read(file.parent / 'request.json')
        if request.get('mode') == 'train' and receipt['status'] != 'success':
            add('MLEvolve', file.parent.name, receipt['source'], file.parent / 'checkpoint.pt', 'recovered')
    # AST-invalid MLEvolve proposals never entered the worker and have no checkpoint.
    for file in sorted((evolve / 'invalid').glob('*.json')):
        missing.append({'family': 'MLEvolve', 'id': file.stem, 'reason': 'source rejected before training; no checkpoint'})
    return entries, missing, {'transform': selection['plan']['transform'], 'members': ensemble}


def evaluation_rows(phase):
    train = read_split(DATA / 'trainval/Movies_and_TV_train.txt')
    valid = read_split(DATA / 'trainval/Movies_and_TV_valid.txt')
    itemnum = read(DATA / 'trainval/dataset_stats.json')['Movies_and_TV']['itemnum']
    rows = rows_for(train, valid, itemnum, 200) if phase == 'validation' else rows_for(
        train, read_split(DATA / 'test/Movies_and_TV_test.txt'), itemnum, 10000, valid)
    return itemnum, rows


def stage_features(root, phase):
    itemnum, rows = evaluation_rows(phase)
    row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    expected = ('39f2bdc5befdc160fa132b63ae3d83ad42f2d4fc2b95b43bb6466731a04f8b7b' if phase == 'validation'
                else '7e2fb8a72364535829f8677ac7532232a4ac1aff5b37dbd564c5ad3cd817b5aa')
    if row_hash != expected:
        raise RuntimeError('Evaluation rows differ from the historical comparison')
    rng, visible, positive_indices = random.Random(2024), [], []
    for row in rows:
        order = list(range(100))
        rng.shuffle(order)
        visible.append({'history': row['history'], 'candidates': [row['candidates'][i] for i in order]})
        positive_indices.append(order.index(0))
    path = root / 'features' / (phase + '.json')
    save(path, {'itemnum': itemnum, 'rows': visible})
    return path, positive_indices, row_hash


def evaluate_one(root, entry, phase, gpu, features, positives, row_hash):
    directory = root / phase / entry['id']
    directory.mkdir(parents=True, exist_ok=True)
    if sha(entry['source']) != entry['source_sha256'] or sha(entry['checkpoint']) != entry['checkpoint_sha256']:
        raise RuntimeError('Source/checkpoint changed: ' + entry['id'])
    if (directory / 'metrics.json').exists():
        return read(directory / 'metrics.json')
    if (directory / 'started.json').exists():
        raise RuntimeError('Unknown evaluation outcome; inspect before repeating: ' + str(directory))
    bundle = directory / 'bundle'
    bundle.mkdir()
    (bundle / 'model.py').symlink_to(entry['source'])
    checkpoint = bundle / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'checkpoint.pt').symlink_to(entry['checkpoint'])
    scores_path = directory / 'raw-predictions.npy'
    evaluator = WORKSPACE / 'staged-autoresearch/staged_research/movies_evaluator.py'
    argv = [sys.executable, str(evaluator), '--predict-child', '--candidate', str(bundle),
        '--checkpoint', str(checkpoint), '--features', str(features), '--scores', str(scores_path)]
    save(directory / 'started.json', {'command': argv, 'gpu': gpu, 'time': time.time(), 'entry': entry})
    started = time.monotonic()
    with (directory / 'prediction.log').open('w') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=1800,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    transport = {'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started}
    save(directory / 'transport.json', transport)
    if result.returncode:
        failure = {'id': entry['id'], 'status': 'failed', 'phase': phase,
                   'error': (directory / 'prediction.log').read_text()[-6000:]}
        save(directory / 'failure.json', failure)
        print(json.dumps(failure), flush=True)
        return failure
    if sha(entry['source']) != entry['source_sha256'] or sha(entry['checkpoint']) != entry['checkpoint_sha256']:
        raise RuntimeError('Source/checkpoint changed during evaluation')
    scores = np.load(scores_path, allow_pickle=False)
    if scores.shape != (len(positives), 100) or not np.isfinite(scores).all():
        raise RuntimeError('Invalid independent prediction matrix')
    ordered = np.empty_like(scores)
    for i, target in enumerate(positives):
        order = [target] + [j for j in range(100) if j != target]
        ordered[i] = scores[i, order]
    np.save(directory / 'scores.npy', ordered)
    payload = {'id': entry['id'], 'family': entry['family'], 'kind': entry['kind'], 'status': 'success',
        'phase': phase, 'check_version': CHECK_VERSION, 'metrics': metrics(ordered), 'n_users': len(positives),
        'rows_sha256': row_hash, 'source_sha256': entry['source_sha256'],
        'checkpoint_sha256': entry['checkpoint_sha256'], 'gpu_seconds': transport['elapsed_seconds']}
    save(directory / 'metrics.json', payload)
    print(json.dumps({'event': 'checkpoint_evaluated', 'id': entry['id'], 'phase': phase,
                     'HR@10': payload['metrics']['HR@10']}), flush=True)
    return payload


def evaluate_phase(root, entries, phase, devices):
    features, positives, row_hash = stage_features(root, phase)
    queues = [entries[i::len(devices)] for i in range(len(devices))]
    def worker(gpu, queue):
        return [evaluate_one(root, entry, phase, gpu, features, positives, row_hash) for entry in queue]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(devices, queues)]
        return [item for future in futures for item in future.result()]


def select_validation(entries, results):
    valid = {result['id']: result for result in results if result['status'] == 'success'}
    selected, priority = {}, {}
    for entry in entries:
        if entry['kind'] == 'ensemble_auxiliary' or entry['id'] not in valid:
            continue
        family = entry['family']
        rank = (valid[entry['id']]['metrics']['HR@10'], entry['kind'] == 'original')
        if family not in selected or rank > priority[family]:
            selected[family], priority[family] = entry['id'], rank
    return selected


def run(root, devices):
    root.mkdir(parents=True, exist_ok=False)
    entries, missing, ensemble = collect()
    manifest = {'check_version': CHECK_VERSION, 'training_calls': 0, 'entries': entries,
        'missing_checkpoints': missing, 'fixed_ensemble': ensemble,
        'code_hashes': {str(p): sha(p) for p in [Path(__file__), WORKSPACE / 'mle_star_codex/worker.py',
            WORKSPACE / 'staged-autoresearch/staged_research/movies_evaluator.py']}}
    save(root / 'manifest.json', manifest)
    print(json.dumps({'validation_checkpoints': len(entries), 'missing_checkpoints': len(missing)}), flush=True)
    validation = evaluate_phase(root, entries, 'validation', devices)
    save(root / 'validation-results.json', validation)
    by_id = {entry['id']: entry for entry in entries}
    validated = {r['id']: r for r in validation if r['status'] == 'success'}
    selected = select_validation(entries, validation)
    # Persist validation-only selection BEFORE requesting any test predictions.
    save(root / 'selection.json', {'criterion': 'validation HR@10 only; original winner retained on ties',
                                 'selected': selected, 'test_predictions_started': False})
    test_ids = list(dict.fromkeys(list(selected.values()) + [m['id'] for m in ensemble['members']]))
    if any(key not in validated for key in test_ids):
        raise RuntimeError('Required final checkpoint did not pass validation')
    tests = evaluate_phase(root, [by_id[key] for key in test_ids], 'test', devices)
    save(root / 'test-results.json', tests)
    tested = {r['id']: r for r in tests if r['status'] == 'success'}
    if any(key not in tested for key in test_ids):
        raise RuntimeError('Final test failure; retained for inspection')
    methods = []
    for family, key in selected.items():
        methods.append({'method': family, 'selected_id': key, 'selected_kind': by_id[key]['kind'],
            'validation': validated[key]['metrics'], 'test': tested[key]['metrics']})
    plan = {'transform': ensemble['transform'], 'weights': [m['weight'] for m in ensemble['members']]}
    blended = {}
    for phase in ('validation', 'test'):
        arrays = [np.load(root / phase / m['id'] / 'scores.npy', allow_pickle=False) for m in ensemble['members']]
        blended[phase] = metrics(blend(arrays, plan))
    methods.append({'method': 'MLE-STAR fixed ensemble', **blended})
    summary = {'status': 'complete', 'check_version': CHECK_VERSION, 'training_calls_added': 0,
        'validation_checkpoints': len(entries), 'recovered_checkpoints': sum(e['kind'] == 'recovered' for e in entries),
        'failed_validation_checks': [v for v in validation if v['status'] != 'success'],
        'missing_checkpoints': missing, 'methods': methods,
        'limitations': ['Checkpoint reevaluation only, not a fresh search with corrected feedback.',
            'Candidates rejected before training remain without scores; no fitting was added.',
            'Same previously used test split; single training seed.',
            'Historical LLM reasoning settings and search budgets are unchanged.']}
    save(root / 'summary.json', summary)
    lines = ['# Evaluator v2 checkpoint reevaluation', '', '| Method | Validation HR@10 | Test HR@10 |',
             '|---|---:|---:|']
    lines += ['| %s | %.4f | %.4f |' % (m['method'], m['validation']['HR@10'], m['test']['HR@10']) for m in methods]
    lines += ['', 'No training was performed. Selection used validation only. These are checkpoint reevaluation results, not a rerun of each search.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'event': 'reevaluation_complete', 'summary': str(root / 'summary.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    devices = [int(x) for x in args.gpus.split(',')]
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('Four distinct GPU ordinals required')
    run(args.root.resolve(), devices)


if __name__ == '__main__':
    main()
