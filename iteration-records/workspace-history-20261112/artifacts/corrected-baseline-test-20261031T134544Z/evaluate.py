"""Evaluate the four corrected cold starts using frozen baseline checkpoints only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
COMPARISON = WORKSPACE / 'artifacts/model-family-comparison-20261029'
REFERENCE = WORKSPACE / 'artifacts/RecHarness-6lyc'
DATA = WORKSPACE / 'artifacts/amazon-aligned-20261022/data'
FAMILIES = ['gru4rec', 'bert4rec', 'nextitnet', 'hstu']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
METRICS = ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sources(family):
    trial = COMPARISON / 'runs' / (family + '-mlestar') / 'mlestar/trials/baseline'
    receipt = json.loads((trial / 'receipt.json').read_text())
    request = json.loads((trial / 'request.json').read_text())
    audit = json.loads((trial / 'single-model-audit.json').read_text())
    provenance = json.loads((COMPARISON / 'audit/baseline-provenance.json').read_text())[family]
    assert receipt['status'] == 'success' and request['smoke'] is False
    assert audit['eligible'] is True
    assert sha(trial / 'train.py') == provenance['corrected_sha256'] == request['source_sha256']
    files = {str(trial / relative): value for relative, value in audit['source_files'].items()}
    files.update(audit['checkpoint_sha256'])
    for path in (trial / 'working').glob('*_meta.pkl'):
        files[str(path)] = sha(path)
    assert len(audit['checkpoint_sha256']) == 4
    for name, checksum in files.items():
        assert sha(name) == checksum, 'Baseline artifact changed: ' + name
    return trial, files, receipt


def verify_data():
    prepared = json.loads((DATA / 'prepared.json').read_text())
    files = {}
    for receipt in prepared['receipts']:
        for relative, checksum in receipt['files'].items():
            assert sha(DATA / relative) == checksum, 'Prepared data changed: ' + relative
            files[str(DATA / relative)] = checksum
    statistics = DATA / 'trainval/dataset_stats.json'
    assert sha(statistics) == prepared['stats_sha256']
    files[str(statistics)] = prepared['stats_sha256']
    return files


def family_test(family):
    directory = ROOT / family
    directory.mkdir(exist_ok=False)
    trial, before, baseline = sources(family)
    import torch
    assert not torch.cuda.is_available(), 'This independent evaluation is explicitly CPU-only'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(REFERENCE))
    from gagc.benchmarks.amazon_reviews.harness import evaluate
    start = time.monotonic()
    save(directory / 'receipt.json', {'status': 'running', 'family': family,
         'started_utc': datetime.now(timezone.utc).isoformat(), 'baseline': str(trial),
         'artifacts': before, 'training_performed': False, 'backend': 'cpu',
         'cpu_affinity': sorted(os.sched_getaffinity(0)), 'torch': torch.__version__})
    result = evaluate(predict_script=str(trial / 'working/predict.py'),
        data_dir=str(DATA / 'trainval'), test_dir=str(DATA / 'test'), datasets=DATASETS,
        mode='test', num_candidates=99, max_eval_users=10000, seed=42, verbose=True)
    for name, checksum in before.items():
        assert sha(name) == checksum, 'Baseline artifact changed during evaluation: ' + name
    record = {'family': family, 'evaluation': 'test', 'model_stage': 'corrected_unmodified_baseline',
        'source_trial': str(trial), 'training_source_sha256': sha(trial / 'train.py'),
        'predictor_sha256': sha(trial / 'working/predict.py'),
        'score': result.primary_metric, 'aggregate': result.aggregate,
        'per_dataset': {name: {'metrics': entry.metrics, 'n_users': entry.n_users}
                        for name, entry in result.per_dataset.items()},
        'evaluation_seconds': time.monotonic() - start, 'backend': 'cpu',
        'training_performed': False, 'selected_before_test': True,
        'baseline_validation_hr10': baseline['score']}
    save(directory / 'result.json', record)
    save(directory / 'receipt.json', {'status': 'complete', 'family': family,
         'completed_utc': datetime.now(timezone.utc).isoformat(), 'artifacts': before,
         'evaluation_seconds': record['evaluation_seconds'], 'backend': 'cpu',
         'torch': torch.__version__, 'threads': torch.get_num_threads(),
         'cpu_affinity': sorted(os.sched_getaffinity(0)), 'training_performed': False,
         'artifact_integrity_verified_before_and_after': True})
    print(json.dumps({'family': family, 'status': 'complete',
        'metrics': {k: record['aggregate'][k] for k in METRICS},
        'evaluation_seconds': record['evaluation_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=FAMILIES)
    args = parser.parse_args()
    os.nice(19)
    if args.family:
        return family_test(args.family)
    for family in FAMILIES:
        sources(family)
    frozen_data = verify_data()
    grader = {str(p): sha(p) for p in (REFERENCE / 'gagc/benchmarks/amazon_reviews').glob('*.py')}
    save(ROOT / 'manifest.json', {'families': FAMILIES, 'datasets': DATASETS,
         'protocol': {'mode': 'test', 'seed': 42, 'negative_candidates': 99,
                      'max_users_per_dataset': 10000, 'aggregation': 'macro dataset mean'},
         'backend': 'cpu', 'threads': 2, 'niceness': 19,
         'data_files': frozen_data, 'grader_files': grader,
         'selection': 'original corrected baseline checkpoint; no search-selected candidate or retraining',
         'started_utc': datetime.now(timezone.utc).isoformat()})
    save(ROOT / 'status.json', {'status': 'running', 'complete': [], 'current': None})
    results = []
    for family in FAMILIES:
        save(ROOT / 'status.json', {'status': 'running', 'complete': [r['family'] for r in results], 'current': family})
        print('START ' + family, flush=True)
        with (ROOT / (family + '.log')).open('w') as log:
            code = subprocess.run([sys.executable, str(Path(__file__)), '--family', family],
                cwd=WORKSPACE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            save(ROOT / 'status.json', {'status': 'failed', 'family': family, 'exit_code': code})
            raise RuntimeError('Baseline test failed; inspect ' + family + '.log; no automatic retry')
        result = json.loads((ROOT / family / 'result.json').read_text())
        results.append(result)
        print('COMPLETE ' + family + ' ' + json.dumps({m: result['aggregate'][m] for m in METRICS}), flush=True)
    for name, checksum in {**frozen_data, **grader}.items():
        assert sha(name) == checksum, 'Frozen benchmark input changed: ' + name
    save(ROOT / 'summary.json', {'status': 'complete', 'results': results,
        'protocol': json.loads((ROOT / 'manifest.json').read_text())['protocol'],
        'backend': 'cpu', 'training_performed': False,
        'completed_utc': datetime.now(timezone.utc).isoformat()})
    with (ROOT / 'macro_test_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['family', *METRICS])
        writer.writeheader()
        writer.writerows({'family': r['family'], **{m: r['aggregate'][m] for m in METRICS}} for r in results)
    with (ROOT / 'per_dataset_test_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['family', 'dataset', 'n_users', *METRICS])
        writer.writeheader()
        for result in results:
            for name, entry in result['per_dataset'].items():
                writer.writerow({'family': result['family'], 'dataset': name,
                    'n_users': entry['n_users'], **{m: entry['metrics'][m] for m in METRICS}})
    save(ROOT / 'status.json', {'status': 'complete', 'complete': FAMILIES, 'current': None})
    print('ALL_BASELINE_TESTS_COMPLETE ' + str(ROOT), flush=True)


if __name__ == '__main__':
    main()
