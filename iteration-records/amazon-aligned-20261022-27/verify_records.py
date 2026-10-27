#!/usr/bin/env python3
"""Check publication hashes, source copies, trial accounting and result labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    manifest = read('manifest.json')
    for name, expected in manifest['files'].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == expected['bytes'], name
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
        assert len(data) < 100 * 1024 * 1024, name
    for name, source in read('copy-sources.json').items():
        assert manifest['files'][name]['sha256'] == source['sha256'], name
    rows = read('comparisons.json')
    trials = read('iterations.json')
    for row in rows:
        root = row['record']
        included = [t for t in trials if t['run'] == row['run']]
        assert len(included) == row['trial_count'], row['run']
        assert sum(t['status'] == 'success' for t in included) == row['successful_trials'], row['run']
        if row['status'] == 'complete':
            assert (ROOT / root / 'summary.json').exists(), row['run']
            assert row['test_hr10'] is not None, row['run']
        else:
            assert row['test_hr10'] is None, row['run']
        if row['spent_gpu_seconds'] is not None:
            budget = read(root + '/budget.json')
            assert abs(sum(v['allocated_gpu_seconds'] for v in budget['settled'].values()) - row['spent_gpu_seconds']) < 1e-5, row['run']
    for row in read('single-model-results.json'):
        record = read(row['record'])
        assert record['selection']['audit']['eligible'] is True, row['run']
        assert record['selection']['original_search_status'] == row['original_search_status'], row['run']
    banned = {'.pt', '.pth', '.pkl', '.npz', '.bin', '.safetensors', '.ckpt', '.lock'}
    assert not [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in banned]
    print(f"Verified {len(manifest['files'])} files, {len(rows)} runs and {len(trials)} trials")


if __name__ == '__main__':
    main()
