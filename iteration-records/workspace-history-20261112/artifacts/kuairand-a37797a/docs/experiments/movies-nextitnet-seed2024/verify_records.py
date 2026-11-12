#!/usr/bin/env python3
"""Verify published hashes, iteration counts, source paths, and summary values."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = manifest['files']
    expected = {e['path'] for e in entries}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual == expected | {'manifest.json'}, 'missing or unexpected files'
    for entry in entries:
        path = root / entry['path']
        assert path.is_relative_to(root)
        data = path.read_bytes()
        assert len(data) == entry['bytes'], f"size mismatch: {entry['path']}"
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], f"hash mismatch: {entry['path']}"
        if path.suffix == '.json':
            json.loads(data)
        elif path.suffix == '.jsonl':
            for line in data.decode().splitlines():
                if line.strip():
                    json.loads(line)
    with (root / 'iterations.csv').open(encoding='utf-8-sig', newline='') as f:
        iterations = list(csv.DictReader(f))
    counts = Counter(row['method'] for row in iterations)
    assert len(iterations) == 312
    assert len({(row['method'], row['id']) for row in iterations}) == 312
    for method in ('mle-star', 'autoresearch', 'mlevolve', 'staged-A', 'staged-B', 'staged-C', 'staged-D'):
        assert counts[method] == 44
    assert counts['d-target-norm-supplement'] == 4
    for row in iterations:
        assert (root / row['model']).is_file(), row['model']
        assert (root / row['record']).is_file(), row['record']
    for arm in 'ABCD':
        state = json.loads((root / f'staged/{arm}/state.json').read_text())
        assert len(state['records']) == 44
        assert state['trial_budget'] == 44
        assert state['status'] == 'completed'
        if arm in 'BCD':
            assert len(state['plans']) == 5
            assert len(state['pre_reviews']) == 4
            assert len(state['final_reviews']) == 5
    for method in ('mle-star', 'autoresearch', 'mlevolve'):
        for receipt in (root / method / 'trials').glob('*/receipt.json'):
            value = json.loads(receipt.read_text())
            model = root / method / 'models' / Path(value['source']).name
            if model.exists():
                assert hashlib.sha256(model.read_bytes()).hexdigest() == value['source_sha256']
    reevaluation = json.loads((root / 'evaluator-v2/reevaluation/summary.json').read_text())
    assert reevaluation['training_calls_added'] == 0
    assert reevaluation['validation_checkpoints'] == 27
    assert reevaluation['recovered_checkpoints'] == 17
    supplement = json.loads((root / 'd-target-norm/summary.json').read_text())
    assert supplement['training_calls_added'] == 4
    assert supplement['selected_id'] == 'beta-0'
    assert supplement['control_reproduces_d_trial_23_rankings']
    with (root / 'comparison.csv').open(encoding='utf-8-sig', newline='') as f:
        comparison = list(csv.DictReader(f))
    assert len(comparison) == len(reevaluation['methods'])
    for row, value in zip(comparison, reevaluation['methods']):
        assert row['method'] == value['method']
        assert float(row['validation_hr10']) == value['validation']['HR@10']
        assert float(row['test_hr10']) == value['test']['HR@10']
    print(json.dumps({'status': 'passed', 'verified_files': len(entries),
                      'bytes': sum(x['bytes'] for x in entries),
                      'iterations': dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
