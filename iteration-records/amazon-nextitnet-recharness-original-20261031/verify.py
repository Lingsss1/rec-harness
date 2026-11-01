"""Verify the exported evidence without training or accessing external data."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    manifest = read('manifest.json')
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
              and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(manifest['files']), 'File inventory changed'
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    rounds = read('rounds.json')
    candidates = read('candidates.json')
    assert len(rounds) == 29 and [r['round'] for r in rounds] == list(range(1, 30))
    assert len(candidates) == 107
    assert Counter(c['status'] for c in candidates) == {'success': 104, 'timeout': 3}
    assert [c['round'] for c in candidates if c['jump']] == [10, 16]
    budget = read('records/budget.json')
    assert not budget['reservations']
    assert abs(sum(c['gpu_seconds'] for c in candidates) - budget['spent_gpu_seconds']) < 1e-5
    assert abs(sum(v['allocated_gpu_seconds'] for v in budget['settled'].values())
               - budget['spent_gpu_seconds']) < 1e-5
    summary = read('records/summary.json')
    assert summary['selected']['id'] == 'round-0011-slot-0-attempt-1'
    assert summary['selected']['score'] == .5737
    assert summary['test']['score'] == .499
    assert summary['test']['aggregate']['NDCG@10'] == .3126
    for metric in ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']:
        mean = sum(v['metrics'][metric] for v in summary['test']['per_dataset'].values()) / 4
        assert abs(mean - summary['test']['aggregate'][metric]) <= .0001
    final = read('records/native/final-selection.json')
    assert final['state']['current_basin_id'] == 'default|change_loss_function@10'
    assert rounds[19]['basin_before'].endswith('|change_architecture@16')
    assert rounds[19]['basin_after'] == 'default|change_loss_function@10'
    assert (ROOT / 'models/selected.py').read_bytes() == (ROOT / 'records/trials' /
            summary['selected']['id'] / 'train.py').read_bytes()
    for path in (ROOT / 'records/trials').glob('*/receipt.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('source_sha256'):
            assert hashlib.sha256((path.parent / 'train.py').read_bytes()).hexdigest() == receipt['source_sha256']
    print(f"Verified {len(actual)} files, 29 rounds, 107 trials, two jumps, rollback, budget and final metrics")


if __name__ == '__main__':
    main()
