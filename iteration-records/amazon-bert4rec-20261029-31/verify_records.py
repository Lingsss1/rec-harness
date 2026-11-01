#!/usr/bin/env python3
"""Offline checks of exported bytes, selected programs, accounting and usage."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = read('manifest.json')
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(manifest['files']), 'File inventory mismatch'
    assert len(actual) == manifest['file_count']
    assert sum(d['bytes'] for d in manifest['files'].values()) == manifest['total_bytes']
    for name, expected in manifest['files'].items():
        p = ROOT / name
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], name
        assert p.stat().st_size < 95 * 1024 * 1024, name
        assert p.suffix not in {'.pt', '.pth', '.pkl', '.npz', '.bin', '.ckpt', '.safetensors', '.pyc', '.lock'}, name
        assert p.name not in {'simulation-input.json', 'auth.json'}, name
        assert p.name != 'events.jsonl' or '/calls/' not in name, name
        assert '/data/trainval/' not in name and '/data/test/' not in name, name
    for name, source in read('copy-sources.json').items():
        assert manifest['files'][name]['sha256'] == source['sha256'], name
    for p in ROOT.glob('searches/self_evolverec/trials/*/directional-feedback.json'):
        d = json.loads(p.read_text())
        assert 'users' not in d and isinstance(d['user_count'], int), str(p)
    trials, calls, events, attempts = [read(n + '.json') for n in ['iterations', 'calls', 'token-events', 'llm-attempts']]
    keys = ['input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
            'output_tokens', 'reasoning_output_tokens', 'total_tokens']
    identities = [(e['thread_id'], e['turn_index']) for e in events if e['thread_id']]
    assert len(identities) == len(set(identities)), 'Duplicated token usage'
    for row in read('comparisons.json'):
        method = row['method']
        prefix = 'searches/' + method + '/'
        summary, budget = read(prefix + 'summary.json'), read(prefix + 'budget.json')
        assert summary['status'] == row['status'] == 'complete'
        assert not budget['reservations']
        assert row['budget_gpu_seconds'] == budget['total_gpu_seconds'] == 43200
        assert row['spent_gpu_seconds'] <= 43200
        own = [t for t in trials if t['method'] == method]
        assert len(own) == row['trial_count'] == len(budget['settled'])
        assert abs(sum(t['gpu_seconds'] for t in own) - row['spent_gpu_seconds']) < 1e-5
        assert abs(sum(x['allocated_gpu_seconds'] for x in budget['settled'].values()) - row['spent_gpu_seconds']) < 1e-5
        for t in own:
            receipt = read(t['receipt'])
            assert receipt['id'] == t['trial'] and receipt['status'] == t['status']
            assert abs(receipt['allocated_gpu_seconds'] - t['gpu_seconds']) < 1e-5
            if t.get('source_sha256'):
                assert sha(ROOT / t['source']) == t['source_sha256'], t['source']
        selected = [t for t in own if t['selected']]
        assert len(selected) == 1 and selected[0]['trial'] == row['selected_trial'] == summary['selected']['id']
        assert selected[0]['validation_hr10'] == row['validation_hr10'] == summary['selected']['score']
        audit = read(prefix + 'trials/' + row['selected_trial'] + '/single-model-audit.json')
        assert audit['eligible'] is True and audit['registered_family'] == 'bert4rec'
        assert audit['training_source_sha256'] == selected[0]['source_sha256']
        assert row['test_hr10'] == summary['test']['score'] == summary['test']['aggregate']['HR@10']
        assert row['test_ndcg10'] == summary['test']['aggregate']['NDCG@10']
        for metric in ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']:
            mean = sum(d['metrics'][metric] for d in summary['test']['per_dataset'].values()) / 4
            assert abs(mean - summary['test']['aggregate'][metric]) <= 0.000051, (method, metric)
        own_calls = [x for x in calls if x['method'] == method]
        own_attempts = [x for x in attempts if x['method'] == method]
        assert len(own_calls) == row['calls']
        assert sum(x['receipt_exists'] for x in own_calls) == row['calls_with_receipt']
        assert len(own_attempts) == row['attempt_event_logs']
        assert sum(not x['usage_available'] for x in own_attempts) == row['attempts_without_usage']
        assert {x['model'] for x in own_calls} == {'gpt-6.1-sol'}
        assert {x['effort'] for x in own_calls} == {'medium'}
        for key in keys:
            assert sum(x[key] for x in own_calls) == row[key], (method, key, 'calls')
            assert sum(x[key] for x in own_attempts) == row[key], (method, key, 'attempts')
            assert sum(x[key] for x in events if x['method'] == method) == row[key], (method, key, 'events')
        assert row['total_tokens'] == row['input_tokens'] + row['output_tokens']
        assert sha(ROOT / prefix / 'baseline.py') == 'd51d266361556daac7a1c7e3dfbbc1aa6e46bcb7117f6cdbc71954385530c4ca'
    for r in read('registered-source-audit.json'):
        if r.get('export'):
            assert sha(ROOT / r['export']) == r['registered_sha256'], r['export']
    print(f"Verified {manifest['file_count']} files, 3 completed searches, {len(trials)} trials, "
          f"{len(calls)} calls and {len(events)} token usage events.")


if __name__ == '__main__':
    main()
