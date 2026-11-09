"""Verify archived hashes and recompute every task/paired aggregate without data or weights."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def verify(root):
    read = lambda p: json.loads(p.read_text())
    manifest = read(root / 'manifest.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('Manifest file inventory differs')
    for name, item in manifest['files'].items():
        raw = (root / name).read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Hash mismatch: ' + name)
    summary = read(root / 'summary.json')
    count = 0
    for row in summary['datasets']:
        folder = root / 'datasets' / row['dataset']
        protocol = read(folder / 'protocol.json')
        labels = protocol['tasks']
        for name, expected in protocol['source_sha256'].items():
            if hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen host source mismatch')
        pair = {}
        for kind in ['candidate', 'din']:
            output = root / 'runs' / row['dataset'] / kind / 'fit-01'
            result = read(output / 'result.json')
            selected = read(output / 'selection.json')
            if selected['checkpoint_sha256'] != result['checkpoint_sha256'] or result['test_used_for_selection']:
                raise ValueError('Selection gate mismatch')
            if result['protocol_hash'] != protocol['identity']:
                raise ValueError('Protocol identity mismatch')
            for split, key in [('valid', 'validation'), ('test', 'test')]:
                values = result[key]
                if values['samples'] != protocol['sample_size'][split] or set(values['per_task']) != set(labels):
                    raise ValueError('Task/sample coverage mismatch')
                for metric in ['auc', 'logloss']:
                    expected = math.fsum(values['per_task'][t][metric] for t in labels) / len(labels)
                    if abs(values[metric] - expected) > 1e-12:
                        raise ValueError('Incorrect native-task mean')
            if abs(row[kind + '_auc'] - result['test']['auc']) > 1e-12:
                raise ValueError('Published AUC differs from raw result')
            pair[kind] = result
            count += 1
        for task, change in row['per_task'].items():
            c = pair['candidate']['test']['per_task'][task]
            d = pair['din']['test']['per_task'][task]
            if abs(change['delta_auc'] - (c['auc'] - d['auc'])) > 1e-12:
                raise ValueError('Incorrect task delta')
        if abs(row['delta_macro_auc'] - (pair['candidate']['test']['auc'] - pair['din']['test']['auc'])) > 1e-12:
            raise ValueError('Incorrect macro delta')
    with (root / 'per-task.csv').open() as stream:
        tasks = list(csv.DictReader(stream))
    if len(tasks) != sum(row['task_count'] for row in summary['datasets']):
        raise ValueError('Incomplete task CSV')
    for record in tasks:
        dataset = next(row for row in summary['datasets'] if row['dataset'] == record['dataset'])
        expected = dataset['per_task'][record['task']]
        for key, value in expected.items():
            if abs(float(record[key]) - value) > 1e-12:
                raise ValueError('Task CSV differs from raw metrics: ' + key)
    with (root / 'summary.csv').open() as stream:
        totals = list(csv.DictReader(stream))
    if len(totals) != len(summary['datasets']):
        raise ValueError('Incomplete summary CSV')
    for record in totals:
        expected = next(row for row in summary['datasets'] if row['dataset'] == record['dataset'])
        for key, value in record.items():
            if key != 'dataset' and abs(float(value) - expected[key]) > 1e-12:
                raise ValueError('Summary CSV differs from raw metrics: ' + key)
    for record in read(root / 'copy-sources.json'):
        if hashlib.sha256((root / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Copied evidence differs from its source digest')
    print(f'Verified {len(actual)} files, {count} complete test runs and {len(tasks)} task comparisons.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root.resolve())
