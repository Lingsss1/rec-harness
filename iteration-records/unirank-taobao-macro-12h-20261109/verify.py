"""Independently verify the exported iteration bundle; never train or open data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = read(root / 'manifest.json')
    files = manifest['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'manifest.json'}
    assert actual == set(files), 'Archive inventory mismatch'
    for name, record in files.items():
        path = root / name
        assert path.stat().st_size == record['bytes'], name
        assert digest(path) == record['sha256'], name
        assert path.suffix not in {'.pt', '.pkl', '.parquet', '.npy', '.npz', '.pdf', '.html', '.bin'}, name
        assert path.stat().st_size < 95 * 2**20, name
    for row in read(root / 'copy-sources.json'):
        assert files[row['path']]['sha256'] == row['sha256'], row['path']
    summary = read(root / 'summary.json')
    rows = read(root / 'iterations.json')
    assert len(rows) == summary['counts']['research_trials']
    assert len(list(csv.DictReader((root / 'iterations.csv').open()))) == len(rows)
    for row in rows:
        assert digest(root / row['model_path']) == row['source_sha256'], row['id']
        if row['status'] == 'success':
            mean = sum(row[t + '_validation_auc'] for t in ['is_click', 'cart', 'fav', 'buy']) / 4
            assert abs(mean - row['validation_macro_auc']) < 1e-12, row['id']
    calls = read(root / 'calls.json')
    assert len(calls) == summary['counts']['agent_calls']
    incumbent_count = len(list(csv.DictReader((root / 'topic-incumbents.csv').open())))
    assert incumbent_count == summary['counts']['completed_rounds']
    assert len(list(csv.DictReader((root / 'final-test-tasks.csv').open()))) == 4 * len(summary['final_tests'])
    assert summary['generalization'][0]['completed_test_runs'] == 4
    diagnostics = read(root / 'cli-diagnostics.json')
    diagnostic_lines = {(row['path'], row['line']): row['text'] for row in diagnostics['lines']}
    patterns = [re.compile(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{24,}'),
                re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})'),
                re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]
    json_count = event_count = raw_diagnostic_count = 0
    for name in files:
        path = root / name
        if path.suffix in {'.py', '.txt', '.md', '.yaml', '.json', '.jsonl', '.log', '.patch', '.csv'}:
            text = path.read_text()
            for pattern in patterns:
                assert not pattern.search(text), 'Credential pattern in ' + name
            if path.suffix == '.json':
                json.loads(text)
                json_count += 1
            if path.suffix == '.jsonl':
                for number, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        try:
                            json.loads(line)
                            event_count += 1
                        except json.JSONDecodeError:
                            # The original CLI mixes stderr diagnostics/tracebacks with JSON events.
                            assert path.name == 'codex.jsonl', name
                            assert diagnostic_lines.get((name, number)) == line, name
                            raw_diagnostic_count += 1
    assert raw_diagnostic_count == diagnostics['count'] == len(diagnostic_lines)
    return {'status': 'passed', 'files': len(files) + 1, 'total_bytes': manifest['total_bytes'],
            'json_files_parsed': json_count, 'jsonl_events_parsed': event_count,
            'raw_cli_diagnostic_lines_preserved': raw_diagnostic_count,
            'all_hashes_match': True, 'all_research_models_and_macro_scores_match': True,
            'trials': len(rows), 'calls': len(calls), 'topic_incumbent_rounds': incumbent_count,
            'no_large_or_binary_data_assets': True, 'credential_patterns_found': False,
            'completed_transfer_runs': 4, 'rank1_transfer_state': summary['generalization'][1]['phase']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False))
