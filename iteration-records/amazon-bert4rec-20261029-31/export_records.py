#!/usr/bin/env python3
"""Export completed BERT4Rec searches without training or making model calls."""
import collections
import csv
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re

WORK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FAMILY = WORK / 'artifacts/model-family-comparison-20261029'
REC = WORK / 'artifacts/recharness-family-comparison-20261031'
COPIES, DERIVED, SEEN = {}, {}, set()
SECRET_PATTERNS = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\bsk-[A-Za-z0-9_-]{24,}\b', r'\bgh[pousr]_[A-Za-z0-9]{24,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{30,}\b', r'\bAKIA[0-9A-Z]{16}\b',
    r'(?i)Authorization:\s*(?:Bearer|Basic)\s+[A-Za-z0-9_./+=-]{20,}']


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(data, name):
    assert len(data) < 95 * 1024 * 1024, name
    s = data.decode('utf-8')
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, s):
            raise ValueError('Potential credential: ' + str(name))


def write(name, text):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode('utf-8')
    check(data, name)
    p.write_bytes(data)


def save(name, value):
    write(name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def copy(source, target):
    source = Path(source)
    if not source.is_file():
        return
    data = source.read_bytes()
    check(data, source)
    p = OUT / target
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    COPIES[str(target)] = {'source': str(source.relative_to(WORK)), 'sha256': sha(data), 'bytes': len(data)}


def derived(source, target, value, policy):
    save(target, value)
    DERIVED[str(target)] = {'source': str(source.relative_to(WORK)),
        'source_sha256': sha(source.read_bytes()), 'policy': policy}


def selected(source, target, patterns):
    for pattern in patterns:
        for p in sorted(source.glob(pattern)):
            if (p.is_file() and '__pycache__' not in p.parts
                    and p.name not in {'simulation-input.json', 'diagnosis.json', 'diagnosis-presentation.json', 'auth.json'}):
                copy(p, Path(target) / p.relative_to(source))


def tabulate(name, rows):
    save(name + '.json', rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / (name + '.csv')).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})


TOKEN_KEYS = ['input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
              'output_tokens', 'reasoning_output_tokens', 'total_tokens']


def calls(root, method, config, call_rows, attempt_rows, token_rows):
    total, counts = collections.Counter(), collections.Counter()
    for request_path in sorted((root / 'calls').glob('*/request.json')):
        call = request_path.parent
        request = read(request_path)
        prompt = request.get('prompt', '')
        receipt_path = call / 'receipt.json'
        receipt = read(receipt_path) if receipt_path.exists() else {}
        role = prompt.split('NATIVE ROLE:', 1)[1].split('\n')[0].strip() if 'NATIVE ROLE:' in prompt else ''
        target = Path('searches') / method / 'calls' / call.name
        metadata = {'request_sha256': request.get('request_sha256'),
            'request_file_sha256': sha(request_path.read_bytes()),
            'prompt_sha256': sha(prompt.encode()), 'prompt_characters': len(prompt),
            'scope': request.get('scope'), 'role': role, 'search': request.get('search', False),
            'schema': request.get('schema'), 'raw_prompt_exported': False}
        derived(request_path, target / 'request-metadata.json', metadata, 'Prompt replaced by hash/length; schema and scope retained.')
        for name in ['receipt.json', 'schema.json', 'started.json', 'failure.json']:
            copy(call / name, target / name)
        # Individual simulated-user replies are deliberately kept local.
        if not str(request.get('scope', '')).startswith('feedback/'):
            copy(call / 'response.json', target / 'response.json')
        logs = sorted(call.rglob('events.jsonl'))
        counts['calls'] += 1
        counts['calls_with_receipt'] += bool(receipt)
        counts['calls_without_event_log'] += not logs
        call_total = collections.Counter()
        for path in logs:
            usage = collections.Counter()
            thread, turn, n = None, 0, 0
            for lineno, line in enumerate(path.open(errors='replace'), 1):
                if not any(k in line for k in ['thread.started', '"usage"', '"token_usage"']):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'thread.started':
                    thread, turn = event.get('thread_id'), 0
                u = event.get('usage')
                if event.get('type') not in ['turn.completed', 'turn.failed'] or not isinstance(u, dict):
                    continue
                assert isinstance(u.get('input_tokens'), int) and isinstance(u.get('output_tokens'), int), str(path)
                identity = (thread, turn) if thread else (str(path), lineno)
                turn += 1
                if identity in SEEN:
                    counts['duplicate_usage_events_skipped'] += 1
                    continue
                SEEN.add(identity)
                values = {k: int(u.get(k, 0)) for k in TOKEN_KEYS if k != 'total_tokens'}
                values['total_tokens'] = values['input_tokens'] + values['output_tokens']
                assert values['cached_input_tokens'] <= values['input_tokens']
                assert values['reasoning_output_tokens'] <= values['output_tokens']
                token_rows.append({'method': method, 'call': call.name, 'thread_id': thread,
                    'turn_index': turn - 1, 'event_type': event['type'],
                    'source_event_file': str(path.relative_to(WORK)),
                    'source_event_sha256': sha(path.read_bytes()), 'line': lineno, **values})
                usage.update(values)
                n += 1
            attempt_rows.append({'method': method, 'call': call.name,
                'attempt': str(path.parent.relative_to(call)), 'reported_turns': n,
                'usage_available': bool(n), 'source_event_sha256': sha(path.read_bytes()),
                **{k: usage[k] for k in TOKEN_KEYS}})
            counts['attempt_event_logs'] += 1
            counts['attempts_without_usage'] += not n
            call_total.update(usage)
        for p in call.glob('attempts/*/failure.json'):
            copy(p, target / p.relative_to(call))
        row = {'method': method, 'call': call.name, 'scope': request.get('scope'), 'role': role,
            'receipt_exists': bool(receipt), 'model': receipt.get('model', config['llm_model']),
            'effort': receipt.get('effort', config['llm_effort']),
            'wall_seconds': receipt.get('wall_seconds'), 'attempt_event_logs': len(logs),
            'prompt_characters': len(prompt), 'prompt_sha256': metadata['prompt_sha256'],
            **{k: call_total[k] for k in TOKEN_KEYS}}
        call_rows.append(row)
        total.update(call_total)
    return {**{k: total[k] for k in TOKEN_KEYS}, **dict(counts),
            'usage_status': 'reported_usage_lower_bound' if counts['attempts_without_usage'] or counts['calls_without_event_log'] else 'all_attempt_logs_have_usage'}


def main():
    comparisons, trials, datasets, call_rows, attempts, tokens, edits, rounds = [], [], [], [], [], [], [], []
    jobs = [('recharness', REC / 'runs/bert4rec'),
            ('mlestar', FAMILY / 'runs/bert4rec-mlestar'),
            ('self_evolverec', FAMILY / 'runs/bert4rec-self_evolverec')]
    for method, parent in jobs:
        root, target = parent / method, Path('searches') / method
        config, summary, budget = read(parent / 'config.json'), read(root / 'summary.json'), read(root / 'budget.json')
        assert summary['status'] == 'complete' and not budget['reservations']
        copy(parent / 'config.json', target / 'config.json')
        selected(parent, target / 'controller', ['status.json', 'failure.json', 'controller.log'])
        selected(root, target, ['*.json', 'events.jsonl', 'resource-observations.jsonl',
            'recovery-selection/**/*.json', 'final-test/*.json', 'final-test/worker.log',
            'trials/*/train.py', 'trials/*/models.py', 'trials/*/diagnosis_tools.py',
            'trials/*/request.json', 'trials/*/receipt.json', 'trials/*/result.json',
            'trials/*/resources.json', 'trials/*/pid.json', 'trials/*/single-model-audit.json',
            'trials/*/single-model-audit.log', 'trials/*/training.log', 'trials/*/worker.log',
            'trials/*/feedback-error.json', 'trials/*/working/predict.py'])
        if method == 'recharness':
            selected(root, target, ['native/**/*.py', 'native/**/*.json'])
            for p in sorted((root / 'native/execution').glob('round-*-slot-*.json')):
                x = read(p)
                evidence = x.get('native_result', {}).get('memory_evidence', {})
                before, after = evidence.get('input_source_sha256'), evidence.get('candidate_source_sha256')
                if before and after:
                    a, b = [root / 'native/memory/programs' / (v + '.py') for v in [before, after]]
                    assert sha(a.read_bytes()) == before and sha(b.read_bytes()) == after
                    diff = ''.join(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                        fromfile=before + '.py', tofile=after + '.py'))
                    write(str(target / 'actual-changes' / (p.stem + '.diff')), diff)
                for a in x.get('attempts', []):
                    proposal = a.get('proposal', {})
                    edits.append({'method': method, 'execution': str(target / p.relative_to(root)),
                        'attempt': a.get('attempt'), 'change_summary': proposal.get('change_summary'),
                        'category_edits': proposal.get('category_edits'), 'edits': proposal.get('edits')})
        elif method == 'self_evolverec':
            selected(root, target, ['native/**/*.py', 'native/**/metadata.json',
                'native/**/native-config.json', 'native/**/initial_idea.json', 'native/**/initial_idea_diagnosis.json'])
            keys = ['id', 'idea', 'idea_diagnosis', 'timestamp', 'parent_id', 'iteration_found', 'metrics', 'language', 'code']
            for p in sorted((root / 'native').glob('**/programs/*.json')):
                d = read(p)
                derived(p, target / p.relative_to(root), {k: d[k] for k in keys if k in d},
                        'Keep program/code/idea/lineage/metrics; omit user_feedback, diagnosis_feedback, report and histories.')
            for p in sorted((root / 'trials').glob('*/directional-feedback.json')):
                d = read(p)
                derived(p, target / p.relative_to(root), {'feedback': d.get('feedback'),
                    'user_count': len(d['users']) if isinstance(d.get('users'), (dict, list)) else d.get('users')},
                        'Aggregate directional feedback only; simulation-input and per-user replies excluded.')
        baseline = Path(config['baseline']['source'])
        copy(baseline, target / 'baseline.py')
        base_text = baseline.read_text()
        for p in sorted((root / 'trials').glob('*/receipt.json')):
            d = read(p)
            source = p.parent / 'train.py'
            if source.exists():
                diff = ''.join(difflib.unified_diff(base_text.splitlines(True), source.read_text().splitlines(True),
                    fromfile='corrected-baseline.py', tofile=p.parent.name + '/train.py'))
                write(str(target / 'changes-from-baseline' / (p.parent.name + '.diff')), diff)
            trials.append({'method': method, 'trial': d['id'], 'status': d['status'],
                'validation_hr10': d.get('score'), 'gpu_seconds': d['allocated_gpu_seconds'],
                'source_sha256': d.get('source_sha256'), 'gpu': d.get('gpu'),
                'receipt': str(target / p.relative_to(root)),
                'source': str(target / 'trials' / p.parent.name / 'train.py'),
                'selected': d['id'] == summary['selected']['id']})
        usage = calls(root, method, config, call_rows, attempts, tokens)
        own = [t for t in trials if t['method'] == method]
        if method == 'recharness':
            best_observed = None
            for p in sorted((root / 'native/logs').glob('iteration_*.json')):
                d = read(p)
                idx = d['selection']['round_idx']
                group = [t for t in own if t['trial'].startswith(f'round-{idx:04d}-')]
                scores = [t['validation_hr10'] for t in group if t['status'] == 'success']
                if scores:
                    best_observed = max(scores + ([best_observed] if best_observed is not None else []))
                rounds.append({'execution_round': idx, 'native_iteration': d['iteration'],
                    'selected_arms': d['selection'].get('selected_arms'),
                    'is_jumping_round': d['selection'].get('is_jumping_round'),
                    'trial_count': len(group), 'best_group_validation_hr10': max(scores) if scores else None,
                    'best_observed_validation_hr10': best_observed,
                    'group_gpu_seconds': sum(t['gpu_seconds'] for t in group),
                    'cumulative_gpu_seconds': 43200 - d['budget']['after_secs'],
                    'record': str(target / p.relative_to(root))})
        comparisons.append({'method': method, 'family': 'bert4rec', 'status': summary['status'],
            'validation_hr10': summary['selected']['score'], 'test_hr10': summary['test']['score'],
            'test_ndcg10': summary['test']['aggregate']['NDCG@10'],
            'selected_trial': summary['selected']['id'], 'budget_gpu_seconds': budget['total_gpu_seconds'],
            'spent_gpu_seconds': budget['spent_gpu_seconds'], 'trial_count': len(own),
            'trial_status_counts': dict(collections.Counter(t['status'] for t in own)),
            'llm_model': config['llm_model'], 'reasoning_effort': config['llm_effort'],
            'original_root': str(root.relative_to(WORK)), **usage})
        for split, data in [('validation', summary['selected']), ('test', summary['test'])]:
            for name, ds in data['per_dataset'].items():
                datasets.append({'method': method, 'split': split, 'dataset': name,
                                 'n_users': ds['n_users'], **ds['metrics']})
    # Freeze manifests are copied intact; source snapshots are matched against their registered hashes.
    pinned = []
    manifests = [(FAMILY / 'frozen.json', 'family'), (REC / 'frozen.json', 'recharness'),
        (FAMILY / 'recovery-20261030/registration.json', 'recovery')]
    for path, label in manifests:
        registration = read(path)
        copy(path, Path('protocol') / label / path.name)
        for name, expected in registration['files'].items():
            p = Path(name)
            if isinstance(expected, dict):
                expected = expected.get('sha256')
            actual = sha(p.read_bytes()) if p.is_file() else None
            record = {'manifest': label, 'source': str(p.relative_to(WORK)), 'registered_sha256': expected,
                      'current_sha256': actual, 'matches_registration': actual == expected}
            # Only code/protocol metadata. Other model run directories are outside this export's scope.
            allowed = p.suffix in {'.py', '.md', '.txt', '.toml', '.sh'} or p.name in {'LICENSE', 'NOTICE'}
            if '/runs/' not in str(p) and '/data/' not in str(p) and allowed and actual == expected:
                target = Path('source-at-export') / p.relative_to(WORK)
                copy(p, target)
                record['export'] = str(target)
            pinned.append(record)
    for p in ['amazon_family_compare', 'amazon_family_recovery', 'amazon_aligned', 'mle_star_codex']:
        selected(WORK / p, Path('source-at-export') / p, ['*.py', '*.md'])
    fork = WORK / 'artifacts/RecHarness-family-jump-v4-20261031'
    selected(fork, 'source-at-export/artifacts/RecHarness-family-jump-v4-20261031',
             ['*.json', '*.md', '*.txt', '*.toml', 'LICENSE', 'tests/*.py'])
    for label, base in [('family', FAMILY), ('recharness', REC)]:
        selected(base, Path('protocol') / label, ['baselines/bert4rec/*.py', 'baselines/bert4rec/*.json',
            'baselines/bert4rec/*.diff', 'arm-policy.json', 'preflight/acceptance.json',
            'preflight/bert4rec/**/*.json', 'audit/*.json'])
    selected(FAMILY / 'recovery-20261030', 'protocol/recovery',
             ['jobs/bert4rec-mlestar/**/*.json', 'jobs/bert4rec-mlestar/*.log',
              'jobs/bert4rec-self_evolverec/**/*.json', 'jobs/bert4rec-self_evolverec/*.log', 'tests.xml'])
    selected(WORK / 'artifacts/amazon-aligned-20261022/data', 'protocol/data', ['*.prepared.json', 'prepared.json'])
    for name in ['RecHarness-6lyc', 'MLEvolve', 'Self-EvolveRec-code']:
        selected(WORK / 'artifacts' / name, Path('protocol/licenses') / name, ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE'])
    for name, rows in [('comparisons', comparisons), ('per-dataset', datasets), ('iterations', trials),
                       ('calls', call_rows), ('llm-attempts', attempts), ('token-events', tokens),
                       ('recharness-edits', edits), ('recharness-rounds', rounds), ('registered-source-audit', pinned)]:
        tabulate(name, rows)
    save('copy-sources.json', COPIES)
    save('derived-sources.json', DERIVED)
    files = {}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts:
            data = p.read_bytes()
            check(data, p)
            files[str(p.relative_to(OUT))] = {'sha256': sha(data), 'bytes': len(data)}
    save('manifest.json', {'exported_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'files': files, 'file_count': len(files), 'total_bytes': sum(x['bytes'] for x in files.values())})
    print(json.dumps({'files': len(files), 'bytes': sum(x['bytes'] for x in files.values()), 'runs': comparisons}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
