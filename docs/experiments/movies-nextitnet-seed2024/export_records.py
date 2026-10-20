#!/usr/bin/env python3
"""Export an allowlisted, byte-preserving Movies experiment record bundle.

Run from the original rec-harness workspace. No model fitting or LLM calls.
"""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

WORK = Path.cwd()
OUT = WORK / 'docs/experiments/movies-nextitnet-seed2024'
OUT.mkdir(parents=True, exist_ok=True)
sources = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(src, dst):
    src, dst = Path(src), OUT / dst
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    sources[str(dst.relative_to(OUT))] = str(src.relative_to(WORK))


def selected(root, target, patterns):
    for pattern in patterns:
        for src in sorted(root.glob(pattern)):
            copy(src, Path(target) / src.relative_to(root))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


roots = {
    'mle-star': WORK / 'artifacts/mle-star-movies-2024',
    'autoresearch': WORK / 'artifacts/autoresearch-movies-2024',
    'mlevolve': WORK / 'artifacts/mlevolve-movies-2024',
    'd-target-norm': WORK / 'artifacts/staged-d-target-norm-2024-v2',
}
worker_patterns = [
    'models/*.py', 'trials/*/receipt.json', 'trials/*/request.json',
    'trials/*/result.json', 'trials/*/manifest.json',
    'trials/*/correctness*.json', 'trials/*/curve.jsonl', 'trials/*/worker.log',
]
for method, root in roots.items():
    selected(root, method, worker_patterns + [
        'config.json', 'frozen.json', 'selection.json', 'summary.json',
        'events.jsonl', 'search-result.json', 'outer-*.json',
        'rounds/*.json', 'results.tsv', 'progress.json', 'invalid/*.json',
        'comparison.json', 'validation-results.json', 'test-results.json',
        'control-audit.json', 'audit-repair.json', 'run-before-audit-repair.py',
        'report.md', 'final-test-recovery*.json', 'retest_final*.py',
        'embedding-model.json', 'embedding-acceptance.json',
    ])
    # Structured final proposals only; raw CLI events/prompts/sessions stay local.
    if method in ('mle-star', 'autoresearch'):
        for src in sorted(root.glob('calls/*/response.json')):
            if src.parent.name != 'backend-smoke':
                copy(src, Path(method) / 'proposals' / (src.parent.name + '.json'))

staged = WORK / 'artifacts/staged-movies-2024-aligned44'
selected(staged, 'staged', [
    'budget-correction.json', 'execution-outcomes.json', 'schedule.json',
    'report/*.json', 'report/*.csv', 'report/*.md', 'iteration-analysis/*.json',
    'iteration-analysis/*.csv', 'generate_report.py',
    'study/study.draft.json', 'study/registration.json',
])
for arm in 'ABCD':
    root = staged / 'study/trajectories/movies_nextitnet' / arm / '1'
    selected(root, f'staged/{arm}', [
        'state.json', 'results.tsv', 'artifacts/*/model.py',
        'validation/*/metrics.json', 'validation/*/transport.json',
        'validation/*/checkpoint/identity.json',
        'validation/*/adapter_scratch/training.log',
        'validation/*/adapter_scratch/training/manifest.json',
        'validation/*/adapter_scratch/training/correctness*.json',
        'validation/*/adapter_scratch/training/result.json',
        'validation/*/adapter_scratch/training/curve.jsonl',
        'hidden_test/metrics.json', 'hidden_test/transport.json',
    ])

fix = WORK / 'artifacts/evaluator-fix-v2'
selected(fix, 'evaluator-v2', [
    'legacy-manifest.json', 'change-manifest.json', 'regression.json', 'recheck.py',
    'legacy-source/**/*.py',
    'reevaluation/*.json', 'reevaluation/*.md',
    'reevaluation/validation/*/metrics.json',
    'reevaluation/validation/*/transport.json',
    'reevaluation/validation/*/prediction.log',
    'reevaluation/test/*/metrics.json', 'reevaluation/test/*/transport.json',
])
copy(WORK / 'artifacts/mle-star-movies-2024/data/provenance.json', 'protocol/data-provenance.json')
for p in (WORK / 'artifacts/mle-star-movies-2024/data').glob('*/dataset_stats.json'):
    copy(p, 'protocol/' + p.parent.name + '-dataset-stats.json')
copy(WORK / 'artifacts/staged-movies-2024/environment.json', 'protocol/environment.json')
old = WORK / 'artifacts/staged-movies-2024'
selected(old, 'staged/stopped-2000-budget', ['study/study.draft.json'])
for arm in 'ABCD':
    selected(old / 'study/trajectories/movies_nextitnet' / arm / '1',
             f'staged/stopped-2000-budget/{arm}', ['state.json', 'results.tsv', 'artifacts/*/model.py'])
for name, directory in [('RecHarness', 'RecHarness-6lyc'), ('MLEvolve', 'MLEvolve'),
                        ('autoresearch', 'karpathy-autoresearch')]:
    for filename in ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE']:
        copy(WORK / 'artifacts' / directory / filename, f'protocol/licenses/{name}/{filename}')
for package in ['mle_star_codex', 'autoresearch_codex', 'mlevolve_codex', 'nextitnet_target_norm', 'staged-autoresearch/staged_research']:
    root = WORK / package
    selected(root, 'source-at-export/' + package, ['*.py', '*.md'])
selected(WORK / 'staged-autoresearch/tests', 'source-at-export/staged-autoresearch/tests', ['test_movies*.py'])

rows = []
def row(method, ident, **fields):
    rows.append(dict(method=method, id=ident, **fields))

for method in ['mle-star', 'autoresearch']:
    root = roots[method]
    decisions = {}
    if method == 'autoresearch':
        for p in sorted(root.glob('rounds/*.json')):
            for d in read(p)['decisions']:
                decisions[d['trial']['id']] = d
    for model in sorted(root.glob('models/*.py')):
        ident = model.stem
        if ident == 'baseline':
            continue
        receipt = read(root / 'trials' / ident / 'receipt.json')
        decision = decisions.get(ident, {})
        row(method, ident, trajectory=ident.rsplit('-t', 1)[-1],
            stage=ident.rsplit('-t', 1)[0], parent_id=decision.get('parent_id', ''),
            status=receipt['status'], decision=decision.get('decision', ''),
            validation_hr10=receipt.get('metrics', {}).get('HR@10'),
            best_epoch=receipt.get('best_epoch'), completed_epochs=receipt.get('completed_epochs'),
            gpu_seconds=receipt.get('allocated_gpu_seconds'),
            model=f'{method}/models/{ident}.py', record=f'{method}/trials/{ident}/receipt.json')

for node in read(roots['mlevolve'] / 'progress.json')['nodes']:
    ident = node['id']
    receipt_path = roots['mlevolve'] / 'trials' / ident / 'receipt.json'
    receipt = read(receipt_path) if receipt_path.exists() else {}
    row('mlevolve', ident, stage=node['stage'], parent_id=node['parent'],
        status='failed' if node['buggy'] else 'success',
        validation_hr10=None if node['buggy'] else node['metric'],
        best_epoch=receipt.get('best_epoch'), completed_epochs=receipt.get('completed_epochs'),
        gpu_seconds=receipt.get('allocated_gpu_seconds'), hypothesis=node['plan'],
        model=f'mlevolve/models/{ident}.py', record='mlevolve/progress.json')

for arm in 'ABCD':
    state = read(OUT / f'staged/{arm}/state.json')
    for rec in state['records']:
        ident = f"t{rec['trial']:03d}"
        row(f'staged-{arm}', ident, trajectory=1, stage=rec['stage'],
            parent_id=rec.get('parent_trial'), status=rec['status'],
            validation_hr10=rec.get('metric'), best_so_far=rec.get('best_metric'),
            gpu_seconds=rec.get('training_seconds'), hypothesis=rec['proposal'].get('hypothesis', ''),
            model=f'staged/{arm}/artifacts/{ident}/model.py', record=f'staged/{arm}/state.json')

for rec in read(roots['d-target-norm'] / 'summary.json')['comparisons']:
    ident = rec['id']; receipt = read(roots['d-target-norm'] / f'trials/{ident}/receipt.json')
    row('d-target-norm-supplement', ident, stage='supplement',
        status=receipt['status'], validation_hr10=rec['validation']['HR@10'],
        test_hr10=rec['test']['HR@10'], best_epoch=receipt.get('best_epoch'),
        completed_epochs=receipt.get('completed_epochs'), gpu_seconds=receipt['allocated_gpu_seconds'],
        model=f'd-target-norm/models/{ident}.py', record=f'd-target-norm/trials/{ident}/receipt.json')

counts = Counter(r['method'] for r in rows)
assert len(rows) == 312, counts
assert all(counts[m] == 44 for m in ['mle-star', 'autoresearch', 'mlevolve', 'staged-A', 'staged-B', 'staged-C', 'staged-D'])
assert counts['d-target-norm-supplement'] == 4
for r in rows:
    assert (OUT / r['model']).is_file(), r
fields = ['method', 'id', 'trajectory', 'stage', 'parent_id', 'status', 'decision',
          'validation_hr10', 'best_so_far', 'test_hr10', 'best_epoch',
          'completed_epochs', 'gpu_seconds', 'hypothesis', 'model', 'record']
with (OUT / 'iterations.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)

comparison = read(OUT / 'evaluator-v2/reevaluation/summary.json')['methods']
with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['method', 'selected_id', 'validation_hr10', 'test_hr10', 'selected_kind'])
    writer.writeheader()
    for x in comparison:
        writer.writerow(dict(method=x['method'], selected_id=x.get('selected_id', ''),
            validation_hr10=x['validation']['HR@10'], test_hr10=x['test']['HR@10'],
            selected_kind=x.get('selected_kind', 'fixed ensemble')))
save('record-counts.json', dict(main_search_attempts=308, supplemental_training_attempts=4,
     attempts_by_method=dict(counts), historical_data_modified=False,
     original_absolute_paths_preserved=True,
     excluded=['datasets and per-user features', 'checkpoints', 'prediction arrays',
               'embedding model weights and vector indexes', 'credentials',
               'raw CLI prompts, reasoning events, and private session directories']))

exporter = Path(__file__).resolve()
if exporter != (OUT / 'export_records.py').resolve():
    shutil.copyfile(exporter, OUT / 'export_records.py')
save('copy-sources.json', sources)
print(json.dumps({'files_copied': len(sources), 'iterations': dict(counts),
                  'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}, indent=2))
