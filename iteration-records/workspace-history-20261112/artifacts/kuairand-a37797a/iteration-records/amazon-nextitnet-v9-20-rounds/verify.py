"""Validate this review archive without GPU, network, or external dependencies."""
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def check(condition, message):
    if not condition:
        raise ValueError(message)

manifest = read(ROOT / 'source-manifest.json')['files']
check(len({x['path'] for x in manifest}) == len(manifest), 'Duplicate manifest paths')
for entry in manifest:
    path = ROOT / entry['path']
    check(path.is_file(), f'Missing source file: {path}')
    data = path.read_bytes()
    check(len(data) == entry['bytes'], f'Size mismatch: {path}')
    check(hashlib.sha256(data).hexdigest() == entry['sha256'], f'Hash mismatch: {path}')
rows = read(ROOT / 'rounds.json')
check([r['round'] for r in rows] == list(range(1, 21)), 'Incomplete round coverage')
completed, promotions = 0, []
for row in rows:
    i = row['round']
    base = ROOT / 'records/research/rounds' / f'round-{i}'
    check((ROOT / f'rounds/{i:02d}/README.md').exists(), f'Missing review {i}')
    check((base / 'researcher/response.json').exists(), f'Missing proposal {i}')
    path = base / 'development-readout.json'
    if not path.exists():
        check(i in (18, 20) and row['candidate_hr10'] is None, f'Unexpected missing result {i}')
        check(len(read(ROOT / f'rounds/{i:02d}/rejections.json')) == 2, f'Missing retry rejection {i}')
        continue
    completed += 1
    rd = read(path)
    for model, value in rd['summary']['means'].items():
        values = [x['metrics']['HR@10'] for x in rd['rows'] if x['model_version'] == model]
        check(len(values) == 4, f'Expected four categories: {i} {model}')
        check(abs(mean(values) - value) < 1e-12, f'Mean mismatch {i} {model}')
    for key, model, filename in [('candidate_hr10', f'amazon-next-v9-round-{i}-model', 'model.py'),
                                  ('control_hr10', f'amazon-next-v9-round-{i}-control-1', 'control-1.py')]:
        check(abs(row[key] - rd['summary']['means'][model]) < 1e-12, f'Summary mismatch {i}')
        digest = hashlib.sha256((base / 'experiment/model' / filename).read_bytes()).hexdigest()
        check(digest == rd['source_hashes'][model], f'Trained source mismatch {i} {model}')
    check(abs(row['baseline_hr10'] - rd['summary']['means'][row['base_model']]) < 1e-12, f'Baseline mismatch {i}')
    promotion = read(base / 'promotion.json')
    check(row['promoted_model'] == promotion.get('promoted_model'), f'Promotion mismatch {i}')
    check(promotion['confirmation_executed'] is False, f'Unexpected confirmation {i}')
    if row['promoted_model']:
        promotions.append(i)
check(completed == 18 and promotions == [8, 14, 15], 'Outcome mismatch')
usage = read(ROOT / 'role-usage.json')
resources = list((ROOT / 'records/research/rounds').glob('round-*/*/resources.json'))
check(len(usage) == len(resources) == 78, 'Role call count mismatch')
for key in ['input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens']:
    check(sum(x.get(key, 0) or 0 for x in usage) == sum((read(p).get('token_usage') or {}).get(key, 0) or 0 for p in resources), f'Token mismatch {key}')
check(len(read(ROOT / 'training-jobs.json')) == 148, 'Training coverage mismatch')
with (ROOT / 'rounds.csv').open() as f:
    csvrows = list(csv.DictReader(f))
check(len(csvrows) == 20, 'CSV coverage mismatch')
for csvrow, row in zip(csvrows, rows):
    for key, value in row.items():
        check(csvrow[key] == ('' if value is None else str(value)), f'CSV mismatch {key}')
with (ROOT / 'category-metrics.csv').open() as f:
    metrics = list(csv.DictReader(f))
check(len(metrics) == 18 * 12, 'Category metric coverage mismatch')
for row in metrics:
    rd = read(ROOT / f'records/research/rounds/round-{row["round"]}/development-readout.json')
    original = next(x for x in rd['rows'] if x['model_version'] == row['model_version'] and x['dataset'] == row['dataset'])
    for key, value in original['metrics'].items():
        check(float(row[key]) == value, f'Metric CSV mismatch {row["round"]} {key}')
links = 0
for page in [ROOT / 'README.md', ROOT / 'REVIEW.md', *ROOT.glob('rounds/*/README.md')]:
    for target in re.findall(r'\]\(([^)]+)\)', page.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        check((page.parent / target.split('#')[0]).exists(), f'Broken link {page}: {target}')
        links += 1
print(json.dumps({'status': 'passed', 'source_files': len(manifest), 'rounds': len(rows),
                  'completed_experiments': completed, 'promotions': promotions,
                  'role_calls': len(usage), 'training_jobs': 148, 'checked_links': links,
                  'scope': 'Archive integrity only; not campaign audit or causal validation'}, indent=2))
