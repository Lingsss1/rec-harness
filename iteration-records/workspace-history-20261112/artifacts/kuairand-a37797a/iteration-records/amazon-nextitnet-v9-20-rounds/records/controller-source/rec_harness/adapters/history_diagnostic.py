"""Exploratory paired validation slices, without exporting user/item records."""
from collections import Counter
import math
from pathlib import Path

from rec_harness.core import ContractError, digest, file_hash, read_json, write_json
from .amazon_controls import load_config
from .amazon_report import audit_job

BINS = [(1, 5), (6, 20), (21, 50), (51, 128), (129, None)]


def summarize(left, right, lengths):
    if left['user_ids'] != right['user_ids'] or len(set(left['user_ids'])) != len(left['user_ids']):
        raise ContractError('paired diagnostic user alignment failed')
    if not len(left['user_ids']) == len(left['ranks']) == len(right['ranks']):
        raise ContractError('paired diagnostic rank count mismatch')
    groups = [[] for _ in BINS]
    for user, a, b in zip(left['user_ids'], left['ranks'], right['ranks']):
        if user not in lengths or lengths[user] < 1:
            raise ContractError('validation user has no training history')
        if any(type(rank) is not int or not 0 <= rank < 100 for rank in [a, b]):
            raise ContractError('invalid validation rank')
        for i, (low, high) in enumerate(BINS):
            if lengths[user] >= low and (high is None or lengths[user] <= high):
                groups[i].append((a, b)); break
    result = []
    for (low, high), rows in zip(BINS, groups):
        n = len(rows)
        metric = lambda r, kind: float(r < 10) if kind == 'HR@10' else (1 / math.log2(r + 2) if r < 10 else 0)
        left_metrics = {kind: sum(metric(a, kind) for a, b in rows) / n if n else None for kind in ['HR@10', 'NDCG@10']}
        right_metrics = {kind: sum(metric(b, kind) for a, b in rows) / n if n else None for kind in left_metrics}
        wins = sum(a >= 10 and b < 10 for a, b in rows)
        losses = sum(a < 10 and b >= 10 for a, b in rows)
        result.append({'history_length_min': low, 'history_length_max': high, 'users': n,
            'baseline': left_metrics, 'candidate': right_metrics,
            'delta': {key: right_metrics[key] - left_metrics[key] if n else None for key in left_metrics},
            'hit_wins': wins, 'hit_losses': losses, 'hit_unchanged': n - wins - losses})
    return result


def diagnose(root, output, baseline='legacy_no_prefix', candidate='legacy_incumbent', *, partitions=('development','confirmation')):
    partitions=list(partitions)
    if not partitions or len(set(partitions))!=len(partitions) or any(p not in {'development','confirmation'} for p in partitions):
        raise ContractError('diagnostic partitions must be unique development/confirmation phases')
    if baseline==candidate:raise ContractError('diagnostic requires distinct model versions')
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(root)
    plan = {'schema': 'paired-history-diagnostic-v1', 'baseline': baseline, 'candidate': candidate,
            'bins': BINS, 'partitions':partitions, 'metric_cutoff': 10, 'protocol_hash': config['protocol_hash'],
            'prior_exposure': 'aggregate validation results already inspected; exploratory, not preregistered confirmation',
            'hypothesis': 'candidate-minus-baseline ranking differences vary with training-history length',
            'limitation': 'observational slices do not identify causal length effects; seeds share validation users; bins use total training history, not truncated model input length'}
    write_json(output / 'plan.json', plan)
    rows, dependencies, input_hashes = [], [], {}
    for dataset in config['datasets']:
        path = Path(config['data']) / (dataset + '_train.txt')
        expected = config['data_hashes'][path.name]
        if file_hash(path) != expected: raise ContractError('training data changed')
        lengths = Counter()
        with path.open() as stream:
            for line in stream:
                user, item = map(int, line.split())
                lengths[user] += 1
        if file_hash(path) != expected: raise ContractError('training data changed during diagnostic')
        input_hashes[path.name] = expected
        for partition in partitions:
            for seed in config[partition + '_seeds']:
                a, am = audit_job(root, config, baseline, dataset, seed)
                b, bm = audit_job(root, config, candidate, dataset, seed)
                if am['candidates_hash'] != bm['candidates_hash'] or am['data_hashes'] != bm['data_hashes']:
                    raise ContractError('unpaired diagnostic inputs')
                if a.get('execution_id','direct-reviewed-v1')!=b.get('execution_id','direct-reviewed-v1'):
                    raise ContractError('unpaired diagnostic execution versions')
                ranks = []
                for version, receipt in [(baseline, a), (candidate, b)]:
                    p = root / 'jobs' / f'{version}-{dataset}-{seed}' / 'validation_ranks.json'
                    ranks.append(read_json(p)); input_hashes[str(p.relative_to(root))] = file_hash(p)
                    dependencies.append(receipt.get('task_id', receipt['job_id']))
                rows.append({'dataset': dataset, 'seed': seed, 'partition': partition,
                             'slices': summarize(*ranks, lengths)})
    report = {'plan': plan, 'plan_hash': digest(plan), 'rows': rows, 'input_hashes': input_hashes,
              'source_experiment_ids': sorted(set(dependencies)), 'diagnostic_code_hash': file_hash(__file__),
              'scope': 'validation only, aggregate slices; not new training or independent confirmation'}
    report['hash'] = digest(report)
    write_json(output / 'report.json', report)
    return report
