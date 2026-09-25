"""Independent validation receipts audit and seed-paired Amazon comparison."""
import argparse
import math
from pathlib import Path
import statistics

from .amazon_identity import model_sources, record_version

from rec_harness.core import ContractError, digest, file_hash, read_json, write_json


def audit_job(root, config, model_version, dataset, seed):
    directory = root / 'jobs' / f'{model_version}-{dataset}-{seed}'
    result = read_json(directory / 'completed.json')
    manifest = read_json(directory / 'manifest.json')
    checks = read_json(directory / 'correctness.json')
    ranks = read_json(directory / 'validation_ranks.json')
    candidates = read_json(directory / 'validation_candidates.json')
    def require(condition, message):
        if not condition:
            raise ContractError(f'{directory.name}: {message}')
    require(result['status'] == 'success' and result['process']['status'] == 'success', 'unsuccessful process')
    request = read_json(directory / 'request.json') if (directory / 'request.json').exists() else {}
    require(request.get('execution_id', 'direct-reviewed-v1') == result.get('execution_id', 'direct-reviewed-v1'), 'execution receipt identity')
    if request.get('contract_gate') == 'pre-and-post-v1':
        for stage in ('pre', 'post'):
            gate = result.get('contract_gates', {}).get(stage)
            require(isinstance(gate, dict) and gate.get('status') == 'success', f'{stage} contract gate missing or failed')
            report = read_json(directory / (stage + '-contract') / 'report.json')
            require(digest({k: v for k, v in report.items() if k != 'hash'}) == report['hash'] == gate['report_hash'], f'{stage} contract report hash')
            require(report['source_hash'] == result['source_hash'], f'{stage} contract source')
            if stage == 'post':
                require(report.get('checkpoint_hash') == result['checkpoint_sha256'], 'post contract checkpoint')
    require(record_version(result) == model_version and result['dataset'] == dataset and result['seed'] == seed, 'receipt identity')
    require(manifest['dataset'] == dataset and manifest['seed'] == seed, 'manifest identity')
    for key, expected in [('source_hash', model_sources(config)[model_version]['sha256']), ('worker_hash', config['worker_sha256'])]:
        require(result[key] == manifest[key] == expected, key)
    require(file_hash(directory / 'checkpoint.pt') == result['checkpoint_sha256'], 'checkpoint hash')
    require(file_hash(directory / 'validation_candidates.json') == result['candidates_hash'] == manifest['candidates_hash'], 'candidates hash')
    for split, suffix in [('train', 'train'), ('validation', 'valid')]:
        require(manifest['data_hashes'][split] == config['data_hashes'][f'{dataset}_{suffix}.txt'], 'data hash')
    for key in ('epochs', 'batches', 'eval_users', 'eval_seed', 'patience', 'eval_every'):
        require(manifest['protocol'][key] == config['protocol'][key], f'protocol {key}')
    require(manifest['initialization'] == 'fresh', 'initialization')
    require(manifest['cudnn_deterministic'] == config['protocol']['cudnn_deterministic'], 'determinism setting')
    require(checks['causal_prefix'] is True and checks['tail_readout'] is True, 'model correctness')
    require(ranks['user_ids'] == [r['user_id'] for r in candidates], 'rank user alignment')
    require(len(ranks['ranks']) == len(candidates) > 0, 'rank count')
    require(len(set(ranks['user_ids'])) == len(candidates), 'duplicate users')
    require(all(type(r) is int and 0 <= r < 100 for r in ranks['ranks']), 'rank range')
    for metric, reported in result['metrics'].items():
        kind, k = metric.split('@'); k = int(k)
        require(kind in {'HR', 'NDCG'}, 'unsupported metric')
        expected = sum((1 if kind == 'HR' else 1 / math.log2(r + 2)) if r < k else 0 for r in ranks['ranks']) / len(candidates)
        require(math.isfinite(reported) and abs(expected - reported) < 1e-12, f'rank-derived {metric}')
    return result, manifest


def paired_report(root, baseline='fixed_baseline', candidate='legacy_incumbent', output=None):
    root = Path(root)
    config = read_json(root / 'study.json')
    if digest(config) != read_json(root / 'study-lock.json')['config_hash']:
        raise ContractError('study configuration changed')
    if baseline not in model_sources(config) or candidate not in model_sources(config):
        from .amazon_controls import load_config
        config = load_config(root)
    report = {'protocol_hash': config['protocol_hash'], 'baseline': baseline, 'candidate': candidate,
              'scope': 'validation_only; confirmation seeds are new initializations, not held-out users or final test',
              'mechanism_status': 'provisional; architecture and hyperparameter changes are confounded', 'partitions': {}}
    prefix_contrast = {baseline, candidate} == {'legacy_no_prefix', 'legacy_incumbent'}
    if prefix_contrast:
        report['mechanism_status'] = 'single-factor prefix-mixing ablation; finite seeds and shared validation users limit inference'
        report['prior_exposure'] = 'incumbent results for all seeds existed before control registration; no claim of a new held-out benchmark'
    for partition in ('development', 'confirmation'):
        seeds = config[f'{partition}_seeds']
        rows, missing = [], []
        for seed in seeds:
            for dataset in config['datasets']:
                dirs = [root / 'jobs' / f'{model_version}-{dataset}-{seed}' for model_version in (baseline, candidate)]
                if not all((d / 'completed.json').exists() for d in dirs):
                    missing.append({'seed': seed, 'dataset': dataset}); continue
                left, lm = audit_job(root, config, baseline, dataset, seed)
                right, rm = audit_job(root, config, candidate, dataset, seed)
                if left.get('execution_id', 'direct-reviewed-v1') != right.get('execution_id', 'direct-reviewed-v1'):
                    raise ContractError('paired comparison mixes execution versions')
                if lm['candidates_hash'] != rm['candidates_hash'] or lm['data_hashes'] != rm['data_hashes']:
                    raise ContractError('unpaired validation identity')
                if prefix_contrast and lm['hyperparameters'] != rm['hyperparameters']:
                    raise ContractError('mechanism control changed other hyperparameters')
                rows.append({'seed': seed, 'dataset': dataset, 'baseline': left['metrics'], 'candidate': right['metrics'],
                             'delta': {m: right['metrics'][m] - left['metrics'][m] for m in left['metrics']}})
        entry = {'rows': rows, 'missing_pairs': missing, 'complete': not missing}
        if not missing:
            metric = config['protocol']['metric']
            effects = [statistics.mean(r['delta'][metric] for r in rows if r['seed'] == s) for s in seeds]
            entry.update(seed_effects=dict(zip(map(str, seeds), effects)),
                         baseline_mean=statistics.mean(r['baseline'][metric] for r in rows),
                         candidate_mean=statistics.mean(r['candidate'][metric] for r in rows),
                         mean_effect=statistics.mean(effects),
                         seed_effect_stdev=statistics.stdev(effects) if len(effects) > 1 else None,
                         minimum_effect=config['protocol']['minimum_effect'],
                         effect_gate_passed=statistics.mean(effects) > config['protocol']['minimum_effect'])
        report['partitions'][partition] = entry
    write_json(output or root / 'paired-audit-report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory')
    result = paired_report(parser.parse_args().directory)
    print({k: {f: v for f, v in p.items() if f != 'rows'} for k, p in result['partitions'].items()})
