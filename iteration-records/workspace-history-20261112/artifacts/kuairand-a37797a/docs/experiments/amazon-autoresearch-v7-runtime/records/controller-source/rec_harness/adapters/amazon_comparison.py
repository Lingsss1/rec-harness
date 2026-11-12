"""Archive audited paired contrasts as derived facts, without inventing new runs."""
from pathlib import Path
import statistics

from rec_harness.core import Artifacts, ContractError, digest, read_json, write_json
from rec_harness.runtime import node, edge
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_evidence import ingest
from .amazon_report import paired_report


def contrast_facts(config, report, source_results):
    """One independent unit is a seed aggregate, never four dataset slices."""
    if not all(part['complete'] for part in report['partitions'].values()):
        raise ContractError('paired contrast incomplete')
    baseline, candidate = report['baseline'], report['candidate']
    identity = digest([config['protocol_hash'], baseline, candidate,
                       config['model_versions'][baseline]['sha256'], config['model_versions'][candidate]['sha256']])
    metric = config['protocol']['metric']
    summaries = []
    for partition, part in report['partitions'].items():
        expected_seeds = config[partition + '_seeds']
        for seed in expected_seeds:
            rows = [r for r in part['rows'] if r['seed'] == seed]
            if len(rows) != len(config['datasets']) or {r['dataset'] for r in rows} != set(config['datasets']):
                raise ContractError('seed aggregate requires exactly the frozen datasets')
            dependencies = []
            for row in rows:
                for version in (baseline, candidate):
                    result = source_results[(version, row['dataset'], seed)]
                    dependencies.append(result.get('task_id', result['job_id']))
            effect = statistics.mean(r['delta'][metric] for r in rows)
            summaries.append({'comparison_id': 'comparison-' + identity[:24] + '-' + str(seed),
                'independent_run_id': 'paired-seed-' + identity + '-' + str(seed),
                'seed': seed, 'partition': partition, 'baseline_model_version': baseline,
                'candidate_model_version': candidate, 'metric': metric,
                'baseline_mean': statistics.mean(r['baseline'][metric] for r in rows),
                'candidate_mean': statistics.mean(r['candidate'][metric] for r in rows),
                'effect': effect, 'minimum_effect': config['protocol']['minimum_effect'],
                'effect_gate_passed': effect > config['protocol']['minimum_effect'],
                'source_experiment_ids': sorted(dependencies), 'slices': rows,
                'cost_accounting': 'derived analysis; source training already charged, not an additional training run'})
    return summaries


def ingest_comparison(root, baseline='legacy_no_prefix', candidate='legacy_incumbent'):
    root = Path(root)
    config = load_config(root)
    output = root / 'comparisons' / (baseline + '-versus-' + candidate + '.json')
    report = paired_report(root, baseline, candidate, output=output)
    if not all(p['complete'] for p in report['partitions'].values()):
        return {'status': 'incomplete', 'missing_pairs': {k: p['missing_pairs'] for k, p in report['partitions'].items()}}
    results = {}
    for version in (baseline, candidate):
        for dataset in config['datasets']:
            for seed in config['development_seeds'] + config['confirmation_seeds']:
                results[(version, dataset, seed)] = read_json(root / 'jobs' / f'{version}-{dataset}-{seed}' / 'completed.json')
    facts = contrast_facts(config, report, results)
    ingest(root)
    store = Store(root / 'memory')
    try:
        known = {n['id'] for n in store.snapshot()['nodes']}
        artifact = Artifacts(root / 'artifacts/objects').put(output)
        task = 'amazon-paired-' + digest([baseline, candidate])[:16]
        protocol = config['protocol_hash']
        nodes, edges = [], []
        for fact in facts:
            key = fact['comparison_id']
            if key in known: continue
            if not set(fact['source_experiment_ids']).issubset(known):
                raise ContractError('source training evidence not present')
            attrs = {k: v for k, v in fact.items() if k not in {'slices', 'comparison_id'}}
            attrs.update(kind='derived_paired_comparison', valid=True, status='success',
                         mechanism_control={baseline, candidate} == {'legacy_no_prefix', 'legacy_incumbent'},
                         report_ref=artifact, scientific_unit='one seed aggregated over all frozen categories',
                         counterevidence=fact['effect'] <= 0, evidence_dependency_ids=fact['source_experiment_ids'])
            nodes.append(node(key, 'Experiment', attrs, task, protocol))
            for source in fact['source_experiment_ids']:
                edges.append(edge('comparison-source-' + digest([key, source])[:32], key, source,
                                  'DERIVED_FROM', [source], task, store.version + 1))
            for row in fact['slices']:
                slice_id = key + '-slice-' + digest(row['dataset'])[:12]
                nodes.append(node(slice_id, 'Observation', {'kind': 'paired_dataset_slice', **row,
                    'counterevidence': row['delta'][fact['metric']] < 0,
                    'evidence_dependency_ids': fact['source_experiment_ids'],
                    'independent_unit': False, 'report_ref': artifact}, task, protocol))
                edges.append(edge(slice_id + '-produced', key, slice_id, 'PRODUCES', [key], task, store.version + 1))
        if nodes:
            store.patch({'base_version': store.version, 'nodes': nodes, 'edges': edges})
        result = {'status': 'complete', 'task_id': task, 'derived_comparisons': len(facts),
                  'independent_seeds': [f['seed'] for f in facts], 'nodes_added': len(nodes),
                  'claim_created': False, 'integrity': store.verify()}
        write_json(root / 'comparison-ingest.json', result)
        return result
    finally:
        store.close()
