"""Separate model promotion from exploratory mechanism comparisons.

Only plans explicitly frozen with this policy use it. Historical plans retain
exactly their original readouts and selection gates.
"""
import statistics
from rec_harness.core import ContractError, digest

POLICY = 'baseline-positive-v1'
SINGLE_SEED_POLICY = 'baseline-single-seed-v1'


def validate_single_seed(config):
    seeds=config.get('development_seeds')
    if not isinstance(seeds,list) or len(seeds)!=1 or type(seeds[0]) is not int or config.get('confirmation_seeds')!=[]:
        raise ContractError('single-seed promotion requires one development seed and no confirmation seeds')


def validate_policy(policy):
    if policy not in (POLICY, SINGLE_SEED_POLICY, 'legacy-candidate-controls-v1'):
        raise ContractError('unknown Amazon selection policy')


def development_summary(config, plan, means, effects, slices, rows):
    policy = plan['selection_policy']
    if policy not in (POLICY, SINGLE_SEED_POLICY):
        raise ContractError('baseline promotion requires baseline-positive-v1')
    if policy == SINGLE_SEED_POLICY:validate_single_seed(config)
    baseline = plan['baseline']
    candidate = plan['versions']['candidate']['model_version']
    controls = [v['model_version'] for v in plan['versions']['controls']]
    # Previously exposed comparator outcomes are not newly generated proposals.
    arms = [candidate, *[v for v in controls if v != baseline and
                         v not in plan.get('reused_control_versions', [])]]
    gains = {v: means[v] - means[baseline] for v in arms}
    # Freeze ranking before confirmation. Ties favor candidate, then plan order.
    order = sorted([v for v in arms if gains[v] > 0], key=lambda v: -means[v])
    proposal = plan['cards']['proposal']
    gap = plan['cards']['gap']
    comparisons = []
    for control in controls:
        if control == baseline:
            continue
        comparisons.append({
            'candidate': candidate, 'control': control, 'baseline': baseline,
            'source_hashes': {v['model_version']: v['source']['sha256']
                              for v in [plan['versions']['candidate'], *plan['versions']['controls']]},
            'design': {'mechanism': proposal['mechanism'],
                       'insertion_contract': proposal['insertion_contract'],
                       'falsification_condition': proposal['falsification_condition'],
                       'alternative_explanations': proposal['alternative_explanations'],
                       'predicted_observations': gap['predicted_observations']},
            'candidate_minus_baseline': means[candidate] - means[baseline],
            'control_minus_baseline': means[control] - means[baseline],
            'candidate_minus_control': means[candidate] - means[control],
            'slices': [{'dataset': s['dataset'], 'seed': s['seed'],
                        'candidate_minus_control': s['effects'][control]} for s in slices],
            'design_hash': digest(plan['cards']),
            'mechanism_status': 'unknown',
            'next_step': 'Assess whether the implemented contrast isolates the proposed mechanism; design a discriminating follow-up for competing explanations.',
            'qualification': 'Paired development observation, not causal proof or independent replication.'})
    return {'metric': config['protocol']['metric'], 'means': means,
            'candidate_minus_comparator': effects, 'slices': slices,
            'minimum_effect': 0, 'protocol_minimum_effect': config['protocol']['minimum_effect'],
            'selection_policy': policy, 'model_minus_baseline': gains,
            'eligible_models': order, 'promotion_order': order,
            'eligible_for_confirmation': bool(order),
            'mechanism_comparisons': comparisons,
            'selection_rule': ('Promote the highest development mean among new candidate/controls exceeding baseline on the single frozen seed; no confirmation phase.' if policy == SINGLE_SEED_POLICY else 'Each new candidate/control qualifies when development mean minus baseline > 0; each confirmation seed must also exceed baseline. Choose first passing model in frozen development ranking.'),
            'qualification': 'Performance selection is separate from mechanism support; confirmation is not final test.'}


def confirm_promotion(root, config, baseline, summary, audit):
    """Private confirmation never reranks models or reaches research agents."""
    order = summary['promotion_order']
    if not config['confirmation_seeds']:
        raise ContractError('independent confirmation seeds required for promotion')
    results = {}
    for version in order:
        effects = {}
        for seed in config['confirmation_seeds']:
            deltas = []
            for dataset in config['datasets']:
                result, manifest = audit(root, config, version, dataset, seed)
                base, bm = audit(root, config, baseline, dataset, seed)
                def identity(r, m):
                    return (r.get('execution_id', 'direct-reviewed-v1'), m['candidates_hash'], m['data_hashes'])
                if identity(result, manifest) != identity(base, bm):
                    raise ContractError('confirmation pairing differs')
                metric = config['protocol']['metric']
                deltas.append(result['metrics'][metric] - base['metrics'][metric])
            effects[str(seed)] = statistics.mean(deltas)
        results[version] = {'effects': effects, 'eligible': all(v > 0 for v in effects.values())}
    promoted = next((v for v in order if results[v]['eligible']), None)
    return {'selection_policy': POLICY, 'models': results, 'promotion_order': order,
            'promoted_model': promoted, 'selected': promoted is not None,
            'scope': 'host-only confirmation; not sent to Codex roles or final test'}
