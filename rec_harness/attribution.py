"""Local attribution increment admission; no claim of independent causal proof."""
from .core import ContractError, required, number


def validate(report, candidate):
    required(report, ['passed', 'implementation_hash', 'evaluation_ids', 'total_cost',
                      'attribution'])
    if report['passed'] is not True or report['implementation_hash'] != candidate['implementation_hash']:
        raise ContractError('attribution execution failed or implementation differs')
    ids = report['evaluation_ids']
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
        raise ContractError('attribution requires source evaluation IDs')
    a = report['attribution']
    if not isinstance(a, dict):raise ContractError('attribution must be an object')
    required(a, ['gap_id', 'before', 'after', 'observations', 'controls',
                 'decision_before', 'decision_after', 'rationale', 'limitations'])
    for key in ['gap_id', 'decision_before', 'decision_after', 'rationale']:
        if not isinstance(a[key], str) or not a[key].strip():raise ContractError('missing attribution text: '+key)
    before, after = a['before'], a['after']
    states = {'unresolved', 'supported', 'refuted'}
    if (not isinstance(before, dict) or len(before) < 2 or not isinstance(after, dict)
            or set(before) != set(after) or any(not isinstance(k,str) or not k for k in before)
            or any(v not in states for v in list(before.values()) + list(after.values()))):
        raise ContractError('attribution requires the same competing explanations before and after')
    changed = {k for k in before if before[k] != after[k]}
    if not changed or a['decision_before'].strip() == a['decision_after'].strip():
        raise ContractError('no attribution or decision increment')
    observations = a['observations']
    if not isinstance(observations,list) or not observations:raise ContractError('actual observations required')
    covered = set()
    for observation in observations:
        required(observation,['evaluation_id','measurement','value','explanation_ids','interpretation'])
        number(observation['value'],'measurement value')
        if observation['evaluation_id'] not in ids:raise ContractError('observation outside evaluated sources')
        if not isinstance(observation['measurement'],str) or not observation['measurement'].strip():raise ContractError('named measurement required')
        refs=observation['explanation_ids']
        if not isinstance(refs,list) or not refs or any(r not in before for r in refs):raise ContractError('observation must address supplied explanations')
        if not isinstance(observation['interpretation'],str) or not observation['interpretation'].strip():raise ContractError('observation interpretation required')
        covered.update(refs)
    if not changed <= covered:raise ContractError('changed attribution lacks observations')
    controls=a['controls']
    if not isinstance(controls,list) or not controls:raise ContractError('basic measurement controls required')
    for control in controls:
        required(control,['name','passed','observed','expected'])
        if control['passed'] is not True or not control['name'] or control['observed'] is None or control['expected'] is None:
            raise ContractError('measurement control failed or missing result')
    if not isinstance(a['limitations'],list) or not a['limitations'] or any(not isinstance(x,str) or not x.strip() for x in a['limitations']):
        raise ContractError('attribution limitations required')
