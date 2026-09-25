"""Register measured model components; compare insertion sites as explicit analogies."""
from pathlib import Path

from .core import Artifacts, ContractError, digest, file_hash, read_json, safe_file
from .runtime import node, edge


def register_contract(store, report_path, task_id, protocol_hash, *, checkpoint_path=None):
    report = read_json(report_path)
    if digest({k: v for k, v in report.items() if k != 'hash'}) != report['hash']:
        raise ContractError('model contract report hash mismatch')
    checks = report['checks']
    if not all(checks.get(k) is True for k in ['causal_input_gradient', 'tail_readout', 'empty_history_finite', 'padding_gradient_zero']):
        raise ContractError('model correctness checks incomplete')
    identity = [task_id, protocol_hash, report['model_id'], report['source_hash']]
    if report.get('checkpoint_hash'):
        identity.append(report['checkpoint_hash'])
    model_id = 'structure-' + digest(identity)[:32]
    observation_id = 'structure-observation-' + digest([task_id, protocol_hash, report['hash']])[:32]
    known = {n['id'] for n in store.snapshot()['nodes']}
    if observation_id in known:
        return {'model_id': model_id, 'observation_id': observation_id, 'imported': False}
    artifacts = Artifacts(store.root.parent / 'artifacts/objects')
    checkpoint_ref = None
    if report.get('checkpoint_hash'):
        if checkpoint_path is None or file_hash(checkpoint_path) != report['checkpoint_hash']:
            raise ContractError('trained contract requires matching checkpoint artifact')
        checkpoint_ref = artifacts.put(checkpoint_path)
        if checkpoint_ref['sha256'] != report['checkpoint_hash']:
            raise ContractError('checkpoint changed during contract registration')
    source_refs = {}
    for label in ['source', 'inspector']:
        path = safe_file(Path(report_path).parent, report[label + '_filename'])
        if file_hash(path) != report[label + '_hash']:
            raise ContractError(f'{label} artifact hash mismatch')
        source_refs[label] = artifacts.put(path)
    artifact = artifacts.put(report_path)
    calls = report['structure']['components']
    mapping = {c['id']: 'component-' + digest([model_id, c['module_path'], c['callsite']])[:32] for c in calls}
    if len(mapping) != len(calls): raise ContractError('duplicate component callsite')
    nodes = [node(model_id, 'ModelVersion', {'code_hash': report['source_hash'], 'design_id': report['model_id'],
                'settings': report['settings'], 'source_ref': source_refs['source'],
                'structure_signature': digest(calls),
                'checkpoint_hash': report.get('checkpoint_hash'),
                'checkpoint_ref': checkpoint_ref,
                'weight_source': 'trained_checkpoint_probe' if report.get('checkpoint_hash') else 'fresh_synthetic_probe'}, task_id, protocol_hash),
             node(observation_id, 'Observation', {'kind': 'synthetic_model_contract', 'report_ref': artifact,
                'checks': checks, 'probe_protocol': report['input_protocol'], 'inspector_hash': report['inspector_hash'],
                'inspector_ref': source_refs['inspector'],
                'dataflow_coverage': report['structure']['dataflow_coverage']}, task_id, protocol_hash)]
    edges = [edge(observation_id + '-about', observation_id, model_id, 'ABOUT', [observation_id], task_id, store.version + 1)]
    for call in calls:
        component_id = mapping[call['id']]
        nodes.append(node(component_id, 'Component', {'model_version': model_id, 'source_hash': report['source_hash'],
            'module_path': call['module_path'], 'callsite': call['callsite'], 'module_type': call['module_type'],
            'input': call['input'], 'output': call['output'], 'parameters': call['parameters'],
            'shape_scope': report['input_protocol']}, task_id, protocol_hash))
        edges.append(edge(component_id + '-owner', model_id, component_id, 'HAS_COMPONENT', [observation_id], task_id, store.version + 1))
        for producer in call['direct_input_producers']:
            if producer not in mapping: raise ContractError('unresolved dataflow producer')
            relation_id = 'flow-' + digest([mapping[producer], component_id])[:32]
            relation = edge(relation_id, mapping[producer], component_id, 'FLOWS_TO', [observation_id], task_id, store.version + 1)
            relation['attrs']['measurement'] = 'direct_tensor_identity'
            edges.append(relation)
    store.patch({'base_version': store.version, 'nodes': nodes, 'edges': edges})
    return {'model_id': model_id, 'observation_id': observation_id, 'components': len(calls), 'imported': True}


def compare_sites(left, right):
    """Same path/type is a candidate correspondence, never automatic equivalence."""
    a = {(c['module_path'], c['callsite']): c for c in left['structure']['components']}
    b = {(c['module_path'], c['callsite']): c for c in right['structure']['components']}
    result = []
    for key in sorted(a.keys() & b.keys()):
        first, second = a[key], b[key]
        same_type = first['module_type'] == second['module_type']
        same_shape = first['input'] == second['input'] and first['output'] == second['output']
        result.append({'source_component': first['id'], 'target_component': second['id'],
            'module_path': key[0], 'callsite': key[1], 'same_module_type': same_type,
            'observed_shape_compatible': same_shape, 'parameter_layout_compatible': first['parameters'] == second['parameters'],
            'status': 'candidate_analogy_requires_semantic_review', 'automatic_insertion_allowed': False})
    return {'source_model': left['model_id'], 'target_model': right['model_id'], 'correspondences': result,
            'unmatched_source_sites': sorted(set(a) - set(b)), 'unmatched_target_sites': sorted(set(b) - set(a)),
            'limitation': 'module hooks do not observe every functional operation, including context blending'}
