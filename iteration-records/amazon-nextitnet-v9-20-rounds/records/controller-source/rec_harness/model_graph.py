"""Register measured model components; compare insertion sites as explicit analogies."""
from pathlib import Path

from .core import Artifacts, ContractError, digest, file_hash, read_json, safe_file
from .runtime import node, edge


def validate_tensor_graph(graph, model_id, source_hash):
    if graph.get('schema')!='observed-tensor-graph-v1' or graph.get('model_id')!=model_id or graph.get('source_hash')!=source_hash:
        raise ContractError('tensor graph identity mismatch')
    if digest({k:v for k,v in graph.items() if k!='hash'})!=graph.get('hash'):
        raise ContractError('tensor graph hash mismatch')
    tensors={row['id']:row for row in graph['tensors']}
    if len(tensors)!=len(graph['tensors']):raise ContractError('duplicate graph tensor')
    seen=set();produced=set();writes={}
    for ordinal,operation in enumerate(graph['operations']):
        identity=operation['id']
        if identity in seen or operation['ordinal']!=ordinal:raise ContractError('tensor graph operation order')
        if not set(operation['mutation_dependencies'])<=seen:raise ContractError('tensor graph forward mutation dependency')
        for ref in operation['inputs']:
            if ref not in tensors or tensors[ref]['producer'] is not None and tensors[ref]['producer'] not in seen:
                raise ContractError('tensor graph unresolved input producer')
        input_storages={tensors[ref]['storage'] for ref in operation['inputs']}
        expected={writes[key] for key in input_storages if key in writes}
        if set(operation['mutation_dependencies'])!=expected:
            raise ContractError('tensor graph storage mutation dependencies differ')
        if not set(operation['written_storages'])<=input_storages:
            raise ContractError('tensor graph mutates unknown input storage')
        for ref in operation['outputs']:
            if ref not in tensors or ref in produced or tensors[ref]['producer']!=identity:
                raise ContractError('tensor graph output producer differs')
            produced.add(ref)
        seen.add(identity)
        for key in operation['written_storages']:writes[key]=identity
    if produced!={ref for ref,row in tensors.items() if row['producer'] is not None}:
        raise ContractError('tensor graph unbound output')
    if not graph['outputs'] or not set(graph['outputs'])<=set(tensors):raise ContractError('tensor graph final outputs missing')
    return tensors


def parameter_sharing_summary(report):
    """Resolve observed use sites against registered Parameter object identities."""
    structure=report.get('structure',{})
    sharing=structure.get('parameter_sharing')
    if sharing is None:
        return {'status':'unavailable','scope':'Historical probe did not record registered parameter identity.'}
    if sharing.get('schema')!='registered-parameter-sharing-v1':
        raise ContractError('unknown parameter sharing schema')
    parameters={};names=set();uses={}
    for row in sharing['parameters']:
        identity=row['id'];aliases=row['registered_names']
        if (identity in parameters or not isinstance(aliases,list) or not aliases or
            any(not isinstance(name,str) or not name for name in aliases) or
            len(set(aliases))!=len(aliases) or names.intersection(aliases)):
            raise ContractError('invalid registered parameter aliases')
        expected='parameter-'+digest([report['model_id'],report['source_hash'],sorted(aliases)])[:32]
        if identity!=expected:raise ContractError('parameter identity differs from model and source')
        names.update(aliases);parameters[identity]=row;uses[identity]=[]
    sites=set()
    for call in structure['components']:
        if call['id'] in sites:raise ContractError('duplicate parameter use component')
        sites.add(call['id'])
        bindings=call.get('parameter_bindings')
        if not isinstance(bindings,dict) or not set(call['parameters'])<=set(bindings):
            raise ContractError('component parameter bindings incomplete')
        for local,identity in bindings.items():
            name=call['module_path']+'.'+local if call['module_path'] else local
            if identity not in parameters or name not in parameters[identity]['registered_names']:
                raise ContractError('component parameter binding does not resolve')
            uses[identity].append({'component_id':call['id'],'module_path':call['module_path'],
                                  'callsite':call['callsite'],'parameter_name':local})
    groups=[{**parameters[identity],'observed_uses':uses[identity]}
            for identity in sorted(parameters)
            if len(parameters[identity]['registered_names'])>1 or len(uses[identity])>1]
    return {'status':'verified','registered_parameter_count':len(parameters),
            'shared_groups':groups,'scope':sharing['scope']}


def sequence_semantics_summary(report, *, required=False):
    checks=report.get('checks',{})
    value=checks.get('sequence_semantics') if isinstance(checks,dict) else None
    if value is None:
        if required:raise ContractError('sequence semantics required by current inspector')
        return {'status':'unavailable','scope':'Historical probe did not record full-sequence batch semantics.'}
    if not isinstance(value,dict):raise ContractError('sequence semantic conditions must be an object')
    fields=('input_tokens_unchanged','batch_independent','batch_permutation_equivariant')
    conditions={}
    for condition in ('dense','right_aligned_histories'):
        row=value.get(condition,{})
        if not isinstance(row,dict) or any(row.get(key) is not True for key in fields):
            raise ContractError('sequence semantic checks incomplete: '+condition)
        if row.get('input_axes')!=['user','time'] or row.get('representation_axes')!=['user','time','feature']:
            raise ContractError('sequence semantic axes differ')
        conditions[condition]={key:row[key] for key in (*fields,'input_axes','representation_axes')}
    return {'status':'verified','conditions':conditions,
        'scope':'Finite synthetic eval-mode probes of input immutability and full-sequence batch independence; not all-input or proposal-specific semantic proof.'}


def tensor_graph_summary(report, model_id, source_hash, *, required=False):
    graph=report.get('tensor_graph')
    if graph is None:
        if required:raise ContractError('tensor graph required by current inspector')
        return {'status':'unavailable','scope':'Historical probe did not record tensor operations; no coverage inferred.'}
    if report.get('model_id')!=model_id or report.get('source_hash')!=source_hash:
        raise ContractError('tensor graph report model identity differs')
    tensors=validate_tensor_graph(graph,model_id,source_hash)
    counts={}
    for op in graph['operations']:counts[op['operator']]=counts.get(op['operator'],0)+1
    return {'status':'verified','graph_hash':graph['hash'],'model_version':model_id,'source_sha256':source_hash,
        'operation_count':len(graph['operations']),'operator_counts':dict(sorted(counts.items())),
        'mutation_operation_count':sum(bool(op['written_storages']) for op in graph['operations']),
        'output_metadata':[tensors[ref]['metadata'] for ref in graph['outputs']],
        'scope':graph['scope']}


def register_contract(store, report_path, task_id, protocol_hash, *, checkpoint_path=None):
    report = read_json(report_path)
    if digest({k: v for k, v in report.items() if k != 'hash'}) != report['hash']:
        raise ContractError('model contract report hash mismatch')
    tensor_graph=report.get('tensor_graph')
    tensor_values=validate_tensor_graph(tensor_graph,report['model_id'],report['source_hash']) if tensor_graph is not None else {}
    sharing=parameter_sharing_summary(report)
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
    if sharing['status']=='verified':
        sharing={**sharing,'shared_groups':[{**group,'observed_uses':[
            {**use,'component_id':mapping[use['component_id']]} for use in group['observed_uses']]}
            for group in sharing['shared_groups']]}
    nodes = [node(model_id, 'ModelVersion', {'code_hash': report['source_hash'], 'design_id': report['model_id'],
                'settings': report['settings'], 'source_ref': source_refs['source'],
                'structure_signature': digest(calls), 'parameter_sharing':sharing,
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
            'shape_scope': report['input_protocol'],
            **({'parameter_bindings':call['parameter_bindings']} if 'parameter_bindings' in call else {})}, task_id, protocol_hash))
        edges.append(edge(component_id + '-owner', model_id, component_id, 'HAS_COMPONENT', [observation_id], task_id, store.version + 1))
        for producer in call['direct_input_producers']:
            if producer not in mapping: raise ContractError('unresolved dataflow producer')
            relation_id = 'flow-' + digest([mapping[producer], component_id])[:32]
            relation = edge(relation_id, mapping[producer], component_id, 'FLOWS_TO', [observation_id], task_id, store.version + 1)
            relation['attrs']['measurement'] = 'direct_tensor_identity'
            edges.append(relation)
    if tensor_graph is not None:
        nodes[1]['attrs']['tensor_graph']={'hash':tensor_graph['hash'],'operations':len(tensor_graph['operations']),
            'scope':tensor_graph['scope'],'report_ref':artifact}
        operation_ids={op['id']:'component-'+digest([model_id,tensor_graph['hash'],op['id']])[:32]
                       for op in tensor_graph['operations']}
        for op in tensor_graph['operations']:
            component_id=operation_ids[op['id']]
            nodes.append(node(component_id,'Component',{'model_version':model_id,'source_hash':report['source_hash'],
                'kind':'tensor_operation','operator':op['operator'],'ordinal':op['ordinal'],
                'module_path':op['module_path'],'tensor_graph_hash':tensor_graph['hash'],
                'input':[tensor_values[ref]['metadata'] for ref in op['inputs']],
                'output':[tensor_values[ref]['metadata'] for ref in op['outputs']],
                'shape_scope':report['input_protocol']},task_id,protocol_hash))
            edges.append(edge(component_id+'-owner',model_id,component_id,'HAS_COMPONENT',[observation_id],task_id,store.version+1))
            dependencies={tensor_values[ref]['producer'] for ref in op['inputs']} - {None}
            dependencies.update(op['mutation_dependencies'])
            for producer in sorted(dependencies):
                relation=edge('flow-'+digest([operation_ids[producer],component_id])[:32],operation_ids[producer],component_id,
                    'FLOWS_TO',[observation_id],task_id,store.version+1)
                relation['attrs']['measurement']='observed_tensor_operation_or_storage_mutation'
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
