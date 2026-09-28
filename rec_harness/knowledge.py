"""Append-only, scoped interpretation boundary shared by research adapters."""
from .core import ContractError


def validate_knowledge_patch(patch, graph):
    if set(patch)!={'base_version','nodes','edges'} or patch['base_version']!=graph['graph_version']:
        raise ContractError('knowledge graph base differs from supplied development graph')
    known={n['id'] for n in graph['nodes']}
    task=graph['task_id']; protocol=graph.get('protocol_hash') or graph['nodes'][0]['attrs']['protocol_hash']
    new_ids=set()
    historical=set(graph.get('historical_explanation_ids',[]))
    current_facts={item['id'] for item in graph['nodes'] if item['id'] not in historical
                   and item['type'] in {'Experiment','Observation'}
                   and item['attrs'].get('origin')=='local_evidence'
                   and item['attrs'].get('partition')=='development'}
    for item in patch['nodes']:
        attrs=item['attrs']
        if item['type'] not in {'Claim','GapHypothesis','MeasurementGap'} or item['id'] in known or item['id'] in new_ids:
            raise ContractError('knowledge may only append new explanation nodes')
        if attrs.get('task_id')!=task or attrs.get('protocol_hash')!=protocol or attrs.get('origin')!='agent_hypothesis':
            raise ContractError('knowledge scope or provenance differs from supplied graph')
        if item['type']=='Claim' and attrs.get('status')!='provisional':
            raise ContractError('development claims must remain provisional')
        if not set(attrs.get('evidence_refs',[])).issubset(known):
            raise ContractError('knowledge references evidence outside development context')
        new_ids.add(item['id'])
    for relation in patch['edges']:
        if relation['source'] not in known|new_ids or relation['target'] not in known|new_ids or not set(relation['attrs'].get('evidence_refs',[])).issubset(known|new_ids):
            raise ContractError('knowledge edge leaves supplied development context')
        if relation['source'] in historical or relation['target'] in historical:
            attrs=relation['attrs']
            if (relation['source'] not in new_ids or relation['target'] not in historical
                or relation['relation'] not in {'SUPERSEDES','BASED_ON','ALTERNATIVE_TO'}
                or attrs.get('origin')!='agent_hypothesis' or attrs.get('status')!='provisional'
                or attrs.get('scope')!=task):
                raise ContractError('historical explanations require a new scoped provisional revision')
            if not set(attrs.get('evidence_refs',[])) & current_facts:
                raise ContractError('historical revision requires current development references')
