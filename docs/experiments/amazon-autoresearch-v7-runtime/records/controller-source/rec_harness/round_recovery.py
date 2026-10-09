"""Restore interrupted-round decisions only from committed archive references."""
from .core import ContractError,read_json,write_json


def archived(research,attempt,state):
    records=[e['payload'] for e in research.store.events()
        if e['kind']=='attempt_archived' and e['payload']['node_id']==attempt]
    if not records:return None
    refs=records[-1]['archive'];values={}
    for name in ['node.json','observations.json','decision.json']:
        if name not in refs:raise ContractError('committed round archive is incomplete')
        values[name]=read_json(research.artifacts.verify(refs[name]))
    node=values['node.json'];evaluation=values['observations.json'];decision=values['decision.json']
    if (node['node_id']!=attempt or node['primary_parent_id']!=state['incumbent']
            or node['workspace_before']!=state['workspace'] or node['protocol_hash']!=research.protocol_hash
            or evaluation['attempt']!=attempt):
        raise ContractError('committed round archive differs from interrupted round')
    if decision['selection']=='keep':
        if not evaluation.get('valid'):raise ContractError('committed keep lacks valid evaluation')
        for ref in node['workspace_after'].values():research.artifacts.verify(ref)
    directory=research.root/'research_archive/attempts'/attempt
    for name,value in values.items():write_json(directory/name,value)
    return node,evaluation,decision
