"""Export only development research memory for subsequent autonomous campaigns."""
from pathlib import Path
from rec_harness.core import ContractError,digest,read_json,write_json


def execution_failure(folder,events):
    folder=Path(folder)
    if not (folder/'failure.json').exists() or (folder/'private-confirmation-started.json').exists():return None
    result={'reason':read_json(folder/'failure.json')['reason'],'status':'unresolved',
            'scope':'execution failure; no mechanism refutation'}
    path=folder/'development-readout.json'
    if not path.exists():return result
    readout=read_json(path)
    if readout.get('hash')!=digest({k:v for k,v in readout.items() if k!='hash'}):
        raise ContractError('failure resolution readout changed')
    completed=[e['payload']['result'] for e in events if e['kind']=='auto_round_completed'
               and 'round-'+str(e['payload']['state']['round'])==folder.name]
    if completed and completed[-1].get('cycle_identity') and completed[-1].get('development_hash')==readout['hash']:
        result.update(status='resolved',resolution={'cycle_identity':completed[-1]['cycle_identity'],
            'development_hash':readout['hash'],'scope':'Development cycle subsequently completed; historical failed-call cost retained.'})
    return result


def validate_memory(pack,protocol,development_seeds=None):
    if set(pack)!={'schema','protocol_hash','entries','hash'} or pack['schema']!='amazon-development-memory-v1':
        raise ContractError('invalid development memory schema')
    if pack['protocol_hash']!=protocol or pack['hash']!=digest({k:v for k,v in pack.items() if k!='hash'}):
        raise ContractError('development memory identity mismatch')
    if not isinstance(pack['entries'],list):raise ContractError('development memory entries required')
    allowed={'round','design','development','development_decision','provisional_knowledge','execution_failure'}
    for entry in pack['entries']:
        if not isinstance(entry,dict) or not set(entry)<=allowed or 'round' not in entry:
            raise ContractError('non-development field in research memory')
        if 'development' in entry:validate_readout(entry['development'],protocol,development_seeds)
    return pack


def validate_readout(value,protocol,development_seeds=None):
    if value.get('schema')!='amazon-development-selection-v1' or value.get('protocol_hash')!=protocol or value.get('hash')!=digest({k:v for k,v in value.items() if k!='hash'}):
        raise ContractError('development readout identity mismatch')
    if not isinstance(value.get('rows'),list):raise ContractError('development rows required')
    if development_seeds is not None and any(row.get('seed') not in development_seeds for row in value['rows']):
        raise ContractError('non-development seed in memory')
    return value


def validate_export_bindings(folder,events):
    """Use available committed identities; do not silently trust edited projections."""
    responses={}
    for role in ['evidence','hypothesis','lead','knowledge']:
        path=folder/role/'response.json'
        completed=[e['payload'] for e in events if e['kind']=='role_completed'
                   and e['payload'].get('role')==role
                   and e['payload'].get('task','').startswith(folder.name+'-'+role+'-')]
        if not path.exists():
            if completed:raise ContractError('development memory missing committed response: '+role)
            continue
        value=read_json(path)
        if completed:
            if not any(e.get('response_hash')==digest(value) for e in completed):
                raise ContractError('development memory role differs from committed response: '+role)
            responses[role]=value
        elif (folder/role/'reuse.json').exists():
            from .amazon_auto_audit import verify_reused_role
            accepted=read_json(folder/role/'accepted.json')
            if digest(value)!=accepted['response_hash']:raise ContractError('reused memory response changed: '+role)
            verify_reused_role(folder/role,accepted);responses[role]=value
    if (folder/'handoff.json').exists() and {'evidence','hypothesis'}<=responses.keys():
        expected={'gap':responses['evidence'],**responses['hypothesis']}
        if read_json(folder/'handoff.json')['cards']!=expected:
            raise ContractError('development memory handoff differs from committed roles')
    completed=[e['payload']['result'] for e in events if e['kind']=='auto_round_completed'
               and 'round-'+str(e['payload']['state']['round'])==folder.name]
    if completed and completed[-1].get('cycle_identity'):
        path=folder/'development-readout.json'
        if not path.exists() or read_json(path).get('hash')!=completed[-1]['development_hash']:
            raise ContractError('development memory differs from committed readout')


def export_development_memory(campaign,output):
    campaign=Path(campaign);config=read_json(campaign/'amazon-auto.json')
    study=read_json(Path(config['study'])/'study.json');entries=[]
    if digest(study)!=config['study_config_hash']:raise ContractError('bound study changed')
    if config.get('review_only'):
        if not (campaign/'review-result.json').exists():raise ContractError('recovered review must finish before memory export')
        from .amazon_autonomous import AmazonAutoresearch
        from .amazon_auto_audit import verify_campaign
        controller=AmazonAutoresearch(campaign,load_agent=False)
        try:
            audit=verify_campaign(controller)
            if audit['verified_cycles']!=1 or audit['roles_without_completion_event']:
                raise ContractError('recovered review lacks a fully bound cycle')
        finally:controller.close()
    events=[]
    if (campaign/'memory/memory.sqlite').exists():
        from rec_harness.store import Store
        store=Store(campaign/'memory')
        try:store.verify();events=store.events()
        finally:store.close()
    for folder in sorted((campaign/'rounds').glob('round-*')):
        validate_export_bindings(folder,events)
        entry={'round':config['task_id']+'/'+folder.name}
        for name,key in [('handoff.json','design'),('lead/response.json','development_decision'),('knowledge/response.json','provisional_knowledge')]:
            if (folder/name).exists():
                value=read_json(folder/name);entry[key]=value['cards'] if key=='design' else value
        path=folder/'development-readout.json'
        if path.exists():
            value=read_json(path)
            validate_readout(value,config['protocol_hash'],study['development_seeds'])
            entry['development']=value
        failure=execution_failure(folder,events)
        if failure is not None:entry['execution_failure']=failure
        if len(entry)>1:entries.append(entry)
    pack={'schema':'amazon-development-memory-v1','protocol_hash':config['protocol_hash'],'entries':entries}
    pack['hash']=digest(pack);validate_memory(pack,config['protocol_hash']);write_json(output,pack);return pack
