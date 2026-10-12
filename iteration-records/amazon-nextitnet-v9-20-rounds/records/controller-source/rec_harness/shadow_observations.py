"""Read-only freezing of committed Amazon public observation prefixes."""
from pathlib import Path
from types import SimpleNamespace
from .core import Artifacts,ContractError,digest,read_json
from .diagnostic_inputs import OBSERVATION_CONTRACT,ROW_FIELDS,measurement_request,verify_observations
from .store import Store


def collect(source,study,study_config,model,runs):
    if not isinstance(source,dict) or set(source)-{'campaign','observation_ids','cutoff_seq'}:
        raise ContractError('shadow observation source requires campaign, observation_ids and optional cutoff_seq')
    ids=source.get('observation_ids')
    if not isinstance(ids,list) or not ids or not all(isinstance(v,str) and v for v in ids) or len(ids)!=len(set(ids)):
        raise ContractError('distinct shadow observation IDs required')
    campaign=Path(source['campaign']).resolve();config=read_json(campaign/'amazon-auto.json')
    config_hash=digest(config)
    if config_hash!=read_json(campaign/'auto-lock.json')['config_hash']:
        raise ContractError('shadow observation campaign changed')
    if (Path(config['study']).resolve()!=Path(study).resolve()
            or config['study_config_hash']!=digest(read_json(Path(study)/'study.json'))
            or config['protocol_hash']!=study_config['protocol_hash']):
        raise ContractError('shadow observations must belong to the bound study and protocol')
    objects=Artifacts(campaign/'artifacts/objects');store=Store(campaign/'memory',read_only=True)
    try:
        store.db.execute('BEGIN');integrity=store.verify();events=store.events()
        cutoff=source.get('cutoff_seq',integrity['events'])
        if type(cutoff) is not int or not 0<cutoff<=integrity['events']:raise ContractError('invalid observation prefix cutoff')
        prefix=events[:cutoff];version=0;hashes={0:digest({'nodes':[],'edges':[]})}
        summaries={};kinds={'activation_summary_recorded':'activation_diagnostics','block_score_summary_recorded':'block_score_diagnostics'}
        for event in prefix:
            value=event['payload']
            if event['kind']=='graph_committed':
                if value['patch']['base_version']!=version or value['before_hash']!=hashes[version]:
                    raise ContractError('observation graph prefix binding differs')
                version=value['version'];hashes[version]=value['after_hash']
            elif event['kind']=='graph_rollback':
                version=value['version'];hashes[version]=hashes[value['restored_version']]
            elif event['kind'] in kinds and value['observation_id'] in ids:
                if value['observation_id'] in summaries:raise ContractError('duplicate observation summary receipt')
                summaries[value['observation_id']]=(kinds[event['kind']],value,event['hash'])
        snapshot=store.snapshot(version)
        if snapshot['hash']!=hashes[version]:raise ContractError('observation snapshot differs from committed prefix')
        known={n['id']:n for n in snapshot['nodes']};observations={name:[] for name in kinds.values()};refs={}
        for observation in sorted(ids):
            if observation not in summaries or observation not in known:
                raise ContractError('observation is not committed in requested prefix')
            field,receipt,event_hash=summaries[observation]
            row=read_json(objects.verify(receipt['summary_ref']))
            attrs=known[observation]['attrs']
            if attrs.get('summary_ref')!=receipt['summary_ref'] or set(row)!=ROW_FIELDS or row['observation_id']!=observation:
                raise ContractError('observation summary differs from graph receipt')
            observations[field].append(row)
            refs[observation]={'summary_ref':receipt['summary_ref'],'event_hash':event_hash}
        request=measurement_request(config['task_id'],model,study_config['protocol_hash'],runs,OBSERVATION_CONTRACT,observations)
        verify_observations(request,SimpleNamespace(snapshot=lambda:snapshot))
        # Source provenance stays in the host suite; the tool sees only public rows.
        request['task_id']='tool-shadow'
        binding={'campaign':str(campaign),'config_hash':config_hash,'task_id':config['task_id'],
            'cutoff_seq':cutoff,'event_head':prefix[-1]['hash'],'graph_hash':snapshot['hash'],
            'observation_ids':sorted(ids),'receipts':refs}
        return request,binding
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()
