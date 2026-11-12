"""Development-only host feedback, committed before confirmation is released."""
from rec_harness.core import Artifacts,ContractError,public_only,read_json
from .amazon_memory import validate_readout


def commit(controller,folder,report,study):
    validate_readout(report,study['protocol_hash'],study['development_seeds'])
    fields={'schema','proposal_id','plan_handoff_hash','protocol_hash','source_hashes','summary',
            'source_receipts','rows','selection_policy_provenance','hash'}
    if set(report)!=fields:raise ContractError('development feedback readout fields differ')
    public_only(report)
    objects=Artifacts(controller.root/'artifacts/objects')
    path=folder/'development-readout.json'
    if read_json(path)!=report:raise ContractError('development feedback differs from readout')
    record={'episode_id':folder.name,'protocol_hash':study['protocol_hash'],
        'development_seeds':study['development_seeds'],'readout_hash':report['hash'],'readout_ref':objects.put(path)}
    previous=[e['payload'] for e in controller.store.events() if e['kind']=='development_feedback_committed'
              and e['payload']['episode_id']==folder.name]
    if previous:
        if any(value!=record for value in previous):raise ContractError('committed development feedback changed')
    else:controller.store.event('development_feedback_committed',record)


def project(events,episode,cutoff,protocol,objects):
    matches=[e for e in events if e['seq']<=cutoff and e['kind']=='development_feedback_committed'
             and e['payload']['episode_id']==episode]
    if not matches:return {'availability':'not_committed_in_prefix'}
    event=matches[-1];record=event['payload']
    if record['protocol_hash']!=protocol:raise ContractError('development feedback protocol changed')
    report=read_json(objects.verify(record['readout_ref']))
    validate_readout(report,protocol,record['development_seeds'])
    if report['hash']!=record['readout_hash']:raise ContractError('development feedback identity changed')
    return {'availability':'committed','partition':'development','event_hash':event['hash'],'event_seq':event['seq'],
        'readout':report,'readout_ref':record['readout_ref'],
        'qualification':'Shared episode development results only; not confirmation selection, final test, or independent evidence per role.'}
