"""Public, event-bound implementation checks for development reviewers."""
from pathlib import Path

from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json
from rec_harness.store import Store


def experiment_report(directory, events):
    """Carry implementation limitations forward without treating prose as tests."""
    from rec_harness.agents import schema_for, validate_schema
    from rec_harness.core import safe_file
    directory=Path(directory);folder=directory/'experiment'
    completed=[event['payload'] for event in events if event['kind']=='role_completed'
               and event['payload'].get('role')=='experiment'
               and event['payload'].get('task','').startswith(directory.name+'-experiment-')]
    if not (folder/'response.json').exists():
        if completed:raise ContractError('committed experiment report missing')
        return {'status':'unavailable','scope':'No archived implementation report.'}
    response=read_json(folder/'response.json')
    if completed:
        if not any(item.get('response_hash')==digest(response) for item in completed):
            raise ContractError('experiment report differs from completion event')
    elif (folder/'reuse.json').exists():
        from .amazon_auto_audit import verify_reused_role
        verify_reused_role(folder,read_json(folder/'accepted.json'))
    else:
        return {'status':'unavailable','scope':'Implementation report has no bound completion event.'}
    accepted=read_json(folder/'accepted.json')
    if completed and not any(all(item.get(key)==value for key,value in accepted.items()) for item in completed):
        raise ContractError('experiment report receipt differs from completion event')
    if accepted.get('response_hash')!=digest(response) or accepted.get('context_hash')!=digest(read_json(folder/'context.json')):
        raise ContractError('experiment report accepted identity changed')
    validate_schema(response,schema_for('experiment'))
    hashes=accepted.get('model_source_hashes',{})
    # Older single-candidate roles bind the edited source under source_hash.
    # Retain that binding when no multi-file receipt was needed.
    if not hashes and accepted.get('source_hash'):
        hashes={'model.py':accepted['source_hash']}
    for name,sha in hashes.items():
        if file_hash(safe_file(folder/'model',name))!=sha:
            raise ContractError('experiment report source changed')
    if hashes and set(hashes)!=set(response['changed_files']):
        raise ContractError('experiment report changed files differ from accepted sources')
    return {'status':'identity_verified','origin':'agent_hypothesis',
        'response_hash':digest(response),'context_hash':accepted['context_hash'],
        'changed_files':response['changed_files'],'source_hashes':hashes,
        'reported_checks':response['implementation_checks'],'reported_limitations':response['limitations'],
        'checks_verified_by_this_report':False,
        'scope':'Event-bound implementation author report, not independently verified test evidence. Empty limitations do not prove complete verification. Compare with host admission/training evidence and the proposal; preserve unexecuted or unresolved checks in the decision.'}


def protocol_summary(config,result,manifest):
    """Project an already audited development receipt, without copying private data."""
    keys=('epochs','batches','eval_users','eval_seed','patience','eval_every')
    if not all(k in manifest.get('protocol',{}) for k in keys):
        return {'status':'unavailable','scope':'No complete execution protocol manifest supplied.'}
    if result['seed'] not in config['development_seeds']:
        raise ContractError('non-development protocol evidence')
    if any(manifest['protocol'][k]!=config['protocol'][k] for k in keys):
        raise ContractError('public execution protocol differs')
    if result['worker_hash']!=manifest['worker_hash'] or result['worker_hash']!=config['worker_sha256']:
        raise ContractError('public execution worker differs')
    if manifest['initialization']!='fresh':raise ContractError('public initialization differs')
    precision={}
    for key in ('cuda_matmul_allow_tf32','cudnn_allow_tf32'):
        if key in config['protocol']:
            if type(manifest.get(key)) is not bool or manifest[key]!=config['protocol'][key]:
                raise ContractError('public execution precision differs: '+key)
            precision[key]=manifest[key]
    return {'status':'verified','dataset':result['dataset'],'seed':result['seed'],
            'execution_id':result.get('execution_id','direct-reviewed-v1'),
            'worker_sha256':result['worker_hash'],'protocol_hash':config['protocol_hash'],
            'protocol':{**{k:manifest['protocol'][k] for k in keys},**precision},'initialization':'fresh',
            'receipt_hash':digest(result),'manifest_hash':digest(manifest),
            'scope':'Host audit of completed development receipt and manifest, not candidate-source helper defaults. No claim of scientific confirmation or independent verification by the role.'}


def admission_evidence(study,proposal_id,protocol_hash,candidate_id,candidate_path):
    study=Path(study)
    store=Store(study/'memory',read_only=True)
    try:
        store.db.execute('BEGIN')
        integrity=store.verify()
        records=[e['payload'] for e in store.events()
                 if e['kind']=='amazon_model_candidate_admitted' and e['payload']['proposal_id']==proposal_id]
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()
    if not records:
        return {'status':'unavailable','scope':'No admission event in this study; do not infer that checks passed or failed.'}
    if len(records)!=1:raise ContractError('ambiguous reviewer admission evidence')
    record=records[0]
    if record['protocol_hash']!=protocol_hash or candidate_id not in record['versions']:
        raise ContractError('reviewer admission identity mismatch')
    item=record['versions'][candidate_id]
    artifacts=Artifacts(study/'artifacts/objects')
    source=artifacts.verify(item['source_ref'])
    if file_hash(source)!=file_hash(candidate_path) or file_hash(source)!=item['source']['sha256']:
        raise ContractError('reviewer candidate differs from admitted source')
    probe=read_json(artifacts.verify(item['probe_ref']))
    from rec_harness.model_graph import tensor_graph_summary,sequence_semantics_summary
    graph_summary=tensor_graph_summary(probe,candidate_id,item['source']['sha256'])
    gate=item['gate']
    if gate['status']!='success':raise ContractError('reviewer numerical admission failed')
    smoke=item.get('training_smoke')
    if smoke:
        artifacts.verify(smoke['report_ref'])
        if smoke['status']!='success':raise ContractError('reviewer training smoke failed')
    source_review=item.get('source_contract',record['source_contract'])
    result={'status':'verified','proposal_id':proposal_id,'protocol_hash':protocol_hash,
            'model_version':candidate_id,'source_sha256':item['source']['sha256'],
            'admission_hash':digest(record),'study_event_head':integrity['head'],
            'source_contract':source_review['contract'],
            'numerical_contract':{'status':gate['status'],'report_sha256':item['probe_ref']['sha256']},
            'tensor_graph':graph_summary,'sequence_semantics':sequence_semantics_summary(probe),
            'training_smoke':({'status':smoke['status'],'report_sha256':smoke['report_ref']['sha256']} if smoke else {'status':'unavailable'}),
            'scope':'Host-verified source admission and synthetic CPU checks only. No proof of mechanism, recommendation benefit, training protocol overrides or final-test performance.'}
    if 'structural_changes' in source_review:
        result['structural_changes']=source_review['structural_changes']
    if 'insertion_scope' in source_review:
        result['insertion_scope']=source_review['insertion_scope']
    result['hash']=digest(result)
    return result
def scope_incumbent_measurements(context):
    """Retain incumbent evidence without labeling it as candidate measurements."""
    from rec_harness.diagnostic_inputs import OBSERVATION_FIELDS
    fields=('diagnostic','measurement_baseline',*OBSERVATION_FIELDS)
    result=dict(context)
    if any(key in context for key in fields):
        result['incumbent_measurement_context']={
            key:context[key] for key in ('task_id','model_version','protocol_hash','development_baseline',*fields)
            if key in context}
        for key in fields:result.pop(key,None)
    return result
