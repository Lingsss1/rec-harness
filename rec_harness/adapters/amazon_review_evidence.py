"""Public, event-bound implementation checks for development reviewers."""
from pathlib import Path

from rec_harness.core import Artifacts,ContractError,digest,file_hash
from rec_harness.store import Store


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
    artifacts.verify(item['probe_ref'])
    gate=item['gate']
    if gate['status']!='success':raise ContractError('reviewer numerical admission failed')
    smoke=item.get('training_smoke')
    if smoke:
        artifacts.verify(smoke['report_ref'])
        if smoke['status']!='success':raise ContractError('reviewer training smoke failed')
    result={'status':'verified','proposal_id':proposal_id,'protocol_hash':protocol_hash,
            'model_version':candidate_id,'source_sha256':item['source']['sha256'],
            'admission_hash':digest(record),'study_event_head':integrity['head'],
            'source_contract':record['source_contract']['contract'],
            'numerical_contract':{'status':gate['status'],'report_sha256':item['probe_ref']['sha256']},
            'training_smoke':({'status':smoke['status'],'report_sha256':smoke['report_ref']['sha256']} if smoke else {'status':'unavailable'}),
            'scope':'Host-verified source admission and synthetic CPU checks only. No proof of mechanism, recommendation benefit, training protocol overrides or final-test performance.'}
    result['hash']=digest(result)
    return result
