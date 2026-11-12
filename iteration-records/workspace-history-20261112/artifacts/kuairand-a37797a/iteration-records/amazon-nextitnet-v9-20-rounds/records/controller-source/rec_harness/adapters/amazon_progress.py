"""Committed incumbent development scores and separately audited cost axes."""
from pathlib import Path
from statistics import mean
from rec_harness.core import ContractError,digest,number,read_json
from rec_harness.cost_audit import freeze_cost_prefix,verify_cost_prefix
from rec_harness.store import Store


def round_costs(controller):
    roles=freeze_cost_prefix(controller.store,controller.store.budget()['limit'])
    study=Store(controller.study/'memory',read_only=True)
    try:
        study.db.execute('BEGIN')
        gpu=freeze_cost_prefix(study,study.budget()['limit'])
    finally:study.close()
    return {'roles':roles,'study_gpu_seconds':gpu,
        'scope':'Separate role and study ledgers, including inherited charges and failed attempts. External diagnostic/evolution ledgers are not included; these axes are not a total project cost.'}


def progress_report(directory):
    root=Path(directory).resolve();config=read_json(root/'amazon-auto.json')
    if digest(config)!=read_json(root/'auto-lock.json')['config_hash']:raise ContractError('autonomous configuration changed')
    study_root=Path(config['study']).resolve()
    controller=Store(root/'memory',read_only=True);study=Store(study_root/'memory',read_only=True)
    try:
        controller.db.execute('BEGIN');study.db.execute('BEGIN')
        integrity=controller.verify();study_integrity=study.verify()
        cfg=read_json(study_root/'study.json')
        if digest(cfg)!=read_json(study_root/'study-lock.json')['config_hash']:raise ContractError('study configuration changed')
        cycles={event['payload']['identity']:event['payload'] for event in study.events()
                if event['kind']=='amazon_research_cycle_completed'}
        rows=[]
        for event in controller.events():
            if event['kind']!='auto_round_completed':continue
            payload=event['payload'];state=payload['state'];result=payload['result']
            row={'round':state['round'],'completion_sequence':event['seq'],'event_hash':event['hash'],'incumbent':state['incumbent'],
                 'candidate':result.get('candidate'),'selected':result.get('selected',False),
                 'status':result.get('status','completed'),'failure_kind':result.get('failure_kind'),
                 'development_score':None,'development_readout_hash':None}
            prefix={'schema':'controller-cost-prefix-v1','memory':str(controller.root.resolve()),
                    'cutoff_seq':event['seq']-1,'event_head':event['previous_hash'],
                    'limit':controller.budget()['limit']}
            # Legacy role costs can be recovered exactly from their own event
            # prefix; no wall-clock alignment of separate ledgers is inferred.
            charges={};pending=set()
            for prior in controller.events()[:event['seq']-1]:
                value=prior['payload'];kind=prior['kind']
                if kind=='budget_reserved':charges[value['task_id']]=value['amount'];pending.add(value['task_id'])
                elif kind=='budget_settled':
                    if value['actual'] is not None:charges[value['task_id']]=value['actual']
                    pending.discard(value['task_id'])
            prefix.update(charged_or_reserved=sum(charges.values()),pending=len(pending))
            roles=verify_cost_prefix(controller,prefix)
            gpu=None;gpu_pending=None
            if 'cost_prefixes' in payload:
                bound=payload['cost_prefixes']
                if bound['roles']!=dict(prefix,qualification=bound['roles'].get('qualification')):
                    raise ContractError('round role cost prefix differs from completion boundary')
                gpu=verify_cost_prefix(study,bound['study_gpu_seconds'])
                gpu_pending=bound['study_gpu_seconds']['pending']
            row['costs']={'role_calls_charged_or_reserved':roles,'study_gpu_seconds_charged_or_reserved':gpu,
                          'role_pending_reservations':prefix['pending'],'study_pending_reservations':gpu_pending,
                          'study_cost_status':'recorded' if gpu is not None else 'historical_boundary_unavailable'}
            if result.get('cycle_identity'):
                cycle=cycles.get(result['cycle_identity'])
                if not cycle or cycle['readout_hash']!=result['development_hash']:raise ContractError('progress cycle identity differs')
                path=root/'rounds'/('round-'+str(state['round']))/'development-readout.json'
                report=read_json(path)
                if digest({k:v for k,v in report.items() if k!='hash'})!=report.get('hash') or report['hash']!=cycle['readout_hash']:
                    raise ContractError('progress readout differs from committed cycle')
                if report['protocol_hash']!=cfg['protocol_hash']:raise ContractError('progress protocol differs')
                selected=[item for item in report['rows'] if item['model_version']==state['incumbent']]
                expected={(dataset,seed) for dataset in cfg['datasets'] for seed in cfg['development_seeds']}
                if len(selected)!=len(expected) or {(item['dataset'],item['seed']) for item in selected}!=expected:
                    raise ContractError('incumbent development conditions incomplete')
                # Macro average datasets, then paired development seeds; never
                # choose the retrospectively best candidate or read confirmation.
                metric=cfg['protocol']['metric']
                row['development_score']=mean(mean(number(item['metrics'][metric],'development score')
                    for item in selected if item['dataset']==dataset) for dataset in cfg['datasets'])
                row['development_readout_hash']=report['hash']
            rows.append(row)
        return {'schema':'amazon-progress-v1','campaign':str(root),'metric':cfg['protocol']['metric'],
                'partition':'development','rows':rows,'controller_integrity':integrity,'study_integrity':study_integrity,
                'scope':'Incumbent recorded at each completed attempt, with public development scores only. No best-candidate hindsight, private confirmation or final-test metrics. Reopened failed attempts remain separate points. Cost axes include inherited ledger charges, not unrecorded external diagnostic/evolution costs.'}
    finally:
        controller.close();study.close()
