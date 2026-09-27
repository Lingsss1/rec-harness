"""Replay one audited, observed Amazon development cycle; no new outcomes."""
from pathlib import Path
import statistics

from rec_harness.core import Artifacts,ContractError,digest,file_hash,number,read_json,write_json
from rec_harness.cost_audit import audit_cost
from rec_harness.replay import build_world
from rec_harness.store import Store
from .amazon_autonomous import AmazonAutoresearch
from .amazon_auto_audit import verify_campaign
from .amazon_controls import load_config
from .amazon_selection import development_readout


def build_cycle_world(*,artifacts,protocol,proposal,baseline,candidate,visible,report,selection,
                      gpu_total,candidate_gpu,role_calls,gpu_weight,role_weight,
                      request_ref,workspace_before,workspace_after):
    """The root may contain only the baseline actually visible before design."""
    for value in [gpu_total,candidate_gpu,role_calls,gpu_weight,role_weight]:number(value,'replay cost',0)
    if gpu_weight+role_weight<=0:raise ContractError('positive replay cost weight required')
    if candidate_gpu>gpu_total:raise ContractError('candidate cost exceeds retained study cost')
    if report['hash']!=digest({k:v for k,v in report.items() if k!='hash'}):raise ContractError('development readout changed')
    if report['protocol_hash']!=protocol or visible['protocol_hash']!=protocol:
        raise ContractError('replay protocol differs')
    binding=visible['bindings']
    if binding['base_model_id']!=baseline or binding['candidate_model_version']!=candidate or binding['proposal_id']!=proposal:
        raise ContractError('visible design binding differs')
    metric=report['summary']['metric']
    before={(r['dataset'],r['seed']):r['metrics'][metric] for r in visible['development_baseline']}
    actual={(r['dataset'],r['seed']):r['metrics'][metric] for r in report['rows'] if r['model_version']==baseline}
    if not before or len(before)!=len(visible['development_baseline']) or before!=actual:
        raise ContractError('replay baseline was not visible before design')
    root_score=statistics.mean(before.values());summary=report['summary']
    if root_score!=summary['means'][baseline]:raise ContractError('baseline mean differs')
    if selection not in {'keep','discard','provisional'}:raise ContractError('completed development selection required')
    if selection=='keep' and not summary['eligible_for_confirmation']:raise ContractError('selection violates frozen gate')
    score=summary['means'][candidate] if selection=='keep' else root_score
    node={'node_id':proposal,'primary_parent_id':'root','creation_seq':1,'dependencies':['root'],
          'context_dependencies_complete':True,'protocol_hash':protocol,'request_ref':request_ref,
          'workspace_before':workspace_before,'workspace_after':workspace_after,
          'cost':candidate_gpu*gpu_weight+role_calls*role_weight,
          'observation':{'score':score,'candidate_score':summary['means'][candidate],'status':'success',
              'selection':selection,'eligible_for_confirmation':summary['eligible_for_confirmation'],
              'development_hash':report['hash']}}
    world=build_world([node],artifacts=artifacts,workers=1,
        root_observation={'score':root_score,'status':'success'},initial_cost=(gpu_total-candidate_gpu)*gpu_weight)
    world.update(schema='amazon-observed-cycle-world-v1',online_utility_evidence=False,
        cost_model={'gpu_second_weight':gpu_weight,'role_call_weight':role_weight,
                    'scope':'Retained study GPU slot seconds and this campaign CLI calls; CPU and token costs excluded.'},
        resources={'retained_gpu_seconds':gpu_total,'candidate_gpu_seconds':candidate_gpu,'campaign_cli_calls':role_calls},
        support={'actions':['continue observed cycle','stop'],
                 'unsupported':['new prompt','new tool','new code','alternative within-cycle role decision','parallel GPU scheduling'],
                 'context':'All archived prior research memory is fixed at the root; this world has one recorded continuation.',
                 'score':'Selected development mean only; discard/provisional retains baseline. No confirmation or final-test score.'})
    world['hash']=digest({k:v for k,v in world.items() if k!='hash'})
    return world


def archive_cycle(campaign,output,*,gpu_weight,role_weight):
    """Freeze one complete normal or recovered cycle with its actual contexts."""
    controller=AmazonAutoresearch(campaign,load_agent=False,read_only=True)
    output=Path(output).resolve();artifacts=Artifacts(output/'artifacts/objects')
    try:
        with controller.lock():
            audit=verify_campaign(controller)
            if audit['verified_cycles']!=1 or audit['roles_without_completion_event'] or audit['role_costs']['pending']:
                raise ContractError('one complete event-bound development cycle required')
            cycles=list((controller.root/'rounds').glob('round-*/cycle.json'))
            if len(cycles)!=1:raise ContractError('one observed cycle required')
            folder=cycles[0].parent;cycle=read_json(cycles[0])
            local=read_json(folder/'development-readout.json')
            study=Store(controller.study/'memory',read_only=True)
            try:
                study_before=study.verify();config=load_config(controller.study,study)
                report=development_readout(controller.study,local['proposal_id'],store=study)
                if report!=local:raise ContractError('archived development differs from audited training')
                plans=[e['payload'] for e in study.events() if e['kind']=='amazon_research_plan_registered'
                       and e['payload']['proposal_id']==local['proposal_id']]
                if len(plans)!=1:raise ContractError('unique archived plan required')
                plan=plans[0];baseline=plan['baseline'];candidate=plan['versions']['candidate']['model_version']
                gpu=audit_cost(study,study.budget()['limit'])
                if gpu['pending']:raise ContractError('study has pending training')
                candidate_gpu=0;receipts={}
                for row in report['rows']:
                    job=f"{row['model_version']}-{row['dataset']}-{row['seed']}"
                    path=controller.study/'jobs'/job/'completed.json';receipt=read_json(path)
                    task=receipt.get('task_id',receipt['job_id'])
                    cost=study.db.execute('SELECT * FROM budget WHERE id=?',(task,)).fetchone()
                    if cost is None or cost['status']!=receipt['status'] or cost['actual']!=receipt['allocated_gpu_seconds']:
                        raise ContractError('training receipt cost differs from ledger')
                    if row['model_version']==candidate:candidate_gpu+=cost['actual']
                    receipts[job]=artifacts.put(path)
                roles={}
                for role in cycle['roles']:
                    # Actual public role inputs/outputs only; never copy private ranks or checkpoints.
                    roles[role]=artifacts.snapshot(folder/role,[name for name in
                        ['context.json','request.json','response.json','accepted.json','resources.json','schema.json','literature-search.json','stdout.jsonl','reuse.json']
                        if (folder/role/name).is_file()])
                before={};after={}
                for version,target in [(baseline,before),(candidate,after)]:
                    source=config['model_versions'][version]
                    if file_hash(source['path'])!=source['sha256']:raise ContractError('replay model source changed')
                    target['model.py']=artifacts.put(source['path'])
                visible=read_json(folder/'evidence/context.json')
                world=build_cycle_world(artifacts=artifacts,protocol=config['protocol_hash'],proposal=plan['proposal_id'],
                    baseline=baseline,candidate=candidate,visible=visible,report=report,selection=cycle['selection'],
                    gpu_total=gpu['charged_or_reserved'],candidate_gpu=candidate_gpu,
                    role_calls=audit['role_costs']['charged_or_reserved'],gpu_weight=gpu_weight,role_weight=role_weight,
                    request_ref=roles['evidence']['context.json'],workspace_before=before,workspace_after=after)
                world.update(role_archives=roles,training_receipts=receipts,
                    readout_ref=artifacts.put(folder/'development-readout.json'),cycle_ref=artifacts.put(cycles[0]),
                    source={'campaign':str(controller.root),'controller_head':audit['head'],'study_head':study_before['head']})
                world['hash']=digest({k:v for k,v in world.items() if k!='hash'})
                if study.verify()!=study_before or controller.store.verify()['head']!=audit['head']:
                    raise ContractError('source changed during replay conversion')
                path=output/'world.json'
                if path.exists() and read_json(path)!=world:raise ContractError('frozen replay world changed; use a new output')
                if not path.exists():write_json(path,world)
                return world
            finally:study.close()
    finally:controller.close()
