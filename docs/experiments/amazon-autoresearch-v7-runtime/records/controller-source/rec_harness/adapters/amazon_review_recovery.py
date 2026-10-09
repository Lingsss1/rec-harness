"""Continue three review roles after a preregistered execution was recovered."""
from pathlib import Path
import shutil
import sys
import uuid

from rec_harness.core import Artifacts,ContractError,read_json,write_json,digest,file_hash
from rec_harness.store import Store
from rec_harness.cost_audit import audit_cost
from rec_harness.agents import run_process
from .amazon_autonomous import AmazonAutoresearch
from .amazon_controls import load_config
from .amazon_selection import development_readout
from .amazon_execution import inspect_execution


def audit_frozen_rescore(controller,directory,row,execution):
    """Recheck receipts using their original verified scorer package and interpreter.

    Input source locations are part of CPU recovery identity. A new review
    checkout must not replace them merely because its worker bytes are equal.
    """
    request=read_json(directory/'inference/request.json')
    package=Path(request['inputs']['worker.py']['path']).resolve().parents[1]
    if set(execution['code_hashes'])!={str(p.relative_to(package)) for p in package.rglob('*.py')}:
        raise ContractError('original scorer package inventory differs')
    for name,expected in execution['code_hashes'].items():
        if file_hash(package/name)!=expected:raise ContractError('original scorer package changed')
    scoring_plan=read_json(directory/'rescore-plan.json')
    from .independent_scoring import require_scoring_runtime
    require_scoring_runtime(scoring_plan['host_scoring_runtime'])
    audit=controller.root/'scoring-audits'/uuid.uuid4().hex;audit.mkdir(parents=True)
    parameters={'study':str(controller.study),'directory':str(directory.resolve()),'row':row,
                'recipe':controller.config['rescore_recipe'],'uid_registry':scoring_plan['uid_registry']}
    write_json(audit/'request.json',parameters)
    script="""import sys
from pathlib import Path
from rec_harness.core import read_json,write_json,ContractError
from rec_harness.adapters import amazon_rescore
def forbidden(*args,**kwargs):raise ContractError('review cannot launch new inference')
amazon_rescore.run_cpu_job=forbidden
request=read_json(sys.argv[1]);row=request['row']
result=amazon_rescore.rescore_job(request['study'],row['model_version'],row['dataset'],row['seed'],request['directory'],request['recipe'],request['uid_registry'],resume=True)
write_json(sys.argv[2],result)
"""
    result_path=audit/'result.json'
    process=run_process([sys.executable,'-c',script,str(audit/'request.json'),str(result_path)],
        package.parent,audit/'process',timeout=180,
        env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(package.parent),'LANG':'C.UTF-8'})
    if process['status']!='success':raise ContractError('frozen scoring audit failed; inspect '+str(audit))
    return read_json(result_path)


def reuse_design(controller,source,folder,plan):
    """Preserve original successful calls and their event-bound provenance."""
    source=Path(source).resolve();handoff=read_json(source/'handoff.json')
    if handoff['cards']!=plan['cards']:raise ContractError('recovered design cards differ')
    if len(handoff['versions']['controls'])!=len(plan['versions']['controls']):raise ContractError('recovered control count differs')
    if read_json(source/'evidence/response.json')!=handoff['cards']['gap']:
        raise ContractError('reused evidence differs from handoff')
    hypothesis=read_json(source/'hypothesis/response.json')
    if any(hypothesis[k]!=handoff['cards'][k] for k in ['method','proposal']):
        raise ContractError('reused hypothesis differs from handoff')
    if file_hash(source/'experiment/model/model.py')!=plan['versions']['candidate']['source']['sha256']:
        raise ContractError('reused experiment differs from registered candidate')
    for expected,actual in zip([plan['versions']['candidate'],*plan['versions']['controls']],
                               [handoff['versions']['candidate'],*handoff['versions']['controls']]):
        if expected['model_version']!=actual['model_version'] or expected['source']['sha256']!=actual['source']['sha256']:
            raise ContractError('recovered design source differs')
    names=['evidence','hypothesis','experiment']
    if (source/'measurement').exists():names.insert(1,'measurement')
    old=Store(source.parent.parent/'memory',read_only=True)
    try:
        events=old.events();cost=audit_cost(old,old.budget()['limit'])
        if cost['pending']:raise ContractError('source controller still has pending calls')
        for role in names:
            src=source/role;accepted=read_json(src/'accepted.json')
            from .amazon_role_sources import verify_generated_sources
            verify_generated_sources(src,role,read_json(src/'context.json'),accepted)
            if digest(read_json(src/'response.json'))!=accepted['response_hash'] or digest(read_json(src/'context.json'))!=accepted['context_hash']:
                raise ContractError('reused role accepted bytes changed')
            if 'source_hash' in accepted and file_hash(src/'model/model.py')!=accepted['source_hash']:
                raise ContractError('reused role model changed')
            matches=[e for e in events if e['kind']=='role_completed' and e['payload']['role']==role
                     and e['payload']['task'].startswith(source.name+'-'+role+'-')
                     and all(e['payload'].get(k)==v for k,v in accepted.items())]
            if len(matches)!=1:raise ContractError('unique committed reused role required')
            objects=Artifacts(old.root.parent/'artifacts/objects')
            for name,ref in matches[0]['payload'].get('archive',{}).items():
                objects.verify(ref)
                if file_hash(src/name)!=ref['sha256']:raise ContractError('reused committed role archive changed')
            if read_json(src/'resources.json')['status']!='success':raise ContractError('reused role did not succeed')
            target=folder/role;target.mkdir(parents=True,exist_ok=True)
            files=['context.json','request.json','response.json','resources.json','schema.json','accepted.json']
            files.extend(name for name in ['literature-search.json','stdout.jsonl','process.json','launch.json','model/model.py'] if (src/name).is_file())
            files.extend('model/'+name for name in accepted.get('model_source_hashes',{}) if name!='model.py')
            files.extend(matches[0]['payload'].get('archive',{}))
            for name in dict.fromkeys(files):
                destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists() and file_hash(destination)!=file_hash(src/name):raise ContractError('reused role archive changed')
                if not destination.exists():shutil.copyfile(src/name,destination)
            reuse={'source_role_directory':str(src),'source_response_hash':accepted['response_hash'],
                   'source_event_hash':matches[0]['hash'],'source_controller_head':cost['integrity']['head'],
                   'scope':'Original call preserved, not a new invocation or new outcome-independent hypothesis.'}
            marker=target/'reuse.json'
            if marker.exists() and read_json(marker)!=reuse:raise ContractError('reuse provenance changed')
            if not marker.exists():write_json(marker,reuse)
    finally:old.close()
    return handoff,cost


def review_recovered(root,proposal,source_round,rescores):
    controller=AmazonAutoresearch(root)
    try:
        if not controller.config.get('review_only'):raise ContractError('dedicated review-only controller required')
        with controller.lock():
            cfg=load_config(controller.study)
            store=Store(controller.study/'memory',read_only=True)
            try:
                plans=[e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal]
                execution=inspect_execution(store,controller.config['execution_id'])
            finally:store.close()
            if len(plans)!=1:raise ContractError('unique preregistered plan required')
            plan=plans[0];folder=controller.root/'rounds/round-1'
            folder.mkdir(parents=True,exist_ok=True)
            handoff,cost=reuse_design(controller,source_round,folder,plan)
            inherited='recovered-design-prior-calls'
            if not controller.store.db.execute('SELECT 1 FROM budget WHERE id=?',(inherited,)).fetchone():
                amount=cost['charged_or_reserved']
                controller.store.reserve(inherited,amount);controller.store.settle(inherited,amount,'retained')
            report=development_readout(controller.study,proposal)
            candidate=plan['versions']['candidate']['model_version'];baseline=plan['baseline']
            binding={'proposal_id':proposal,'base_model_id':baseline,'candidate_model_version':candidate,
                     'controls':[v['model_version'] for v in plan['versions']['controls']],
                     'train_protocol_hash':cfg['protocol_hash'],'seeds':cfg['development_seeds']+cfg['confirmation_seeds']}
            freeze={'proposal':proposal,'plan_hash':digest(plan),'readout_hash':report['hash'],
                    'source_round':str(Path(source_round).resolve()),'rescores':str(Path(rescores).resolve()),
                    'prior_role_head':cost['integrity']['head'],'prior_role_cost':cost['charged_or_reserved']}
            marker=controller.root/'review-recovery.json'
            if marker.exists() and read_json(marker)!=freeze:raise ContractError('frozen review recovery changed')
            if not marker.exists():write_json(marker,freeze)
            from .amazon_review_history import retain_predecessor
            retain_predecessor(controller,freeze,folder)
            for row in report['rows']:
                name=f"{row['model_version']}-{row['dataset']}-{row['seed']}";directory=Path(rescores)/name
                if not (directory/'acceptance.json').is_file() or not (directory/'inference').is_dir():
                    raise ContractError('complete independent scoring required before review')
                training=read_json(controller.study/'jobs'/name/'completed.json')
                if training.get('execution_id')!=controller.config.get('execution_id'):
                    raise ContractError('recovered review execution differs')
                scored=audit_frozen_rescore(controller,directory,row,execution)
                if scored['rank_disagreements'] or scored['positive_tied_users']:raise ContractError('independent scoring gate failed')
            context={'task_id':controller.task,'model_version':candidate,'protocol_hash':cfg['protocol_hash'],
                     'protocol':cfg['protocol'],'bindings':binding,'recovery_provenance':freeze}
            write_json(folder/'handoff.json',handoff);write_json(folder/'development-readout.json',report)
            source=Path(cfg['model_versions'][candidate]['path'])
            lead,cycle=controller._review_roles(folder,context,binding,handoff['cards'],handoff['versions'],source,report)
            result={'status':'completed','cycle_identity':cycle['identity'],'selection':cycle['selection'],
                    'development_hash':report['hash'],'prior_role_cost':cost['charged_or_reserved'],
                    'role_budget':controller.store.budget(),'confirmation_released':False,
                    'scope':'Recovered development review and knowledge; confirmation requires a separate release.'}
            write_json(controller.root/'review-result.json',result)
            return result
    finally:controller.close()


def retry_review(root,reason):
    """Archive exhausted local role attempts before an explicit host-fix retry."""
    if not reason.strip():raise ContractError('review repair reason required')
    controller=AmazonAutoresearch(root,load_agent=False)
    try:
        with controller.lock():
            if not controller.config.get('review_only'):raise ContractError('review-only repair required')
            if (controller.root/'review-result.json').exists():raise ContractError('completed review cannot be reopened')
            intent_path=controller.root/'review-retry-intent.json'
            if intent_path.exists():intent=read_json(intent_path)
            else:
                folder=controller.root/'rounds/round-1'
                failed=[role for role in ['evaluation','lead','knowledge'] if (folder/role/'rejection.json').exists()
                        and not (folder/role/'accepted.json').exists()]
                if len(failed)!=1:raise ContractError('one failed review role required')
                role=failed[0];paths=[folder/role,*sorted(folder.glob(role+'-rejected-*'))]
                archive=controller.root/'review-retries'/uuid.uuid4().hex
                records={path.name:controller.artifacts.snapshot(path,[str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()]) for path in paths}
                intent={'reason':reason,'role':role,'archive':str(archive),'records':records,
                        'retained_budget':controller.store.budget(),'scope':'Host repair; failed attempts stay charged and archived.'}
                write_json(intent_path,intent)
            for name,refs in intent['records'].items():
                source=controller.root/'rounds/round-1'/name;destination=Path(intent['archive'])/name
                path=source if source.exists() else destination
                for relative,ref in refs.items():
                    controller.artifacts.verify(ref)
                    if file_hash(path/relative)!=ref['sha256']:raise ContractError('review retry source changed')
                if source.exists():
                    destination.parent.mkdir(parents=True,exist_ok=True);source.rename(destination)
            event={'intent_hash':digest(intent),**intent}
            if not any(e['kind']=='amazon_review_role_reopened' and e['payload']==event for e in controller.store.events()):
                controller.store.event('amazon_review_role_reopened',event)
            return event
    finally:controller.close()
