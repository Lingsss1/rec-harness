"""Continue three review roles after a preregistered execution was recovered."""
from pathlib import Path
import shutil

from rec_harness.core import Artifacts,ContractError,read_json,write_json,digest,file_hash
from rec_harness.store import Store
from rec_harness.cost_audit import audit_cost
from .amazon_autonomous import AmazonAutoresearch
from .amazon_controls import load_config
from .amazon_selection import development_readout
from .amazon_rescore import rescore_job


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
            for name in files:
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
        with controller.lock():
            cfg=load_config(controller.study)
            store=Store(controller.study/'memory',read_only=True)
            try:
                plans=[e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal]
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
            for row in report['rows']:
                name=f"{row['model_version']}-{row['dataset']}-{row['seed']}";directory=Path(rescores)/name
                if not (directory/'acceptance.json').is_file() or not (directory/'inference').is_dir():
                    raise ContractError('complete independent scoring required before review')
                scoring_plan=read_json(directory/'rescore-plan.json')
                training=read_json(controller.study/'jobs'/name/'completed.json')
                if training.get('execution_id')!=controller.config.get('execution_id'):
                    raise ContractError('recovered review execution differs')
                scored=rescore_job(controller.study,row['model_version'],row['dataset'],row['seed'],directory,
                    controller.config['rescore_recipe'],scoring_plan['uid_registry'],resume=True)
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
