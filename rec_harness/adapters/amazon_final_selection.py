"""Seal selected validation checkpoints before any host-only final-test access."""
import fcntl
from pathlib import Path
from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from rec_harness.process_identity import assert_stopped
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_report import audit_job


def package_code_hashes():
    root=Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)):file_hash(path) for path in sorted(root.rglob('*.py'))}


def seal_selection(study,output,models,test_directory,*,eval_seed,eval_users=1000):
    study=Path(study).resolve();output=Path(output).resolve()
    if output==study or study in output.parents or output in study.parents:
        raise ContractError('final evaluation must live outside research study')
    if not isinstance(models,list) or len(models)<2 or any(not isinstance(x,str) for x in models) or len(set(models))!=len(models):
        raise ContractError('distinct comparator and selected model IDs required')
    if type(eval_seed) is not int or eval_seed<0 or type(eval_users) is not int or eval_users<1:
        raise ContractError('fixed nonnegative evaluation seed and positive user count required')
    request={'study':str(study),'models':models,'test_directory':str(Path(test_directory).resolve()),'eval_seed':eval_seed,'eval_users':eval_users}
    with (study/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(study/'memory')
        try:
            store.verify();events=store.events()
            sealed=[e for e in events if e['kind']=='amazon_final_selection_sealed']
            plan_path=output/'selection.json'
            if sealed:
                if len(sealed)!=1 or not plan_path.exists():raise ContractError('study already sealed for another final evaluation')
                plan=read_json(plan_path)
                if plan['request']!=request or plan['hash']!=digest({k:v for k,v in plan.items() if k!='hash'}) or sealed[0]['payload']!={'selection_hash':plan['hash']}:
                    raise ContractError('final selection differs from committed seal')
                return load_selection(output)
            if any(e['kind']=='amazon_campaign_continued' for e in events):raise ContractError('continued study cannot be sealed for final evaluation')
            assert_stopped(study/'jobs')
            if store.db.execute("SELECT 1 FROM budget WHERE status='running'").fetchone():raise ContractError('research work is still unsettled')
            cfg=load_config(study,store)
            if any(m not in cfg['model_versions'] for m in models):raise ContractError('unregistered final model')
            if not cfg['confirmation_seeds']:raise ContractError('confirmed checkpoints required before final evaluation')
            jobs=[];identities=set();vocabularies={}
            for dataset in cfg['datasets']:
                for seed in cfg['confirmation_seeds']:
                    for model in models:
                        result,manifest=audit_job(study,cfg,model,dataset,seed)
                        source=cfg['model_versions'][model];checkpoint=study/'jobs'/f'{model}-{dataset}-{seed}'/'checkpoint.pt'
                        if file_hash(source['path'])!=source['sha256'] or file_hash(checkpoint)!=result['checkpoint_sha256']:
                            raise ContractError('final model or checkpoint bytes changed')
                        itemnum=manifest['itemnum']
                        if type(itemnum) is not int or itemnum<1 or vocabularies.setdefault(dataset,itemnum)!=itemnum:
                            raise ContractError('final comparison requires identical dataset vocabularies')
                        identities.add(result.get('execution_id','direct-reviewed-v1'))
                        jobs.append({'model_version':model,'dataset':dataset,'seed':seed,'source':source,
                            'checkpoint':{'path':str(checkpoint),'sha256':result['checkpoint_sha256']},
                            'itemnum':manifest['itemnum'],'validation_receipt_hash':digest(result)})
            if len(identities)!=1:raise ContractError('final comparison mixes execution versions')
            plan={'schema':'amazon-final-selection-v1','request':request,'study_config_hash':digest(read_json(study/'study.json')),
                'research_event_head':store.verify()['head'],'protocol_hash':cfg['protocol_hash'],
                'datasets':cfg['datasets'],'seeds':cfg['confirmation_seeds'],'data':cfg['data'],'data_hashes':cfg['data_hashes'],
                'jobs':jobs,'execution_id':next(iter(identities)),
                'python':cfg['python'],
                'package_code':package_code_hashes(),
                'evaluation_code':{name:file_hash(Path(__file__).with_name(name)) for name in ['amazon_final.py','amazon_final_data.py','independent_scoring.py','score_worker.py']},
                'final_protocol':{'histories':'train followed by validation interactions','positive':'first test interaction',
                    'candidates':'positive plus 99 unique negatives excluding known history','unknown_items':'reject, do not silently drop users',
                    'users':'sorted test users, seeded uniform sample up to eval_users; missing train/validation history rejects the population',
                    'ties':'strictly greater negatives; report optimistic-tie counts',
                    'aggregation':'unweighted category mean per initialization; report all frozen initializations',
                    'checkpoint_selection':'frozen validation-selected checkpoints; no retraining or test-based selection'},
                'scope':'Host-only final evaluation plan; test data not opened during sealing. No return to model optimization in this study.'}
            plan['hash']=digest(plan)
            output.mkdir(parents=True,exist_ok=True);output.chmod(0o700)
            if plan_path.exists() and read_json(plan_path)!=plan:raise ContractError('uncommitted final selection changed')
            write_json(plan_path,plan)
            store.event('amazon_final_selection_sealed',{'selection_hash':plan['hash']})
            return plan
        finally:store.close()


def load_selection(output):
    plan=read_json(Path(output)/'selection.json')
    if plan.get('schema')!='amazon-final-selection-v1' or plan.get('hash')!=digest({k:v for k,v in plan.items() if k!='hash'}):
        raise ContractError('final selection hash changed')
    study=Path(plan['request']['study'])
    if digest(read_json(study/'study.json'))!=plan['study_config_hash']:raise ContractError('final study configuration changed')
    store=Store(study/'memory')
    try:
        integrity=store.verify();sealed=[e for e in store.events() if e['kind']=='amazon_final_selection_sealed']
        if len(sealed)!=1 or sealed[0]['payload']!={'selection_hash':plan['hash']} or sealed[0]['previous_hash']!=plan['research_event_head'] or integrity['head']!=sealed[0]['hash']:
            raise ContractError('research changed after final selection')
    finally:store.close()
    for job in plan['jobs']:
        for key in ['source','checkpoint']:
            if file_hash(job[key]['path'])!=job[key]['sha256']:raise ContractError('frozen final model bytes changed')
    for name,expected in plan['evaluation_code'].items():
        if file_hash(Path(__file__).with_name(name))!=expected:raise ContractError('frozen final evaluator changed')
    if plan.get('package_code')!=package_code_hashes():
        raise ContractError('frozen final package code changed or absent; use the original frozen evaluator')
    return plan
