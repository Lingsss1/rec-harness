"""Authorized campaign continuation with retained costs and audited comparators."""
import fcntl
import shutil
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, read_json, write_json
from rec_harness.process_identity import assert_stopped
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_report import audit_job
from .amazon_study import init_study


def continue_campaign(root, predecessor, project, additional_budget, authorization):
    root,predecessor=Path(root).resolve(),Path(predecessor).resolve()
    if additional_budget<=0 or not authorization.strip():
        raise ContractError('explicit positive budget extension and authorization required')
    with (predecessor/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=Store(predecessor/'memory')
        try:
            if any(e['kind']=='amazon_final_selection_sealed' for e in prior.events()):
                raise ContractError('final evaluation selection sealed; cannot continue research')
            if any(e['kind']=='amazon_campaign_continued' for e in prior.events()):
                raise ContractError('predecessor already continued; do not fork its available budget twice')
            assert_stopped(predecessor/'jobs')
            if prior.db.execute("SELECT 1 FROM budget WHERE status='running'").fetchone():
                raise ContractError('predecessor has unsettled work')
            config=load_config(predecessor,prior);budget=prior.budget();integrity=prior.verify()
            if root.exists():raise ContractError('new campaign destination already exists')
            # Audit every reused outcome before creating the new campaign.
            receipts=[]
            for dataset in config['datasets']:
                for seed in config['development_seeds']+config['confirmation_seeds']:
                    result,manifest=audit_job(predecessor,config,'legacy_incumbent',dataset,seed)
                    receipts.append({'job_id':result['job_id'],'receipt_hash':digest(result),
                                     'manifest_hash':digest(manifest),'execution_id':result.get('execution_id','direct-reviewed-v1')})
            new_config=init_study(root,project,budget=budget['limit']+additional_budget,
                                  predecessor=predecessor,continuation_required=True)
            try:
                for key in ('protocol_hash','data_hashes','worker_sha256','datasets','development_seeds','confirmation_seeds'):
                    if new_config[key]!=config[key]:raise ContractError('continuation changed frozen '+key)
                for name in ('fixed_baseline','legacy_incumbent'):
                    if new_config['model_versions'][name]['sha256']!=config['model_versions'][name]['sha256']:
                        raise ContractError('continuation changed model source')
                for receipt in receipts:
                    job=receipt['job_id']
                    shutil.copytree(predecessor/'jobs'/job,root/'jobs'/job)
                record={'predecessor':str(predecessor),'successor':str(root),'authorization':authorization,
                        'additional_gpu_seconds':additional_budget,'prior_budget':budget,
                        'new_total_limit_gpu_seconds':new_config['budget_gpu_seconds'],
                        'predecessor_integrity':integrity,'reused_comparators':receipts,
                        'scope':'all predecessor costs retained once; existing comparators reused, not independent new runs; old rejected plans remain frozen'}
                write_json(root/'continuation.json',record)
                current=Store(root/'memory')
                try:
                    ref=Artifacts(root/'artifacts/objects').put(root/'continuation.json')
                    current.event('amazon_campaign_continuation_prepared',{**record,'artifact':ref})
                finally:current.close()
                # Seal spending in the predecessor before allowing the successor to run.
                prior.event('amazon_campaign_continued',record)
                activate_campaign(root)
                return record
            except BaseException:
                # Keep incomplete artifacts for diagnosis; run_study rejects unactivated continuations.
                raise
        finally:prior.close()


def campaign_guard(config,store):
    events=store.events()
    if any(e['kind']=='amazon_final_selection_sealed' for e in events):
        raise ContractError('final evaluation selection sealed; research execution is closed')
    if any(e['kind']=='amazon_campaign_continued' for e in events):
        raise ContractError('campaign continued; only its successor may spend further budget')
    if config.get('continuation_required'):
        if not any(e['kind']=='amazon_campaign_continuation_activated' for e in events):
            raise ContractError('campaign continuation is not activated')


def activate_campaign(root):
    """Recover activation after a crash between predecessor seal and successor activation."""
    root=Path(root).resolve();record=read_json(root/'continuation.json')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=Store(root/'memory');prior=Store(Path(record['predecessor'])/'memory')
        try:
            config=load_config(root,current)
            prepared=[e['payload'] for e in current.events() if e['kind']=='amazon_campaign_continuation_prepared']
            sealed=[e['payload'] for e in prior.events() if e['kind']=='amazon_campaign_continued']
            if (len(prepared)!=1 or len(sealed)!=1 or sealed[0]!=record or
                    {k:v for k,v in prepared[0].items() if k!='artifact'}!=record or
                    not config.get('continuation_required') or str(root)!=record['successor'] or
                    config['budget_gpu_seconds']!=record['new_total_limit_gpu_seconds'] or
                    config['predecessor']['charged_gpu_seconds']!=record['prior_budget']['charged_or_reserved']):
                raise ContractError('continuation activation identity mismatch')
            Artifacts(root/'artifacts/objects').verify(prepared[0]['artifact'])
            identity=digest(record)
            existing=[e['payload'] for e in current.events() if e['kind']=='amazon_campaign_continuation_activated']
            if existing:
                if existing!=[{'continuation_hash':identity}]:raise ContractError('conflicting campaign activation')
                return existing[0]
            current.event('amazon_campaign_continuation_activated',{'continuation_hash':identity})
            return {'continuation_hash':identity}
        finally:prior.close();current.close()
