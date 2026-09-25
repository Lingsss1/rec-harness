"""Run a preregistered phase while reusing audited existing comparators."""
from pathlib import Path

from rec_harness.core import ContractError
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_prefix_candidate import plan_guard
from .amazon_report import audit_job
from .amazon_study import run_study


def prepare_phase(root, proposal_id, phase, execution_id=None):
    if phase not in ('development','confirmation'):
        raise ContractError('unknown research phase')
    root=Path(root);store=Store(root/'memory')
    try:
        config=load_config(root,store)
        plans=[e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal_id]
        if len(plans)!=1:raise ContractError('unique preregistered plan required')
        plan=plans[0];reused=set(plan.get('reused_control_versions',[]))
        variants=[plan['versions']['candidate'],*plan['versions']['controls']]
        for version in variants:
            if config['model_versions'].get(version['model_version'],{}).get('sha256')!=version['source']['sha256']:
                raise ContractError('phase contains unadmitted or changed model source')
        new_versions=[v['model_version'] for v in variants if v['model_version'] not in reused]
        seeds=config[phase+'_seeds']
        guarded=plan_guard(store,new_versions,seeds,config['confirmation_seeds'])
        if guarded is None or guarded['handoff_hash']!=plan['handoff_hash']:
            raise ContractError('phase admission differs from registered plan')
        expected_execution=execution_id or 'direct-reviewed-v1'
        comparators=sorted({plan['baseline'],*reused})
        for version in comparators:
            for dataset in config['datasets']:
                for seed in seeds:
                    result,_=audit_job(root,config,version,dataset,seed)
                    if result.get('execution_id','direct-reviewed-v1')!=expected_execution:
                        raise ContractError('existing comparator execution cannot be reused')
        return {'proposal_id':proposal_id,'phase':phase,'new_versions':new_versions,'reused_comparators':comparators,
                'seeds':seeds,'execution_id':execution_id,'plan_handoff_hash':plan['handoff_hash']}
    finally:store.close()


def run_phase(root, proposal_id, phase, execution_id=None):
    request=prepare_phase(root,proposal_id,phase,execution_id)
    result=run_study(root,model_versions=request['new_versions'],seeds=request['seeds'],execution_id=execution_id)
    return {'phase_request':request,'budget':result['budget'],
            'completed_seeds':{v:result['completed_seeds'][v] for v in request['new_versions']},
            'scope':'phase execution; paired selection and confirmation conclusions require independent audit'}
