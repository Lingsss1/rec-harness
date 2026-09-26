"""Resume a preregistered matched development matrix without regenerating roles."""
import fcntl
from pathlib import Path

from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_execution import resolve_execution
from .amazon_study import run_study
from .amazon_phases import prepare_phase,run_phase
from .amazon_rescore import rescore_job
from .amazon_selection import development_readout
from .independent_scoring import scoring_runtime


def run_matched_development(study,proposal,output,*,execution_id,recipe,uid_registry):
    """Study creation, execution registration and code admission are explicit prerequisites.

    Repaired executions need a successor study with retained costs; successful
    comparator jobs from another execution must never silently be reused.
    Confirmation release and Codex role generation are outside this operation.
    """
    study=Path(study).resolve();output=Path(output).resolve()
    store=Store(study/'memory',read_only=True)
    try:
        cfg=load_config(study,store)
        plans=[e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered'
               and e['payload']['proposal_id']==proposal]
        if len(plans)!=1:raise ContractError('unique preregistered plan required')
        plan=plans[0];execution=resolve_execution(store,execution_id)
    finally:store.close()
    if recipe!=execution['recipe']:raise ContractError('scoring recipe differs from registered execution')
    controls=list(dict.fromkeys([plan['baseline'],*[v['model_version'] for v in plan['versions']['controls']]]))
    candidate=plan['versions']['candidate']['model_version']
    if set(controls)!=set(plan['reused_control_versions']):
        raise ContractError('matched runner requires previously registered control models')
    for version in [plan['versions']['candidate'],*plan['versions']['controls']]:
        if cfg['model_versions'].get(version['model_version'],{}).get('sha256')!=version['source']['sha256']:
            raise ContractError('matched runner requires admitted unchanged source')
    binding={'schema':'amazon-matched-development-v1','study':str(study),'proposal_id':proposal,
             'plan_hash':digest(plan),'execution_id':execution_id,'protocol_hash':cfg['protocol_hash'],
             'sources':{name:cfg['model_versions'][name]['sha256'] for name in [*controls,candidate]},
             'datasets':cfg['datasets'],'seeds':cfg['development_seeds'],'recipe_hash':digest(recipe),
             'host_scoring_runtime':scoring_runtime()}
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=output/'plan.json'
        if frozen.exists():
            if read_json(frozen)!=binding:raise ContractError('matched development frozen binding changed')
        else:
            if (output/'acceptance.json').exists() or (output/'rescores').exists():
                raise ContractError('matched development outputs lack frozen plan')
            write_json(frozen,binding)
        terminal=output/'acceptance.json'
        if terminal.exists():
            prior=read_json(terminal)
            if prior.get('hash')!=digest({k:v for k,v in prior.items() if k!='hash'}) or prior.get('binding_hash')!=digest(binding):
                raise ContractError('matched development acceptance changed')
            # Terminal verification must not train again if an artifact vanished.
            prepare_phase(study,proposal,'development',execution_id)
        else:
            # Training retains interrupted attempts and all incurred costs.
            for control in controls:
                run_study(study,model_versions=[control],seeds=cfg['development_seeds'],execution_id=execution_id)
            run_phase(study,proposal,'development',execution_id=execution_id)
        receipts={}
        for version in [*controls,candidate]:
            for dataset in cfg['datasets']:
                for seed in cfg['development_seeds']:
                    name=f'{version}-{dataset}-{seed}';directory=output/'rescores'/name
                    if terminal.exists() and not (directory/'inference').is_dir():
                        raise ContractError('completed development inference archive missing: '+name)
                    receipt=rescore_job(study,version,dataset,seed,directory,recipe,uid_registry,resume=directory.exists())
                    if receipt['rank_disagreements']!=0:raise ContractError('independent development rankings disagree: '+name)
                    receipts[name]=receipt['hash']
        readout=development_readout(study,proposal)
        report={'status':'completed','binding_hash':digest(binding),'development_hash':readout['hash'],
                'independent_rescores':receipts,'codex_calls':0,
                'scope':'Matched development only; eligibility does not release confirmation or final evaluation.'}
        report['hash']=digest(report)
        for name,value in [('development-readout.json',readout),('acceptance.json',report)]:
            path=output/name
            if path.exists() and read_json(path)!=value:raise ContractError('matched development committed output changed: '+name)
            if not path.exists():write_json(path,value)
        return report
