"""Host-owned development readout and immutable confirmation release."""
import fcntl
from pathlib import Path
import statistics

from rec_harness.agents import schema_for, validate_schema
from rec_harness.core import Artifacts, ContractError, digest, read_json, validate_card, write_json
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_report import audit_job


def summarize_development(config, plan, rows):
    candidate=plan['versions']['candidate']['model_version']
    controls=[v['model_version'] for v in plan['versions']['controls']]
    versions=list(dict.fromkeys([plan['baseline'],candidate,*controls]))
    expected={(v,d,s) for v in versions for d in config['datasets'] for s in config['development_seeds']}
    index={(r['model_version'],r['dataset'],r['seed']):r for r in rows}
    if len(index)!=len(rows) or set(index)!=expected:
        raise ContractError('development requires exactly the frozen model/category/seed matrix')
    metric=config['protocol']['metric']
    means={v:statistics.mean(index[(v,d,s)]['metrics'][metric] for d in config['datasets'] for s in config['development_seeds']) for v in versions}
    effects={v:means[candidate]-means[v] for v in [plan['baseline'],*controls]}
    slices=[]
    for seed in config['development_seeds']:
        for dataset in config['datasets']:
            value=index[(candidate,dataset,seed)]['metrics'][metric]
            slices.append({'seed':seed,'dataset':dataset,'candidate':value,
                           'effects':{v:value-index[(v,dataset,seed)]['metrics'][metric] for v in effects}})
    threshold=config['protocol']['minimum_effect']
    if plan.get('selection_policy')=='baseline-positive-v1':
        from .amazon_promotion import development_summary
        return development_summary(config,plan,means,effects,slices,rows)
    eligible=effects[plan['baseline']]>threshold and all(effects[v]>0 for v in controls)
    return {'metric':metric,'means':means,'candidate_minus_comparator':effects,'slices':slices,
            'minimum_effect':threshold,'eligible_for_confirmation':eligible,
            'selection_rule':'development mean candidate-minus-baseline > frozen minimum_effect and candidate-minus-each-control > 0',
            'qualification':'development selection only; not mechanism confirmation or final test'}


def development_readout(root, proposal_id, store=None):
    root=Path(root); own=store is None; store=store or Store(root/'memory')
    try:
        config=load_config(root,store)
        plans=[e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal_id]
        if len(plans)!=1: raise ContractError('unique research plan required')
        plan=plans[0]
        planned=[plan['versions']['candidate'],*plan['versions']['controls']]
        for version in planned:
            if config['model_versions'].get(version['model_version'],{}).get('sha256')!=version['source']['sha256']:
                raise ContractError('planned source has not been admitted or changed')
        versions=list(dict.fromkeys([plan['baseline'],*(v['model_version'] for v in planned)]))
        rows=[]; identities={}
        for seed in config['development_seeds']:
            for dataset in config['datasets']:
                pair_identity=None
                for version in versions:
                    result,manifest=audit_job(root,config,version,dataset,seed)
                    identity=(result.get('execution_id','direct-reviewed-v1'),manifest['candidates_hash'],digest(manifest['data_hashes']))
                    if pair_identity is not None and identity!=pair_identity:
                        raise ContractError('development comparisons must share execution and evaluation identities')
                    pair_identity=identity
                    rows.append({'model_version':version,'dataset':dataset,'seed':seed,'metrics':result['metrics']})
                    identities[f'{version}-{dataset}-{seed}']={'receipt_hash':digest(result),'manifest_hash':digest(manifest)}
        summary=summarize_development(config,plan,rows)
        body={'schema':'amazon-development-selection-v1','proposal_id':proposal_id,
              'plan_handoff_hash':plan['handoff_hash'],'protocol_hash':config['protocol_hash'],
              'source_hashes':{v:config['model_versions'][v]['sha256'] for v in versions},
              'summary':summary,'source_receipts':identities,'rows':rows,
              'selection_policy_provenance':'host minimum-effect rule plus positive-control contrast; adapter release gate implemented during development, not an additional preregistered confirmation criterion'}
        if plan.get('selection_policy')=='baseline-positive-v1':
            body['selection_policy_provenance']='Preregistered baseline-only promotion; mechanism contrasts are descriptive development evidence.'
        return {**body,'hash':digest(body)}
    finally:
        if own:store.close()


def release_confirmation(root, proposal_id, decision, readout_hash):
    root=Path(root)
    validate_schema(decision,schema_for('lead')); validate_card('decision',decision)
    if decision['selection']!='keep' or decision['conclusion_status'] not in {'provisional','unknown'} or readout_hash not in decision['evidence_refs']:
        raise ContractError('confirmation requires provisional keep with the exact development evidence hash')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(root/'memory')
        try:
            report=development_readout(root,proposal_id,store)
            if report['hash']!=readout_hash or not report['summary']['eligible_for_confirmation']:
                raise ContractError('development selection gate failed or evidence changed')
            prior=[e['payload'] for e in store.events() if e['kind']=='amazon_confirmation_released' and e['payload']['proposal_id']==proposal_id]
            if prior:
                if prior[0]['readout_hash']!=readout_hash or prior[0]['decision_hash']!=digest(decision):
                    raise ContractError('frozen selection cannot be rewritten')
                return prior[0]
            config=load_config(root,store)
            plans=[e['payload'] for e in store.events() if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal_id]
            if len(plans)!=1:raise ContractError('unique research plan required')
            reused={plans[0]['baseline'],*plans[0].get('reused_control_versions',[])}
            for name in report['source_hashes']:
                if name in reused: continue
                for dataset in config['datasets']:
                    for seed in config['confirmation_seeds']:
                        if (root/'jobs'/f'{name}-{dataset}-{seed}').exists() or (root/'attempt-history'/f'{name}-{dataset}-{seed}').exists():
                            raise ContractError('confirmation work exists before frozen selection')
            directory=root/'confirmation-selections'/proposal_id; directory.mkdir(parents=True,exist_ok=False)
            write_json(directory/'readout.json',report);write_json(directory/'decision.json',decision)
            artifacts=Artifacts(root/'artifacts/objects')
            record={'proposal_id':proposal_id,'plan_handoff_hash':report['plan_handoff_hash'],
                    'readout_hash':report['hash'],'decision_hash':digest(decision),
                    'readout_ref':artifacts.put(directory/'readout.json'),'decision_ref':artifacts.put(directory/'decision.json'),
                    'source_hashes':report['source_hashes'],'confirmation_seeds':config['confirmation_seeds'],
                    'scope':'frozen development selection; no confirmation result yet'}
            if 'promotion_order' in report['summary']:
                record['promotion_order']=report['summary']['promotion_order']
                record['selection_policy']=report['summary']['selection_policy']
            store.event('amazon_confirmation_released',record)
            return record
        finally:store.close()
