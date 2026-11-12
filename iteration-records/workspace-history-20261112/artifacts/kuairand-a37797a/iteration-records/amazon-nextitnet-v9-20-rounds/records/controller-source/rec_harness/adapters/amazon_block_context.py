"""Freeze host-verified development ranking aggregates; private scores stay local."""
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.diagnostics import validate_measurement
from rec_harness.runtime import node,edge
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_report import audit_job
from .independent_scoring import score_response


def project_reports(directory,study):
    directory=Path(directory);config=read_json(directory/'config.json')
    source=load_config(config['study']);base=config['model_version'];dataset=config['dataset'];seed=config['seed']
    if source['protocol_hash']!=study['protocol_hash'] or source['data_hashes']!=study['data_hashes']:
        raise ContractError('block scores use another protocol or data snapshot')
    if dataset not in study['datasets'] or seed not in study['development_seeds']:
        raise ContractError('block scores must use development conditions')
    if base not in study['model_versions'] or source['model_versions'][base]['sha256']!=study['model_versions'][base]['sha256']:
        raise ContractError('block score model source differs')
    baseline,_=audit_job(Path(config['study']),source,base,dataset,seed)
    baseline_ranks=read_json(Path(config['study'])/'jobs'/f'{base}-{dataset}-{seed}'/'validation_ranks.json')
    store=Store(directory/'memory');objects=Artifacts(directory/'objects')
    scores=Artifacts(Path(config['study'])/'artifacts/objects')
    try:
        integrity=store.verify();events=store.events()
        registered=[e['payload'] for e in events if e['kind']=='block_scores_registered']
        if registered!=[{'config_hash':digest(config)}]:raise ContractError('block score configuration changed')
        if store.db.execute("SELECT 1 FROM budget WHERE status!='success'").fetchone():
            raise ContractError('block score work is unsettled or failed')
        completed=[e['payload'] for e in events if e['kind']=='block_score_completed']
        expected={'reference',*('block-'+str(i) for i in config['blocks'])}
        if {row['id'] for row in store.db.execute('SELECT id FROM budget')}!=expected:
            raise ContractError('block score cost ledger does not cover declared conditions')
        if len(completed)!=len(expected) or {r['task'] for r in completed}!=expected:
            raise ContractError('block score conditions incomplete')
        measurements={};reference=None;conditions=[]
        for record in sorted(completed,key=lambda r:r['task']!='reference'):
            refs=record['artifacts']
            for ref in refs.values():objects.verify(ref)
            report=read_json(objects.verify(refs['acceptance.json']))
            if report['hash']!=digest({k:v for k,v in report.items() if k!='hash'}):raise ContractError('block score report changed')
            if (report['model_version'],report['dataset'],report['seed'],report['source_hash'],report['checkpoint_hash'])!=(base,dataset,seed,baseline['source_hash'],baseline['checkpoint_sha256']):
                raise ContractError('block score checkpoint or condition differs')
            if report['score_worker_hash']!=file_hash(Path(__file__).with_name('score_worker.py')) or report['host_scorer_hash']!=file_hash(Path(__file__).with_name('independent_scoring.py')):
                raise ContractError('block score implementation differs from reviewed scorer')
            receipt=read_json(objects.verify(refs['inference/receipt.json']))
            if receipt['process']['status']!='success':raise ContractError('block score inference failed')
            response=read_json(scores.verify(receipt['collection']['artifacts']['scores.json']))
            request=read_json(objects.verify(refs['inference-request.json']));key=read_json(objects.verify(refs['private-key.json']))
            ranked=score_response(request,key,response)
            if ranked!=read_json(objects.verify(refs['host-ranking.json'])) or ranked['hash']!=report['host_report_hash'] or ranked['metrics']!=report['host_metrics'] or len(ranked['ranks'])!=report['users']:
                raise ContractError('block score aggregate differs from host recomputation')
            if record['task']=='reference':
                if (report['rank_disagreements']!=0 or ranked['user_ids']!=baseline_ranks['user_ids']
                        or ranked['ranks']!=baseline_ranks['ranks']
                        or set(report['host_metrics'])!=set(baseline['metrics'])
                        or any(abs(v-baseline['metrics'][k])>1e-12 for k,v in report['host_metrics'].items())):
                    raise ContractError('reference ranking differs from audited baseline')
                reference=ranked
                measurements[dataset+'.users']=report['users']
                measurements.update({dataset+'.reference.'+k:v for k,v in ranked['metrics'].items()})
            else:
                block=report['bypass_block']
                if record['task']!='block-'+str(block) or ranked['user_ids']!=reference['user_ids']:
                    raise ContractError('block score user or intervention alignment differs')
                measurements.update({dataset+'.block-'+str(block)+'.delta.'+k:v-reference['metrics'][k] for k,v in ranked['metrics'].items()})
                conditions.append(block)
        measurement={'measurements':measurements,'limitations':[
            'Fixed-weight identity bypasses on development candidates; not deletion-and-retraining effects.',
            'One selected development checkpoint per category; repeated scoring is not independent training.',
            'All declared blocks retained, including negative effects. No mechanism confirmation or final test.']}
        validate_measurement(measurement)
        public={'observation_id':'block-score-'+digest([integrity['head'],base,dataset,seed])[:32],
            'model_version':base,'dataset':dataset,'seed':seed,**measurement,
            'scope':'host-scored fixed-weight development block interventions'}
        if store.verify()!=integrity:raise ContractError('block score source changed while freezing')
        return {'public':public,'protocol_hash':study['protocol_hash'],'source_hash':baseline['source_hash'],
            'source_event_head':integrity['head'],'source_config_hash':digest(config),
            'retained_bcu':store.budget()['charged_or_reserved'],'blocks':sorted(conditions)}
    finally:store.close()


def freeze_reports(directories,study,artifacts):
    records=[];seen=set()
    for directory in directories:
        projected=project_reports(directory,study);public=projected.pop('public')
        identity=(public['model_version'],public['dataset'],public['seed'])
        if identity in seen:raise ContractError('duplicate block score condition')
        seen.add(identity)
        path=artifacts.root.parent/'block-scores'/(digest(public)+'.json');write_json(path,public)
        records.append({**projected,'model_version':public['model_version'],'public_ref':artifacts.put(path)})
    return records


def block_context(controller,base,study):
    rows=[];known={n['id']:n for n in controller.store.snapshot()['nodes']}
    for record in controller.config.get('block_score_diagnostics',[])+controller.config.get('block_interaction_diagnostics',[]):
        public=read_json(controller.artifacts.verify(record['public_ref']))
        validate_measurement({k:public[k] for k in ['measurements','limitations']})
        if record['protocol_hash']!=controller.protocol_hash:raise ContractError('frozen block score protocol differs')
        if public['model_version']!=record['model_version']:raise ContractError('frozen block score model differs')
        if public['model_version']!=base:continue
        if record['source_hash']!=study['model_versions'][base]['sha256']:raise ContractError('frozen block score source differs')
        if record.get('diagnostic_kind')=='block_interaction':
            for ref in record['source_receipts']:controller.artifacts.verify(ref)
            if 'source_archives' in record:
                costs={};receipts=[]
                for archive in record['source_archives']:
                    for ref in archive.values():controller.artifacts.verify(ref)
                    receipt=read_json(controller.artifacts.verify(archive['inference/receipt.json']))
                    costs[digest(receipt)]=receipt['wall_seconds'];receipts.append(archive['acceptance.json'])
                if costs!=record['inference_costs'] or receipts!=record['source_receipts']:
                    raise ContractError('frozen interaction cost or condition archive differs')
        observation=public['observation_id']
        value=node(observation,'Observation',{**public,'summary_ref':record['public_ref'],'valid':True},controller.task,controller.protocol_hash)
        if observation in known and known[observation]!=value:raise ContractError('committed block score observation differs')
        if observation not in known:
            nodes=[value];edges=[]
            if record.get('diagnostic_kind')=='block_interaction':
                tool_id='block-interaction-tool-'+record['implementation_hash'][:32]
                tool=node(tool_id,'ToolVersion',{'implementation_hash':record['implementation_hash'],
                    'version':'block-interaction-v1','scope':'Exploratory builtin diagnostic; not utility-promoted.'},controller.task,controller.protocol_hash)
                if tool_id in known and known[tool_id]!=tool:raise ContractError('interaction tool identity changed')
                if tool_id not in known:nodes.append(tool)
                edges.append(edge(observation+'-measured',observation,tool_id,'MEASURED_BY',[observation],controller.task,controller.store.version+1))
            controller.store.patch({'base_version':controller.store.version,'nodes':nodes,'edges':edges},
                accompanying_events=[('block_score_summary_recorded',{'observation_id':observation,'summary_ref':record['public_ref']})])
            known.update({n['id']:n for n in nodes})
        rows.append(public)
    return {'block_score_diagnostics':rows} if rows else {}


def cost_summary(config):
    costs={}
    for record in config.get('block_interaction_diagnostics',[]):
        for identity,seconds in record['inference_costs'].items():
            if identity in costs and costs[identity]!=seconds:raise ContractError('interaction cost receipt differs')
            costs[identity]=seconds
    return {'retained_diagnostic_bcu':sum(r['retained_bcu'] for r in config.get('block_score_diagnostics',[])),
        'retained_interaction_cpu_seconds':sum(v for v in costs.values() if v is not None),
        'interaction_jobs_with_unknown_full_time':sum(v is None for v in costs.values()),
        'new_diagnostic_bcu':0,'accounting':'Previously completed source diagnostic ledgers; retained once, not new role or GPU charges.'}
