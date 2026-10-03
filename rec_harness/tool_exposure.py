"""Read-only proof that committed measurements reached actual research inputs."""
import argparse
from pathlib import Path
from .core import Artifacts,ContractError,digest,read_json,write_json
from .store import Store
from .diagnostic_inputs import CONTRACT,OBSERVATION_CONTRACT,measurement_request,verify_observations


def audit_exposure(campaign):
    root=Path(campaign);amazon=(root/'amazon-auto.json').exists()
    config=read_json(root/('amazon-auto.json' if amazon else 'campaign.json'))
    lock=root/('auto-lock.json' if amazon else 'research_archive/manifest.json')
    if read_json(lock)['config_hash']!=digest(config):
        raise ContractError('exposure campaign configuration changed')
    study=None
    if amazon:
        study=read_json(Path(config['study'])/'study.json')
        if digest(study)!=config['study_config_hash']:raise ContractError('exposure bound study changed')
    objects=Artifacts(root/'artifacts/objects');store=Store(root/'memory',read_only=True)
    try:
        store.db.execute('BEGIN')
        integrity=store.verify();records={};submitted={};completed=set();exposures=[]
        tool=config.get('diagnostic_tool')
        if tool:
            for ref in [tool['worker'],*tool['implementation'].values()]:objects.verify(ref)
            if digest(tool['implementation'])!=tool['implementation_hash']:raise ContractError('tool implementation binding changed')
        for event in store.events():
            value=event['payload']
            if event['kind']=='diagnostic_completed':
                if tool is None or value['implementation_hash']!=tool['implementation_hash'] or value['tool_version']!=tool['candidate_id']:
                    raise ContractError('measurement came from an unbound tool')
                for ref in value.get('artifacts',{}).values():objects.verify(ref)
                if value['status']=='success':
                    result=read_json(objects.verify(value['artifacts']['measurement.json']))
                    if result!=value['output']:raise ContractError('measurement differs from collected output')
                records[value['identity']]=value
            elif event['kind']=='agent_submitted':
                context=read_json(objects.verify(value['visible_state_ref']))
                if digest(context)!=value['input_hash']:raise ContractError('submitted role input changed')
                submitted[value['task_id']]=value
                if 'incumbent_measurement_context' in context:
                    scoped=context['incumbent_measurement_context']
                    binding=context.get('bindings',{})
                    if (not amazon or value['role'] not in ('evaluation','lead','analyst') or
                            not isinstance(scoped,dict) or
                            any(key in context for key in ('diagnostic','measurement_baseline','activation_diagnostics','block_score_diagnostics')) or
                            scoped.get('model_version')!=binding.get('base_model_id') or
                            context.get('model_version')!=binding.get('candidate_model_version') or
                            scoped.get('task_id')!=context.get('task_id') or
                            scoped.get('protocol_hash')!=context.get('protocol_hash') or
                            scoped.get('development_baseline')!=context.get('development_baseline')):
                        raise ContractError('incumbent measurement scope differs from review bindings')
                    context=scoped
                diagnostic=context.get('diagnostic')
                if diagnostic is not None:
                    projected=value['role']=='measurement'
                    baseline=context.get('baseline' if projected or not amazon else 'measurement_baseline') or ([] if projected else {})
                    runs=baseline if projected else baseline.get('runs',[])
                    if amazon:
                        if context['protocol_hash']!=config['protocol_hash'] or any(
                                row['seed'] not in study['development_seeds'] or row.get('dataset') not in study['datasets'] for row in runs):
                            raise ContractError('measurement exposure is outside frozen development conditions')
                        if not projected:
                            public_rows=[{k:row[k] for k in ['dataset','seed','metrics']} for row in runs]
                            if public_rows!=context.get('development_baseline'):
                                raise ContractError('measurement differs from Amazon development baseline')
                    if tool is None:raise ContractError('role saw measurement without a bound tool')
                    request=measurement_request(context['task_id'],context['model_version'],context['protocol_hash'],
                        runs,tool.get('contract',CONTRACT),context)
                    verify_observations(request,store)
                    identity=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']})
                    record=records.get(identity if projected else diagnostic.get('identity'))
                    expected=({k:record[k] for k in ['status','tool_version','implementation_hash','output','tool_registration'] if k in record}
                              if projected and record else record)
                    if diagnostic!=expected:raise ContractError('role saw uncommitted or changed measurement')
                    diagnostic=record
                    if diagnostic['model_version']!=context['model_version']:raise ContractError('role saw stale model measurement')
                    if diagnostic['input_hash']!=digest(request):raise ContractError('measurement does not describe the visible baseline')
                    if tool.get('contract')==OBSERVATION_CONTRACT:
                        if read_json(objects.verify(diagnostic['request_ref']))!=request:
                            raise ContractError('measurement differs from committed observation input')
                    if diagnostic['identity']!=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']}):
                        raise ContractError('measurement execution identity changed')
                    exposures.append({'task_id':value['task_id'],'role':value['role'],'model_version':context['model_version'],
                        'measurement_identity':diagnostic['identity'],'input_hash':value['input_hash'],
                        'input_ref':value['visible_state_ref'],'measurement_status':diagnostic['status']})
            elif event['kind']=='agent_completed' and value['status']=='success':
                if value['task_id'] not in submitted:raise ContractError('role completed without submitted input')
                completed.add(value['task_id'])
            elif amazon and event['kind']=='role_completed':
                task=value['task'];submission=submitted.get(task)
                if not submission or value['role']!=submission['role'] or value.get('context_hash')!=submission['input_hash']:
                    raise ContractError('Amazon role completion differs from submitted input')
                completed.add(task)
        for row in exposures:row['transport_completed']=row['task_id'] in completed
        if store.verify()!=integrity:raise ContractError('exposure source changed during audit')
        return {'task_id':config['task_id'],'tool_hash':tool['implementation_hash'] if tool else None,
            'measurements':len(records),'exposures':exposures,'integrity':integrity,
            'qualification':'Committed measurement present in exact role input; not proof of semantic use or causal decision benefit'}
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign');parser.add_argument('output');args=parser.parse_args()
    write_json(args.output,audit_exposure(args.campaign))


if __name__=='__main__':main()
