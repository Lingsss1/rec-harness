"""Read-only proof that committed measurements reached actual research inputs."""
import argparse
from pathlib import Path
from .core import Artifacts,ContractError,digest,read_json,write_json
from .store import Store


def audit_exposure(campaign):
    root=Path(campaign);config=read_json(root/'campaign.json')
    if read_json(root/'research_archive/manifest.json')['config_hash']!=digest(config):
        raise ContractError('exposure campaign configuration changed')
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
                diagnostic=context.get('diagnostic')
                if diagnostic is not None:
                    projected=value['role']=='measurement'
                    baseline=context.get('baseline') or ([] if projected else {})
                    runs=baseline if projected else baseline.get('runs',[])
                    request={'task_id':context['task_id'],'model_version':context['model_version'],
                        'protocol_hash':context['protocol_hash'],'runs':[{'seed':row['seed'],'valid':row['valid'],
                        'metrics':row.get('metrics',{}),**({'dataset':row['dataset']} if 'dataset' in row else {})}
                        for row in runs]}
                    if tool is None:raise ContractError('role saw measurement without a bound tool')
                    identity=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']})
                    record=records.get(identity if projected else diagnostic.get('identity'))
                    expected=({k:record[k] for k in ['status','tool_version','implementation_hash','output','tool_registration'] if k in record}
                              if projected and record else record)
                    if diagnostic!=expected:raise ContractError('role saw uncommitted or changed measurement')
                    diagnostic=record
                    if diagnostic['model_version']!=context['model_version']:raise ContractError('role saw stale model measurement')
                    if diagnostic['input_hash']!=digest(request):raise ContractError('measurement does not describe the visible baseline')
                    if diagnostic['identity']!=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']}):
                        raise ContractError('measurement execution identity changed')
                    exposures.append({'task_id':value['task_id'],'role':value['role'],'model_version':context['model_version'],
                        'measurement_identity':diagnostic['identity'],'input_hash':value['input_hash'],
                        'input_ref':value['visible_state_ref'],'measurement_status':diagnostic['status']})
            elif event['kind']=='agent_completed' and value['status']=='success':
                if value['task_id'] not in submitted:raise ContractError('role completed without submitted input')
                completed.add(value['task_id'])
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
