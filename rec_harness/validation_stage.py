"""Stable validator directories and durable reports across controller loss."""
import sys
from .core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from .cost_audit import freeze_cost_prefix
from .cpu_job import recover_completed_cpu_job
from .process_identity import assert_stopped
from .store import Store
from .validation_case import settle_once


class ValidationRecoveryRequired(ContractError):
    """An incomplete stage must not be converted into a scientific rejection."""


def recoverable_cases(target):
    for database in sorted(target.rglob('memory.sqlite')):
        root=database.parent.parent;store=Store(database.parent,read_only=True)
        try:
            store.verify();events=store.events()
            completed={e['payload'].get('case_id',e['payload'].get('key')) for e in events
                if e['kind'] in {'reference_case_completed','shadow_measurement_completed'}}
            for event in events:
                value=event['payload']
                if event['kind']!='validation_case_submitted' or value['key'] in completed:continue
                job=root/'cases'/value['key']/'job-1'
                request=read_json(job/'request.json')
                recover_completed_cpu_job(job,request['recipe'],request['inputs'],request['entry'],request['arguments'],artifacts=Artifacts(root/'objects'))
        finally:store.close()


def report(evolution,candidate,plan,stage,runner):
    store=evolution.store;artifacts=evolution.artifacts;candidate_id=candidate['candidate_id']
    target=evolution.root/'candidates'/candidate_id/'validation'/stage
    rule=plan['stages'][stage];result=target/'result.json';cost_context=target/'controller-cost.json'
    argv=[s.format_map({'python':sys.executable,'candidate':str(target/'candidate'),
        'output':str(result),'cost_context':str(cost_context)}) for s in rule['command']]
    expected={'candidate_id':candidate_id,'stage':stage,'plan_hash':candidate['plan_hash'],
        'implementation_hash':candidate['implementation_hash'],'argv':argv,'timeout':rule['timeout']}
    events=[e for e in store.events() if e['payload'].get('candidate_id')==candidate_id and e['payload'].get('stage')==stage]
    starts=[e['payload'] for e in events if e['kind']=='validation_stage_started']
    launches=[e['payload'] for e in events if e['kind']=='validation_stage_launched']
    reports=[e['payload'] for e in events if e['kind']=='validation_stage_report_recorded']
    if starts:
        if len(starts)!=1 or {k:starts[0].get(k) for k in expected}!=expected:
            raise ValidationRecoveryRequired('frozen validation stage invocation changed')
        frozen=read_json(artifacts.verify(starts[0]['cost_context_ref']))
        if cost_context.exists() and read_json(cost_context)!=frozen:
            raise ValidationRecoveryRequired('frozen validation cost prefix changed')
        write_json(cost_context,frozen)
        for name,ref in candidate['implementation'].items():
            if file_hash(target/'candidate'/name)!=ref['sha256']:
                raise ValidationRecoveryRequired('validation candidate workspace changed')
        if reports:
            if len(reports)!=1 or reports[0]['request_hash']!=digest(starts[0]):
                raise ValidationRecoveryRequired('validation report binding changed')
            return read_json(artifacts.verify(reports[0]['report_ref']))
        for index,launch in enumerate(launches,1):
            expected_launch={'candidate_id':candidate_id,'stage':stage,'request_hash':digest(starts[0]),
                'reservation':f'{candidate_id}-validate-{stage}-{index}',
                'process_directory':'process' if index==1 else 'process-resume-'+str(index)}
            if 'cost_context_ref' in launch:
                launch_cost=target/('controller-cost-resume-'+str(index)+'.json')
                value=read_json(artifacts.verify(launch['cost_context_ref']))
                from .cost_audit import verify_cost_prefix
                verify_cost_prefix(store,value)
                if launch_cost.exists() and read_json(launch_cost)!=value:
                    raise ValidationRecoveryRequired('validation invocation cost prefix changed')
                write_json(launch_cost,value)
                expected_launch.update(cost_context_ref=launch['cost_context_ref'],argv=[s.format_map({
                    'python':sys.executable,'candidate':str(target/'candidate'),'output':str(result),
                    'cost_context':str(launch_cost)}) for s in rule['command']])
            if launch!=expected_launch:
                raise ValidationRecoveryRequired('validation launch binding changed')
        try:assert_stopped(target)
        except ContractError as error:raise ValidationRecoveryRequired(str(error)) from error
        if launches:
            last=launches[-1];process_path=target/last['process_directory']/'process.json'
            if process_path.exists() and read_json(process_path).get('status')=='success':
                return commit_report(store,artifacts,starts[0],result)
            module=argv[argv.index('-m')+1] if '-m' in argv and argv.index('-m')+1<len(argv) else None
            safe=(module in {'rec_harness.tool_validation','rec_harness.tool_shadow'} or
                  module=='rec_harness.tool_pipeline' and stage in {'correctness','synthetic','robustness','shadow'})
            online=(module=='rec_harness.online_validation' and stage in {'development','confirmation'} or
                    module=='rec_harness.harness_pipeline' and stage in {'development','confirmation'} or
                    module in {'rec_harness.tool_online','rec_harness.amazon_tool_online'} and stage in {'utility','confirmation'} or
                    module=='rec_harness.tool_pipeline' and stage=='utility')
            prefix=module in {'rec_harness.harness_prefix','rec_harness.harness_pipeline'} and stage=='prefix_audit'
            if not (safe or online or prefix):raise ValidationRecoveryRequired('unfinished validator has no safe automatic resume contract')
            try:
                if prefix:
                    from .prefix_recovery import preflight
                    preflight(target)
                elif online:
                    from .online_recovery import completed_groups
                    completed_groups(target,module,stage)
                else:recoverable_cases(target)
            except (ContractError,OSError,KeyError,TypeError,ValueError) as error:
                raise ValidationRecoveryRequired('unfinished validation case requires recovery: '+str(error)) from error
    elif target.exists():
        raise ValidationRecoveryRequired('unbound legacy validation directory requires review; not moving or rerunning it')
    else:
        target.mkdir(parents=True)
        artifacts.restore(candidate['implementation'],target/'candidate')
    index=len(launches)+1;reservation=f'{candidate_id}-validate-{stage}-{index}'
    if store.db.execute('SELECT 1 FROM budget WHERE id=?',(reservation,)).fetchone():
        raise ValidationRecoveryRequired('validation reservation lacks a committed launch; review required')
    store.reserve(reservation,rule['cost'])
    status='interrupted'
    try:
        if not starts:
            write_json(cost_context,freeze_cost_prefix(store,plan['budget']))
            start={**expected,'cost_context_ref':artifacts.put(cost_context)}
            store.event('validation_stage_started',start);starts=[start]
        process_directory='process' if index==1 else 'process-resume-'+str(index)
        launch={'candidate_id':candidate_id,'stage':stage,
            'request_hash':digest(starts[0]),'reservation':reservation,'process_directory':process_directory}
        if index>1:
            launch_cost=target/('controller-cost-resume-'+str(index)+'.json')
            write_json(launch_cost,freeze_cost_prefix(store,plan['budget']))
            argv=[s.format_map({'python':sys.executable,'candidate':str(target/'candidate'),
                'output':str(result),'cost_context':str(launch_cost)}) for s in rule['command']]
            launch.update(cost_context_ref=artifacts.put(launch_cost),argv=argv)
        store.event('validation_stage_launched',launch)
        process=runner(argv,target,target/process_directory,timeout=rule['timeout'])
        status=process['status']
        if status!='success':raise ContractError('independent validator failed')
        return commit_report(store,artifacts,starts[0],result)
    finally:settle_once(store,reservation,rule['cost'],status)


def commit_report(store,artifacts,start,path):
    value=read_json(path)
    store.event('validation_stage_report_recorded',{'candidate_id':start['candidate_id'],'stage':start['stage'],
        'request_hash':digest(start),'report_ref':artifacts.put(path)})
    return value
