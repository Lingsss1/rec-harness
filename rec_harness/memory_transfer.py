"""Frozen source-only experience versus fresh research on independent toy conditions."""
import argparse
import copy
import fcntl
import statistics
from pathlib import Path
from .core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from .memory_policy import export_memory,validate_package
from .memory_comparison import campaign_outcomes
from .memory_view import restore
from .online_validation import load_suite,episode
from .decision_traces import collect_traces
from .runtime import Research


def _conditions(seeds):
    return {condition for seed in seeds for condition in (seed,seed+10000)}


def freeze(source_campaign,target_suite,directory,*,context_limit=262144,max_nodes=100,max_bytes=65536):
    if type(context_limit) is not int or context_limit<=0:raise ContractError('positive shared context limit required')
    suite,inputs=load_suite(target_suite);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('transfer study directory must be empty')
    source=Research(source_campaign,agent=object())
    try:
        with source.lock():
            integrity=source.verify();state=read_json(source.state_path);config=source.config
            if state['status']!='stopped' or state.get('stop_reason')!='max_rounds' or not state['baseline']['valid']:
                raise ContractError('settled valid source research required')
            if source.store.db.execute("SELECT 1 FROM budget WHERE status='running'").fetchone():
                raise ContractError('source has unsettled work')
            if config.get('execution',{}).get('kind')!='isolated-toy-v1':raise ContractError('transfer validator currently supports isolated toy research')
            if any(config.get(k) for k in ['initial_memory','diagnostic_tool','measurement_discovery']):
                raise ContractError('source must use fixed built-in tools without imported experience')
            if config.get('research_policy',{})!=suite['baseline_policy']:raise ContractError('source and target Harness policies differ')
            source_options={**config.get('codex',{}),'web_search':'disabled'}
            execution={k:v for k,v in config['execution'].items() if k not in {'host_scorer_hash','worker_hash'}}
            for case in suite['cases']:
                target=case['config']
                if target['task_id']==source.task or _conditions(target['protocol']['seeds'])&_conditions(source.protocol['seeds']):
                    raise ContractError('target reuses source task or generated data conditions')
                if {**target.get('codex',{}),'web_search':'disabled'}!=source_options or target['agent_backend']!=config['agent_backend']:
                    raise ContractError('source and target base agent differ')
                if target['execution']!=execution:raise ContractError('source and target tools differ')
                if any(target.get(k) for k in ['measurement_discovery','initial_memory_files','initial_memory']):
                    raise ContractError('target must start without imported experience or tool evolution')
            root.mkdir(parents=True,exist_ok=True)
            package=export_memory(source.store,root/'source-memory.json',task_id=source.task,protocol_hash=source.protocol_hash,
                cutoff_seq=integrity['events'],limit=max_nodes,max_bytes=max_bytes)
            if not package['nodes']:raise ContractError('source contains no eligible transferable evidence')
            validate_package(package)
            artifacts=Artifacts(root/'objects');package_ref=artifacts.put(root/'source-memory.json')
            target_objects=Artifacts(root/'target/objects')
            for case in suite['cases']:
                for ref in case['inputs'].values():
                    if target_objects.put(inputs.verify(ref))!=ref:raise ContractError('target input changed while freezing')
            write_json(root/'target/suite.json',suite)
            events=source.store.events();completed=[e['payload'] for e in events if e['kind']=='agent_completed']
            real_source=bool(completed)
            for event in completed:
                ref=event['archive'].get('request.json')
                request=read_json(source.artifacts.verify(ref)) if ref else {}
                argv=request.get('argv',[])
                real_source=real_source and len(argv)>1 and Path(argv[0]).name=='codex' and argv[1]=='exec'
            body={'schema':'memory-transfer-study-v1','target_suite_hash':suite['hash'],
                'source_memory':package_ref,'source_package_hash':package['hash'],'source_task':source.task,
                'source_protocol':source.protocol_hash,'source_conditions':sorted(_conditions(source.protocol['seeds'])),
                'source_event_head':integrity['head'],'source_cost':source.store.budget()['charged_or_reserved'],
                'source_codex_requests':bool(real_source),'context_byte_limit':context_limit,
                'source_exporter_hash':file_hash(Path(__file__).with_name('memory_policy.py')),
                'qualification':('Source stage cost is reported separately; independent synthetic conditions in one task family, not cross-domain or final-test evidence.'
                    if all(c['config']['protocol'].get('synthetic_task',{}).get('family','bilinear-v1')=='bilinear-v1' for c in suite['cases'])
                        and config['protocol'].get('synthetic_task',{}).get('family','bilinear-v1')=='bilinear-v1'
                    else 'Source stage cost is separate; frozen synthetic regression task contracts, not real recommendation, cross-domain or final-test evidence.')}
            body['hash']=digest(body);write_json(root/'study.json',body)
            if source.verify()!=integrity:raise ContractError('source changed during transfer freeze')
            return body
    finally:source.store.close()


def load_study(path):
    path=Path(path).resolve();study=read_json(path)
    if study.get('schema')!='memory-transfer-study-v1' or study.get('hash')!=digest({k:v for k,v in study.items() if k!='hash'}):
        raise ContractError('transfer study identity changed')
    package_path=Artifacts(path.parent/'objects').verify(study['source_memory'])
    package=read_json(package_path);validate_package(package)
    if package['hash']!=study['source_package_hash']:raise ContractError('source package changed')
    if (package['source_task'],package['source_protocol'],package['source_cutoff']['hash'])!=(study['source_task'],study['source_protocol'],study['source_event_head']):
        raise ContractError('source package provenance changed')
    suite,inputs=load_suite(path.parent/'target/suite.json')
    if suite['hash']!=study['target_suite_hash']:raise ContractError('target suite changed')
    for case in suite['cases']:
        if case['config']['task_id']==study['source_task'] or _conditions(case['config']['protocol']['seeds'])&set(study['source_conditions']):
            raise ContractError('target source overlap')
    return study,suite,inputs,package_path


def run_validation(study_path,output,stage,*,agent_factory=None):
    if stage not in {'development','confirmation'}:raise ContractError('invalid transfer stage')
    study,suite,inputs,package=load_study(study_path);output=Path(output).resolve()
    root=output.parent/'transfer-episodes';root.mkdir(parents=True,exist_ok=True)
    binding={'study_hash':study['hash'],'stage':stage,'decision_backend':'injected' if agent_factory else 'codex',
        'implementation':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('transfer comparison already active') from error
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('transfer inputs changed')
        write_json(root/'binding.json',binding);rows=[]
        for index,original in enumerate(suite['cases']):
            if original['stage']!=stage:continue
            case=copy.deepcopy(original);case['config'].update(context_byte_limit=study['context_byte_limit'],memory_presentation='graph')
            case['config'].setdefault('codex',{})['web_search']='disabled'
            order=['fresh','experienced'] if index%2==0 else ['experienced','fresh'];pair={}
            for mode in order:
                campaign=root/case['id']/mode
                agent=agent_factory(case,mode) if agent_factory else None
                pair[mode]=episode(case,suite['baseline_policy'],campaign,inputs,agent=agent,
                    memory_package=package if mode=='experienced' else None)
                trace=collect_traces(campaign,roles=['evidence'])['decision_traces'][0]
                visible=restore(trace['visible_state']['evidence'])
                exposed=[n['id'] for n in visible['nodes'] if n['attrs'].get('source_package_hash')==study['source_package_hash']]
                if bool(exposed)!=(mode=='experienced'):raise ContractError('source memory treatment was not exposed as assigned')
                pair[mode]['visible_source_nodes']=exposed
                pair[mode]['outcomes']=campaign_outcomes(campaign)
                if pair[mode]['event_head']!=pair[mode]['outcomes']['event_head']:raise ContractError('target changed while auditing')
            if pair['fresh']['protocol_hash']!=pair['experienced']['protocol_hash']:raise ContractError('paired target protocols differ')
            sign=1 if case['config']['protocol']['direction']=='maximize' else -1
            rows.append({'evaluation_id':case['id'],'order':order,
                'delta_experienced_minus_fresh':sign*(pair['experienced']['score']-pair['fresh']['score']),**pair})
        if not rows:raise ContractError('no independent target cases')
        target_cost=sum(row[mode]['charged_cost'] for row in rows for mode in ['fresh','experienced'])
        report={'schema':'memory-transfer-report-v1','online':agent_factory is None and study['source_codex_requests'],
            'binding':binding,'rows':rows,'gain':statistics.mean(r['delta_experienced_minus_fresh'] for r in rows),
            'negative_transfer_rate':sum(r['delta_experienced_minus_fresh']<0 for r in rows)/len(rows),
            'source_cost':study['source_cost'],'target_cost':target_cost,'total_attributable_cost':study['source_cost']+target_cost,
            'qualification':study['qualification']+' Fresh and experienced have separate baseline executions and overlays; only the experienced group receives frozen source memory. Small samples do not establish general transfer utility.'}
        write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='action',required=True)
    freeze_parser=commands.add_parser('freeze');freeze_parser.add_argument('source');freeze_parser.add_argument('suite');freeze_parser.add_argument('directory')
    run=commands.add_parser('run');run.add_argument('--study',required=True);run.add_argument('--output',required=True);run.add_argument('--stage',choices=['development','confirmation'],required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.source,args.suite,args.directory)
    else:run_validation(args.study,args.output,args.stage)


if __name__=='__main__':main()
