"""Independent paired research episodes for Harness policy validation.

The host computes utility from actual terminal incumbents. Candidate policy code
is never used as an evaluator; each comparison gets a separate event/graph store.
"""
import argparse
import copy
import fcntl
import json
import statistics
from pathlib import Path

from .autonomous import configuration
from .core import Artifacts,ContractError,digest,file_hash,identifier,number,read_json,write_json
from .evolution import validate_policy
from .runtime import Research


def freeze_suite(specification,directory):
    spec_path=Path(specification).resolve();spec=read_json(spec_path)
    root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('online suite directory must be empty')
    cases=[];seen=set();seed_sets=set()
    for item in spec['cases']:
        identifier(item['id'])
        if item['id'] in seen or item['stage'] not in {'development','confirmation'}:raise ContractError('unique case IDs and independent validation stages required')
        seen.add(item['id']);config=configuration(spec_path.parent/item['config'])
        if config.get('adapter') is not None or config.get('execution',{}).get('kind')!='isolated-toy-v1':
            raise ContractError('online Harness validator currently requires host-scored isolated toy research')
        if config.get('agent_backend',{}).get('kind')!='jailed_codex':raise ContractError('online validation requires isolated Codex roles')
        if not isinstance(config.get('codex',{}).get('model'),str) or not config['codex']['model'].strip():
            raise ContractError('online validation must freeze an explicit codex.model, not a mutable service default')
        if any(key in config for key in ['harness_registry','research_policy','diagnostic_registry','diagnostic_tool','measurement_requirements','initial_memory_files','initial_memory']):
            raise ContractError('validation cases must freeze an unmodified reference Harness with fixed built-in tools')
        for seed in config['protocol']['seeds']:
            # The toy host generator uses seed for training and seed+10000 for
            # validation. Renaming data_version does not create new examples.
            conditions={seed,seed+10000}
            if conditions & seed_sets:raise ContractError('validation cases reuse a data/seed condition')
            seed_sets.update(conditions)
        paths=list(dict.fromkeys(config['model_files']+config.get('evaluator_files',[])))
        artifacts=Artifacts(root/'objects');snapshot=artifacts.snapshot(config['source'],paths)
        config['source']='archived'
        cases.append({'id':item['id'],'stage':item['stage'],'config':config,'inputs':snapshot})
    if {c['stage'] for c in cases}!={'development','confirmation'}:raise ContractError('development and confirmation cases required')
    policy=spec.get('baseline_policy',{});validate_policy(policy)
    body={'schema':'online-harness-suite-v1','cases':cases,'baseline_policy':policy,
        'qualification':'independent synthetic data/seed conditions; one task family, not held-out benchmark evidence'}
    body['identity']=suite_identity(body);body['hash']=digest(body)
    write_json(root/'suite.json',body)
    return body


def suite_identity(suite):
    cases=suite['cases']
    return {
        'base_llm':digest({c['id']:{'codex':c['config'].get('codex',{}),'backend':c['config']['agent_backend']} for c in cases}),
        'tools':digest({c['id']:c['config']['execution'] for c in cases}),
        'start':digest({c['id']:c['inputs'] for c in cases}),
        'graph':digest('independent empty graph overlays with identical root model per pair'),
        'budget':digest({c['id']:{k:c['config'].get(k) for k in ['budget','max_rounds','max_role_retries']} for c in cases}),
        'protocol':digest({c['id']:c['config']['protocol'] for c in cases})}


def load_suite(path):
    path=Path(path).resolve();suite=read_json(path)
    if suite.get('schema')!='online-harness-suite-v1' or suite.get('hash')!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('online suite identity changed')
    if suite['identity']!=suite_identity(suite):raise ContractError('paired conditions changed')
    artifacts=Artifacts(path.parent/'objects')
    for case in suite['cases']:
        for ref in case['inputs'].values():artifacts.verify(ref)
    return suite,artifacts


def episode(case,policy,root,inputs,*,agent=None,validation_tool=None,memory_package=None):
    root=Path(root);config=copy.deepcopy(case['config'])
    config['research_policy']=policy
    if memory_package is not None:
        from .memory_policy import validate_package
        validate_package(read_json(memory_package))
        config['initial_memory_files']=[str(memory_package)]
    source=root.parent/(root.name+'-inputs');inputs.restore(case['inputs'],source)
    config['source']=str(source)
    if not (root/'campaign.json').exists():Research.init(root,config,_validation_tool=validation_tool)
    research=Research(root,agent=agent)
    try:
        frozen=research.config
        expected_memory=[research.artifacts.put(memory_package)] if memory_package is not None else []
        if frozen.get('initial_memory',[])!=expected_memory:
            raise ContractError('paired source memory changed')
        if validation_tool is not None:
            from .diagnostics import freeze_validation_tool
            expected=freeze_validation_tool(validation_tool,research.artifacts)
            if frozen.get('diagnostic_tool')!=expected or not frozen.get('validation_only'):
                raise ContractError('paired diagnostic implementation changed')
        elif frozen.get('diagnostic_tool') is not None:
            raise ContractError('unexpected diagnostic tool in reference episode')
        manifest=read_json(root/'research_archive/manifest.json')
        if manifest['root_workspace']!={name:case['inputs'][name] for name in config['model_files']}:
            raise ContractError('paired initial model changed')
        if {k:v for k,v in frozen['execution'].items() if k not in {'host_scorer_hash','worker_hash'}}!=config['execution']:
            raise ContractError('paired execution backend changed')
        # Reconstruct initialization changes through the persisted config/manifest.
        for key,value in config.items():
            if key in {'source','protocol','execution','evaluator_files','initial_memory_files'}:continue
            if frozen.get(key)!=value:raise ContractError('paired episode configuration changed: '+key)
        if frozen['protocol']['evaluator_snapshot']!={name:case['inputs'][name] for name in config.get('evaluator_files',[])}:
            raise ContractError('paired evaluator changed')
        if {k:v for k,v in frozen['protocol'].items() if k!='evaluator_snapshot'}!=config['protocol']:
            raise ContractError('paired protocol changed')
        result=research.run();integrity=research.verify();state=result['state'];baseline=state['baseline']
        if state['stop_reason'] not in {'budget','max_rounds'} or not baseline or not baseline['valid']:
            raise ContractError('online episode did not reach a valid terminal incumbent')
        events=research.store.events();submitted=[e for e in events if e['kind']=='agent_submitted']
        if not submitted:raise ContractError('online utility requires actual research decisions')
        return {'score':number(baseline['score'],'terminal incumbent score'),
            'charged_cost':result['budget']['charged_or_reserved'],'incumbent':state['incumbent'],
            'rounds':state['round'],'failures':sum(e['kind']=='round_failed' for e in events),
            'diagnostic_calls':sum(e['kind']=='diagnostic_completed' for e in events),
            'diagnostic_failures':sum(e['kind']=='diagnostic_completed' and e['payload']['status']!='success' for e in events),
            'agent_calls':len(submitted),'policy_hash':digest(policy),'event_head':integrity['head'],
            'protocol_hash':research.protocol_hash,'graph_directory':str(root/'memory')}
    finally:research.store.close()


def run_validation(suite_path,candidate,output,stage,*,agent_factory=None):
    if stage not in {'development','confirmation'}:raise ContractError('online stage must be development or confirmation')
    suite,inputs=load_suite(suite_path);policy=read_json(Path(candidate)/'policy.json');validate_policy(policy)
    output=Path(output).resolve();root=output.parent/'online-episodes';root.mkdir(parents=True,exist_ok=True)
    binding={'suite_hash':suite['hash'],'candidate_policy_hash':digest(policy),'stage':stage}
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('online validation already active') from error
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('online validation inputs changed')
        write_json(root/'binding.json',binding)
        rows=[]
        for case in suite['cases']:
            if case['stage']!=stage:continue
            pair={}
            for group,patch in [('baseline',suite['baseline_policy']),('candidate',policy)]:
                agent=agent_factory(case,group) if agent_factory else None
                pair[group]=episode(case,patch,root/case['id']/group,inputs,agent=agent)
            if pair['baseline']['protocol_hash']!=pair['candidate']['protocol_hash']:raise ContractError('paired protocols differ')
            sign=1 if case['config']['protocol']['direction']=='maximize' else -1
            rows.append({'evaluation_id':case['id'],'delta':sign*(pair['candidate']['score']-pair['baseline']['score']),**pair})
        if not rows:raise ContractError('no independent evaluation cases')
        report={'passed':True,'gain':statistics.mean(row['delta'] for row in rows),
            'regression':max(0,max(-row['delta'] for row in rows)),
            'total_cost':sum(row[group]['charged_cost'] for row in rows for group in ['baseline','candidate']),
            'evaluation_ids':[row['evaluation_id'] for row in rows],'identity':suite['identity'],
            'comparison_groups':['baseline','candidate'],'online':agent_factory is None,'rows':rows,
            'suite_hash':suite['hash'],'candidate_policy_hash':digest(policy),'qualification':suite['qualification']}
        write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='action',required=True)
    freeze=commands.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=commands.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--candidate',required=True)
    run.add_argument('--output',required=True);run.add_argument('--stage',choices=['development','confirmation'],required=True)
    args=parser.parse_args()
    result=freeze_suite(args.specification,args.directory) if args.action=='freeze' else run_validation(args.suite,args.candidate,args.output,args.stage)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
