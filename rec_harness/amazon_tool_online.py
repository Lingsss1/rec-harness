"""Independent Amazon studies for host-scored, three-group tool comparisons."""
import argparse
import copy
import fcntl
import statistics
from pathlib import Path

from .autonomous import configuration
from .core import Artifacts,ContractError,digest,file_hash,identifier,number,read_json,write_json
from .cost_audit import audit_cost
from .diagnostics import freeze_validation_tool
from .store import Store
from .adapters.amazon_autonomous import AmazonAutoresearch
from .adapters.amazon_controls import load_config
from .adapters.amazon_execution import code_identity,register_execution
from .adapters.amazon_progress import progress_report
from .adapters.amazon_report import audit_job
from .adapters.amazon_study import run_study

GROUPS=('no_tool','existing_tool','candidate_tool')
STUDY_KEYS={'datasets','development_seeds','confirmation_seeds','model_versions','worker','worker_sha256',
            'python','data','data_hashes','protocol','protocol_hash','budget_gpu_seconds','job_timeout_seconds',
            'gpu_ids','trusted_baselines_only','hidden_evaluation'}
CONTROLLER_KEYS={'adapter','base_model_id','control_model_ids','max_rounds','role_call_limit','rescore_recipe',
                 'uid_registry','agent_backend','role_timeout','diagnostic_budget','generated_control_count',
                 'structured_insertion_contract','require_method_sources','minimum_round_gpu_seconds','activation_diagnostic','semantic_encoder','source_novelty_guard'}


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('Amazon online suite directory must be empty')
    weights=spec['cost_weights']
    required_weights={'role_calls','gpu_seconds','diagnostic_units'}
    if set(weights) not in [required_weights,required_weights|{'activation_units'}]:raise ContractError('explicit ledger cost weights required')
    for value in weights.values():
        if number(value,'ledger cost weight',0)<=0:raise ContractError('positive ledger cost weights required')
    runtime=copy.deepcopy(spec['tool_runtime'])
    # A venv interpreter may be a symlink to the base executable. Resolving
    # that symlink changes sys.prefix and discards the venv's dependencies.
    runtime['execution']['python']=str((path.parent/runtime['execution']['python']).absolute())
    runtime['execution']['uid_registry']=str((path.parent/runtime['execution']['uid_registry']).resolve())
    existing=freeze_validation_tool({**runtime,'source':str((path.parent/spec['existing_tool']).resolve()),
        'files':spec['tool_files']},Artifacts(root/'objects'))
    cases=freeze_cases(spec['cases'],path.parent,Artifacts(root/'objects'),weights)
    body={'schema':'amazon-online-tool-suite-v1','cases':cases,'tool_runtime':runtime,'tool_files':spec['tool_files'],
          'existing_tool':existing,'cost_weights':weights,'code_hashes':code_identity(),
          'qualification':'Independent study and seed conditions on declared Amazon data; shared validation users can remain. Measures terminal development HR@10 after each study internal confirmation rule; not final-test evidence. CPU admission/rescoring wall time is not a charged ledger axis.'}
    body['identity']={'base_llm':digest([c['controller']['agent_backend'] for c in cases]),
        'tools':digest([existing,runtime]),'start':digest([c['models'] for c in cases]),
        'graph':digest('fresh independent study and controller stores per group'),
        'budget':digest([weights,[(c['study']['budget_gpu_seconds'],c['controller']['role_call_limit'],c['controller']['diagnostic_budget']) for c in cases]]),
        'protocol':digest([c['study'] for c in cases])}
    if any(c['controller'].get('activation_diagnostic') for c in cases):
        activation=[c['controller'].get('activation_diagnostic') for c in cases]
        body['identity']['tools']=digest([body['identity']['tools'],activation])
        body['identity']['budget']=digest([body['identity']['budget'],[o['budget'] if o else 0 for o in activation]])
    if any('semantic_encoder' in c['controller'] for c in cases):
        body['identity']['tools']=digest([body['identity']['tools'],[c['controller'].get('semantic_encoder') for c in cases]])
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def freeze_cases(entries,base,objects,weights,*,require_diagnostics=True):
    """Freeze independent Amazon inputs shared by tool and policy comparisons."""
    cases=[];seen=set();conditions=set()
    for entry in entries:
        if set(entry)-{'id','stage','config','execution_recipe','development_seeds','confirmation_seeds'}:
            raise ContractError('unsupported Amazon comparison condition option')
        identifier(entry['id'])
        if entry['id'] in seen or entry['stage'] not in {'development','confirmation'}:raise ContractError('unique case IDs and both stages required')
        seen.add(entry['id'])
        config=configuration(base/entry['config'])
        if set(config)-CONTROLLER_KEYS-{'study','task_id'}:raise ContractError('Amazon validation requires fresh reference controller settings')
        if config.get('adapter')!='amazon-nextitnet-v1':raise ContractError('Amazon adapter required')
        backend=config['agent_backend']
        if backend.get('kind','jailed_codex')!='jailed_codex' or not backend.get('model'):raise ContractError('explicit isolated Codex model required')
        raw=read_json(Path(config['study'])/'study.json')
        if digest(raw)!=read_json(Path(config['study'])/'study-lock.json')['config_hash']:raise ContractError('source study lock changed')
        study={key:copy.deepcopy(raw[key]) for key in STUDY_KEYS}
        if ('development_seeds' in entry)!=('confirmation_seeds' in entry):raise ContractError('override both study seed partitions together')
        for key in ['development_seeds','confirmation_seeds']:
            if key in entry:study[key]=copy.deepcopy(entry[key])
            if not isinstance(study[key],list):raise ContractError('study seeds must be lists')
        if digest(study['protocol'])!=study['protocol_hash']:raise ContractError('study protocol identity differs')
        if study['protocol']['metric']!='HR@10':raise ContractError('Amazon tool comparison currently scores HR@10')
        for key in ['budget_gpu_seconds','job_timeout_seconds']:
            if number(study[key],key,0)<=0:raise ContractError('positive study limits required')
        for key in ['max_rounds','role_call_limit']:
            if type(config.get(key)) is not int or config[key]<1:raise ContractError('positive controller limits required')
        if config['role_call_limit']<6 or not config.get('rescore_recipe') or not config.get('uid_registry'):
            raise ContractError('six-role capacity and independent CPU rescoring configuration required')
        number(config.get('diagnostic_budget',None if require_diagnostics else 0),'diagnostic_budget',0)
        seeds=study['development_seeds']+study['confirmation_seeds']
        if any(type(seed) is not int or seed<0 for seed in seeds):raise ContractError('nonnegative integer seeds required')
        if not study['development_seeds'] or not study['confirmation_seeds'] or len(set(seeds))!=len(seeds):raise ContractError('disjoint nonempty study seed partitions required')
        from .adapters.amazon_activation import validate_options
        validate_options(config.get('activation_diagnostic'),study)
        if config.get('activation_diagnostic'):
            if 'activation_units' not in weights:raise ContractError('activation cost weight required')
            options=config['activation_diagnostic']
            calls=len(options.get('datasets',study['datasets']))*len(options.get('seeds',study['development_seeds']))*config['max_rounds']
            if calls*options['cost']>options['budget']:raise ContractError('activation budget cannot cover frozen research rounds')
            if set(options['settings'])!={'HIDDEN_UNITS','NUM_BLOCKS','KERNEL_SIZE','DROPOUT_RATE','MAXLEN'}:
                raise ContractError('explicit activation model settings required')
            if type(options['settings']['MAXLEN']) is not int or options['settings']['MAXLEN']<16:
                raise ContractError('activation protocol requires MAXLEN at least 16')
        for dataset in study['datasets']:
            data_identity=[study['data_hashes'][dataset+'_'+split+'.txt'] for split in ['train','valid']]
            for seed in seeds:
                condition=digest([dataset,data_identity,seed])
                if condition in conditions:raise ContractError('Amazon validation cases reuse a training condition')
                conditions.add(condition)
        for name,sha in study['data_hashes'].items():
            if file_hash(Path(study['data'])/name)!=sha:raise ContractError('private data identity differs')
        snapshots={}
        for version,source in study['model_versions'].items():
            identifier(version)
            if file_hash(source['path'])!=source['sha256']:raise ContractError('initial model source changed')
            snapshots[version]=objects.put(source['path'])
        if file_hash(study['worker'])!=study['worker_sha256']:raise ContractError('study worker changed')
        worker=objects.put(study['worker']);study.pop('worker');study.pop('model_versions')
        config.pop('study');config.pop('task_id',None)
        config.setdefault('source_novelty_guard',True)
        from .semantic_encoder import pin_options
        pin_options(config)
        if config.get('base_model_id','legacy_incumbent') not in snapshots:raise ContractError('initial model absent from frozen sources')
        recipe=entry['execution_recipe']
        if set(recipe)!={'runtime_dirs','runtime_files'}:raise ContractError('explicit GPU execution recipe required')
        for values in recipe.values():
            if not isinstance(values,list) or not values or any(not isinstance(p,str) or not Path(p).is_absolute() for p in values):raise ContractError('absolute GPU runtime paths required')
        cases.append({'id':entry['id'],'stage':entry['stage'],'controller':config,'study':study,
                      'models':snapshots,'worker':worker,'execution_recipe':recipe})
    if {case['stage'] for case in cases}!={'development','confirmation'}:raise ContractError('development and confirmation cases required')
    return cases


def load_suite(path):
    path=Path(path).resolve();suite=read_json(path)
    if suite.get('schema')!='amazon-online-tool-suite-v1' or suite.get('hash')!=digest({k:v for k,v in suite.items() if k!='hash'}):raise ContractError('Amazon online suite changed')
    if suite['code_hashes']!=code_identity():raise ContractError('Amazon online implementation changed')
    objects=Artifacts(path.parent/'objects')
    for case in suite['cases']:
        for ref in [case['worker'],*case['models'].values()]:objects.verify(ref)
        for name,sha in case['study']['data_hashes'].items():
            if file_hash(Path(case['study']['data'])/name)!=sha:raise ContractError('private data identity differs')
    for ref in [suite['existing_tool']['worker'],*suite['existing_tool']['implementation'].values()]:objects.verify(ref)
    return suite,objects


def create_study(case,root,objects):
    """Create from frozen inputs only; never copy jobs, graphs, or old charges."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(case['study']);config['study_id']=root.parent.name+'-'+case['id']
    models={}
    for version,ref in case['models'].items():
        destination=root/'frozen'/(version+'.py');objects.restore({destination.name:ref},destination.parent)
        models[version]={'path':str(destination.resolve()),'sha256':ref['sha256']}
    objects.restore({'worker.py':case['worker']},root/'frozen')
    config.update(model_versions=models,worker=str((root/'frozen/worker.py').resolve()))
    write_json(root/'study.json',config);write_json(root/'study-lock.json',{'config_hash':digest(config)})
    store=Store(root/'memory')
    try:
        store.configure_budget(config['budget_gpu_seconds'])
        store.event('study_registered',{'config_hash':digest(config),'protocol_hash':config['protocol_hash'],
            'model_versions':models,'data_hashes':config['data_hashes']})
        execution=register_execution(store,case['execution_recipe'])
    finally:store.close()
    return execution['execution_id']


def read_result(root):
    controller=AmazonAutoresearch(root/'campaign',load_agent=False,read_only=True)
    try:
        integrity=controller.verify();report=read_json(root/'campaign/run-report.json')
        if report['state']['stop_reason']!='max_rounds' or report['failed_rounds'] or report['state']['round']!=controller.config['max_rounds']:
            raise ContractError('Amazon online episode did not complete its frozen rounds')
        completed=[e['payload'] for e in controller.store.events() if e['kind']=='auto_round_completed']
        if (not completed or report['state']!={**completed[-1]['state'],'status':'stopped','stop_reason':'max_rounds'}
                or any(r['result'].get('status') in {'failed','deferred'} for r in completed)):
            raise ContractError('terminal report differs from committed research')
        progress=progress_report(root/'campaign')
        if not progress['rows'] or progress['rows'][-1]['development_score'] is None:raise ContractError('terminal incumbent lacks committed development score')
        cfg=load_config(root/'study');scores=[];evaluation_conditions=[]
        for dataset in cfg['datasets']:
            dataset_scores=[]
            for seed in cfg['development_seeds']:
                result,manifest=audit_job(root/'study',cfg,report['state']['incumbent'],dataset,seed)
                evaluation_conditions.append({'dataset':dataset,'seed':seed,'candidates_hash':manifest['candidates_hash'],
                    'data_hashes':manifest['data_hashes']})
                dataset_scores.append(number(result['metrics']['HR@10'],'terminal HR@10'))
            scores.append(statistics.mean(dataset_scores))
        if statistics.mean(scores)!=progress['rows'][-1]['development_score']:
            raise ContractError('terminal score differs from audited training receipts')
        events=controller.store.events();calls=sum(e['kind']=='agent_submitted' for e in events)
        if not calls:raise ContractError('Amazon online utility requires research decisions')
        costs={'role_calls':audit_cost(controller.store,controller.config['role_call_limit'])}
        study=Store(root/'study/memory',read_only=True)
        try:costs['gpu_seconds']=audit_cost(study,load_config(root/'study')['budget_gpu_seconds'])
        finally:study.close()
        if (root/'campaign/diagnostic-budget/memory.sqlite').exists():
            diagnostics=Store(root/'campaign/diagnostic-budget',read_only=True)
            try:costs['diagnostic_units']=audit_cost(diagnostics,controller.config['diagnostic_budget'])
            finally:diagnostics.close()
        if controller.config.get('activation_diagnostic'):
            from .adapters.amazon_activation import audit_costs
            costs['activation_units']=audit_costs(controller.root,controller.config['activation_diagnostic'])
        if any(value['pending'] for value in costs.values()):raise ContractError('Amazon online costs remain unsettled')
        return {'score':progress['rows'][-1]['development_score'],'incumbent':report['state']['incumbent'],
            'evaluation_conditions':evaluation_conditions,'agent_calls':calls,'diagnostic_calls':sum(e['kind']=='diagnostic_completed' for e in events),
            'diagnostic_failures':sum(e['kind']=='diagnostic_completed' and e['payload']['status']!='success' for e in events),
            'costs':costs,'integrity':integrity,'protocol_hash':controller.protocol_hash}
    finally:controller.close()


def episode(case,root,objects,tool,*,agent=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    binding={'case_hash':digest(case),'tool_hash':digest(tool),'scripted_agent':agent is not None}
    if (root/'completed.json').exists():
        saved=read_json(root/'completed.json')
        if read_json(root/'started.json')!=binding or saved!=read_result(root):raise ContractError('completed Amazon comparison group changed')
        return saved
    if (root/'started.json').exists():raise ContractError('Amazon comparison group outcome unknown; no automatic restart')
    if any(root.iterdir()):raise ContractError('unbound Amazon comparison artifacts')
    write_json(root/'started.json',binding)
    execution=create_study(case,root/'study',objects)
    config={**copy.deepcopy(case['controller']),'study':str((root/'study').resolve()),
            'task_id':'online-'+digest([case['id'],str(root)])[:24],'execution_id':execution}
    AmazonAutoresearch.init(root/'campaign',config,_validation_tool=tool)
    controller=AmazonAutoresearch(root/'campaign',agent=agent)
    try:
        with controller.lock():
            versions=list(dict.fromkeys([config.get('base_model_id','legacy_incumbent'),*config.get('control_model_ids',[])]))
            run_study(root/'study',model_versions=versions,seeds=case['study']['development_seeds'],execution_id=execution)
            for version in versions:controller._rescore(root/'baseline-rescore',version,case['study']['development_seeds'])
        controller.run()
    finally:controller.close()
    result=read_result(root);write_json(root/'completed.json',result);return result


def run_validation(suite_path,candidate,output,stage,*,agent_factory=None):
    if stage not in {'utility','confirmation'}:raise ContractError('invalid Amazon tool validation stage')
    suite,objects=load_suite(suite_path);root=Path(output).resolve().parent/'amazon-tool-online';root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('Amazon tool comparison already running') from error
        archive=Artifacts(root/'objects');snapshot=archive.snapshot(candidate,suite['tool_files'])
        binding={'suite_hash':suite['hash'],'candidate_hash':digest(snapshot),'stage':stage,'scripted_agent':agent_factory is not None}
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('Amazon tool comparison binding changed')
        write_json(root/'binding.json',binding)
        archive.restore(snapshot,root/'candidate');objects.restore(suite['existing_tool']['implementation'],root/'existing')
        rows=[]
        for case in suite['cases']:
            if case['stage']!=('development' if stage=='utility' else 'confirmation'):continue
            results={}
            for group in GROUPS:
                tool=None if group=='no_tool' else {**copy.deepcopy(suite['tool_runtime']),'files':suite['tool_files'],
                    'source':str(root/('candidate' if group=='candidate_tool' else 'existing'))}
                results[group]=episode(case,root/case['id']/group,objects,tool,
                    agent=agent_factory(case,group) if agent_factory else None)
            if len({r['protocol_hash'] for r in results.values()})!=1:raise ContractError('Amazon comparison protocols differ')
            if len({digest(r['evaluation_conditions']) for r in results.values()})!=1:raise ContractError('Amazon comparison evaluation conditions differ')
            rows.append({'evaluation_id':case['id'],**results})
        if not rows:raise ContractError('no Amazon tool comparison conditions')
        gains={group:statistics.mean(row['candidate_tool']['score']-row[group]['score'] for row in rows) for group in GROUPS[:2]}
        costs={axis:sum(row[group]['costs'].get(axis,{}).get('charged_or_reserved',0) for row in rows for group in GROUPS) for axis in suite['cost_weights']}
        report={'passed':all(row[g]['diagnostic_calls']>0 and row[g]['diagnostic_failures']==0 for row in rows for g in GROUPS[1:]),
            'gain':min(gains.values()),'gain_by_reference':gains,
            'regression':max([0]+[row[g]['score']-row['candidate_tool']['score'] for row in rows for g in GROUPS[:2]]),
            'total_cost':number(sum(costs[axis]*weight for axis,weight in suite['cost_weights'].items()),'weighted ledger cost',0),'cost_axes':costs,'cost_weights':suite['cost_weights'],
            'comparison_groups':list(GROUPS),'evaluation_ids':[row['evaluation_id'] for row in rows],'rows':rows,
            'identity':suite['identity'],'binding':binding,'online':agent_factory is None,
            'evaluation_mode':'end_to_end_candidate_generation','qualification':suite['qualification']}
        write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--candidate',required=True)
    run.add_argument('--output',required=True);run.add_argument('--stage',choices=['utility','confirmation'],required=True)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.candidate,args.output,args.stage)


if __name__=='__main__':main()
