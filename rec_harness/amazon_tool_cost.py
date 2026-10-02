"""Reconcile Amazon utility against isolated transports and typed cost ledgers."""
import statistics
from pathlib import Path
from .amazon_tool_online import GROUPS,load_suite,read_result
from .core import Artifacts,ContractError,digest,file_hash,number,read_json,write_json,safe_file
from .cost_audit import audit_cost,verify_cost_prefix
from .store import Store
from .tool_exposure import audit_exposure
from .adapters.amazon_execution import resolve_execution


def check_binding(root,case,suite,group,report,*,study_group=None,scripted_agent=False):
    config=read_json(root/'campaign/amazon-auto.json');study=read_json(root/'study/study.json')
    options=None if group=='no_tool' else {**suite['tool_runtime'],'files':suite['tool_files'],
        'source':str(root.parents[1]/('candidate' if group=='candidate_tool' else 'existing'))}
    if read_json(root/'started.json')!={'case_hash':digest(case),'tool_hash':digest(options),'scripted_agent':scripted_agent}:
        raise ContractError('Amazon utility launch binding differs')
    expected={**case['study'],'study_id':(study_group or group)+'-'+case['id'],
        'model_versions':{name:{'path':str((root/'study/frozen'/(name+'.py')).resolve()),'sha256':ref['sha256']}
                          for name,ref in case['models'].items()},'worker':str((root/'study/frozen/worker.py').resolve())}
    if study!=expected:raise ContractError('Amazon utility study differs from frozen inputs')
    for source in study['model_versions'].values():
        if file_hash(source['path'])!=source['sha256']:raise ContractError('Amazon initial model bytes changed')
    if file_hash(study['worker'])!=study['worker_sha256']:raise ContractError('Amazon initial worker bytes changed')
    wanted={**case['controller'],'study':str((root/'study').resolve()),
        'task_id':'online-'+digest([case['id'],str(root)])[:24],
        'study_config_hash':digest(study),'protocol_hash':study['protocol_hash'],
        'development_memory':[],'trace_development_feedback':True}
    wanted.setdefault('structured_insertion_contract',True);wanted.setdefault('require_method_sources',True)
    execution=config['execution_id'];wanted['execution_id']=execution
    store=Store(root/'study/memory',read_only=True)
    try:
        if resolve_execution(store,execution)['recipe']!=case['execution_recipe']:raise ContractError('Amazon GPU execution recipe changed')
    finally:store.close()
    if group!='no_tool':
        source=root.parents[1]/('candidate' if group=='candidate_tool' else 'existing')
        implementation={}
        for name in suite['tool_files']:
            path=safe_file(source,name)
            implementation[name]={'sha256':file_hash(path),'size':path.stat().st_size,'availability':'available'}
        tool={**suite['existing_tool'],'implementation':implementation,'implementation_hash':digest(implementation),
              'candidate_id':'validation-'+digest(implementation)[:24]}
        objects=Artifacts(root/'campaign/artifacts/objects')
        for ref in [tool['worker'],*tool['implementation'].values()]:objects.verify(ref)
        if tool['implementation_hash']!=(report['binding']['candidate_hash'] if group=='candidate_tool' else suite['existing_tool']['implementation_hash']):
            raise ContractError('Amazon utility tool implementation changed')
        wanted.update(diagnostic_tool=tool,validation_only=True)
    from .semantic_encoder import freeze_options
    freeze_options(wanted)
    if 'source_novelty_guard' in config:wanted.setdefault('source_novelty_guard',True)
    if config!=wanted:raise ContractError('Amazon utility controller differs from frozen inputs')
    return config


def real_transports(root,config):
    store=Store(root/'campaign/memory',read_only=True);objects=Artifacts(root/'campaign/artifacts/objects')
    try:
        calls=[]
        for event in store.events():
            if event['kind']!='role_completed':continue
            archive=event['payload']['archive']
            if not all(key in archive for key in ['request.json','resources.json','response.json']):raise ContractError('Amazon role lacks actual transport receipts')
            request=read_json(objects.verify(archive['request.json']));resource=read_json(objects.verify(archive['resources.json']))
            objects.verify(archive['response.json']);isolation=resource.get('isolation',{});argv=request.get('argv',[])
            if (resource.get('status')!='success' or isolation.get('kind')!='chroot-unprivileged'
                    or request.get('isolation')!=isolation or '--model' not in argv
                    or argv[argv.index('--model')+1]!=config['agent_backend']['model']):
                raise ContractError('Amazon isolated model transport differs')
            calls.append(event['payload']['task'])
        for index in range(1,config['max_rounds']+1):
            for role in ['evidence','hypothesis','experiment','evaluation','lead','knowledge']:
                if not any(task.startswith(f'round-{index}-{role}-') for task in calls):raise ContractError('Amazon utility lacks completed real research roles')
        return len(calls)
    finally:store.close()


def audit_utility(utility,suite_path):
    report=read_json(utility);suite,_=load_suite(suite_path)
    if (report.get('online') is not True or report.get('comparison_groups')!=list(GROUPS)
            or report.get('evaluation_mode')!='end_to_end_candidate_generation'
            or report['binding'].get('scripted_agent') is not False):raise ContractError('real Amazon three-group report required')
    if report['binding']['suite_hash']!=suite['hash'] or report['identity']!=suite['identity']:raise ContractError('Amazon utility suite identity differs')
    stage=report['binding']['stage']
    if stage not in {'utility','confirmation'}:raise ContractError('invalid Amazon utility stage')
    cases=[c for c in suite['cases'] if c['stage']==('development' if stage=='utility' else 'confirmation')]
    ids=[c['id'] for c in cases]
    if report['evaluation_ids']!=ids or [row['evaluation_id'] for row in report['rows']]!=ids:raise ContractError('Amazon utility conditions omitted or changed')
    root=Path(utility).resolve().parent/'amazon-tool-online'
    if read_json(root/'binding.json')!=report['binding']:raise ContractError('Amazon utility root binding differs')
    used=set();receipts=[]
    for case,row in zip(cases,report['rows']):
        for group in GROUPS:
            folder=root/case['id']/group;config=check_binding(folder,case,suite,group,report)
            actual=read_result(folder)
            if row[group]!=actual or read_json(folder/'completed.json')!=actual:raise ContractError('Amazon utility row differs from committed evidence')
            calls=real_transports(folder,config);exposure=audit_exposure(folder/'campaign')
            if group=='no_tool':
                if exposure['exposures']:raise ContractError('no-tool reference received a tool')
            elif report['passed'] and not any(e['measurement_status']=='success' and e['transport_completed'] for e in exposure['exposures']):
                raise ContractError('Amazon tool measurement never reached a completed role')
            probe_ledgers=[Path(item['directory']) for item in actual['costs'].get('activation_units',{}).get('ledgers',[])]
            for memory in [folder/'study/memory',folder/'campaign/memory',folder/'campaign/diagnostic-budget',*probe_ledgers]:
                if (memory/'memory.sqlite').exists():
                    memory=memory.resolve()
                    if memory in used:raise ContractError('Amazon groups share a ledger')
                    used.add(memory)
            receipts.append({'evaluation_id':case['id'],'group':group,'real_calls':calls,'costs':actual['costs'],'exposure':exposure})
        if len({digest(row[g]['evaluation_conditions']) for g in GROUPS})!=1:raise ContractError('Amazon comparison populations differ')
    gains={g:statistics.mean(row['candidate_tool']['score']-row[g]['score'] for row in report['rows']) for g in GROUPS[:2]}
    costs={axis:sum(row[g]['costs'].get(axis,{}).get('charged_or_reserved',0) for row in report['rows'] for g in GROUPS) for axis in suite['cost_weights']}
    expected={'gain':min(gains.values()),'gain_by_reference':gains,
        'regression':max([0]+[row[g]['score']-row['candidate_tool']['score'] for row in report['rows'] for g in GROUPS[:2]]),
        'passed':all(row[g]['diagnostic_calls']>0 and row[g]['diagnostic_failures']==0 for row in report['rows'] for g in GROUPS[1:]),
        'cost_axes':costs,'cost_weights':suite['cost_weights'],
        'total_cost':number(sum(costs[k]*w for k,w in suite['cost_weights'].items()),'utility cost',0)}
    if any(report.get(k)!=v for k,v in expected.items()):raise ContractError('Amazon utility aggregate differs from audited groups')
    return report,suite,receipts,used


def ledger_axis(memory):
    """Identify abandoned Amazon ledgers from their adjacent controller contract."""
    parent=Path(memory).parent
    if parent.parent.name=='activation-probes':
        root=parent.parent.parent;controller=read_json(root/'amazon-auto.json')
        if digest(controller)!=read_json(root/'auto-lock.json')['config_hash']:raise ContractError('abandoned activation controller changed')
        config=read_json(parent/'config.json');binding=read_json(parent/'binding.json')
        if binding['config_hash']!=digest(config) or parent.name!=digest(config):raise ContractError('abandoned activation binding changed')
        if config['budget']!=controller['activation_diagnostic']['cost']:raise ContractError('abandoned activation cost changed')
        return 'activation_units',config['budget']
    if (parent/'study.json').exists():
        config=read_json(parent/'study.json')
        if digest(config)!=read_json(parent/'study-lock.json')['config_hash']:raise ContractError('abandoned study changed')
        return 'gpu_seconds',config['budget_gpu_seconds']
    controller=parent if (parent/'amazon-auto.json').exists() else parent.parent
    if (controller/'amazon-auto.json').exists():
        config=read_json(controller/'amazon-auto.json')
        if digest(config)!=read_json(controller/'auto-lock.json')['config_hash']:raise ContractError('abandoned Amazon controller changed')
        if Path(memory).name=='diagnostic-budget':return 'diagnostic_units',config['diagnostic_budget']
        if parent==controller and Path(memory).name=='memory':return 'role_calls',config['role_call_limit']
        raise ContractError('unrecognized Amazon cost ledger')
    return 'development_bcu',None


def run_validation(utility,candidate,output,development_ledgers=(),evolution_context=None,*,suite_path=None):
    report,suite,receipts,used=audit_utility(utility,suite_path)
    snapshot=Artifacts(Path(output).parent/'cost-audit-objects').snapshot(candidate,suite['tool_files'])
    if digest(snapshot)!=report['binding']['candidate_hash']:raise ContractError('Amazon cost audit candidate changed')
    extra=[];total=report['total_cost']
    for directory in development_ledgers:
        memory=Path(directory).resolve()
        if memory in used:raise ContractError('cost ledger counted twice')
        used.add(memory)
        if not (memory/'memory.sqlite').exists():raise ContractError('development cost ledger missing')
        axis,limit=ledger_axis(memory);store=Store(memory,read_only=True)
        try:cost=audit_cost(store,store.budget()['limit'] if limit is None else limit)
        finally:store.close()
        if cost['pending']:raise ContractError('development cost ledger unfinished')
        weight=1 if axis=='development_bcu' else suite['cost_weights'][axis]
        amount=cost['charged_or_reserved']*weight;total+=amount
        extra.append({'directory':str(memory),'axis':axis,'weight':weight,'cost':cost,'weighted_cost':amount})
    prefix=None
    if evolution_context is not None:
        prefix=read_json(evolution_context);memory=Path(prefix['memory']).resolve()
        if memory in used:raise ContractError('controller cost ledger counted twice')
        if not (memory/'memory.sqlite').exists():raise ContractError('controller cost ledger missing')
        store=Store(memory,read_only=True)
        try:total+=verify_cost_prefix(store,prefix)
        finally:store.close()
    result={key:report[key] for key in ['passed','gain','regression','evaluation_ids','identity','comparison_groups','online','evaluation_mode']}
    result.update(total_cost=number(total,'total cost',0),controller_cost_prefix=prefix,utility_report_hash=digest(report),
        binding=report['binding'],utility_receipts=receipts,utility_cost_axes=report['cost_axes'],cost_weights=suite['cost_weights'],development_costs=extra,
        qualification='Amazon ledger axes converted with frozen weights; development and outer controller ledgers already in BCU. Includes declared abandoned ledgers. Does not price CPU admission/rescoring wall time or unrecorded costs.')
    write_json(output,result);return result
