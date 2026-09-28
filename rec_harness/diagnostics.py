"""Freeze validated tool versions, execute in CPU jails, retain versioned measurements."""
from pathlib import Path
from .core import ContractError,digest,file_hash,number,public_only,read_json,required,write_json
from .cpu_job import run_cpu_job
from .evolution import Evolution

CONTRACT='public-aggregate-measurement-v1'


def freeze_tool(options,artifacts):
    required(options,['registry','identity','execution','cost','timeout'])
    number(options['cost'],'diagnostic cost',0);number(options['timeout'],'diagnostic timeout',0)
    if options['timeout']<=0:raise ContractError('positive diagnostic timeout required')
    required(options['execution'],['recipe','python','uid_registry'])
    registry=Evolution(options['registry'])
    try:
        head=registry.store.verify()['head'];active=registry._active('tool')
        if active is None:raise ContractError('no committed active diagnostic tool')
        candidate=registry.verified_candidate(active['candidate_id'])
        plan=read_json(registry.root/'candidates'/candidate['candidate_id']/'plan.json')
        if plan['identity']!=options['identity']:raise ContractError('tool validation conditions differ')
        if plan.get('tool_contract')!=CONTRACT:raise ContractError('tool must validate public aggregate measurement contract')
        if 'diagnostic.py' not in candidate['implementation']:raise ContractError('diagnostic.py entry required')
        if candidate['implementation']!=active['implementation']:raise ContractError('active tool implementation differs')
        snapshot={name:artifacts.put(registry.artifacts.verify(ref)) for name,ref in candidate['implementation'].items()}
        from .evolution import STAGES
        stages={stage:{'status':'passed','report_hash':registry._accepted_stage(candidate,plan,stage)['report_hash']}
                for stage in STAGES['tool']}
        context=read_json(registry.root/'candidates'/candidate['candidate_id']/'context.json')
        if digest(context)!=candidate['context_hash']:raise ContractError('tool registration context changed')
        registration={'schema':'validated-diagnostic-registration-v1','entrypoint':'diagnostic.py:measure',
            'contract':CONTRACT,'validated_conditions':plan['identity'],'validation_stages':stages,
            'declared_call_cost':options['cost'],'cost_unit':'campaign budget units',
            'rollback_candidate_id':(active.get('previous') or {}).get('candidate_id'),
            'source_tool_predecessor':context.get('tool_predecessor'),
            'scope':'Diagnostic validation under frozen conditions; not independent model-effect or mechanism evidence. Source ancestry does not establish a local rollback target.'}
        if registry.store.verify()['head']!=head:raise ContractError('tool registry changed during activation')
        worker=Path(__file__).with_name('diagnostic_worker.py')
        return {'candidate_id':candidate['candidate_id'],'implementation':snapshot,'implementation_hash':digest(snapshot),
            'contract':CONTRACT,'plan_hash':candidate['plan_hash'],'registry_event_head':head,'registration':registration,
            'worker':artifacts.put(worker),'execution':options['execution'],'cost':options['cost'],'timeout':options['timeout']}
    finally:registry.store.close()


def validate_measurement(value):
    if not isinstance(value,dict) or set(value)!={'measurements','limitations'}:raise ContractError('diagnostic output fields differ')
    if not isinstance(value['measurements'],dict) or not value['measurements']:raise ContractError('nonempty measurements required')
    for name,result in value['measurements'].items():
        if not isinstance(name,str) or not name or len(name)>128:raise ContractError('invalid measurement name')
        number(result,'diagnostic measurement')
    if not isinstance(value['limitations'],list) or not all(isinstance(v,str) for v in value['limitations']):raise ContractError('diagnostic limitations required')
    public_only(value)
    return value


def record_graph(research,record):
    from .runtime import node,edge
    observation='diagnostic-'+record['identity'];tool_id='tool-'+record['implementation_hash']
    existing={n['id'] for n in research.store.snapshot()['nodes']}
    if observation in existing:return
    nodes=[node(observation,'Observation',{**record,'valid':record['status']=='success',
        'evidence_dependency_ids':[record['model_version']]},research.task,research.protocol_hash)]
    if tool_id not in existing:nodes.append(node(tool_id,'ToolVersion',{
        'version':record['tool_version'],'implementation_hash':record['implementation_hash'],
        **({'registration':record['tool_registration']} if 'tool_registration' in record else {})},research.task,research.protocol_hash))
    research.store.patch({'base_version':research.store.version,'nodes':nodes,'edges':[
        edge(observation+'-measured',observation,tool_id,'MEASURED_BY',[observation],research.task,research.store.version+1),
        edge(observation+'-about',observation,record['model_version'],'ABOUT',[observation],research.task,research.store.version+1)]})


def measure(research,state):
    """Caller holds the campaign lock; only public aggregate metrics enter the jail."""
    tool=research.config.get('diagnostic_tool')
    if tool is None:return None
    baseline=state.get('baseline') or {}
    request={'task_id':research.task,'model_version':state['incumbent'],'protocol_hash':research.protocol_hash,
        'runs':[{'seed':row['seed'],'valid':row['valid'],'metrics':row.get('metrics',{}),
                 **({'dataset':row['dataset']} if 'dataset' in row else {})} for row in baseline.get('runs',[])]}
    public_only(request)
    identity=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']})
    records=[e['payload'] for e in research.store.events() if e['kind']=='diagnostic_completed' and e['payload']['identity']==identity]
    if records:
        record=records[-1]
        for ref in record.get('artifacts',{}).values():research.artifacts.verify(ref)
        record_graph(research,record)
        return record
    directory=research.root/'research_archive/diagnostics'/identity
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'request.json',request)
    files={'request.json':{'path':str(directory/'request.json'),'sha256':file_hash(directory/'request.json')},
        'worker.py':{'path':str(research.artifacts.verify(tool['worker'])),'sha256':tool['worker']['sha256']}}
    for name,ref in tool['implementation'].items():files['tool/'+name]={'path':str(research.artifacts.verify(ref)),'sha256':ref['sha256']}
    attempt=len(list(directory.glob('job-*')))+1;task='diagnostic-'+identity[:24]+'-'+str(attempt)
    research.store.reserve(task,tool['cost']);status='invalid'
    record={'identity':identity,'tool_version':tool['candidate_id'],'implementation_hash':tool['implementation_hash'],
        'input_hash':digest(request),'model_version':state['incumbent'],
        **({'tool_registration':tool['registration']} if 'registration' in tool else {}),
        'qualification':'tool measurement, not independent model evaluation or mechanism proof'}
    try:
        execution=tool['execution'];job=directory/('job-'+str(attempt))
        receipt=run_cpu_job(job,execution['recipe'],files,'worker.py',[],python=execution['python'],
            uid_registry=execution['uid_registry'],artifacts=research.artifacts,timeout=tool['timeout'])
        if receipt['process']['status']!='success':raise ContractError('isolated diagnostic failed')
        record['output']=validate_measurement(read_json(job/'results/measurement.json'))
        record['artifacts']=receipt['collection']['artifacts'];status='success'
    except (ContractError,OSError,ValueError,KeyError,TypeError) as error:
        record['error']=str(error)
    finally:research.store.settle(task,tool['cost'],status)
    record['status']=status;research.store.event('diagnostic_completed',record)
    record_graph(research,record)
    return record


def freeze_validation_tool(options,artifacts):
    """Host-only experimental activation; never creates a registry promotion."""
    required(options,['source','files','execution','cost','timeout'])
    required(options['execution'],['recipe','python','uid_registry'])
    number(options['cost'],'diagnostic cost',0);number(options['timeout'],'diagnostic timeout',0)
    if options['timeout']<=0:raise ContractError('positive diagnostic timeout required')
    snapshot=artifacts.snapshot(options['source'],options['files'])
    if 'diagnostic.py' not in snapshot:raise ContractError('diagnostic.py entry required')
    return {'candidate_id':'validation-'+digest(snapshot)[:24],'implementation':snapshot,
        'implementation_hash':digest(snapshot),'contract':CONTRACT,'validation_only':True,
        'worker':artifacts.put(Path(__file__).with_name('diagnostic_worker.py')),
        'execution':options['execution'],'cost':options['cost'],'timeout':options['timeout']}
