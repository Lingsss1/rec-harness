"""Freeze validated tool versions, execute in CPU jails, retain versioned measurements."""
from pathlib import Path
from .core import ContractError,digest,file_hash,number,public_only,read_json,required,write_json
from .cpu_job import run_cpu_job,recover_completed_cpu_job
from .evolution import Evolution

from .diagnostic_inputs import CONTRACT,OBSERVATION_CONTRACT,CONTRACTS,measurement_request,verify_observations


def runtime_contract(contract=CONTRACT):
    """Public authoring contract for newly initialized diagnostic candidates."""
    if contract not in CONTRACTS:raise ContractError('unsupported diagnostic input contract')
    result={'contract':contract,'entrypoint':'diagnostic.py:measure(request)',
        'request':{'task_id':'string','model_version':'string','protocol_hash':'string',
            'runs':'list of objects: seed, valid, metrics mapping; dataset present when available'},
        'response':{'measurements':'nonempty mapping of quantity names (1–128 characters) to finite numbers',
            'limitations':'list of strings; report unsupported inputs and interpretation limits'},
        'execution':'The controller imports diagnostic.py in a CPU jail and calls measure once with the request object; return the response object, do not write result files.',
        'available_inputs':'Only supplied public development aggregate rows. No raw interactions, user IDs, rankings, checkpoints, or model internals are supplied by this contract.',
        'constraints':'Do not modify training, weights or scoring. Define behavior for empty, malformed or insufficient inputs; reject unsupported cases instead of inventing measurements. Validation and promotion are performed independently by the host.'}
    if contract==OBSERVATION_CONTRACT:
        result['request']['observations']={field:'list of committed current-model development observations: observation_id, model_version, dataset, seed, measurements (finite named quantities), limitations, scope'
            for field in ['activation_diagnostics','block_score_diagnostics']}
        result['available_inputs']='Public development metric rows and supplied committed activation/block-score aggregate observations only. Lists may be empty. Preserve each observation scope and limitations; repeated quantities or probes are not independent training. No raw interactions, user IDs, rankings, tensors, checkpoints or model access.'
    return result


def validate_runtime(options):
    """Check execution settings before spending research or validation costs."""
    required(options,['execution','cost','timeout'])
    number(options['cost'],'diagnostic cost',0);number(options['timeout'],'diagnostic timeout',0)
    if options['timeout']<=0:raise ContractError('positive diagnostic timeout required')
    required(options['execution'],['recipe','python','uid_registry'])
    if not isinstance(options['execution']['recipe'],dict):raise ContractError('diagnostic recipe must be an object')
    for key in ('python','uid_registry'):
        if not isinstance(options['execution'][key],str) or not options['execution'][key].strip():
            raise ContractError('diagnostic '+key+' must be a nonempty string')


def freeze_tool(options,artifacts):
    required(options,['registry','identity'])
    validate_runtime(options)
    registry=Evolution(options['registry'])
    try:
        head=registry.store.verify()['head'];active=registry._active('tool')
        if active is None:raise ContractError('no committed active diagnostic tool')
        candidate=registry.verified_candidate(active['candidate_id'])
        plan=read_json(registry.root/'candidates'/candidate['candidate_id']/'plan.json')
        if plan['identity']!=options['identity']:raise ContractError('tool validation conditions differ')
        contract=plan.get('tool_contract')
        if contract not in CONTRACTS:raise ContractError('tool must validate public aggregate measurement contract')
        if 'diagnostic.py' not in candidate['implementation']:raise ContractError('diagnostic.py entry required')
        if candidate['implementation']!=active['implementation']:raise ContractError('active tool implementation differs')
        snapshot={name:artifacts.put(registry.artifacts.verify(ref)) for name,ref in candidate['implementation'].items()}
        from .evolution import STAGES
        stages={stage:{'status':'passed','report_hash':registry._accepted_stage(candidate,plan,stage)['report_hash']}
                for stage in STAGES['tool']}
        context=read_json(registry.root/'candidates'/candidate['candidate_id']/'context.json')
        if digest(context)!=candidate['context_hash']:raise ContractError('tool registration context changed')
        registration={'schema':'validated-diagnostic-registration-v1','entrypoint':'diagnostic.py:measure',
            'contract':contract,'validated_conditions':plan['identity'],'validation_stages':stages,
            'usage':{'execution':'The controller automatically measures the current incumbent before research roles; use the committed diagnostic output and its limitations.',
                'input':'task_id, model_version, protocol_hash, runs of seed/valid/metrics and optional dataset; public development aggregates only',
                'output':'measurements: nonempty mapping of quantity names to finite numbers; limitations: list of strings',
                'interpretation':'Check model_version and validation conditions before use. Missing quantities may warrant measurement review; invocation failure must be repaired first. A measurement alone does not establish model improvement or mechanism causality.'},
            'declared_call_cost':options['cost'],'cost_unit':'campaign budget units',
            'rollback_candidate_id':(active.get('previous') or {}).get('candidate_id'),
            'source_tool_predecessor':context.get('tool_predecessor'),
            'scope':'Diagnostic validation under frozen conditions; not independent model-effect or mechanism evidence. Source ancestry does not establish a local rollback target.'}
        if contract==OBSERVATION_CONTRACT:
            registration['usage']['input']=runtime_contract(contract)['available_inputs']
        if plan.get('tool_method_contract'):
            registration['method_card']=read_json(registry.artifacts.verify(candidate['proposal_ref']))
            registration['method_card_hash']=candidate['proposal_ref']['sha256']
            registration['method_card_status']='author declaration; source content and scientific utility are not established by this card'
        if registry.store.verify()['head']!=head:raise ContractError('tool registry changed during activation')
        worker=Path(__file__).with_name('diagnostic_worker.py')
        return {'candidate_id':candidate['candidate_id'],'implementation':snapshot,'implementation_hash':digest(snapshot),
            'contract':contract,'plan_hash':candidate['plan_hash'],'registry_event_head':head,'registration':registration,
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


def measure(research,state,*,observations=None):
    """Caller holds the campaign lock; only public aggregate metrics enter the jail."""
    tool=research.config.get('diagnostic_tool')
    if tool is None:return None
    baseline=state.get('baseline') or {}
    request=measurement_request(research.task,state['incumbent'],research.protocol_hash,
        baseline.get('runs',[]),tool.get('contract',CONTRACT),observations)
    verify_observations(request,research.store)
    identity=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']})
    records=[e['payload'] for e in research.store.events() if e['kind']=='diagnostic_completed' and e['payload']['identity']==identity]
    if records:
        record=records[-1]
        for ref in record.get('artifacts',{}).values():research.artifacts.verify(ref)
        if tool.get('contract')==OBSERVATION_CONTRACT:
            if read_json(research.artifacts.verify(record['request_ref']))!=request:
                raise ContractError('cached diagnostic observations changed')
        record_graph(research,record)
        return record
    directory=research.root/'research_archive/diagnostics'/identity
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'request.json',request)
    files={'request.json':{'path':str(directory/'request.json'),'sha256':file_hash(directory/'request.json')},
        'worker.py':{'path':str(research.artifacts.verify(tool['worker'])),'sha256':tool['worker']['sha256']}}
    for name,ref in tool['implementation'].items():files['tool/'+name]={'path':str(research.artifacts.verify(ref)),'sha256':ref['sha256']}
    execution=tool['execution']
    submissions=[e['payload'] for e in research.store.events() if e['kind']=='diagnostic_submitted' and e['payload']['identity']==identity]
    receipt=None
    if submissions:
        submitted=submissions[-1];attempt=submitted['attempt'];task=submitted['task_id']
        if submitted['request_hash']!=digest({'files':files,'execution':execution}):
            raise ContractError('diagnostic recovery inputs changed')
        job=directory/('job-'+str(attempt))
        # Only a completed, stopped, input-bound process can be reused. Unknown
        # outcomes are never silently converted into another diagnostic call.
        receipt=recover_completed_cpu_job(job,execution['recipe'],files,'worker.py',[],artifacts=research.artifacts)
    else:
        if list(directory.glob('job-*')):raise ContractError('uncommitted legacy diagnostic attempt requires review')
        attempt=1;task='diagnostic-'+identity[:24]+'-1';job=directory/'job-1'
        research.store.reserve(task,tool['cost'])
        research.store.event('diagnostic_submitted',{'identity':identity,'attempt':attempt,'task_id':task,
            'request_hash':digest({'files':files,'execution':execution})})
    status='invalid'
    record={'identity':identity,'tool_version':tool['candidate_id'],'implementation_hash':tool['implementation_hash'],
        'input_hash':digest(request),'model_version':state['incumbent'],
        **({'tool_registration':tool['registration']} if 'registration' in tool else {}),
        'qualification':'tool measurement, not independent model evaluation or mechanism proof'}
    if tool.get('contract')==OBSERVATION_CONTRACT:
        record['request_ref']=research.artifacts.put(directory/'request.json')
    try:
        if receipt is None:
            receipt=run_cpu_job(job,execution['recipe'],files,'worker.py',[],python=execution['python'],
                uid_registry=execution['uid_registry'],artifacts=research.artifacts,timeout=tool['timeout'])
        if receipt['process']['status']!='success':raise ContractError('isolated diagnostic failed')
        record['output']=validate_measurement(read_json(research.artifacts.verify(receipt['collection']['artifacts']['measurement.json'])))
        record['artifacts']=receipt['collection']['artifacts'];status='success'
    except (ContractError,OSError,ValueError,KeyError,TypeError) as error:
        record['error']=str(error)
    finally:
        if not any(e['kind']=='budget_settled' and e['payload']['task_id']==task for e in research.store.events()):
            research.store.settle(task,tool['cost'],status)
    record['status']=status;research.store.event('diagnostic_completed',record)
    record_graph(research,record)
    return record


def freeze_validation_tool(options,artifacts):
    """Host-only experimental activation; never creates a registry promotion."""
    required(options,['source','files','execution','cost','timeout'])
    required(options['execution'],['recipe','python','uid_registry'])
    number(options['cost'],'diagnostic cost',0);number(options['timeout'],'diagnostic timeout',0)
    if options['timeout']<=0:raise ContractError('positive diagnostic timeout required')
    contract=options.get('contract',CONTRACT)
    if contract not in CONTRACTS:raise ContractError('unsupported diagnostic input contract')
    snapshot=artifacts.snapshot(options['source'],options['files'])
    if 'diagnostic.py' not in snapshot:raise ContractError('diagnostic.py entry required')
    return {'candidate_id':'validation-'+digest(snapshot)[:24],'implementation':snapshot,
        'implementation_hash':digest(snapshot),'contract':contract,'validation_only':True,
        'worker':artifacts.put(Path(__file__).with_name('diagnostic_worker.py')),
        'execution':options['execution'],'cost':options['cost'],'timeout':options['timeout']}
