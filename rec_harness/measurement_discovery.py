"""Codex proposes missing measurements; the host retains prerequisites and routing."""
from .agents import schema_for,validate_schema
from .core import ContractError,digest,number,public_only,read_json,write_json
from .measurement_gaps import MeasurementDeferred,triage,validate_requirements


def validate_config(config):
    if config is None:return
    if not isinstance(config,dict) or set(config)!={'development_budget'}:
        raise ContractError('measurement_discovery requires development_budget only')
    number(config['development_budget'],'measurement development budget',0)


def review_context(research,state,context,gap):
    config=research.config.get('measurement_discovery');validate_config(config)
    if config is None:return None
    runs=(state.get('baseline') or {}).get('runs',[])
    if not runs or not all(row.get('valid') is True for row in runs):
        raise MeasurementDeferred('valid baseline required before measurement discovery')
    diagnostic=context.get('diagnostic')
    if diagnostic is not None and diagnostic.get('status')!='success':
        raise MeasurementDeferred('repair diagnostic execution before measurement discovery')
    graph=research.store.snapshot();known={n['id']:n for n in graph['nodes']}
    refs=[r['run_id'] for r in runs if r.get('run_id') in known and known[r['run_id']]['type']=='Experiment']
    if not refs or len(refs)!=len(runs):raise MeasurementDeferred('auditable baseline experiment references required')
    # Strict projection: no checkpoint, interaction, user-level result or arbitrary runtime path.
    rows=[{k:r[k] for k in ['run_id','seed','dataset','metrics','valid'] if k in r} for r in runs]
    request={'task_id':research.task,'model_version':state['incumbent'],'protocol_hash':research.protocol_hash,
        'gap':gap,'baseline':rows,'allowed_evidence_refs':refs,
        **{key:context[key] for key in ['activation_diagnostics','block_score_diagnostics'] if context.get(key)},
        'diagnostic':({k:diagnostic[k] for k in ['status','tool_version','implementation_hash','output'] if k in diagnostic} if diagnostic else None),
        'measurement_contract':{'development_budget':config['development_budget'],'budget_unit':'tool-development BCU; not GPU seconds or model-training budget',
            'available_artifacts':['public development aggregate rows','model source supplied to role'],
            'requirements_limit':1,'prerequisite_values':'Each prerequisite needs status and rationale grounded in available evidence. Unresolved preconditions prohibit a measurement gap. Only invocation/implementation may be not_applicable when no diagnostic is configured.',
            'instructions':'Judge whether existing metrics can evaluate the next model experiment. Unexecuted candidate training, its unknown performance and host training feasibility are handled by the model controller; they are not measurement prerequisites. Prefer existing measurements. Missing raw data or checkpoints is a prerequisite problem, not permission to access them. Do not claim a missing measurement just because no diagnostic is configured. If proposing a gap, explain why invocation, implementation, data and applicability permit the cheapest discriminating test using only available artifacts. Requirements are provisional until independently validated.'}}
    for field,row in [(field,row) for field in ['activation_diagnostics','block_score_diagnostics'] for row in context.get(field,[])]:
        label='activation' if field=='activation_diagnostics' else 'block score'
        ref=row.get('observation_id')
        observation=known.get(ref)
        if (not observation or observation['type']!='Observation'
                or observation['attrs'].get('task_id')!=research.task
                or observation['attrs'].get('protocol_hash')!=research.protocol_hash
                or observation['attrs'].get('origin')!='local_evidence'
                or observation['attrs'].get('valid') is not True
                or row.get('model_version')!=state['incumbent']):
            raise ContractError(label+' review requires a valid current-model observation')
        fields={'observation_id','model_version','dataset','seed','measurements','limitations','scope'}
        if set(row)!=fields or any(row[k]!=observation['attrs'].get(k) for k in fields):
            raise ContractError(label+' review differs from committed observation')
        request['allowed_evidence_refs'].append(ref)
    if context.get('activation_diagnostics'):
        request['measurement_contract']['available_artifacts'].append('committed activation aggregates; history scope and limitations retained')
    if context.get('block_score_diagnostics'):
        request['measurement_contract']['available_artifacts'].append('committed fixed-weight development block-score aggregates; not retrained deletion effects')
    public_only(request);return request


def validate_review(request,response):
    validate_schema(response,schema_for('measurement'))
    public_only(response)
    if not response['rationale'].strip():raise ContractError('measurement rationale required')
    if not response['evidence_refs'] or not set(response['evidence_refs'])<=set(request['allowed_evidence_refs']):
        raise ContractError('measurement review must cite supplied evidence references')
    if any(not value['rationale'].strip() for value in response['prerequisites'].values()):
        raise ContractError('measurement prerequisite assessment missing')
    requirements=response['requirements'];validate_requirements(requirements)
    if response['assessment']!='measurement_gap' and requirements:
        raise ContractError('only measurement_gap may propose requirements')
    if response['assessment']=='measurement_gap':
        for key,value in response['prerequisites'].items():
            allowed={'satisfied'}
            if key in {'invocation','implementation'} and request.get('diagnostic') is None:allowed.add('not_applicable')
            if value['status'] not in allowed:raise ContractError('unresolved measurement prerequisite: '+key)
        if len(requirements)!=1:raise ContractError('one discriminating measurement required')
        if requirements[0]['development_budget']>request['measurement_contract']['development_budget']:
            raise ContractError('measurement proposal exceeds host development envelope')
        available=set((request.get('diagnostic') or {}).get('output',{}).get('measurements',{}))
        for row in request.get('activation_diagnostics',[])+request.get('block_score_diagnostics',[]):
            available.update(row['measurements'])
        if requirements[0]['measurement'] in available:
            raise ContractError('proposed measurement already exists; describe a distinct missing quantity')


def discover(research,state,context,gap,attempt,invoke):
    request=review_context(research,state,context,gap)
    if request is None:return []
    key=digest(request)
    records=[e['payload'] for e in research.store.events() if e['kind']=='measurement_reviewed' and e['payload']['input_hash']==key]
    if records:
        record=records[-1];response=read_json(research.artifacts.verify(record['review_ref']))
        if digest(read_json(research.artifacts.verify(record['input_ref'])))!=key:raise ContractError('measurement review input changed')
    else:
        response=invoke(request);validate_review(request,response)
        path=research.root/'research_archive/measurement-reviews'/(key+'.json');write_json(path,response)
        input_path=path.with_name(key+'-input.json');write_json(input_path,request)
        record={'input_hash':key,'input_ref':research.artifacts.put(input_path),'review_ref':research.artifacts.put(path),'attempt':attempt,
            'model_version':state['incumbent'],'assessment':response['assessment']}
        research.store.event('measurement_reviewed',record)
    validate_review(request,response)
    if response['assessment']=='prerequisite_unresolved':
        raise MeasurementDeferred('measurement prerequisites unresolved: '+response['rationale'])
    if response['assessment']=='sufficient':return []
    return triage(research,state,context.get('diagnostic'),requirements=response['requirements'],review_ref=record['review_ref'],
        limitation=response['rationale'],related_gap_ids=[attempt+'-gap'])
