"""Host-owned coverage triage; a failed tool is not evidence of a model deficit."""
from .core import ContractError,digest,number,required,validate_card,read_json


class MeasurementDeferred(ContractError):
    pass


def validate_requirements(requirements):
    if not isinstance(requirements,list):raise ContractError('measurement requirements must be a list')
    seen=set()
    for item in requirements:
        required(item,['measurement','blocked_decision','competing_explanations','cheapest_test','development_budget','acceptance_contract'])
        name=item['measurement']
        if not isinstance(name,str) or not name or name in seen:raise ContractError('distinct measurement names required')
        seen.add(name)
        if not isinstance(item['competing_explanations'],list) or len(item['competing_explanations'])<2 or not all(isinstance(v,str) and v for v in item['competing_explanations']):
            raise ContractError('measurement must distinguish at least two explanations')
        for key in ('blocked_decision','cheapest_test','acceptance_contract'):
            if not isinstance(item[key],str) or not item[key]:raise ContractError('measurement requirement text missing: '+key)
        number(item['development_budget'],'tool development budget',0)


def triage(research,state,diagnostic,*,requirements=None,review_ref=None,limitation=None,related_gap_ids=None):
    if requirements is None:requirements=research.config.get('measurement_requirements',[])
    validate_requirements(requirements)
    if not requirements:return []
    runs=(state.get('baseline') or {}).get('runs',[])
    if not runs or not all(row.get('valid') for row in runs):
        raise MeasurementDeferred('valid baseline measurements unavailable; investigate data or execution before tool evolution')
    if diagnostic is not None and diagnostic['status']!='success':
        raise MeasurementDeferred('diagnostic execution failed; investigate invocation or implementation before claiming measurement insufficiency')
    measurements=diagnostic['output']['measurements'] if diagnostic else {}
    from .runtime import node,edge
    graph=research.store.snapshot();known={n['id'] for n in graph['nodes']};gaps=[]
    for rule in requirements:
        if rule['measurement'] in measurements:continue
        identity=digest({'rule':rule,'model':state['incumbent'],'protocol':research.protocol_hash,
                         'tool':diagnostic['implementation_hash'] if diagnostic else None,**({'review_ref':review_ref} if review_ref else {})})
        gap_id='measurement-gap-'+identity
        experiments=[r['run_id'] for r in runs if r.get('run_id') in known]
        gap={'measurement_gap_id':gap_id,'related_gap_ids':related_gap_ids or [],'experiment_refs':experiments,
            'blocked_decision':rule['blocked_decision'],'competing_explanations':rule['competing_explanations'],
            'current_tools_and_versions':[diagnostic['tool_version']] if diagnostic else [],
            'observed_limitation':limitation or ('required output absent from successful tool result' if diagnostic else 'no validated diagnostic tool configured'),
            'required_information':rule['measurement'],
            'available_artifacts':diagnostic.get('artifacts',{}) if diagnostic else {},
            'cheapest_test':rule['cheapest_test'],'development_budget':rule['development_budget'],
            'acceptance_contract':rule['acceptance_contract']}
        validate_card('measurement_gap',gap);gaps.append(gap)
        if gap_id in known:continue
        attrs={**gap,'status':'proposed','scope':'measurement coverage only; no model or causal conclusion',
               'evidence_dependency_ids':[state['incumbent']],
               **({'discovery_review_ref':review_ref} if review_ref else {})}
        research.store.patch({'base_version':research.store.version,'nodes':[
            node(gap_id,'MeasurementGap',attrs,research.task,research.protocol_hash,'agent_hypothesis' if review_ref else 'local_evidence')],'edges':[
            edge(gap_id+'-exposed-'+str(i),ref,gap_id,'EXPOSES',[ref],research.task,research.store.version+1)
            for i,ref in enumerate(experiments)]},accompanying_events=[('measurement_gap_opened',{
                'gap':gap,'model_version':state['incumbent'],'protocol_hash':research.protocol_hash,
                **({'review_ref':review_ref} if review_ref else {})})])
    return gaps


def block_model_proposal(research,attempt,gap,measurement_gaps):
    if not measurement_gaps:return
    from .runtime import node,edge
    model_gap=attempt+'-gap'
    research.store.patch({'base_version':research.store.version,'nodes':[
        node(model_gap,'GapHypothesis',gap,research.task,research.protocol_hash,'agent_hypothesis')],
        'edges':[edge(attempt+'-measurement-block-'+str(i),m['measurement_gap_id'],model_gap,'BLOCKS',
            m['experiment_refs'],research.task,research.store.version+1) for i,m in enumerate(measurement_gaps)]})
    raise MeasurementDeferred('model proposal deferred until required measurements are available: '+', '.join(m['required_information'] for m in measurement_gaps))


def evolution_context(campaign,gap_id):
    """Freeze an observed coverage gap and its public measurements, never the whole archive."""
    from pathlib import Path
    from .store import Store
    root=Path(campaign);amazon=(root/'amazon-auto.json').exists()
    config=read_json(root/('amazon-auto.json' if amazon else 'campaign.json'))
    frozen=read_json(root/('auto-lock.json' if amazon else 'research_archive/manifest.json'))
    if digest(config)!=frozen['config_hash']:raise ContractError('measurement source configuration changed')
    store=Store(root/'memory',read_only=True)
    try:
        store.db.execute('BEGIN')
        integrity=store.verify()
        events=[e for e in store.events() if e['kind']=='measurement_gap_opened' and e['payload']['gap']['measurement_gap_id']==gap_id]
        if len(events)!=1:raise ContractError('one committed measurement gap required')
        event=events[0];payload=event['payload'];gap=payload['gap'];validate_card('measurement_gap',gap)
        if payload['protocol_hash']!=config['protocol_hash']:raise ContractError('measurement gap protocol differs')
        # Use the graph at the trigger, excluding subsequent decisions and outcomes.
        patches=[e for e in store.events() if e['seq']<event['seq'] and e['kind'] in {'graph_committed','graph_rollback'}]
        version=patches[-1]['payload']['version'] if patches else 0
        graph=store.snapshot(version)
        known={n['id']:n for n in graph['nodes']};observations=[]
        for ref in gap['experiment_refs']:
            item=known.get(ref)
            if not item or item['type']!='Experiment':raise ContractError('measurement source experiment missing')
            attrs=item['attrs']
            observations.append({'id':ref,**{k:attrs[k] for k in ['dataset','seed','metrics','valid','failure_kind'] if k in attrs}})
        if store.verify()!=integrity:raise ContractError('measurement source changed while freezing')
        review={}
        if payload.get('review_ref'):
            from .core import Artifacts
            review={'measurement_review':read_json(Artifacts(root/'artifacts/objects').verify(payload['review_ref']))}
        return {'measurement_gap':gap,'observations':observations,**review,
            'source':{'task_id':config['task_id'],'protocol_hash':config['protocol_hash'],
                      'event_head':event['hash'],'trigger_sequence':event['seq'],'graph_version':version,'graph_hash':graph['hash']},
            'instructions':'Develop a diagnostic, not a model change. Validate invocation and applicability first. Independent validation must not reuse these development observations.'}
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()
