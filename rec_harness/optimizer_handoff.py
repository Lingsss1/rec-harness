"""Source-bound optimizer measurement referrals; no extra model or review call."""
from pathlib import Path
from .core import Artifacts,ContractError,digest,number,public_only,read_json,validate_card
from .trace_context import expand_traces
from .measurement_discovery import validate_review
from .store import Store


def request(context,budget):
    number(budget,'optimizer measurement development budget',0)
    original=expand_traces(context);evidence=[]
    for trace in original.get('decision_traces',[]):
        if trace.get('transport_status')!='success':continue
        visible=trace.get('visible_state',{});diagnostic=visible.get('diagnostic')
        if diagnostic and diagnostic.get('status')!='success':continue
        rows=(visible.get('baseline') or {}).get('runs',[])
        if rows and all(r.get('valid') is True for r in rows):
            rows=[{k:r[k] for k in ['run_id','seed','dataset','metrics','valid'] if k in r} for r in rows]
        else:
            rows=visible.get('development_baseline',[]);receipts=visible.get('baseline_execution_evidence',[])
            if not rows or len(rows)!=len(receipts) or not all(r.get('status')=='success' for r in receipts):continue
        evidence.append({'trace_id':trace['trace_id'],'model_version':trace.get('model_version'),
            'trace_hash':digest(trace),'rows':rows,'diagnostic':diagnostic})
    value={'allowed_evidence_refs':[e['trace_id'] for e in evidence],'trace_evidence':evidence,
        'diagnostic':next((e['diagnostic'] for e in reversed(evidence) if e['diagnostic']),None),
        'measurement_contract':{'development_budget':budget,'available_artifacts':['supplied public development aggregates'],
            'instructions':'Cite only allowed trace IDs. Missing candidate outcomes or inaccessible private artifacts are not measurement gaps. Require satisfied data/applicability and one independently testable requirement.'}}
    public_only(value);return value


def validate(context,response,candidate_id):
    packet=context['measurement_handoff_request'];validate_review(packet,response)
    if response['assessment']!='measurement_gap':raise ContractError('optimizer handoff must establish a measurement gap')
    selected=[row for row in packet['trace_evidence'] if row['trace_id'] in response['evidence_refs']]
    if not selected:raise ContractError('optimizer handoff lacks usable development evidence')
    rule=response['requirements'][0]
    if any(rule['measurement'] in (r['diagnostic'] or {}).get('output',{}).get('measurements',{}) for r in selected):
        raise ContractError('proposed measurement already exists in cited trace')
    gap={'measurement_gap_id':'optimizer-measurement-gap-'+digest([candidate_id,response,packet]),
        'related_gap_ids':[],'experiment_refs':[], 'blocked_decision':rule['blocked_decision'],
        'competing_explanations':rule['competing_explanations'],
        'current_tools_and_versions':sorted({r['diagnostic']['tool_version'] for r in selected if r['diagnostic']}),
        'observed_limitation':response['rationale'],'required_information':rule['measurement'],
        'available_artifacts':{'trace_hashes':{r['trace_id']:r['trace_hash'] for r in selected}},
        'cheapest_test':rule['cheapest_test'],'development_budget':rule['development_budget'],'acceptance_contract':rule['acceptance_contract']}
    validate_card('measurement_gap',gap);return gap


def load_context(campaign):
    root=Path(campaign);config=read_json(root/'evolution-auto.json')
    if digest(config)!=read_json(root/'auto-lock.json')['config_hash']:raise ContractError('optimizer source configuration changed')
    store=Store(root/'memory',read_only=True)
    try:
        store.verify();events=[e for e in store.events() if e['kind']=='measurement_handoff_recorded']
        if len(events)!=1:raise ContractError('one committed optimizer measurement handoff required')
        event=events[0];value=event['payload'];objects=Artifacts(root/'objects')
        response=read_json(objects.verify(value['proposal_ref']))
        if value['context_hash']!=digest(config['context']):raise ContractError('handoff source context changed')
        gap=validate(config['context'],response,value['candidate_id'])
        if gap!=value['measurement_gap']:raise ContractError('handoff gap differs from source review')
        source=config['context']['source']
        return {'measurement_gap':gap,'measurement_review':response,
            'observations':config['context']['measurement_handoff_request']['trace_evidence'],
            'source':{**source,'optimizer_event_hash':event['hash']},
            'instructions':'Develop only the missing diagnostic; these source traces are development evidence and must be excluded from independent utility validation.'}
    finally:store.close()
