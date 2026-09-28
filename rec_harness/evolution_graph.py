"""Deterministic lifecycle graph from committed evolution events, without scores."""
from .core import ContractError,digest,read_json


def lifecycle_graph(events,contexts=None):
    nodes={};edges={};candidates={};versions={}
    def node(identity,kind,attrs,task,protocol,event):
        nodes[identity]={'id':identity,'type':kind,'attrs':{**attrs,'task_id':task,'protocol_hash':protocol,
            'origin':'local_evidence','visibility':'public','source_event_hash':event['hash'],
            'independent_unit':False,'scope':'Evolution lifecycle evidence; not model-effect or mechanism evidence.'}}
    def link(source,target,relation,event,task):
        identity='evolution-edge-'+digest([source,target,relation,event['hash']])[:32]
        edges[identity]={'id':identity,'source':source,'target':target,'relation':relation,'attrs':{
            'scope':task,'evidence_refs':[source],'origin':'local_evidence','status':'provisional',
            'created_at':event['created_at'],'revision_id':event['hash']}}
    for event in events:
        value=event['payload'];kind=event['kind']
        if kind=='candidate_proposed':
            candidate=value['candidate_id'];identity='evolution-candidate-'+digest([candidate,value['plan_hash']])[:32]
            if candidate in candidates:raise ContractError('duplicate evolution proposal')
            candidates[candidate]=(identity,value['kind'],value['plan_hash'])
            node(identity,'ToolCandidate' if value['kind']=='tool' else 'HarnessPatch',{
                'candidate_id':candidate,'kind':value['kind'],'plan_hash':value['plan_hash'],
                'context_hash':value['context_hash'],'source_hash':value.get('source_hash'),
                'lifecycle_event':'proposed'},candidate,value['plan_hash'],event)
            context=(contexts or {}).get(candidate)
            if context is not None:
                if digest(context)!=value['context_hash']:raise ContractError('evolution graph context changed')
                gap=context.get('measurement_gap')
                required={'measurement_gap_id','blocked_decision','required_information',
                          'competing_explanations','experiment_refs','current_tools_and_versions'}
                if gap and (not isinstance(gap,dict) or not required.issubset(gap)):
                    nodes[identity]['attrs']['measurement_gap_projection']='legacy_unstructured'
                elif gap:
                    gap_id='evolution-gap-'+digest([candidate,gap['measurement_gap_id']])[:32]
                    node(gap_id,'MeasurementGap',{'source_gap_id':gap['measurement_gap_id'],
                        'blocked_decision':gap['blocked_decision'],'required_information':gap['required_information'],
                        'competing_explanations':gap['competing_explanations'],
                        'external_experiment_refs':gap['experiment_refs'],
                        'external_tool_versions':gap['current_tools_and_versions'],
                        'context_hash':value['context_hash']},candidate,value['plan_hash'],event)
                    nodes[gap_id]['attrs']['origin']='agent_hypothesis'
                    link(identity,gap_id,'ADDRESSES',event,candidate)
        elif kind in {'candidate_implemented','validation_passed','validation_rejected','candidate_promoted','evolution_stopped'}:
            candidate=value['candidate_id']
            # Invalid inputs can stop before a proposal is committed. The raw
            # stop event remains auditable, but cannot establish a candidate.
            if kind=='evolution_stopped' and candidate not in candidates:continue
            if candidate not in candidates:raise ContractError('evolution lifecycle event lacks proposal')
            parent,category,protocol=candidates[candidate]
            identity='evolution-record-'+event['hash'][:32]
            if kind=='candidate_implemented':
                node(identity,'Observation',{'kind':'evolution_implementation','implementation_hash':value['implementation_hash']},candidate,protocol,event)
                link(identity,parent,'ABOUT',event,candidate)
            elif kind in {'validation_passed','validation_rejected'}:
                attrs={'stage':value['stage'],'status':'accepted' if kind=='validation_passed' else 'rejected'}
                if kind=='validation_passed':attrs.update(report_hash=value['report_hash'],implementation_hash=value['implementation_hash'])
                # Detailed validator scores, task IDs and free-text failure
                # messages stay in the source event; only gate outcomes enter here.
                node(identity,'ToolValidation' if category=='tool' else 'HarnessValidation',attrs,candidate,protocol,event)
                link(identity,parent,'EVALUATES',event,candidate)
            elif kind=='candidate_promoted':
                node(identity,'ToolVersion' if category=='tool' else 'HarnessVersion',{
                    'candidate_id':candidate,'implementation_hash':digest(value['implementation']),
                    'lifecycle_event':'promoted'},candidate,protocol,event)
                link(identity,parent,'PROMOTED_FROM',event,candidate)
                previous=value.get('previous')
                if previous:
                    prior=versions.get(previous['candidate_id'])
                    if prior is None:raise ContractError('promotion predecessor missing from graph history')
                    link(identity,prior,'SUPERSEDES',event,candidate)
                versions[candidate]=identity
            else:
                node(identity,'DecisionTrace',{'kind':'evolution_stop','status':value['status']},candidate,protocol,event)
                link(identity,parent,'ABOUT',event,candidate)
        elif kind=='candidate_rollback':
            prior=versions.get(value['from'])
            if prior is None:raise ContractError('rollback source missing from graph history')
            identity='evolution-record-'+event['hash'][:32]
            task=value['from'];protocol=nodes[prior]['attrs']['protocol_hash']
            node(identity,'DecisionTrace',{'kind':'evolution_rollback','from_candidate':value['from'],
                'to_candidate':value['to']},task,protocol,event)
            link(identity,prior,'ABOUT',event,task)
    return {'nodes':sorted(nodes.values(),key=lambda n:n['id']),'edges':sorted(edges.values(),key=lambda e:e['id'])}


def sync_lifecycle(store,*,check_only=False):
    """Append missing facts, rejecting changed projections; repeated sync is free."""
    store.verify();events=store.events()
    contexts={event['payload']['candidate_id']:read_json(store.root.parent/'candidates'/event['payload']['candidate_id']/'context.json')
              for event in events if event['kind']=='candidate_proposed'}
    projection=lifecycle_graph(events,contexts);snapshot=store.snapshot()
    patch={'base_version':snapshot['version']}
    for key in ['nodes','edges']:
        existing={item['id']:item for item in snapshot[key]}
        for item in projection[key]:
            if item['id'] in existing and existing[item['id']]!=item:
                raise ContractError('evolution graph projection changed')
        patch[key]=[item for item in projection[key] if item['id'] not in existing]
    if patch['nodes'] or patch['edges']:
        if check_only:raise ContractError('evolution graph projection incomplete')
        store.patch(patch)
    return {'graph_version':store.version,'nodes':len(projection['nodes']),'edges':len(projection['edges'])}
