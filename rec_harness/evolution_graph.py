"""Deterministic lifecycle graph from committed evolution events, without scores."""
from .core import Artifacts,ContractError,digest,read_json


def lifecycle_graph(events,contexts=None,proposals=None):
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
        return edges[identity]
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
                if context.get('trace_context_encoding',{}).get('schema')=='shared-trace-fields-v1':
                    from .trace_context import expand_traces
                    context=expand_traces(context)
                predecessor=context.get('tool_predecessor')
                if predecessor and value['kind']=='tool':
                    source_id='evolution-source-tool-'+digest([candidate,predecessor])[:32]
                    node(source_id,'ToolVersion',{**predecessor,'lifecycle_event':'inherited_source',
                        'qualification':'Validated source inherited from a predecessor campaign; not a promotion in this campaign.'},candidate,value['plan_hash'],event)
                    link(identity,source_id,'BASED_ON',event,candidate)
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
        elif kind=='measurement_handoff_recorded':
            candidate=value['candidate_id'];parent,category,protocol=candidates[candidate]
            gap=value['measurement_gap'];identity=gap['measurement_gap_id']
            node(identity,'MeasurementGap',{**gap,'proposal_ref':value['proposal_ref'],'status':'proposed'},candidate,protocol,event)
            nodes[identity]['attrs']['origin']='agent_hypothesis'
            link(parent,identity,'EXPOSES',event,candidate)
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
                proposal=(proposals or {}).get(candidate)
                if category=='tool' and proposal is not None:
                    from .tool_method import validate
                    validate(proposal,nodes[parent]['attrs'].get('source_hash'))
                    method='evolution-method-'+event['hash'][:32]
                    node(method,'Method',{**proposal,'proposal_ref':value['proposal_ref'],
                        'status':'provisional','verification_status':'author_declared_unverified'},candidate,protocol,event)
                    origin='paper_claim' if proposal['route']=='paper' else 'agent_hypothesis'
                    nodes[method]['attrs']['origin']=origin
                    link(parent,method,'BASED_ON',event,candidate)['attrs']['origin']='agent_hypothesis'
                    citations={}
                    for source in proposal['sources']:
                        mappings=citations.setdefault(source['locator'],[])
                        if source['method_mapping'] not in mappings:mappings.append(source['method_mapping'])
                    for citation,mappings in citations.items():
                        source='evolution-method-source-'+digest([candidate,protocol,citation])[:32]
                        node(source,'Source',{'citation':citation,'citation_hash':digest(citation),
                            'verification_status':'agent_reported_unverified','source_version':'unknown',
                            'retrieved_content_ref':None},candidate,protocol,event)
                        nodes[source]['attrs']['origin']='paper_claim'
                        relationship=link(method,source,'DERIVED_FROM',event,candidate)
                        relationship['attrs'].update(origin='paper_claim',method_mappings=mappings,evidence_refs=[source])
                if category=='harness' and proposal is not None:
                    hypothesis='evolution-failure-'+event['hash'][:32]
                    node(hypothesis,'DecisionFailureHypothesis',{
                        'failure_pattern':proposal['failure_pattern'],
                        'causal_hypothesis':proposal['causal_hypothesis'],
                        'expected_decision_change':proposal['expected_decision_change'],
                        'regression_risks':proposal['regression_risks'],
                        'proposal_ref':value['proposal_ref'],'status':'provisional'},candidate,protocol,event)
                    nodes[hypothesis]['attrs']['origin']='agent_hypothesis'
                    link(parent,hypothesis,'ADDRESSES',event,candidate)
                    trace_context=(contexts or {}).get(candidate,{})
                    if trace_context.get('trace_context_encoding',{}).get('schema')=='shared-trace-fields-v1':
                        from .trace_context import expand_traces
                        trace_context=expand_traces(trace_context)
                    traces={t['trace_id']:t for t in trace_context.get('decision_traces',[])}
                    for reference in proposal['trace_refs']:
                        if reference not in traces:continue  # Legacy references remain in the immutable proposal.
                        trace=traces[reference]
                        trace_id='evolution-trace-'+digest([candidate,reference])[:32]
                        node(trace_id,'DecisionTrace',{'external_trace_id':reference,
                            'trace_hash':digest(trace),'context_hash':nodes[parent]['attrs']['context_hash'],
                            'qualification':'Frozen optimizer input; does not establish the causal hypothesis.'},candidate,protocol,event)
                        if trace.get('harness_version'):
                            harness_id='evolution-source-harness-'+digest([candidate,trace['harness_version']])[:32]
                            node(harness_id,'HarnessVersion',{'external_version':trace['harness_version'],
                                'lifecycle_event':'observed_in_trace',
                                'qualification':'Version attribution in frozen source trace; not a local promotion.'},candidate,protocol,event)
                            link(trace_id,harness_id,'GENERATED_BY',event,candidate)
                        support=link(hypothesis,trace_id,'SUPPORTED_BY',event,candidate)
                        # This is the optimizer's supporting attribution, not an
                        # independently confirmed causal relationship.
                        support['attrs']['origin']='agent_hypothesis'
                        support['attrs']['evidence_refs']=[trace_id]
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
    artifacts=Artifacts(store.root.parent/'objects')
    proposals={event['payload']['candidate_id']:read_json(artifacts.verify(event['payload']['proposal_ref']))
               for event in events if event['kind']=='candidate_implemented' and event['payload'].get('proposal_ref')}
    projection=lifecycle_graph(events,contexts,proposals);snapshot=store.snapshot()
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
