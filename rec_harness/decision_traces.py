"""Build optimizer inputs from committed role-visible prefixes, not later outcomes."""
from pathlib import Path
from .core import Artifacts,ContractError,digest,public_only,read_json,write_json
from .memory_view import restore
from .store import Store


def collect_traces(campaign,*,roles=None,cutoff_seq=None):
    root=Path(campaign);amazon=(root/'amazon-auto.json').exists()
    config=read_json(root/('amazon-auto.json' if amazon else 'campaign.json'))
    if read_json(root/('auto-lock.json' if amazon else 'research_archive/manifest.json'))['config_hash']!=digest(config):raise ContractError('trace campaign configuration changed')
    store=Store(root/'memory',read_only=True);artifacts=Artifacts(root/'artifacts/objects')
    try:
        store.db.execute('BEGIN')
        integrity=store.verify();events=store.events()
        cutoff=integrity['events'] if cutoff_seq is None else cutoff_seq
        if type(cutoff) is not int or not 0<=cutoff<=integrity['events']:raise ContractError('invalid trace cutoff')
        submitted={};traces=[]
        for event in events:
            if event['seq']>cutoff:break
            value=event['payload']
            if event['kind']=='agent_submitted':submitted[value['task_id']]=event
            if event['kind']!=('role_completed' if amazon else 'agent_completed'):continue
            if amazon and 'task_id' not in value:continue
            start=submitted.get(value['task_id'])
            # A staging/preflight failure can settle a reservation before any
            # role input was submitted. It is an execution failure, not a decision.
            if start is None:continue
            request=start['payload'];role=request['role']
            if roles is not None and role not in roles:continue
            archive=value['archive'];context_ref=request.get('visible_state_ref') or archive.get('context.json')
            if context_ref:
                context=read_json(artifacts.verify(context_ref))
            elif archive.get('request.json'):
                context=read_json(artifacts.verify(archive['request.json'])).get('context')
                context_ref=archive['request.json']
            else:raise ContractError('trace lacks the actual role input')
            if digest(context)!=request['input_hash']:raise ContractError('trace input differs from submitted identity')
            response=read_json(artifacts.verify(archive['response.json'])) if 'response.json' in archive else None
            resources=read_json(artifacts.verify(archive['resources.json'])) if 'resources.json' in archive else {}
            bundle=context.get('evidence');original=restore(bundle) if bundle and 'presentation' in bundle else bundle
            snapshot=request.get('source_snapshot')
            if snapshot is not None:
                for ref in snapshot.values():artifacts.verify(ref)
            trace={'trace_id':value['task_id'],'role':role,'task_id':config['task_id'],
                'timestamp':start['created_at'],'base_llm_version':config.get('agent_backend' if amazon else 'codex',{}).get('model','configured default; exact service model not recorded'),
                'skill_versions':{'availability':'not recorded by transport'},
                'tool_versions':({config['diagnostic_tool']['candidate_id']:config['diagnostic_tool']['implementation_hash']} if config.get('diagnostic_tool') else {}),
                'harness_version':config.get('harness_version','amazon:'+digest(config.get('research_policy',{}))[:12]),'protocol_hash':config['protocol_hash'],
                'model_version':request.get('model_version'),'graph_snapshot_id':request['graph_version'],
                'visible_state':context,'visible_state_ref':context_ref,'input_hash':request['input_hash'],
                'visible_source_snapshot':snapshot,'source_visibility':'exact snapshot' if snapshot is not None else 'legacy source-before snapshot unavailable',
                'evidence_bundle_hash':digest(bundle) if bundle else None,
                'retrieved_node_ids':[n['id'] for n in original['nodes']] if original else [],
                'role_output':response,'transport_status':value['status'],
                'resource_cost':{'declared_bcu':1 if amazon else config['protocol']['agent_cost'],'wall_seconds':value['wall_seconds'],
                                 'unit':'role_call' if amazon else 'bcu',
                                 'token_usage':resources.get('token_usage')},
                'submitted_event':start['hash'],'completed_event':event['hash'],
                'qualification':'visible input and role output; transport success is not scientific or protocol validation'}
            public_only(trace);traces.append(trace)
        if not traces:raise ContractError('no committed decision traces in requested prefix')
        if store.verify()!=integrity:raise ContractError('trace source changed while freezing')
        result={'schema':'research-decision-prefix-v1','decision_traces':traces,
            'source':{'task_id':config['task_id'],'protocol_hash':config['protocol_hash'],
                'cutoff_seq':cutoff,'event_head':events[cutoff-1]['hash'] if cutoff else '0'*64},
            'instructions':'Diagnose decision-process hypotheses from these exact visible inputs. Do not assign outcomes to unexecuted alternatives. Do not infer model or tool failure from transport failure.'}
        result['hash']=digest(result);return result
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()


def project_traces(research):
    """Project after round decisions so knowledge handoff graph versions stay frozen."""
    from .runtime import node,edge
    events=research.store.events()
    submitted={e['payload']['task_id'] for e in events if e['kind']=='agent_submitted'}
    if not any(e['kind']=='agent_completed' and e['payload']['task_id'] in submitted for e in events):return
    package=collect_traces(research.root)
    known={n['id'] for n in research.store.snapshot()['nodes']};nodes=[];edges=[]
    harness='harness-'+digest(research.config.get('research_policy',{}))
    if harness not in known:
        nodes.append(node(harness,'HarnessVersion',{'version':research.config['harness_version'],
            'policy_hash':digest(research.config.get('research_policy',{}))},research.task,research.protocol_hash))
    for trace in package['decision_traces']:
        trace_id='trace-'+trace['trace_id']
        if trace_id in known:continue
        path=research.root/'research_archive/decision_traces'/(trace['trace_id']+'.json');write_json(path,trace)
        # Contexts live in CAS; nesting whole visible graphs inside graph nodes
        # would recursively inflate subsequent role inputs.
        nodes.append(node(trace_id,'DecisionTrace',{'role':trace['role'],'input_hash':trace['input_hash'],
            'transport_status':trace['transport_status'],'model_version':trace['model_version'],
            'trace_ref':research.artifacts.put(path),'resource_cost':trace['resource_cost']},research.task,research.protocol_hash))
        edges.append(edge(trace_id+'-generated',trace_id,harness,'GENERATED_BY',[trace_id],research.task,research.store.version+1))
    if nodes:research.store.patch({'base_version':research.store.version,'nodes':nodes,'edges':edges})
