"""Paired prompt replay on exact historical role-visible prefixes.

This checks contracts, not scientific truth or counterfactual model utility.
Only read-only reasoning roles are supported; untested patch scopes fail closed.
"""
import argparse
import copy
import fcntl
from pathlib import Path
from .agents import schema_for,validate_schema
from .core import Artifacts,ContractError,canonical,digest,file_hash,identifier,number,read_json,validate_card,write_json
from .decision_traces import collect_traces
from .evolution import validate_policy
from .jailed_agent import JailedCodexAgent
from .memory_view import restore,present
from .process_identity import assert_stopped
from .store import Store

ROLES={'evidence','hypothesis','evaluation','lead','researcher','reviewer','analyst'}


def check_response(role,context,response):
    validate_schema(response,schema_for(role))
    if role=='evidence':
        validate_card('gap',response)
        if response['model_version']!=context['model_version'] or response['protocol_hash']!=context['protocol_hash']:
            raise ContractError('replay gap changed frozen identity')
    elif role=='hypothesis':
        method=response['method'];proposal=response['proposal'];validate_card('method',method);validate_card('proposal',proposal)
        if proposal['base_model_id']!=context['model_version'] or proposal['train_protocol_hash']!=context['protocol_hash'] or proposal['seeds']!=context['protocol']['seeds']:
            raise ContractError('replay proposal changed model, protocol or seeds')
        if context['gap']['gap_id'] not in proposal['gap_ids'] or method['method_id'] not in proposal['method_ids']:
            raise ContractError('replay proposal broke gap/method handoff')
        if proposal['budget']>context['budget']['remaining']:raise ContractError('replay proposal over budget')
        if context.get('supported_controls') is not None and proposal['controls']!=context['supported_controls']:
            raise ContractError('replay proposal changed paired controls')
    else:
        validate_card('decision',response)
        if response['selection']=='keep' and context.get('comparison',{}).get('effect_gate_passed') is not True:
            raise ContractError('replay keep bypassed host effect gate')
    bundle=context.get('evidence',{})
    if 'presentation' in bundle:bundle=restore(bundle)
    known={n['id'] for n in bundle.get('nodes',[])}
    for key in ['baseline','evaluation']:
        known.update(row['run_id'] for row in (context.get(key) or {}).get('runs',[]) if 'run_id' in row)
    refs=response.get('observation_refs',response.get('evidence_refs',[]))
    if not set(refs).issubset(known):raise ContractError('replay cites evidence outside visible prefix')


def historical_bundle(graph,original,limit):
    from .retrieval import select,revision_visibility
    body={key:graph[key] for key in ['nodes','edges']}
    if digest(body)!=graph['hash'] or graph['version']!=original['graph_snapshot_id'] or graph['hash']!=original['graph_hash']:
        raise ContractError('historical graph binding changed')
    nodes={n['id']:n for n in graph['nodes'] if n['attrs']['task_id']==original['task_id'] and n['attrs']['protocol_hash']==original['protocol_hash']}
    revealed=original.get('retrieval',{}).get('revealed')
    if revealed is not None:
        nodes={key:n for key,n in nodes.items() if set(n['attrs'].get('evidence_dependency_ids',[key])).issubset(revealed)}
    edges=[e for e in graph['edges'] if e['source'] in nodes and e['target'] in nodes
           and (revealed is None or set(e['attrs'].get('evidence_refs',[])).issubset(nodes))]
    counter={e['source'] for e in edges if e['relation'] in {'CONTRADICTS','ALTERNATIVE_TO'}}
    counter|={n['id'] for n in nodes.values() if n['attrs'].get('status') in {'disputed','demoted'} or n['type']=='Failure' or n['attrs'].get('counterevidence') is True}
    query=original.get('retrieval',{}).get('query','')
    selected,audit=select(nodes,edges,counter,original['role'],query,limit,canonical,
        semantic=original.get('retrieval',{}).get('semantic'))
    revisions=revision_visibility(nodes,edges,selected)
    return {**{k:original[k] for k in ['graph_snapshot_id','graph_hash','role','task_id','protocol_hash']},
        **({'revision_visibility':revisions} if revisions['entries'] else {}),
        'nodes':[nodes[key] for key in selected],'edges':[e for e in edges if e['source'] in selected and e['target'] in selected],
        'retrieval':{'query':query,'limit':limit,'retrieved_node_ids':selected,'counterevidence_available':len(counter),
            'counterevidence_omitted':len(counter-set(selected)),
            **({'revealed':revealed} if revealed is not None else {}),**audit}}


def replay_context(case,policy,baseline):
    context=copy.deepcopy(case['context'])
    if case.get('adapter')=='amazon-nextitnet-v1':
        if policy.get('retrieval_limit')!=baseline.get('retrieval_limit'):
            raise ContractError('Amazon prefix cannot reconstruct changed historical memory retrieval')
        return context
    if policy.get('retrieval_limit',24)==baseline.get('retrieval_limit',24):return context
    if not case.get('graph'):raise ContractError('retrieval replay requires exact historical graph')
    view=context['evidence'];original=restore(view) if 'presentation' in view else view
    bundle=historical_bundle(case['graph'],original,policy.get('retrieval_limit',24))
    context['evidence']=present(bundle,view['presentation']['mode']) if 'presentation' in view else bundle
    return context


def freeze_suite(specification,directory):
    path=Path(specification).resolve();spec=read_json(path);root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise ContractError('prefix suite directory must be empty')
    campaign=(path.parent/spec['campaign']).resolve()
    if (campaign/'amazon-auto.json').exists():
        from .amazon_prefix import freeze_suite as freeze_amazon
        return freeze_amazon(path,root,spec)
    if not (campaign/'campaign.json').exists():raise ContractError('prefix replay currently requires generic research traces')
    roles=spec.get('roles',sorted(ROLES))
    if not roles or not set(roles).issubset(ROLES):raise ContractError('unsupported prefix role')
    backend=spec['agent_backend']
    if not backend.get('model'):raise ContractError('prefix replay requires explicit model')
    for key in ['budget','call_cost']:number(spec[key],key,0)
    traces=collect_traces(campaign,roles=roles,cutoff_seq=spec.get('cutoff_seq'))
    source=Artifacts(campaign/'artifacts/objects');objects=Artifacts(root/'objects');cases=[]
    for trace in traces['decision_traces']:
        if trace['visible_source_snapshot'] is None:raise ContractError('exact pre-call source snapshot required')
        inputs={name:objects.put(source.verify(ref)) for name,ref in trace['visible_source_snapshot'].items()}
        graph=None;view=trace['visible_state'].get('evidence',{})
        original=restore(view) if 'presentation' in view else view
        if 'graph_snapshot_id' in original:
            store=Store(campaign/'memory',read_only=True)
            try:graph=store.snapshot(original['graph_snapshot_id'])
            finally:store.close()
            rebuilt=historical_bundle(graph,original,original['retrieval']['limit'])
            if rebuilt!=original:raise ContractError('historical retrieval cannot be reproduced exactly')
        cases.append({'graph':graph,'id':trace['trace_id'],'role':trace['role'],'context':trace['visible_state'],
            'input_hash':trace['input_hash'],'source':inputs,'submitted_event':trace['submitted_event']})
    baseline=read_json(campaign/'campaign.json').get('research_policy',{});validate_policy(baseline)
    body={'schema':'harness-prefix-suite-v1','cases':cases,'baseline_policy':baseline,
        **{key:spec[key] for key in ['agent_backend','identity','budget','call_cost']},'trace_source':traces['source']}
    body['hash']=digest(body);write_json(root/'suite.json',body);return body


def run_validation(suite_path,candidate,output,*,agent_factory=None,memory_context_bytes=None):
    path=Path(suite_path).resolve();suite=read_json(path)
    if suite.get('schema')!='harness-prefix-suite-v1' or suite['hash']!=digest({k:v for k,v in suite.items() if k!='hash'}):
        raise ContractError('prefix suite changed')
    if suite.get('adapter')=='amazon-nextitnet-v1':
        from .adapters.amazon_execution import code_identity
        if suite['code_hashes']!=code_identity():raise ContractError('Amazon prefix implementation changed')
    policy=read_json(Path(candidate)/'policy.json');validate_policy(policy)
    baseline=suite['baseline_policy'];roles={c['role'] for c in suite['cases']}
    memory_pairs={}
    if memory_context_bytes is not None:
        from .memory_view import paired_contexts
        if policy!=baseline:raise ContractError('memory presentation comparison requires identical policies')
        memory_pairs={case['id']:paired_contexts(case['context'],memory_context_bytes) for case in suite['cases']}
    changed={r for r in set(baseline.get('prompts',{}))|set(policy.get('prompts',{})) if baseline.get('prompts',{}).get(r)!=policy.get('prompts',{}).get(r)}
    if not changed.issubset(roles):raise ContractError('patch roles lack frozen prefix coverage')
    for key in ['routing']:
        if policy.get(key)!=baseline.get(key):raise ContractError('prefix prompt replay cannot validate changed '+key)
    inputs=Artifacts(path.parent/'objects')
    for case in suite['cases']:
        identifier(case['id'])
        if case['role'] not in ROLES or digest(case['context'])!=case['input_hash']:raise ContractError('prefix input changed')
        for ref in case['source'].values():inputs.verify(ref)
        replay_context(case,policy,baseline)
    root=Path(output).resolve().parent/('memory-prefix-replay' if memory_context_bytes is not None else 'prefix-replay');root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('prefix replay already running') from error
        assert_stopped(root/'cases');store=Store(root/'memory');objects=Artifacts(root/'objects')
        try:
            binding={'suite_hash':suite['hash'],'policy_hash':digest(policy),'validator_hash':file_hash(__file__),
                'transport_hashes':{name:file_hash(Path(__file__).with_name(name)) for name in ['agents.py','jailed_agent.py']},
                'web_search':'disabled'}
            if memory_context_bytes is not None:
                binding.update(memory_context_bytes=memory_context_bytes,memory_view_hash=file_hash(Path(__file__).with_name('memory_view.py')))
            if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('prefix replay binding changed')
            write_json(root/'binding.json',binding);store.configure_budget(suite['budget']);store.recover();rows=[]
            for case in suite['cases']:
                pair={}
                for group,patch in [('baseline',baseline),('candidate',policy)]:
                    context=replay_context(case,patch,baseline)
                    if memory_context_bytes is not None:
                        context=memory_pairs[case['id']]['contexts']['graph' if group=='baseline' else 'flat']
                    key=case['id']+'-'+group
                    records=[e['payload'] for e in store.events() if e['kind']=='prefix_replay_completed' and e['payload']['key']==key]
                    if records:
                        result=records[-1]
                        for ref in result.get('artifacts',{}).values():objects.verify(ref)
                    else:
                        from .prefix_recovery import prepare
                        from .validation_case import settle_once
                        folder=root/'cases'/key;job=folder/'attempt-1'
                        call_binding={'context':context,'source':case['source'],'policy':patch,'role':case['role'],'validation':binding}
                        task,result=prepare(store,objects,folder,key,call_binding)
                        if result is None:
                            job.mkdir(parents=True,exist_ok=True)
                            if case['source']:inputs.restore(case['source'],job/'model')
                            store.reserve(task,suite['call_cost'])
                            submission={'key':key,'task_id':task,'binding_hash':digest(call_binding)}
                            store.event('prefix_replay_submitted',submission)
                            result={'key':key,'passed':False,'input_hash':digest(context)}
                            try:
                                options=suite['agent_backend']
                                agent=agent_factory(case,group) if agent_factory else JailedCodexAgent(options['template'],options['runtime_root'],model=options['model'],timeout=options.get('timeout',300),prompts=patch.get('prompts'),web_search='disabled',role_skills=options.get('role_skills'))
                                response=agent.run(case['role'],context,job)
                                write_json(job/'replay-response.json',response)
                                if case.get('adapter')=='amazon-nextitnet-v1':
                                    from .amazon_prefix import check_response as check_amazon
                                    check_amazon(case,context,response)
                                else:check_response(case['role'],context,response)
                                result['passed']=True
                            except (ContractError,OSError,ValueError,KeyError,TypeError) as error:result['error']=str(error)
                            finally:settle_once(store,task,suite['call_cost'],'accepted' if result['passed'] else 'rejected')
                            write_json(job/'replay-context.json',context)
                            names=[name for name in ['replay-context.json','replay-response.json','request.json','resources.json','literature-search.json','stdout.jsonl'] if (job/name).is_file()]
                            result['artifacts']=objects.snapshot(job,names)
                            store.event('prefix_replay_result_recorded',{'key':key,'request_hash':digest(submission),'result':result})
                        store.event('prefix_replay_completed',result)
                    pair[group]=result
                rows.append({'evaluation_id':case['id'],**pair})
            if not rows:raise ContractError('empty prefix suite')
            delta=sum(int(r['candidate']['passed'])-int(r['baseline']['passed']) for r in rows)/len(rows)
            report={'passed':all(r['candidate']['passed'] for r in rows),'gain':delta,
                'regression':max(int(r['baseline']['passed'])-int(r['candidate']['passed']) for r in rows),
                'total_cost':store.budget()['charged_or_reserved'],'evaluation_ids':[r['evaluation_id'] for r in rows],
                'identity':suite['identity'],'comparison_groups':['baseline','candidate'],'online':False,'rows':rows,
                'binding':binding,'integrity':store.verify(),'qualification':'Paired prompt replay and contract checks only; not semantic truth, scientific utility, or counterfactual model results'}
            if memory_context_bytes is not None:
                report.update(memory_controls={key:value['audit'] for key,value in memory_pairs.items()},
                    presentation_groups={'baseline':'graph','candidate':'flat'},
                    qualification='Same historical information and policy with a common context byte cap; presentation contract replay only, not online utility or equal token usage')
            report['regression']=max(0,report['regression']);write_json(output,report);return report
        finally:store.close()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('specification');freeze.add_argument('directory')
    run=sub.add_parser('run');run.add_argument('--suite',required=True);run.add_argument('--candidate',required=True);run.add_argument('--output',required=True)
    run.add_argument('--memory-context-bytes',type=int)
    args=parser.parse_args()
    if args.action=='freeze':freeze_suite(args.specification,args.directory)
    else:run_validation(args.suite,args.candidate,args.output,memory_context_bytes=args.memory_context_bytes)


if __name__=='__main__':main()
