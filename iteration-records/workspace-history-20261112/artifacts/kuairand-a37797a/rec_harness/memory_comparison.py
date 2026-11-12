"""Audit matched research conditions and exact role-visible memory information."""
import argparse
import copy
from pathlib import Path
from .core import ContractError,canonical,digest,read_json,write_json
from .decision_traces import collect_traces
from .memory_view import restore


def campaign_outcomes(root):
    """Explain terminal scores using committed evaluations, including failed attempts."""
    from .runtime import Research
    research=Research(root,agent=object())
    try:
        integrity=research.verify();events=research.store.events();archives={};attempts=[]
        submitted={};completed=set();interrupted=[]
        for event in events:
            value=event['payload'];kind=event['kind']
            if kind=='agent_submitted':submitted[value['task_id']]=value['role']
            if kind=='agent_completed':completed.add(value['task_id'])
            if kind=='budget_settled' and value['status']=='interrupted':interrupted.append(value)
            if kind=='attempt_archived':archives[value['node_id']]=value['archive']
            if kind!='round_completed':continue
            ref=archives.get(value['attempt'],{}).get('observations.json')
            if ref is None:raise ContractError('completed round lacks committed evaluation archive')
            evaluation=read_json(research.artifacts.verify(ref))
            attempts.append({'attempt':value['attempt'],'selection':value['decision']['selection'],
                'valid':evaluation['valid'],'score':evaluation['score'],
                'run_statuses':[{'seed':r['seed'],'valid':r['valid'],'status':r['status'],
                    'failure_kind':r.get('failure_kind'),'error':r.get('error')} for r in evaluation['runs']]})
        return {'attempts':attempts,'all_candidate_attempts_valid':bool(attempts) and all(a['valid'] for a in attempts),
            'incomplete_roles':[{'task_id':key,'role':submitted[key]} for key in sorted(set(submitted)-completed)],
            'interrupted_reservations':interrupted,'event_head':integrity['head'],
            'qualification':'Committed candidate evaluations and execution failures; equal terminal incumbent scores alone do not establish a valid candidate comparison or memory utility.'}
    finally:research.store.close()


def compare_traces(left,right,limit):
    if type(limit) is not int or limit<=0:raise ContractError('positive shared context limit required')
    groups=[]
    for traces,mode in [(left,'graph'),(right,'flat')]:
        seen={};group={}
        for trace in traces:
            role=trace['role'];ordinal=seen.get(role,0);seen[role]=ordinal+1
            context=copy.deepcopy(trace['visible_state'])
            size=len(canonical(context).encode('utf-8'))
            if size>limit:raise ContractError('archived context exceeds frozen byte limit')
            view=context.get('evidence')
            if not view or view.get('presentation',{}).get('mode')!=mode:
                raise ContractError('trace does not use expected memory presentation')
            context['evidence']=restore(view)
            source=trace.get('visible_source_snapshot')
            if source is None:raise ContractError('exact source snapshot required for memory comparison')
            group[(role,ordinal)]={'trace_id':trace['trace_id'],'input_hash':digest(context),
                'source_hash':digest(source),'context_bytes':size,'transport_status':trace['transport_status'],
                'resource_cost':trace['resource_cost']}
        groups.append(group)
    if not left or not right:raise ContractError('both memory groups need completed role traces')
    rows=[]
    for role,ordinal in sorted(set(groups[0])|set(groups[1])):
        a=groups[0].get((role,ordinal));b=groups[1].get((role,ordinal))
        rows.append({'role':role,'ordinal':ordinal,'graph':a,'flat':b,
            'equal_visible_information':bool(a and b and a['input_hash']==b['input_hash'] and a['source_hash']==b['source_hash'])})
    return {'pairs':rows,'all_inputs_equal':all(row['equal_visible_information'] for row in rows),
        'context_byte_limit':limit,'qualification':'Exact normalized role inputs and source; divergent or unmatched traces are not equal-information presentation evidence. Does not establish decision quality or causal utility.'}


def audit_pair(graph_campaign,flat_campaign):
    roots=[Path(graph_campaign),Path(flat_campaign)]
    configs=[read_json(root/'campaign.json') for root in roots]
    if [c.get('memory_presentation','graph') for c in configs]!=['graph','flat']:
        raise ContractError('graph and flat campaigns required in that order')
    conditions=[{k:v for k,v in config.items() if k!='memory_presentation'} for config in configs]
    if conditions[0]!=conditions[1]:raise ContractError('memory groups differ beyond presentation configuration')
    if 'context_byte_limit' not in configs[0]:raise ContractError('online memory comparison requires a frozen common byte limit')
    traces=[collect_traces(root) for root in roots]
    result=compare_traces(*(trace['decision_traces'] for trace in traces),configs[0]['context_byte_limit'])
    result.update(condition_hash=digest(conditions[0]),sources=[trace['source'] for trace in traces])
    result['outcomes']={mode:campaign_outcomes(root) for mode,root in zip(['graph','flat'],roots)}
    if any(result['outcomes'][mode]['event_head']!=trace['source']['event_head'] for mode,trace in zip(['graph','flat'],traces)):
        raise ContractError('campaign changed during memory audit')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('graph_campaign');parser.add_argument('flat_campaign');parser.add_argument('output')
    args=parser.parse_args();write_json(args.output,audit_pair(args.graph_campaign,args.flat_campaign))


if __name__=='__main__':main()
