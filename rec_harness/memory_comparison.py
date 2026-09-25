"""Audit matched research conditions and exact role-visible memory information."""
import argparse
import copy
from pathlib import Path
from .core import ContractError,canonical,digest,read_json,write_json
from .decision_traces import collect_traces
from .memory_view import restore


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
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('graph_campaign');parser.add_argument('flat_campaign');parser.add_argument('output')
    args=parser.parse_args();write_json(args.output,audit_pair(args.graph_campaign,args.flat_campaign))


if __name__=='__main__':main()
