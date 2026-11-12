"""Lossless sharing of repeated fields in optimizer-visible decision traces."""
import copy
from collections import Counter
from .core import ContractError,digest,canonical


def compact_traces(context):
    traces=context.get('decision_traces',[])
    reserved={'shared_visible_values','trace_context_encoding'}
    if reserved & set(context) or any('visible_state_shared_fields' in t for t in traces):return context
    values={};counts=Counter()
    for trace in traces:
        for value in trace.get('visible_state',{}).values():
            if len(canonical(value).encode())<512:continue
            key=digest(value);values[key]=value;counts[key]+=1
    shared={key:values[key] for key,count in counts.items() if count>1}
    if not shared:return context
    result=copy.deepcopy(context)
    for trace in result['decision_traces']:
        state=trace.get('visible_state',{});refs={}
        for field in list(state):
            key=digest(state[field])
            if key in shared:refs[field]=key;del state[field]
        if refs:trace['visible_state_shared_fields']=refs
    result['shared_visible_values']=shared
    result['trace_context_encoding']={'schema':'shared-trace-fields-v1','original_context_hash':digest(context),
        'instructions':'Each trace visible_state consists of its inline fields plus visible_state_shared_fields. Resolve each referenced hash in shared_visible_values. Shared values are exact original input fields, not summaries; no information is omitted.'}
    if len(canonical(result).encode())>=len(canonical(context).encode()):return context
    return result


def expand_traces(context):
    encoding=context.get('trace_context_encoding')
    if encoding is None:return context
    if encoding.get('schema')!='shared-trace-fields-v1':raise ContractError('unknown trace context encoding')
    result=copy.deepcopy(context);shared=result.pop('shared_visible_values');result.pop('trace_context_encoding')
    for key,value in shared.items():
        if digest(value)!=key:raise ContractError('shared trace value changed')
    used=set()
    for trace in result['decision_traces']:
        refs=trace.pop('visible_state_shared_fields',{})
        for field,key in refs.items():
            if key not in shared or field in trace['visible_state']:raise ContractError('invalid shared trace reference')
            trace['visible_state'][field]=copy.deepcopy(shared[key]);used.add(key)
    if used!=set(shared) or digest(result)!=encoding['original_context_hash']:
        raise ContractError('expanded trace context differs from original')
    return result
