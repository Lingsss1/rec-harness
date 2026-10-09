"""Lossless presentation controls for an already selected evidence bundle.

Flat records retain relationships as data: dropping them would confound
presentation with information availability. This does not equalize token cost.
"""
import copy

from .core import ContractError, canonical, digest, public_only


def present(bundle, mode='graph'):
    if mode not in {'graph', 'flat'}:
        raise ContractError('memory_presentation must be graph or flat')
    public_only(bundle)
    if 'presentation' in bundle or 'records' in bundle:
        raise ContractError('expected original evidence bundle')
    result = copy.deepcopy(bundle)
    if mode == 'flat':
        result['records'] = ([{'record_kind': 'node', 'value': n} for n in result.pop('nodes')]
                             + [{'record_kind': 'edge', 'value': e} for e in result.pop('edges')])
    result['presentation'] = {'schema': 'equal-information-v1', 'mode': mode,
                              'information_hash': digest(bundle)}
    return result


def restore(view):
    result = copy.deepcopy(view)
    metadata = result.pop('presentation')
    if metadata.get('schema') != 'equal-information-v1':
        raise ContractError('unknown memory presentation schema')
    if metadata['mode'] == 'flat':
        if 'nodes' in result or 'edges' in result:
            raise ContractError('conflicting flat and graph records')
        records = result.pop('records')
        if any(set(r) != {'record_kind', 'value'} or r['record_kind'] not in {'node', 'edge'} for r in records):
            raise ContractError('invalid flat record')
        result['nodes'] = [r['value'] for r in records if r['record_kind'] == 'node']
        result['edges'] = [r['value'] for r in records if r['record_kind'] == 'edge']
    elif metadata['mode'] != 'graph' or 'records' in result:
        raise ContractError('unknown or conflicting memory presentation')
    if digest(result) != metadata['information_hash']:
        raise ContractError('memory information hash mismatch')
    public_only(result)
    return result


def paired_views(bundle):
    graph, flat = present(bundle, 'graph'), present(bundle, 'flat')
    if restore(graph) != restore(flat):
        raise ContractError('memory presentations differ in information')
    return {'graph': graph, 'flat': flat, 'information_hash': digest(bundle),
            'qualification': 'same selected evidence; presentation control only',
            'online_utility_evidence': False}


def paired_contexts(context, max_bytes):
    """Freeze equal information under one serialized-context cap, not equal tokens."""
    if type(max_bytes) is not int or max_bytes<=0:
        raise ContractError('positive memory context byte limit required')
    original=copy.deepcopy(context)
    if 'evidence' not in original:raise ContractError('memory comparison requires visible evidence')
    view=original['evidence']
    bundle=restore(view) if 'presentation' in view else view
    pair=paired_views(bundle)
    contexts={mode:{**copy.deepcopy(original),'evidence':pair[mode]} for mode in ['graph','flat']}
    sizes={mode:len(canonical(value).encode('utf-8')) for mode,value in contexts.items()}
    if max(sizes.values())>max_bytes:raise ContractError('paired memory context exceeds common byte limit')
    original['evidence']=bundle
    return {'contexts':contexts,'audit':{'normalized_context_hash':digest(original),
        'information_hash':pair['information_hash'],'context_bytes':sizes,'max_context_bytes':max_bytes,
        'qualification':'identical facts, relations and other context; common byte cap, not equal token usage'}}
