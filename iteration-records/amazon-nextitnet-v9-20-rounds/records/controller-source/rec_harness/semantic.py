"""Frozen neural retrieval receipts; scoring is restricted to eligible nodes."""
import math
import re
from .core import ContractError,canonical,digest,public_only


def documents(nodes):
    """Stable input bytes, including the node's scoped evidence and limitations."""
    return {key:canonical(nodes[key]) for key in sorted(nodes)}


def receipt(nodes,query,encoder,scores,*,execution=None):
    body={'schema':'semantic-retrieval-v1','query_hash':digest(query),
          'documents':{key:digest(text) for key,text in documents(nodes).items()},
          'encoder':encoder,'scores':scores,
          **({'execution':execution} if execution is not None else {})}
    result={**body,'hash':digest(body)}
    validate(result,nodes,query)
    return result


def validate(record,nodes,query):
    public_only(record)
    required={'schema','query_hash','documents','encoder','scores','hash'}
    if (not isinstance(record,dict) or not required<=set(record) or set(record)-required-{'execution'}
            or record['schema']!='semantic-retrieval-v1'):
        raise ContractError('invalid semantic retrieval receipt')
    if record['hash']!=digest({k:v for k,v in record.items() if k!='hash'}):
        raise ContractError('semantic retrieval receipt changed')
    if not isinstance(query,str) or not query.strip() or record['query_hash']!=digest(query):
        raise ContractError('semantic retrieval query differs')
    expected={key:digest(text) for key,text in documents(nodes).items()}
    if record['documents']!=expected:
        raise ContractError('semantic retrieval eligible documents differ')
    encoder=record['encoder']
    if (not isinstance(encoder,dict) or any(not isinstance(encoder.get(key),str)
            or not re.fullmatch('[0-9a-f]{64}',encoder[key]) for key in ('model_hash','implementation_hash'))
            or not isinstance(encoder.get('settings'),dict) or not encoder['settings']):
        raise ContractError('semantic retrieval requires frozen encoder identity')
    scores=record['scores']
    if not isinstance(scores,dict) or set(scores)!=set(nodes):
        raise ContractError('semantic retrieval must score every eligible node exactly once')
    if any(type(value) not in {int,float} or not math.isfinite(value)
           or not -1.000001<=value<=1.000001 for value in scores.values()):
        raise ContractError('semantic retrieval scores must be finite cosine similarities')
    if 'execution' in record:
        execution=record['execution']
        if (not isinstance(execution,dict) or set(execution)!=
                {'elapsed_seconds','cpu_seconds','document_count','query_truncated','truncated_documents'}
                or any(type(execution[key]) not in {int,float} or not math.isfinite(execution[key])
                       or execution[key]<0 for key in ('elapsed_seconds','cpu_seconds'))
                or type(execution['document_count']) is not int or execution['document_count']!=len(nodes)
                or type(execution['query_truncated']) is not bool
                or not isinstance(execution['truncated_documents'],list)
                or any(not isinstance(key,str) or key not in nodes for key in execution['truncated_documents'])
                or len(execution['truncated_documents'])!=len(set(execution['truncated_documents']))):
            raise ContractError('invalid semantic execution receipt')
    return scores
