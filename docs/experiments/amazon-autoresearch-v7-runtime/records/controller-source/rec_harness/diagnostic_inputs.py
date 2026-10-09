"""Versioned, strictly projected public inputs for generated measurements."""
from .core import ContractError,number,public_only

CONTRACT='public-aggregate-measurement-v1'
OBSERVATION_CONTRACT='public-observation-measurement-v2'
CONTRACTS={CONTRACT,OBSERVATION_CONTRACT}
OBSERVATION_FIELDS=('activation_diagnostics','block_score_diagnostics')
ROW_FIELDS={'observation_id','model_version','dataset','seed','measurements','limitations','scope'}


def measurement_request(task,model,protocol,runs,contract=CONTRACT,observations=None):
    if contract not in CONTRACTS:raise ContractError('unsupported diagnostic input contract')
    request={'task_id':task,'model_version':model,'protocol_hash':protocol,
        'runs':[{'seed':row['seed'],'valid':row['valid'],'metrics':row.get('metrics',{}),
                 **({'dataset':row['dataset']} if 'dataset' in row else {})} for row in runs]}
    if contract==OBSERVATION_CONTRACT:
        source=observations or {};result={};seen=set()
        conditions={(row.get('dataset'),row['seed']) for row in request['runs'] if row['valid'] is True}
        for field in OBSERVATION_FIELDS:
            rows=source.get(field,[])
            if not isinstance(rows,list):raise ContractError('diagnostic observations must be lists')
            result[field]=[]
            for row in rows:
                if not isinstance(row,dict) or set(row)!=ROW_FIELDS:
                    raise ContractError('diagnostic observation fields differ')
                if any(not isinstance(row[key],str) or not row[key].strip()
                       for key in ['observation_id','model_version','dataset','scope']):
                    raise ContractError('diagnostic observation identity and scope required')
                if row['model_version']!=model or type(row['seed']) is not int or (row['dataset'],row['seed']) not in conditions:
                    raise ContractError('diagnostic observation outside current development baseline')
                if row['observation_id'] in seen:raise ContractError('duplicate diagnostic observation')
                seen.add(row['observation_id'])
                if not isinstance(row['measurements'],dict) or not row['measurements']:
                    raise ContractError('diagnostic observation measurements required')
                for name,value in row['measurements'].items():
                    if not isinstance(name,str) or not name or len(name)>128:raise ContractError('invalid observation quantity')
                    number(value,'observation quantity')
                if not isinstance(row['limitations'],list) or not all(isinstance(v,str) for v in row['limitations']):
                    raise ContractError('diagnostic observation limitations required')
                result[field].append({key:row[key] for key in sorted(ROW_FIELDS)})
        request['observations']=result
    public_only(request)
    return request


def verify_observations(request,store):
    """Only current, committed local observations may reach executable tool code."""
    if not any(request.get('observations',{}).values()):return
    known={n['id']:n for n in store.snapshot()['nodes']}
    for rows in request.get('observations',{}).values():
        for row in rows:
            entry=known.get(row['observation_id']);attrs=(entry or {}).get('attrs',{})
            if (not entry or entry['type']!='Observation' or attrs.get('valid') is not True
                    or attrs.get('origin')!='local_evidence' or attrs.get('visibility')!='public'
                    or attrs.get('task_id')!=request['task_id'] or attrs.get('protocol_hash')!=request['protocol_hash']
                    or any(row[key]!=attrs.get(key) for key in ROW_FIELDS)):
                raise ContractError('diagnostic observation differs from committed current-model evidence')
