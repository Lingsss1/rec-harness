"""Host-recomputed four-condition fixed-checkpoint block interaction contrast."""
from pathlib import Path
from rec_harness.core import ContractError,digest,read_json
from .independent_scoring import score_response


def summarize_interaction(directories):
    if len(directories)!=4:raise ContractError('reference, two singles and one pair required')
    conditions={};bindings=None;reports=[];paired_inputs=None
    keys=['dataset','model_version','seed','users','protocol_hash','source_hash','checkpoint_hash',
          'candidates_hash','score_worker_hash','host_scorer_hash']
    for directory in map(Path,directories):
        value=read_json(directory/'acceptance.json')
        if value.get('hash')!=digest({k:v for k,v in value.items() if k!='hash'}):
            raise ContractError('intervention receipt changed')
        identity={key:value[key] for key in keys}
        if bindings is not None and identity!=bindings:raise ContractError('unpaired block intervention bindings')
        bindings=identity
        if 'bypass_block' in value and 'bypass_blocks' in value:raise ContractError('ambiguous block intervention')
        indices=value.get('bypass_blocks',[value['bypass_block']] if 'bypass_block' in value else [])
        if any(type(i) is not int or i<0 for i in indices) or len(set(indices))!=len(indices):
            raise ContractError('invalid block intervention indices')
        indices=tuple(sorted(indices))
        if indices in conditions:raise ContractError('duplicate block intervention condition')
        request=read_json(directory/'inference-request.json');private=read_json(directory/'private-key.json')
        report=score_response(request,private,read_json(directory/'inference/results/scores.json'))
        by_id={row['row_id']:row for row in request['rows']}
        input_identity=digest([{'history':by_id[row['row_id']]['history'],
                                'candidates':sorted(by_id[row['row_id']]['candidates'])} for row in private['rows']])
        if paired_inputs is not None and paired_inputs!=input_identity:
            raise ContractError('unpaired block histories or candidate sets')
        paired_inputs=input_identity
        if report['hash']!=value['host_report_hash'] or report['metrics']!=value['host_metrics']:
            raise ContractError('intervention scores differ from host receipt')
        if value['users']!=len(report['ranks']):raise ContractError('intervention user count changed')
        if report['tied_user_count']:raise ContractError('unresolved ties in block interaction')
        conditions[indices]=(value,report);reports.append(value['hash'])
    pairs=[key for key in conditions if len(key)==2]
    if len(pairs)!=1:raise ContractError('exactly one two-block intervention required')
    a,b=pairs[0]
    if set(conditions)!={(),(a,),(b,),(a,b)}:raise ContractError('incomplete factorial block conditions')
    reference=conditions[()][1]
    if conditions[()][0]['rank_disagreements']!=0:raise ContractError('reference differs from archived rankings')
    for value,report in conditions.values():
        if report['user_ids']!=reference['user_ids']:raise ContractError('unpaired block intervention users')
        if value['rank_disagreements']!=sum(x!=y for x,y in zip(reference['ranks'],report['ranks'])):
            raise ContractError('intervention rank difference changed')
    metrics={name:{'reference':reference['metrics'][name],
        'single_a':conditions[(a,)][1]['metrics'][name],'single_b':conditions[(b,)][1]['metrics'][name],
        'pair':conditions[(a,b)][1]['metrics'][name],
        'interaction':conditions[(a,b)][1]['metrics'][name]-conditions[(a,)][1]['metrics'][name]
                      -conditions[(b,)][1]['metrics'][name]+reference['metrics'][name]}
        for name in reference['metrics']}
    result={'schema':'block-interaction-v1','bindings':bindings,'blocks':[a,b],'metrics':metrics,
        'source_receipts':sorted(reports),
        'qualification':'Fixed-checkpoint metric interaction M(ab)-M(a)-M(b)+M(reference). Nonadditivity does not establish redundancy, retrained deletion benefit or mechanism confirmation.'}
    result['hash']=digest(result);return result
