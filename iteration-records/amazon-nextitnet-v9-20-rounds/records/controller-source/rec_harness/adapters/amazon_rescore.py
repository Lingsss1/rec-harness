"""Independent isolated CPU rescore of an already audited validation checkpoint."""
from pathlib import Path
import fcntl

from rec_harness.core import Artifacts, ContractError, digest, file_hash, read_json, write_json
from rec_harness.cpu_job import run_cpu_job,recover_completed_cpu_job
from rec_harness.process_identity import assert_stopped
from .amazon_controls import load_config
from .amazon_report import audit_job
from .independent_scoring import prepare_request, score_response,scoring_runtime,require_scoring_runtime


def rescore_job(root, model_version, dataset, seed, output, recipe, uid_registry, *, bypass_block=None,bypass_blocks=None,resume=False):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=resume);output.chmod(0o700)
    with (output/'rescore.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_stopped(output/'inference')
        return _rescore_job(root,model_version,dataset,seed,output,recipe,uid_registry,bypass_block,bypass_blocks)


def _rescore_job(root,model_version,dataset,seed,output,recipe,uid_registry,bypass_block,bypass_blocks):
    config=load_config(root)
    if dataset not in config['datasets'] or seed not in config['development_seeds']+config['confirmation_seeds']:
        raise ContractError('rescore outside frozen study')
    if bypass_block is not None and bypass_blocks is not None:raise ContractError('choose single or multiple block intervention')
    blocks=[bypass_block] if bypass_block is not None else bypass_blocks
    if blocks is not None and (not isinstance(blocks,(list,tuple)) or not blocks
            or any(type(index) is not int or index<0 for index in blocks)
            or len(set(blocks))!=len(blocks) or seed not in config['development_seeds']):
        raise ContractError('block intervention requires a nonnegative index and development seed')
    if blocks is not None:blocks=sorted(blocks)
    result,_=audit_job(root,config,model_version,dataset,seed)
    worker=Path(__file__).with_name('score_worker.py')
    plan={'study':str(root.resolve()),'model_version':model_version,'dataset':dataset,'seed':seed,
          'bypass_block':bypass_block,'bypass_blocks':blocks if bypass_block is None else None,
          'training_receipt_hash':digest(result),'protocol_hash':config['protocol_hash'],
          'recipe':recipe,'uid_registry':str(Path(uid_registry).resolve()),'python':config['python'],
          'host_scoring_runtime':scoring_runtime(),
          'worker_hash':file_hash(worker),'host_hash':file_hash(__file__),
          'scorer_hash':file_hash(Path(__file__).with_name('independent_scoring.py'))}
    plan_path=output/'rescore-plan.json'
    if plan_path.exists():
        require_scoring_runtime(read_json(plan_path).get('host_scoring_runtime'))
        if read_json(plan_path)!=plan:raise ContractError('frozen rescore plan changed')
    elif (output/'inference').exists() or (output/'inference-request.json').exists():
        raise ContractError('legacy rescore requires its original recovery procedure')
    else:write_json(plan_path,plan)
    job=root/'jobs'/f'{model_version}-{dataset}-{seed}'
    training=Path(config['data'])/(dataset+'_train.txt');expected=config['data_hashes'][training.name]
    if file_hash(training)!=expected:raise ContractError('rescore training history changed')
    histories={}
    with training.open() as stream:
        for line in stream:
            user,item=map(int,line.split());histories.setdefault(user,[]).append(item)
    if file_hash(training)!=expected:raise ContractError('history changed while reading')
    candidates=read_json(job/'validation_candidates.json')
    rows=[{**row,'history':histories[row['user_id']]} for row in candidates]
    request,key=prepared_request(output,rows)
    paths={'model.py':Path(config['model_versions'][model_version]['path']), 'checkpoint.pt':job/'checkpoint.pt',
           'request.json':output/'inference-request.json','worker.py':worker}
    files={name:{'path':str(path.resolve()),'sha256':file_hash(path)} for name,path in paths.items()}
    if files['model.py']['sha256']!=result['source_hash'] or files['checkpoint.pt']['sha256']!=result['checkpoint_sha256']:
        raise ContractError('rescore source or checkpoint identity changed')
    arguments=['--source','/inputs/model.py','--source-hash',files['model.py']['sha256'],
               '--checkpoint','/inputs/checkpoint.pt','--checkpoint-hash',files['checkpoint.pt']['sha256'],
               '--request','/inputs/request.json','--request-hash',files['request.json']['sha256'],
               '--output','/output/scores.json']
    for index in blocks or []:arguments+=['--bypass-block',str(index)]
    artifacts=Artifacts(root/'artifacts/objects')
    if (output/'inference').exists():
        receipt=recover_completed_cpu_job(output/'inference',recipe,files,'worker.py',arguments,artifacts=artifacts)
    else:
        receipt=run_cpu_job(output/'inference',recipe,files,'worker.py',arguments,python=config['python'],
                            uid_registry=uid_registry,artifacts=artifacts,timeout=180)
    if receipt['process']['status']!='success':raise ContractError('isolated checkpoint inference failed')
    report=score_response(request,key,read_json(artifacts.verify(receipt['collection']['artifacts']['scores.json'])))
    if (output/'host-ranking.json').exists() and read_json(output/'host-ranking.json')!=report:
        raise ContractError('cached host rankings changed')
    write_json(output/'host-ranking.json',report)
    archived=read_json(job/'validation_ranks.json')
    if report['user_ids']!=archived['user_ids']:raise ContractError('rescore user alignment changed')
    summary={'dataset':dataset,'model_version':model_version,'seed':seed,'users':len(rows),
             'protocol_hash':config['protocol_hash'],
             'host_scoring_runtime':plan['host_scoring_runtime'],
             'source_hash':files['model.py']['sha256'],'checkpoint_hash':files['checkpoint.pt']['sha256'],
             'candidates_hash':result['candidates_hash'],
             'score_worker_hash':files['worker.py']['sha256'],
             'host_scorer_hash':file_hash(Path(__file__).with_name('independent_scoring.py')),
             'rank_disagreements':sum(a!=b for a,b in zip(report['ranks'],archived['ranks'])),
             'host_metrics':report['metrics'],'archived_metrics':result['metrics'],
             'positive_tied_users':report['tied_user_count'],'host_report_hash':report['hash'],
             'inference_wall_seconds':receipt['wall_seconds'],
             'inference_process_wall_seconds':receipt['process'].get('wall_seconds'),
             'inference_recovery':receipt.get('recovery'),
             'qualification':'independent CPU rescore on existing validation users; not new training or final test'}
    if blocks is not None:
        summary.update(**({'bypass_block':bypass_block} if bypass_block is not None else {'bypass_blocks':blocks}),
            qualification='Fixed-weight block identity intervention on frozen development candidates; host-computed ranking changes, not retrained deletion benefit or causal mechanism confirmation.')
    summary['hash']=digest(summary)
    if (output/'acceptance.json').exists() and read_json(output/'acceptance.json')!=summary:
        raise ContractError('cached rescore acceptance changed')
    write_json(output/'acceptance.json',summary)
    return summary


def prepared_request(output,rows):
    request_path=output/'inference-request.json';key_path=output/'private-key.json'
    if not request_path.exists():
        if key_path.exists():raise ContractError('private rescore key exists without its request')
        request,key=prepare_request(rows)
        write_json(request_path,request);write_json(key_path,key)
        return request,key
    request=read_json(request_path)
    if set(request)!={'schema','rows'} or request['schema']!='candidate-scores-request-v1' or len(request['rows'])!=len(rows):
        raise ContractError('cached rescore request changed')
    private=[]
    for index,(source,visible) in enumerate(zip(rows,request['rows'])):
        if (set(visible)!={'row_id','history','candidates'} or visible['row_id']!='row-'+str(index)
            or visible['history']!=source['history'] or sorted(visible['candidates'])!=sorted(source['candidates'])):
            raise ContractError('cached rescore history or candidates changed')
        private.append({'row_id':visible['row_id'],'user_id':source['user_id'],
                        'positive_index':visible['candidates'].index(source['candidates'][0])})
    key={'schema':'candidate-scores-private-v1','request_hash':digest(request),'rows':private}
    if key_path.exists():
        if read_json(key_path)!=key:raise ContractError('cached private rescore key changed')
    else:write_json(key_path,key)
    return request,key
