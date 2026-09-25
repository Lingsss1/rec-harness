"""Independent isolated CPU rescore of an already audited validation checkpoint."""
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, file_hash, read_json, write_json
from rec_harness.cpu_job import run_cpu_job
from .amazon_controls import load_config
from .amazon_report import audit_job
from .independent_scoring import prepare_request, score_response


def rescore_job(root, model_version, dataset, seed, output, recipe, uid_registry):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False);output.chmod(0o700)
    config=load_config(root)
    if dataset not in config['datasets'] or seed not in config['development_seeds']+config['confirmation_seeds']:
        raise ContractError('rescore outside frozen study')
    result,_=audit_job(root,config,model_version,dataset,seed)
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
    request,key=prepare_request(rows)
    write_json(output/'inference-request.json',request);write_json(output/'private-key.json',key)
    paths={'model.py':Path(config['model_versions'][model_version]['path']), 'checkpoint.pt':job/'checkpoint.pt',
           'request.json':output/'inference-request.json','worker.py':Path(__file__).with_name('score_worker.py')}
    files={name:{'path':str(path.resolve()),'sha256':file_hash(path)} for name,path in paths.items()}
    if files['model.py']['sha256']!=result['source_hash'] or files['checkpoint.pt']['sha256']!=result['checkpoint_sha256']:
        raise ContractError('rescore source or checkpoint identity changed')
    arguments=['--source','/inputs/model.py','--source-hash',files['model.py']['sha256'],
               '--checkpoint','/inputs/checkpoint.pt','--checkpoint-hash',files['checkpoint.pt']['sha256'],
               '--request','/inputs/request.json','--request-hash',files['request.json']['sha256'],
               '--output','/output/scores.json']
    receipt=run_cpu_job(output/'inference',recipe,files,'worker.py',arguments,python=config['python'],
                        uid_registry=uid_registry,artifacts=Artifacts(root/'artifacts/objects'),timeout=180)
    if receipt['process']['status']!='success':raise ContractError('isolated checkpoint inference failed')
    report=score_response(request,key,read_json(output/'inference/results/scores.json'))
    write_json(output/'host-ranking.json',report)
    archived=read_json(job/'validation_ranks.json')
    if report['user_ids']!=archived['user_ids']:raise ContractError('rescore user alignment changed')
    summary={'dataset':dataset,'model_version':model_version,'seed':seed,'users':len(rows),
             'source_hash':files['model.py']['sha256'],'checkpoint_hash':files['checkpoint.pt']['sha256'],
             'score_worker_hash':files['worker.py']['sha256'],
             'host_scorer_hash':file_hash(Path(__file__).with_name('independent_scoring.py')),
             'rank_disagreements':sum(a!=b for a,b in zip(report['ranks'],archived['ranks'])),
             'host_metrics':report['metrics'],'archived_metrics':result['metrics'],
             'positive_tied_users':report['tied_user_count'],'host_report_hash':report['hash'],
             'inference_wall_seconds':receipt['wall_seconds'],
             'qualification':'independent CPU rescore on existing validation users; not new training or final test'}
    summary['hash']=digest(summary);write_json(output/'acceptance.json',summary)
    return summary
