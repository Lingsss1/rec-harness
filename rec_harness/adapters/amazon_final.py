"""Isolated final checkpoint inference and host-only paired test reporting."""
import fcntl
from pathlib import Path

from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.cpu_job import run_cpu_job,recover_completed_cpu_job
from rec_harness.process_identity import assert_stopped
from .amazon_final_data import prepare_final_data
from .amazon_final_selection import load_selection
from .independent_scoring import score_response


def run_final(output,recipe,uid_registry):
    output=Path(output).resolve();plan=load_selection(output)
    # Freeze the inference configuration before opening test data.
    with (output/'final.lock').open('a') as final_lock:
        fcntl.flock(final_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execution={'selection_hash':plan['hash'],'recipe':recipe,'python':plan['python'],
                   'uid_registry':str(Path(uid_registry).resolve())}
        config_path=output/'inference-config.json'
        if config_path.exists():
            if read_json(config_path)!=execution:raise ContractError('final inference configuration changed')
        else:write_json(config_path,execution)
        data=prepare_final_data(output)
        study=Path(plan['request']['study'])
        with (study/'controller.lock').open('a') as study_lock:
            fcntl.flock(study_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            load_selection(output)
            artifacts=Artifacts(output/'artifacts/objects');records=[]
            for job in plan['jobs']:
                identity={key:job[key] for key in ['model_version','dataset','seed']}
                directory=output/'jobs'/digest(identity);directory.mkdir(parents=True,exist_ok=True)
                bundle=data['datasets'][job['dataset']];request=bundle['request'];key=bundle['private_key']
                request_path=directory/'request.json'
                if request_path.exists() and read_json(request_path)!=request:raise ContractError('final inference request changed')
                if not request_path.exists():write_json(request_path,request)
                worker=Path(__file__).with_name('score_worker.py')
                files={'model.py':job['source'],'checkpoint.pt':job['checkpoint'],
                    'request.json':{'path':str(request_path),'sha256':file_hash(request_path)},
                    'worker.py':{'path':str(worker),'sha256':file_hash(worker)}}
                arguments=['--source','/inputs/model.py','--source-hash',files['model.py']['sha256'],
                    '--checkpoint','/inputs/checkpoint.pt','--checkpoint-hash',files['checkpoint.pt']['sha256'],
                    '--request','/inputs/request.json','--request-hash',files['request.json']['sha256'],
                    '--output','/output/scores.json']
                inference=directory/'inference';receipt_path=inference/'receipt.json'
                if receipt_path.exists():
                    assert_stopped(inference)
                    receipt=read_json(receipt_path);submitted=read_json(inference/'request.json')
                    if submitted['inputs']!=files or submitted['arguments']!=arguments or submitted['recipe']!=recipe:
                        raise ContractError('cached final inference binding changed')
                else:
                    if inference.exists():
                        receipt=recover_completed_cpu_job(inference,recipe,files,'worker.py',arguments,artifacts=artifacts)
                    else:
                        receipt=run_cpu_job(inference,recipe,files,'worker.py',arguments,python=plan['python'],
                            uid_registry=uid_registry,artifacts=artifacts,timeout=180)
                if receipt['process']['status']!='success':raise ContractError('final checkpoint inference failed')
                response=read_json(artifacts.verify(receipt['collection']['artifacts']['scores.json']))
                report=score_response(request,key,response)
                report_path=directory/'host-ranking.json'
                if report_path.exists() and read_json(report_path)!=report:raise ContractError('final ranking report changed')
                if not report_path.exists():write_json(report_path,report)
                records.append({**identity,'users':bundle['users'],'test_population':bundle['test_population'],
                    'metrics':report['metrics'],'positive_tied_users':report['tied_user_count'],
                    'source_hash':job['source']['sha256'],'checkpoint_hash':job['checkpoint']['sha256'],
                    'request_hash':digest(request),'ranking_hash':report['hash'],'receipt_hash':digest(receipt),
                    'inference_wall_seconds':receipt['wall_seconds'],
                    'inference_process_wall_seconds':receipt['process'].get('wall_seconds'),
                    'inference_recovery':receipt.get('recovery')})
            load_selection(output)
            # Recheck test/history hashes before committing the final report.
            for ref in data['inputs']:
                if file_hash(ref['path'])!=ref['sha256']:raise ContractError('final data changed during inference')
            means={};effects={};models=plan['request']['models'];comparator=models[0]
            for seed in plan['seeds']:
                means[str(seed)]={}
                for model in models:
                    rows=[r for r in records if r['seed']==seed and r['model_version']==model]
                    if {r['dataset'] for r in rows}!=set(plan['datasets']) or len(rows)!=len(plan['datasets']):
                        raise ContractError('incomplete final comparison')
                    means[str(seed)][model]={metric:sum(r['metrics'][metric] for r in rows)/len(rows) for metric in rows[0]['metrics']}
                effects[str(seed)]={model:{metric:value-means[str(seed)][comparator][metric]
                    for metric,value in means[str(seed)][model].items()} for model in models if model!=comparator}
            summary={'schema':'amazon-final-report-v1','selection_hash':plan['hash'],'data_hash':data['hash'],
                'execution_hash':digest(execution),'records':records,'category_means_by_seed':means,
                'comparator':comparator,'paired_effects_by_seed':effects,
                'scope':'Final sampled-candidate test evaluation of previously selected checkpoints. All frozen initializations reported; no test-driven selection or return to optimization.',
                'tie_policy':'strictly greater negatives; optimistic ties counted in each record'}
            summary['hash']=digest(summary);target=output/'final-report.json'
            if target.exists() and read_json(target)!=summary:raise ContractError('committed final report changed')
            if not target.exists():write_json(target,summary)
            return summary
