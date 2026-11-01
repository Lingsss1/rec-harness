"""Eight fresh family-constrained comparisons; two serial four-GPU queues."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .bootstrap import bootstrap
bootstrap()
from .prepare import WORKSPACE, REPO, sha
from .core import read, save, schema
from .controller import FamilyController

ROOT = WORKSPACE/'artifacts/model-family-comparison-20261029'
FAMILIES = ['gru4rec', 'nextitnet', 'bert4rec', 'hstu']


def inventory():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):[int(v.strip()) for v in row.split(',')[1:]] for row in output.strip().splitlines()}


def idle(devices, gpus):
    return all(devices[g][0] < 512 and devices[g][1] < 5 for g in gpus)


def prepare():
    if (ROOT/'plan.json').exists():
        raise RuntimeError('Existing plan must not be overwritten')
    original = read(WORKSPACE/'artifacts/self-evorec-resilient-20261027/run-01/config.json')
    provenance = read(ROOT/'audit/baseline-provenance.json')
    queues, jobs = [], []
    for queue_id, gpus in [(0,[0,1,2,3]), (1,[4,5,6,7])]:
        ids = []
        for i, family in enumerate(FAMILIES):
            method = 'mlestar' if (i+queue_id)%2 else 'self_evolverec'
            job_id = family+'-'+method
            path = ROOT/'runs'/job_id
            config = deepcopy(original)
            config.update(root=str(path), gpus=gpus, cpu_cores=list(range(gpus[0]*16,(gpus[-1]+1)*16)),
                methods=[method], model_family=family, search_budget_gpu_seconds=43200,
                llm_model='gpt-6.1-sol', llm_effort='medium', single_model_search=True,
                llm_slot_directory=str(ROOT/'llm-slots'), global_llm_concurrency=4,
                resilient_transport=True, repair_diagnosis_contract=True, llm_timeout=1200,
                llm_attempts=3, retry_backoff_seconds=[5,15], transport_cooldown_seconds=60,
                baseline={'source':provenance[family]['corrected_source'], 'sha256':provenance[family]['corrected_sha256']},
                diagnosis_source=str(ROOT/'source/diagnosis_tools.py'))
            config['alignment'].update(cold_start=family+'_perdataset_corrected',
                search_scope='fixed '+family+' backbone family, corrected sequence contracts, one model per category',
                method_adaptations={method:{'source':'isolated amazon_family_compare adapter',
                    'family_constraint':'prompt plus loaded-module structural audit',
                    'single_model':'no independent score fusion; MLE-STAR ensemble phase disabled',
                    'transport':'same-request retries/cooldown; four shared LLM slots',
                    'diagnosis':'family-specific training sampler and real model scoring'}})
            config['replication']={'repetition':1,'fresh_baseline_and_llm_calls':True,
                'cross_run_response_or_memory_reuse':False,'search_gpu_seconds':43200}
            save(path/'config.json',config)
            save(path/'status.json',{'status':'queued','gpus':gpus,'model_family':family,'method':method})
            jobs.append({'id':job_id,'method':method,'family':family,'gpus':gpus,
                         'root':str(path),'config':str(path/'config.json')})
            ids.append(job_id)
        queues.append({'id':queue_id,'gpus':gpus,'jobs':ids})
    save(ROOT/'plan.json',{'jobs':jobs,'queues':queues,'budget_gpu_seconds_per_job':43200,
        'total_gpu_seconds':345600,'llm_model':'gpt-6.1-sol','llm_effort':'medium',
        'data':'four fixed Amazon Reviews categories; existing seed-0 prepared splits',
        'scope':'four model families times two search methods; one fresh run per cell',
        'gpu_policy':'wait for an idle four-GPU block; no sharing with unrelated tasks',
        'preflight':'short training/diagnosis/LLM checks outside formal search budgets'})
    print(json.dumps({'prepared_jobs':len(jobs),'root':str(ROOT)}),flush=True)


def manifest():
    paths = list((WORKSPACE/'amazon_family_compare').glob('*.py'))
    paths += list((ROOT/'baselines').glob('*/train.py'))
    paths += [ROOT/'source/diagnosis_tools.py',ROOT/'plan.json',ROOT/'audit/baseline-provenance.json']
    paths += list((ROOT/'runs').glob('*/config.json'))
    paths += list((REPO/'gagc/benchmarks/amazon_reviews').glob('*.py'))
    paths += [p for p in (WORKSPACE/'artifacts/Self-EvolveRec-code').rglob('*.py') if '.git' not in p.parts and '__pycache__' not in p.parts]
    paths += [WORKSPACE/'artifacts/Self-EvolveRec-code/configs/config_baseline.yaml',
              WORKSPACE/'staged-autoresearch/staged_research/codex.py']
    config = read(ROOT/'runs'/read(ROOT/'plan.json')['jobs'][0]['id']/'config.json')
    data=Path(config['data_dir'])
    paths += [data/'prepared.json',data/'trainval/dataset_stats.json']
    for name in config['alignment']['datasets']:
        paths += [data/'trainval'/(name+s) for s in ['_train.txt','_valid.txt','_text_name_dict.json.gz']]
        paths += [data/'test'/(name+'_test.txt')]
    return {str(p):sha(p) for p in sorted(set(paths))}


def preflight():
    out=ROOT/'preflight'
    out.mkdir(exist_ok=True)
    with (out/'cpu-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','amazon_family_compare/test_families.py',
            'amazon_family_compare/test_accounting.py','amazon_family_compare/test_protocol_fixes.py',
            'amazon_family_compare/test_resilient_self.py','-q',
            '--junitxml',str(out/'cpu-tests.xml')],cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not idle(inventory(),[4,5,6,7]):
        raise RuntimeError('Preflight waits for GPUs 4-7 to be idle; no training started')
    plan=read(ROOT/'plan.json')
    def probe(pair):
        family,gpu=pair
        job=next(j for j in plan['jobs'] if j['family']==family and j['gpus']==[4,5,6,7])
        config={**read(job['config']),'root':str(out/family)}
        c=FamilyController(config,'gpu')
        receipt=c.trial('baseline-'+family,c.baseline(),gpu,requested=900,
            files={'diagnosis_tools.py':Path(config['diagnosis_source']).read_text()},smoke=True)
        if receipt['status']!='success':
            raise RuntimeError('GPU/diagnosis preflight failed: '+json.dumps(receipt))
        trial=Path(receipt['output'])
        if not (trial/'diagnosis.json').exists():
            raise RuntimeError('Missing real model diagnosis')
        return {'family':family,'receipt':receipt}
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(probe,zip(FAMILIES,[4,5,6,7])))
    save(out/'gpu.json',{'passed':True,'models':receipts,
        'gpu_seconds_outside_search':sum(r['receipt']['allocated_gpu_seconds'] for r in receipts)})
    config={**read(plan['jobs'][0]['config']),'root':str(out/'llm-health'),'llm_timeout':180}
    controller=FamilyController(config,'self_evolverec')
    ready=controller.llm('Transport check. Return status=ready.',schema({'status':{'type':'string','enum':['ready']}}),key='health')
    if ready!={'status':'ready'}:
        raise RuntimeError('LLM health check failed')
    save(out/'acceptance.json',{'passed':True,'cpu_tests':True,'four_gpu_training_and_native_diagnosis':True,
        'llm_health':True,'files':manifest(),'time':time.time()})
    print('PREFLIGHT_PASSED',flush=True)


def worker(job):
    folder=Path(job['root'])
    lock=(folder/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if manifest()!=read(ROOT/'frozen.json')['files']:
        raise RuntimeError('Frozen comparison source/config/data changed')
    if (folder/job['method']/'events.jsonl').exists():
        raise RuntimeError('Already started; never reset or silently retrain a job')
    config=read(job['config'])
    os.sched_setaffinity(0,set(config['cpu_cores']))
    save(folder/'status.json',{'status':'running','pid':os.getpid(),'started_at':time.time(),
        'gpus':job['gpus'],'method':job['method'],'model_family':job['family'],'budget_gpu_seconds':43200})
    try:
        if job['method']=='mlestar':
            from . import mlestar
            mlestar.Controller=FamilyController
            result=mlestar.run(config)
        else:
            from . import self_evolverec
            self_evolverec.Controller=FamilyController
            result=self_evolverec.run(config)
        save(folder/'status.json',{'status':'complete','time':time.time(),
            'validation_hr10':result['selected']['score'],'test_hr10':result['test']['score'],
            'spent_gpu_seconds':result['budget']['spent_gpu_seconds']})
    except BaseException:
        error=traceback.format_exc()
        save(folder/'failure.json',{'error':error,'time':time.time()})
        save(folder/'status.json',{'status':'failed','time':time.time(),'error':error})
        raise


def queue(queue_id):
    plan=read(ROOT/'plan.json')
    q=next(q for q in plan['queues'] if q['id']==queue_id)
    folder=ROOT/'queues'/str(queue_id)
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=[]
    for job_id in q['jobs']:
        job=next(j for j in plan['jobs'] if j['id']==job_id)
        while not idle(inventory(),q['gpus']):
            state={'status':'waiting_for_idle_gpus','next_job':job_id,'gpus':q['gpus'],'time':time.time()}
            save(folder/'status.json',state)
            save(Path(job['root'])/'status.json',state)
            time.sleep(15)
        command=[sys.executable,'-u','-m','amazon_family_compare.launch','--worker',job_id]
        with (Path(job['root'])/'controller.log').open('a') as log:
            process=subprocess.Popen(command,cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        save(folder/'status.json',{'status':'running','job':job_id,'pid':process.pid,'gpus':q['gpus'],'completed':completed})
        code=process.wait()
        completed.append({'job':job_id,'exit_code':code})
    save(folder/'status.json',{'status':'complete','completed':completed,'time':time.time()})


def start():
    if (ROOT/'launcher.json').exists():
        raise RuntimeError('Comparison already launched; inspect status')
    preflight()
    files=manifest()
    if files!=read(ROOT/'preflight/acceptance.json')['files']:
        raise RuntimeError('Preflight source/config/data changed')
    save(ROOT/'frozen.json',{'files':files,'time':time.time()})
    processes=[]
    for q in read(ROOT/'plan.json')['queues']:
        folder=ROOT/'queues'/str(q['id']);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'supervisor.log').open('a') as log:
            command=[sys.executable,'-u','-m','amazon_family_compare.launch','--queue',str(q['id'])]
            process=subprocess.Popen(command,cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        processes.append({'queue':q['id'],'pid':process.pid,'gpus':q['gpus']})
    save(ROOT/'launcher.json',{'processes':processes,'started_at':time.time()})
    print(json.dumps({'launched':processes,'jobs':8,'root':str(ROOT)}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--start',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--queue',type=int)
    args=parser.parse_args()
    if sum([args.prepare,args.preflight,args.start,args.worker is not None,args.queue is not None])!=1:
        parser.error('Choose one action')
    if args.prepare:return prepare()
    if args.preflight:return preflight()
    if args.start:return start()
    if args.queue is not None:return queue(args.queue)
    return worker(next(j for j in read(ROOT/'plan.json')['jobs'] if j['id']==args.worker))


if __name__=='__main__':main()
