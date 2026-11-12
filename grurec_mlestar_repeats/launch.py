"""Three independent MLE-STAR searches on the corrected GRU4Rec cold start."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from nextitnet_repeats.launch import (
    WORKSPACE,COMPARISON,verify,inherited_files,environment,inventory,owners,
)
from amazon_family_compare.core import read,save
from amazon_family_compare.prepare import sha

ROOT=WORKSPACE/'artifacts/mlestar-gru4rec-repeats-20261031'
PREVIOUS=WORKSPACE/'artifacts/nextitnet-repeats-20261031'
GPUS=[4,5,6,7]


def prepare():
    if (ROOT/'plan.json').exists():raise RuntimeError('Existing plan must not be overwritten')
    verify(inherited_files())
    stopped=read(PREVIOUS/'user-stop-recharness/complete.json')
    if len(stopped['stopped'])!=3:raise RuntimeError('The replaced RecHarness queue is not stopped')
    template=COMPARISON/'runs/gru4rec-mlestar/config.json'
    original=read(template)
    source=Path(original['baseline']['source'])
    baseline=ROOT/'baselines/gru4rec/train.py'
    baseline.parent.mkdir(parents=True,exist_ok=True);baseline.write_bytes(source.read_bytes())
    if sha(baseline)!=original['baseline']['sha256']:raise RuntimeError('Corrected baseline changed')
    info={'source':str(baseline),'sha256':sha(baseline),'original_source':str(source),
          'origin':'identical corrected GRU4Rec cold start as completed MLE-STAR/Self-EvolveRec/RecHarness comparisons',
          'weights':'fresh random initialization; no earlier checkpoints or candidates'}
    save(baseline.parent/'provenance.json',info)
    jobs=[]
    for number in range(1,4):
        name=f'repeat-{number:02d}';folder=ROOT/'runs'/name;c=deepcopy(original)
        c.update(root=str(folder),methods=['mlestar'],model_family='gru4rec',gpus=GPUS,
            cpu_cores=list(range(64,128)),search_budget_gpu_seconds=43200,
            llm_model='gpt-6.1-sol',llm_effort='medium',single_model_search=True,
            baseline=info,llm_slot_directory=str(COMPARISON/'llm-slots'),global_llm_concurrency=4)
        c['replication']={'additional_repetition':number,'overall_repetition':number+1,
            'fresh_baseline_weights':True,'independent_search':True,
            'fresh_baseline_and_llm_calls':True,'cross_run_response_or_memory_reuse':False,
            'training_seed':42,'search_gpu_seconds':43200,
            'policy':'same code/splits/baseline; new weights, ledger, proposals and request cache'}
        c['alignment']['method_adaptations']['mlestar']['transport_repair']={
            'adapter':'amazon_family_recovery.controller.RecoveryController',
            'policy':'same exact-request retry and constrained missing block-selection identities',
            'search':'unchanged retrieval, initial solutions, merging, ablation/refinement; ensemble disabled'}
        save(folder/'config.json',c)
        save(folder/'status.json',{'status':'queued','method':'mlestar','model_family':'gru4rec',
             'gpus':GPUS,'budget_gpu_seconds':43200,'additional_repetition':number})
        jobs.append({'id':name,'root':str(folder),'config':str(folder/'config.json'),
                     'additional_repetition':number,'gpus':GPUS})
    save(ROOT/'plan.json',{'created_at':time.time(),'jobs':jobs,'new_runs':3,
        'method':'mlestar','model_family':'gru4rec','gpus':GPUS,
        'budget_gpu_seconds_per_job':43200,'total_budget_gpu_seconds':129600,
        'llm_model':'gpt-6.1-sol','llm_effort':'medium','baseline':info,
        'replaces':'three additional RecHarness/NextItNet repeats',
        'cancelled_search_gpu_seconds_retained_separately':stopped['spent_gpu_seconds_retained'],
        'preserved':'existing HSTU run and its successor MLE-STAR/NextItNet queue',
        'resources':'four GPUs, 64 CPU cores, four threads per worker; no GPU sharing',
        'checks':'unchanged frozen training/LLM pipeline; only queue/configuration controls are new'})
    print(json.dumps({'prepared':3,'root':str(ROOT)}),flush=True)


def files():
    result=inherited_files()
    paths=list((WORKSPACE/'grurec_mlestar_repeats').glob('*.py'))
    paths+=list((WORKSPACE/'nextitnet_repeats').glob('*.py'))
    paths+=list((ROOT/'runs').glob('*/config.json'))
    paths+=[ROOT/'plan.json',ROOT/'baselines/gru4rec/train.py',ROOT/'baselines/gru4rec/provenance.json',
            ROOT/'preflight/acceptance.json',PREVIOUS/'user-stop-recharness/complete.json']
    for path in paths:result[str(path)]=sha(path)
    return result


def worker(name):
    job=next(j for j in read(ROOT/'plan.json')['jobs'] if j['id']==name)
    folder=Path(job['root']);c=read(job['config'])
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        verify(read(ROOT/'frozen.json')['files'])
        if (folder/'mlestar').exists():raise RuntimeError('Existing run output must never be reset')
        os.environ.update(environment());os.sched_setaffinity(0,set(c['cpu_cores']))
        save(folder/'status.json',{'status':'running','pid':os.getpid(),'started_at':time.time(),
            'method':'mlestar','model_family':'gru4rec','gpus':GPUS,'budget_gpu_seconds':43200,
            'additional_repetition':job['additional_repetition']})
        from amazon_family_compare import mlestar
        from amazon_family_recovery.controller import RecoveryController
        mlestar.Controller=RecoveryController
        result=mlestar.run(c)
    except BaseException:
        error={'status':'failed','time':time.time(),'error':traceback.format_exc()}
        save(folder/'failure.json',error);save(folder/'status.json',error);raise
    save(folder/'status.json',{'status':'complete','time':time.time(),'method':'mlestar',
        'model_family':'gru4rec','additional_repetition':job['additional_repetition'],
        'validation_hr10':result['selected']['score'],'test_hr10':result['test']['score'],
        'test_ndcg10':result['test']['aggregate']['NDCG@10'],
        'spent_gpu_seconds':result['budget']['spent_gpu_seconds']})


def queue():
    verify(read(ROOT/'frozen.json')['files'])
    directory=ROOT/'queue';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=[]
    for job in read(ROOT/'plan.json')['jobs']:
        folder=Path(job['root'])
        while True:
            existing=owners(GPUS);devices=inventory()
            if not existing and all(devices[g][0]<512 and devices[g][1]<5 for g in GPUS):break
            status={'status':'waiting_for_idle_gpus','next_job':job['id'],'gpus':GPUS,
                    'existing_owners':existing,'time':time.time()}
            save(directory/'status.json',status);save(folder/'status.json',status);time.sleep(15)
        verify(read(ROOT/'frozen.json')['files'])
        with (folder/'controller.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u','-m','grurec_mlestar_repeats.launch','--worker',job['id']],
                cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=environment())
        save(directory/'status.json',{'status':'running','job':job['id'],'pid':p.pid,
                                     'gpus':GPUS,'completed':completed})
        code=p.wait();completed.append({'job':job['id'],'exit_code':code})
        if code and read(folder/'status.json').get('status')!='failed':
            save(folder/'status.json',{'status':'failed','time':time.time(),'exit_code':code})
    save(directory/'status.json',{'status':'complete','completed':completed,'time':time.time()})


def start():
    if (ROOT/'launcher.json').exists():raise RuntimeError('Already launched; do not restart the queue')
    if not read(ROOT/'preflight/acceptance.json').get('passed'):raise RuntimeError('Checks have not passed')
    verify(inherited_files())
    stopped=read(PREVIOUS/'queues/1/status.json')
    if stopped.get('status')!='stopped_by_user':raise RuntimeError('The replaced queue is not stopped')
    for job in read(ROOT/'plan.json')['jobs']:
        if (Path(job['root'])/'mlestar').exists():raise RuntimeError('A repetition already has output')
    save(ROOT/'frozen.json',{'time':time.time(),'files':files()})
    directory=ROOT/'queue';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'supervisor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','grurec_mlestar_repeats.launch','--queue'],
            cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,env=environment())
    save(ROOT/'launcher.json',{'time':time.time(),'pid':p.pid,'gpus':GPUS})
    print(json.dumps({'launched':True,'pid':p.pid,'new_runs':3,'gpus':GPUS,
                      'budget_gpu_seconds_per_run':43200,'root':str(ROOT)}),flush=True)


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--start',action='store_true')
    g.add_argument('--queue',action='store_true');g.add_argument('--worker');args=p.parse_args()
    if args.prepare:return prepare()
    if args.start:return start()
    if args.queue:return queue()
    return worker(args.worker)


if __name__=='__main__':main()
