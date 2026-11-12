"""Three fresh searches per method; MLE-STAR waits for the current HSTU job."""
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

from amazon_family_compare.bootstrap import bootstrap
bootstrap()
from amazon_family_compare.core import read, save
from amazon_family_compare.prepare import WORKSPACE, sha

ROOT = WORKSPACE/'artifacts/nextitnet-repeats-20261031'
RECHARNESS = WORKSPACE/'artifacts/recharness-family-comparison-20261031'
COMPARISON = WORKSPACE/'artifacts/model-family-comparison-20261029'
RECOVERY = COMPARISON/'recovery-20261030'
FORK = WORKSPACE/'artifacts/RecHarness-family-jump-v4-20261031'
HSTU = RECHARNESS/'runs/hstu/status.json'
QUEUES = {0:('mlestar',[0,1,2,3]),1:('recharness',[4,5,6,7])}


def verify(files):
    changed=[p for p,h in files.items() if not Path(p).is_file() or sha(p)!=h]
    if changed:
        raise RuntimeError('Frozen source, configuration or data changed: '+repr(changed))


def inherited_files():
    files={}
    for path in [RECHARNESS/'frozen.json',RECOVERY/'registration.json']:
        for name,digest in read(path)['files'].items():
            if name in files and files[name]!=digest:
                raise RuntimeError('Historical source registrations disagree: '+name)
            files[name]=digest
        files[str(path)]=sha(path)
    return files


def prepare():
    if (ROOT/'plan.json').exists():
        raise RuntimeError('The existing repeat plan must not be overwritten')
    verify(inherited_files())
    baseline=ROOT/'baselines/nextitnet/train.py'
    original=COMPARISON/'baselines/nextitnet/train.py'
    baseline.parent.mkdir(parents=True,exist_ok=True)
    baseline.write_bytes(original.read_bytes())
    baseline_info={'source':str(baseline),'sha256':sha(baseline),'original_source':str(original),
                   'origin':'same corrected NextItNet cold start as the completed comparisons',
                   'weights':'new initialization in each repeat; no checkpoint reuse'}
    save(baseline.parent/'provenance.json',baseline_info)
    jobs=[];queues=[]
    for index,(method,gpus) in QUEUES.items():
        template=(RECHARNESS/'runs/nextitnet/config.json' if method=='recharness'
                  else COMPARISON/'runs/nextitnet-mlestar/config.json')
        ids=[]
        for number in range(1,4):
            name=f'{method}-repeat-{number:02d}'
            folder=ROOT/'runs'/name
            c=deepcopy(read(template))
            if c['baseline']['sha256']!=baseline_info['sha256']:
                raise RuntimeError('Methods have different cold-start sources')
            c.update(root=str(folder),methods=[method],gpus=gpus,
                cpu_cores=list(range(16*gpus[0],16*(gpus[-1]+1))),
                model_family='nextitnet',search_budget_gpu_seconds=43200,
                llm_model='gpt-6.1-sol',llm_effort='medium',single_model_search=True,
                baseline=baseline_info,llm_slot_directory=str(COMPARISON/'llm-slots'),
                global_llm_concurrency=4)
            if method=='recharness':
                c['base_model']=baseline_info
                c['prior_protocol_charge']={'gpu_seconds':0.,'reason':'fresh independent repeat'}
                c['alignment']['method_adaptations']['recharness']['cold_start']=baseline_info
            else:
                c['alignment']['method_adaptations']['mlestar']['transport_repair']={
                    'adapter':'amazon_family_recovery.controller.RecoveryController',
                    'selection':'retain valid trajectory identities; fill omitted identities with constrained schemas',
                    'search_algorithm':'same native retrieval, initial solutions, merging, ablation and refinement'}
            c['replication']={'additional_repetition':number,'overall_repetition':number+1,
                'fresh_baseline_weights':True,'independent_search':True,
                'fresh_baseline_and_llm_calls':True,'cross_run_response_or_memory_reuse':False,
                'training_seed':42,'routing_seed':42,'search_gpu_seconds':43200,
                'policy':'same pinned code, splits and baseline; new ledger, weights, requests and memory',
                'original_completed_run':str(template.parent)}
            save(folder/'config.json',c)
            save(folder/'status.json',{'status':'queued','method':method,'model_family':'nextitnet',
                'gpus':gpus,'budget_gpu_seconds':43200,'additional_repetition':number})
            ids.append(name)
            jobs.append({'id':name,'method':method,'family':'nextitnet','queue':index,
                'additional_repetition':number,'gpus':gpus,'root':str(folder),
                'config':str(folder/'config.json')})
        queues.append({'id':index,'method':method,'gpus':gpus,'jobs':ids,
                       'wait_for_hstu':method=='mlestar'})
    save(ROOT/'plan.json',{'created_at':time.time(),'jobs':jobs,'queues':queues,
        'new_runs':6,'new_runs_per_method':3,'budget_gpu_seconds_per_job':43200,
        'total_new_budget_gpu_seconds':259200,'llm_model':'gpt-6.1-sol','llm_effort':'medium',
        'data':'four existing Amazon categories, unchanged prepared splits and evaluation',
        'baseline':baseline_info,'hstu_prerequisite':str(HSTU),
        'resources':'four GPUs, 64 CPU cores, four threads per worker; no sharing across experiment owners',
        'authorization':'user confirmed six new independent runs, three per method',
        'preflight_policy':'reuse the unchanged frozen pipeline acceptance; test the new queue/configuration controls; no unnecessary extra training'})
    print(json.dumps({'prepared':6,'root':str(ROOT)}),flush=True)


def environment():
    return {**os.environ,'PATH':str(WORKSPACE/'portable/bin')+os.pathsep+os.environ['PATH'],
        'PYTHONPATH':str(WORKSPACE),'CUDA_VISIBLE_DEVICES':'',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'GAGC_ARM_DESIGN':'semantic_v4','GAGC_JUMP_ENABLED':'true'}


def inventory():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
    return {int(line.split(',')[0]):[int(v.strip()) for v in line.split(',')[1:]]
            for line in value.strip().splitlines()}


def owners(gpus):
    found=[]
    for base in [RECHARNESS,COMPARISON]:
        for path in (base/'runs').glob('*/status.json'):
            s=read(path)
            if s.get('status')=='running' and set(s.get('gpus',[]))&set(gpus):
                found.append(str(path.parent.relative_to(WORKSPACE)))
    return found


def wait_reason(method,gpus,devices,hstu_state,existing_owners):
    terminal={'complete','failed','stopped_by_user','stopped'}
    if method=='mlestar' and hstu_state.get('status') not in terminal:
        return 'waiting_for_hstu'
    if existing_owners:
        return 'waiting_for_existing_experiment'
    if not all(devices[g][0]<512 and devices[g][1]<5 for g in gpus):
        return 'waiting_for_idle_gpus'
    return None


def registered_files():
    files=inherited_files()
    paths=list((WORKSPACE/'nextitnet_repeats').glob('*.py'))
    paths+=list((ROOT/'runs').glob('*/config.json'))
    paths+=[ROOT/'plan.json',ROOT/'baselines/nextitnet/train.py',
            ROOT/'baselines/nextitnet/provenance.json',ROOT/'preflight/acceptance.json']
    for p in paths:files[str(p)]=sha(p)
    return files


def worker(name):
    job=next(j for j in read(ROOT/'plan.json')['jobs'] if j['id']==name)
    folder=Path(job['root']);method=job['method'];c=read(job['config'])
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        verify(read(ROOT/'frozen.json')['files'])
        if (folder/method).exists():
            raise RuntimeError('The method output already exists; never reset or silently restart a repeat')
        os.environ.update(environment());os.sched_setaffinity(0,set(c['cpu_cores']))
        save(folder/'status.json',{'status':'running','pid':os.getpid(),'started_at':time.time(),
            'method':method,'model_family':'nextitnet','gpus':c['gpus'],
            'budget_gpu_seconds':43200,'additional_repetition':job['additional_repetition']})
        if method=='recharness':
            sys.path.insert(0,str(FORK))
            from family_runner import FamilyJumpController
            result=FamilyJumpController(c).run()
        else:
            from amazon_family_compare import mlestar
            from amazon_family_recovery.controller import RecoveryController
            mlestar.Controller=RecoveryController
            result=mlestar.run(c)
    except BaseException:
        status={'status':'failed','time':time.time(),'method':method,
                'error':traceback.format_exc(),'additional_repetition':job['additional_repetition']}
        save(folder/'failure.json',status);save(folder/'status.json',status)
        raise
    save(folder/'status.json',{'status':'complete','time':time.time(),'method':method,
        'additional_repetition':job['additional_repetition'],
        'validation_hr10':result['selected']['score'],'test_hr10':result['test']['score'],
        'test_ndcg10':result['test']['aggregate']['NDCG@10'],
        'spent_gpu_seconds':result['budget']['spent_gpu_seconds']})


def queue(index):
    verify(read(ROOT/'frozen.json')['files'])
    plan=read(ROOT/'plan.json');q=next(x for x in plan['queues'] if x['id']==index)
    folder=ROOT/'queues'/str(index);folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=[]
    for name in q['jobs']:
        job=next(x for x in plan['jobs'] if x['id']==name)
        while True:
            existing=owners(q['gpus']);hstu=read(HSTU);devices=inventory()
            reason=wait_reason(q['method'],q['gpus'],devices,hstu,existing)
            if reason is None:break
            status={'status':reason,'next_job':name,'method':q['method'],'gpus':q['gpus'],
                    'hstu_status':hstu.get('status'),'existing_owners':existing,'time':time.time()}
            save(folder/'status.json',status);save(Path(job['root'])/'status.json',status)
            time.sleep(15)
        verify(read(ROOT/'frozen.json')['files'])
        with (Path(job['root'])/'controller.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u','-m','nextitnet_repeats.launch','--worker',name],
                cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=environment())
        save(folder/'status.json',{'status':'running','job':name,'pid':p.pid,'gpus':q['gpus'],
                                  'completed':completed})
        code=p.wait();completed.append({'job':name,'exit_code':code})
        if code and read(Path(job['root'])/'status.json').get('status')!='failed':
            save(Path(job['root'])/'status.json',{'status':'failed','time':time.time(),
                'error':'worker exited unexpectedly; inspect controller.log','exit_code':code})
    save(folder/'status.json',{'status':'complete','completed':completed,'time':time.time()})


def start():
    if (ROOT/'launcher.json').exists():raise RuntimeError('Already launched; inspect the existing queues')
    if not read(ROOT/'preflight/acceptance.json').get('passed'):
        raise RuntimeError('Queue/configuration tests have not passed')
    verify(inherited_files())
    plan=read(ROOT/'plan.json')
    for job in plan['jobs']:
        if (Path(job['root'])/job['method']).exists():
            raise RuntimeError('A repeat already has output; registration must not be reset')
    save(ROOT/'frozen.json',{'time':time.time(),'files':registered_files()})
    launched=[]
    for q in plan['queues']:
        folder=ROOT/'queues'/str(q['id']);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'supervisor.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u','-m','nextitnet_repeats.launch','--queue',str(q['id'])],
                cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=environment())
        launched.append({'queue':q['id'],'method':q['method'],'pid':p.pid,
                         'gpus':q['gpus'],'jobs':q['jobs'],'wait_for_hstu':q['wait_for_hstu']})
    save(ROOT/'launcher.json',{'time':time.time(),'processes':launched})
    print(json.dumps({'launched':launched,'new_runs':6,'new_budget_gpu_seconds':259200}),flush=True)


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--start',action='store_true')
    g.add_argument('--queue',type=int,choices=[0,1]);g.add_argument('--worker')
    args=p.parse_args()
    if args.prepare:return prepare()
    if args.start:return start()
    if args.queue is not None:return queue(args.queue)
    return worker(args.worker)


if __name__=='__main__':main()
