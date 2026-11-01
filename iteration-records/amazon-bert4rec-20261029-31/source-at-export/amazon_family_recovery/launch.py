"""Restore four failed cells, preserving sources, checkpoints and GPU budgets."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from amazon_family_compare.bootstrap import bootstrap
bootstrap()
from amazon_family_compare.core import read, save
from amazon_family_compare.launch import ROOT, inventory, idle
from amazon_family_compare.prepare import WORKSPACE, sha

RECOVERY = ROOT/'recovery-20261030'
QUEUES = {
    0: ['bert4rec-self_evolverec', 'hstu-mlestar'],
    1: ['hstu-self_evolverec', 'bert4rec-mlestar'],
}


def verify_files(files):
    changed = [p for p, expected in files.items()
               if not Path(p).is_file() or sha(p) != expected]
    if changed:
        raise RuntimeError('Registered recovery inputs changed: ' + repr(changed))


def audited_budget(job):
    folder = Path(job['root'])/job['method']
    state = read(folder/'budget.json')
    if state['total_gpu_seconds'] != 43200 or state['reservations']:
        raise RuntimeError('Unexpected budget or unsettled GPU reservations')
    records = [read(p) for p in folder.glob('trials/*/receipt.json')]
    if abs(sum(p.get('allocated_gpu_seconds', 0) for p in records)
           - state['spent_gpu_seconds']) > .001:
        raise RuntimeError('Ledger and training receipts disagree')
    settled = state['settled']
    if abs(sum(p['allocated_gpu_seconds'] for p in settled.values())
           - state['spent_gpu_seconds']) > .001:
        raise RuntimeError('Ledger settled costs disagree')
    if any(not (folder/'trials'/key/'receipt.json').exists() for key in settled):
        raise RuntimeError('Settled trial lacks its receipt')
    if (folder/'summary.json').exists() or (folder/'final-test').exists():
        raise RuntimeError('Already finalized or incomplete final evaluation requires review')
    return state


def runtime_environment():
    return {**os.environ, 'PATH': str(WORKSPACE/'portable/bin')+os.pathsep+os.environ['PATH'],
            'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'}


def check_registration():
    reg = read(RECOVERY/'registration.json')
    verify_files(reg['files'])
    return reg


def register():
    if (RECOVERY/'registration.json').exists():
        raise RuntimeError('Already registered; inspect recovery status')
    verify_files(read(ROOT/'frozen.json')['files'])
    if not read(RECOVERY/'health/acceptance.json')['passed']:
        raise RuntimeError('LLM transport health check has not passed')
    plan = read(ROOT/'plan.json')
    jobs = {}
    for name in sum(QUEUES.values(), []):
        job = next(j for j in plan['jobs'] if j['id'] == name)
        status = read(Path(job['root'])/'status.json')
        if status['status'] != 'failed':
            raise RuntimeError('Only failed jobs may resume: ' + name)
        state = audited_budget(job)
        snapshot = RECOVERY/'jobs'/name/'before'
        snapshot.mkdir(parents=True, exist_ok=False)
        for source in [Path(job['config']), Path(job['root'])/'status.json',
                       Path(job['root'])/'failure.json',
                       Path(job['root'])/job['method']/'budget.json']:
            shutil.copy2(source, snapshot/source.name)
        native = Path(job['root'])/job['method']/'native'
        for latest in native.glob('*/amazon/durable/latest.json'):
            d = read(latest)
            verify_files(d['files'])
            t = latest.relative_to(native).parts[0]
            save(snapshot/(t+'.json'), d)
        jobs[name] = {**job, 'prior_spent_gpu_seconds':state['spent_gpu_seconds'],
                      'remaining_gpu_seconds':43200-state['spent_gpu_seconds'],
                      'budget_sha256':sha(Path(job['root'])/job['method']/'budget.json')}
    files = dict(read(ROOT/'frozen.json')['files'])
    for path in list((WORKSPACE/'amazon_family_recovery').glob('*.py')) + [
        ROOT/'frozen.json', RECOVERY/'health/acceptance.json', WORKSPACE/'portable/bin/codex',
        WORKSPACE/'portable/runtime/node/bin/node', WORKSPACE/'portable/runtime/codex/bin/codex.js']:
        files[str(path)] = sha(path)
    for path in (WORKSPACE/'portable/runtime/codex').rglob('codex'):
        if path.is_file(): files[str(path)] = sha(path)
    save(RECOVERY/'registration.json', {
        'time':time.time(), 'jobs':jobs, 'queues':QUEUES, 'files':files,
        'policy':'same-run continuation only; retain all spent costs, receipts, caches and native checkpoints',
        'repairs':['resolve bundled CLI in child PATH; retry process launch failures',
                   'replay completed MLE-STAR waves before available-budget checks',
                   'complete missing block-selection identities with constrained schemas'],
        'llm_model':'gpt-6.1-sol', 'llm_effort':'medium', 'budget_gpu_seconds_per_job':43200})


def worker(name):
    reg = check_registration()
    job = reg['jobs'][name]
    root = Path(job['root'])
    recovery = RECOVERY/'jobs'/name
    locks = []
    for path in [root/'worker.lock', root/job['method']/'controller.lock']:
        handle = path.open('a'); fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(handle)
    state = audited_budget(job)
    if sha(root/job['method']/'budget.json') != job['budget_sha256']:
        raise RuntimeError('Budget changed since registration; do not automatically retry')
    config = read(job['config'])
    os.sched_setaffinity(0, set(config['cpu_cores']))
    os.environ.update(runtime_environment())
    status = {'status':'running', 'pid':os.getpid(), 'started_at':time.time(),
              'gpus':job['gpus'], 'method':job['method'], 'model_family':job['family'],
              'budget_gpu_seconds':43200, 'recovery':True,
              'prior_spent_gpu_seconds':state['spent_gpu_seconds']}
    save(root/'status.json',status); save(recovery/'status.json',status)
    from .controller import RecoveryController
    try:
        if job['method'] == 'mlestar':
            from amazon_family_compare import mlestar as method
        else:
            from amazon_family_compare import self_evolverec as method
        method.Controller = RecoveryController
        result = method.run(config)
        status = {'status':'complete','time':time.time(),'recovery':True,
                  'validation_hr10':result['selected']['score'],'test_hr10':result['test']['score'],
                  'spent_gpu_seconds':result['budget']['spent_gpu_seconds']}
        save(root/'status.json',status);save(recovery/'status.json',status)
    except BaseException:
        error = {'status':'failed','error':traceback.format_exc(),'time':time.time(),'recovery':True}
        save(recovery/'failure.json',error);save(recovery/'status.json',error)
        save(root/'status.json',error)
        raise


def queue(index):
    reg = check_registration()
    directory = RECOVERY/'queues'/str(index)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory/'queue.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed = []
    for name in QUEUES[index]:
        job = reg['jobs'][name]
        while not idle(inventory(),job['gpus']):
            status = {'status':'waiting_for_idle_gpus','next_job':name,'gpus':job['gpus'],
                      'recovery':True,'time':time.time()}
            save(directory/'status.json',status)
            save(Path(job['root'])/'status.json',status)
            time.sleep(15)
        command = [sys.executable,'-u','-m','amazon_family_recovery.launch','--worker',name]
        with (RECOVERY/'jobs'/name/'controller.log').open('a') as log:
            process = subprocess.Popen(command,cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env=runtime_environment())
        save(directory/'status.json',{'status':'running','job':name,'pid':process.pid,
                                      'gpus':job['gpus'],'completed':completed})
        code = process.wait()
        completed.append({'job':name,'exit_code':code})
    save(directory/'status.json',{'status':'complete','completed':completed,'time':time.time()})


def start():
    if (RECOVERY/'launcher.json').exists():
        raise RuntimeError('Recovery already launched; inspect status')
    register()
    launched = []
    for index in QUEUES:
        directory = RECOVERY/'queues'/str(index)
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'supervisor.log').open('a') as log:
            p = subprocess.Popen([sys.executable,'-u','-m','amazon_family_recovery.launch','--queue',str(index)],
                cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=runtime_environment())
        launched.append({'queue':index,'pid':p.pid,'jobs':QUEUES[index]})
    save(RECOVERY/'launcher.json',{'time':time.time(),'processes':launched})
    print(json.dumps({'launched':launched,'remaining_gpu_seconds':{
        k:v['remaining_gpu_seconds'] for k,v in read(RECOVERY/'registration.json')['jobs'].items()}}),flush=True)


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--start',action='store_true')
    group.add_argument('--queue',type=int,choices=[0,1])
    group.add_argument('--worker')
    args = p.parse_args()
    if args.start: return start()
    if args.queue is not None: return queue(args.queue)
    return worker(args.worker)


if __name__ == '__main__': main()
