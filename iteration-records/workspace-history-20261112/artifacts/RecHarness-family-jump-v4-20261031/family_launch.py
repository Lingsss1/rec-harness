"""Four fresh, family-preserving RecHarness comparisons on fixed Amazon splits."""
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

from family_runner import FamilyJumpController, FORK, WORKSPACE, native, read, save
from amazon_family_compare.prepare import sha
from gagc import semantic_arms, experiment_memory

ROOT = WORKSPACE/'artifacts/recharness-family-comparison-20261031'
COMPARISON = WORKSPACE/'artifacts/model-family-comparison-20261029'
FAMILIES = ['gru4rec','bert4rec','nextitnet','hstu']
QUEUES = {0:['bert4rec','hstu'],1:['gru4rec','nextitnet']}


def environment():
    return {**os.environ,'PATH':str(WORKSPACE/'portable/bin')+os.pathsep+os.environ['PATH'],
            'PYTHONPATH':str(FORK)+os.pathsep+str(WORKSPACE),
            'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
            'GAGC_JUMP_ENABLED':'true','GAGC_ARM_DESIGN':'semantic_v4'}


def inventory():
    result = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):[int(x.strip()) for x in row.split(',')[1:]]
            for row in result.strip().splitlines()}


def idle(devices, gpus):
    return all(devices[g][0]<512 and devices[g][1]<5 for g in gpus)


def blockers(gpus):
    result=[]
    # Inventory can look idle during an LLM phase. Respect the existing
    # comparison's ownership of its entire four-card allocation across phases.
    for path in (COMPARISON/'runs').glob('*/status.json'):
        state=read(path)
        if state.get('status')=='running' and set(state.get('gpus',[]))&set(gpus):
            result.append(path.parent.name)
    return result


def verify(files):
    changed=[p for p,h in files.items() if not Path(p).is_file() or sha(p)!=h]
    if changed:raise RuntimeError('Frozen inputs changed: '+repr(changed))


def prepare():
    if (ROOT/'plan.json').exists():raise RuntimeError('Existing plan must not be overwritten')
    verify(read(COMPARISON/'frozen.json')['files'])
    provenance=read(COMPARISON/'audit/baseline-provenance.json')
    jobs=[]
    for family in FAMILIES:
        queue=next(i for i,names in QUEUES.items() if family in names)
        gpus=list(range(queue*4,queue*4+4))
        folder=ROOT/'runs'/family
        c=deepcopy(read(COMPARISON/'runs'/(family+'-mlestar')/'config.json'))
        source=Path(provenance[family]['corrected_source'])
        baseline=ROOT/'baselines'/family/'train.py'
        baseline.parent.mkdir(parents=True,exist_ok=True);baseline.write_bytes(source.read_bytes())
        if sha(baseline)!=provenance[family]['corrected_sha256']:raise RuntimeError('Baseline mismatch')
        info={'source':str(baseline),'sha256':sha(baseline),'original_source':str(source),
              'origin':'identical corrected cold start as same-family MLE-STAR and Self-EvolveRec',
              'weights':'fresh initialization; no checkpoint or proposal reuse'}
        save(baseline.parent/'provenance.json',info)
        c.update(root=str(folder),gpus=gpus,cpu_cores=list(range(gpus[0]*16,(gpus[-1]+1)*16)),
                 methods=['recharness'],base_model=info,baseline=info,
                 search_budget_gpu_seconds=43200,prior_protocol_charge={'gpu_seconds':0},
                 arm_policy=semantic_arms.policy(),
                 jump_fork={'source':str(FORK),'version':'jump-v4-six-arm-multicombination-memory-v2','enabled':True},
                 memory_policy={'schema_version':experiment_memory.SCHEMA_VERSION,
                    'max_context_characters':experiment_memory.CONTEXT_CHAR_LIMIT,'max_context_records':12,
                    'archive':'per-run immutable observations, source snapshots, outcomes and request contexts',
                    'duplicate_guard':'same-run successful AST-identical programs only','held_out_feedback':False},
                 replication={'repetition':1,'independent_search':True,'fresh_baseline_weights':True,
                    'cross_run_response_or_memory_reuse':False,'training_seed':42,'routing_seed':42},
                 llm_slot_directory=str(COMPARISON/'llm-slots'),global_llm_concurrency=4)
        c['alignment']['search_scope']='fixed '+family+' family; six arms, joint tuning and budgeted 2-4-direction combination jumps'
        c['alignment']['method_adaptations']={'recharness':{
            'implementation':str(FORK),'cold_start':info,
            'jump':'up to four unique combinations; best-only basin switch, four retune groups and safe rollback',
            'memory':'version 2 within-run concrete evidence only; no held-out feedback',
            'execution':'same pinned family worker, grader and single-model/family audit as baselines',
            'transport':'same gpt-6.1-sol/medium and four shared LLM slots; bundled CLI; retry/cooldown',
            'selection':'native incumbent after safe jump finalization; final held-out test once'}}
        save(folder/'config.json',c)
        save(folder/'status.json',{'status':'queued','gpus':gpus,'model_family':family,
                                  'method':'recharness','budget_gpu_seconds':43200})
        jobs.append({'id':family,'family':family,'gpus':gpus,'root':str(folder),
                     'config':str(folder/'config.json'),'queue':queue})
    save(ROOT/'arm-policy.json',semantic_arms.policy())
    save(ROOT/'plan.json',{'jobs':jobs,'queues':QUEUES,'created_at':time.time(),
        'gpu_seconds_per_job':43200,'total_gpu_seconds':172800,'llm_model':'gpt-6.1-sol','llm_effort':'medium',
        'scope':'four fresh independent RecHarness cells; corrected same-family cold starts',
        'resources':'four GPUs, 64 CPU cores and four threads/worker; wait for existing block owners',
        'preflight':'short real GPU/LLM checks separately reported, never reused in formal search'})
    print(json.dumps({'prepared':4,'root':str(ROOT)}),flush=True)


def manifest():
    paths=list(FORK.rglob('*.py'))+list((WORKSPACE/'amazon_family_compare').glob('*.py'))
    paths+=list((ROOT/'baselines').glob('*/*.py'))+list((ROOT/'baselines').glob('*/provenance.json'))
    paths+=list((ROOT/'runs').glob('*/config.json'))
    paths+=[ROOT/'plan.json',ROOT/'arm-policy.json',FORK/'FAMILY_PROVENANCE.json',
            FORK/'JUMP_POLICY.json',FORK/'MEMORY_DESIGN.json',FORK/'PROMPT_REVISION.json',FORK/'FAMILY_REVISION.json',
            WORKSPACE/'portable/bin/codex',WORKSPACE/'portable/runtime/node/bin/node',
            WORKSPACE/'portable/runtime/codex/bin/codex.js']
    paths += [p for p in (WORKSPACE/'portable/runtime/codex').rglob('codex') if p.is_file()]
    files={str(p):sha(p) for p in paths if '__pycache__' not in p.parts}
    # Keep the already registered data and evaluator identical across methods.
    files.update(read(COMPARISON/'frozen.json')['files'])
    return files


def preflight(family, gpu, integration=False):
    folder=ROOT/'preflight'/family
    if folder.exists():raise RuntimeError('Existing preflight must be reviewed before retry')
    folder.mkdir(parents=True)
    c=read(ROOT/'runs'/family/'config.json')
    c.update(root=str(folder),gpus=[4,5,6,7],cpu_cores=list(range(64,128)),llm_timeout=240)
    os.sched_setaffinity(0,set(c['cpu_cores']))
    os.environ.update(environment())
    controller=FamilyJumpController(c,smoke=True)
    controller.setup()
    # Single-slot native cold start; choose a distinct GPU for concurrent probes.
    original_gpus=list(controller.c['gpus'])
    controller.c['gpus']=[gpu]+[g for g in original_gpus if g!=gpu]
    specs=json.loads(native.propose_action_group(controller.store.state().to_json()))
    baseline=controller.group(specs)
    if len(baseline)!=1 or not native._is_valid_trial(baseline[0]):
        raise RuntimeError('Family baseline preflight failed: '+json.dumps(baseline))
    controller.c['gpus']=original_gpus
    receipts={'baseline':baseline,'integration':integration}
    if integration:
        specs=json.loads(native.propose_action_group(controller.store.state().to_json(),json.dumps(baseline[0])))
        results=controller.group(specs,baseline[0])
        if len(results)!=4 or not all(native._is_valid_trial(r) for r in results):
            raise RuntimeError('Regular native/LLM integration failed: '+json.dumps(results))
        state=controller.store.state()
        state.basin_score_history[state.current_basin_id]=[state.current_basin_score]*3
        controller.store.put(('gagc','thompson'),'/state.json',{'content':state.to_json()})
        jump_specs=json.loads(native.propose_action_group(state.to_json()))
        if not jump_specs or not all(s.get('is_jumping') for s in jump_specs):
            raise RuntimeError('Synthetic integration plateau did not open the jump gate')
        jumped=controller.group(jump_specs,max(results,key=lambda r:r['val_score']))
        if len(jumped)!=len(jump_specs) or not all(native._is_valid_trial(r) for r in jumped):
            raise RuntimeError('Combination jump integration failed: '+json.dumps(jumped))
        receipts.update(regular=results,jump=jumped,synthetic_plateau_for_integration_only=True)
    finalized=json.loads(native.finalize_jump_search('preflight_only'))
    if not finalized['ok']:raise RuntimeError('Jump finalization failed')
    controller.sync_memory()
    gpu_seconds=sum(read(p)['allocated_gpu_seconds'] for p in controller.root.glob('smoke/*/receipt.json'))
    save(folder/'acceptance.json',{'passed':True,'family':family,'time':time.time(),
         'checks':receipts,'jump_finalization':finalized,'gpu_seconds_outside_search':gpu_seconds,
         'memory_records':len(controller.memory_store.all_records()),
         'llm_model':c['llm_model'],'effort':c['llm_effort']})
    print('PREFLIGHT_PASSED '+family,flush=True)


def worker(family):
    verify(read(ROOT/'frozen.json')['files'])
    c=read(ROOT/'runs'/family/'config.json');folder=Path(c['root'])
    handle=(folder/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'recharness').exists():raise RuntimeError('Formal output already exists; never reset its budget')
    os.environ.update(environment());os.sched_setaffinity(0,set(c['cpu_cores']))
    save(folder/'status.json',{'status':'running','pid':os.getpid(),'started_at':time.time(),
         'gpus':c['gpus'],'model_family':family,'method':'recharness','budget_gpu_seconds':43200})
    try:
        result=FamilyJumpController(c).run()
    except BaseException:
        status={'status':'failed','time':time.time(),'error':traceback.format_exc()}
        save(folder/'failure.json',status);save(folder/'status.json',status);raise
    save(folder/'status.json',{'status':'complete','time':time.time(),
         'validation_hr10':result['selected']['score'],'test_hr10':result['test']['score'],
         'spent_gpu_seconds':result['budget']['spent_gpu_seconds']})


def queue(index):
    folder=ROOT/'queues'/str(index);folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/'queue.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=[]
    for family in QUEUES[index]:
        c=read(ROOT/'runs'/family/'config.json')
        while True:
            owners=blockers(c['gpus'])
            devices=inventory()
            if not owners and idle(devices,c['gpus']):break
            status={'status':'waiting_for_idle_gpus','next_job':family,'gpus':c['gpus'],
                    'existing_experiment_owners':owners,'time':time.time()}
            save(folder/'status.json',status);save(Path(c['root'])/'status.json',status);time.sleep(15)
        with (Path(c['root'])/'controller.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',family],
                cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=environment())
        save(folder/'status.json',{'status':'running','job':family,'pid':process.pid,
                                  'gpus':c['gpus'],'completed':completed})
        completed.append({'job':family,'exit_code':process.wait()})
    save(folder/'status.json',{'status':'complete','completed':completed,'time':time.time()})


def start():
    if (ROOT/'launcher.json').exists():raise RuntimeError('Already launched; inspect status')
    tests=read(ROOT/'preflight/cpu-acceptance.json')
    if not tests['passed']:raise RuntimeError('CPU checks missing')
    probes=[read(ROOT/'preflight'/f/'acceptance.json') for f in FAMILIES]
    if not all(p['passed'] for p in probes) or not probes[0]['checks']['integration']:
        raise RuntimeError('Four-family training and native integration must pass')
    save(ROOT/'preflight/acceptance.json',{'passed':True,'time':time.time(),
         'gpu_seconds_outside_search':sum(p['gpu_seconds_outside_search'] for p in probes),
         'checks':['CPU regressions','four-family real training, grader and structural audit',
                   'native four-GPU regular mutations and combination jump with real LLM',
                   'fresh corrected baselines, six arms, immutable evidence memory and accounting']})
    save(ROOT/'frozen.json',{'time':time.time(),'files':manifest()})
    launched=[]
    for index in QUEUES:
        folder=ROOT/'queues'/str(index);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'supervisor.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--queue',str(index)],
                cwd=WORKSPACE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=environment())
        launched.append({'queue':index,'pid':p.pid,'families':QUEUES[index]})
    save(ROOT/'launcher.json',{'time':time.time(),'processes':launched})
    print(json.dumps({'launched':launched,'root':str(ROOT)}),flush=True)


def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--start',action='store_true')
    group.add_argument('--preflight',choices=FAMILIES);group.add_argument('--worker',choices=FAMILIES)
    group.add_argument('--queue',type=int,choices=[0,1]);p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--integration',action='store_true');args=p.parse_args()
    if args.prepare:return prepare()
    if args.start:return start()
    if args.preflight:return preflight(args.preflight,args.gpu,args.integration)
    if args.queue is not None:return queue(args.queue)
    return worker(args.worker)


if __name__=='__main__':main()
