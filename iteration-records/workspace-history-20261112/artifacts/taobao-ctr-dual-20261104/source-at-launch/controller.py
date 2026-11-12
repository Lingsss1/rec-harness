"""Persistent matched auto/enhanced campaign; frozen temporal CTR evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import shlex
import statistics

from jsonschema import Draft202012Validator

from common import read,write,sha,identity
from agent import run as agent_run,prepare_template

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('REC_CTR_PROJECT',HERE.parents[1]))
DESIGN=HERE/'task-design'
if not DESIGN.exists():DESIGN=PROJECT/'task_designs/autonomous_research_v1'
BUDGET=86400
SEEDS=[42,43,44]


def status(root,phase,**values):
    write(root/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**values})


def bounded(command,directory,timeout,env=None):
    directory.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    write(directory/'process.json',{'argv':command,'started_unix':time.time(),'timeout_seconds':timeout})
    with (directory/'process.log').open('a') as log:
        process=subprocess.Popen(command,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True)
        write(directory/'process.json',{'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],'argv':command,'started_unix':time.time(),'timeout_seconds':timeout})
        try:code=process.wait(timeout=max(1,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();code=-signal.SIGKILL
    receipt={'returncode':code,'status':'success' if code==0 else 'failed','wall_seconds':time.monotonic()-start}
    write(directory/'process-receipt.json',receipt)
    return receipt


def train_job(data,source,config,out,seed,gpu,protocol,deadline,calibration=0):
    out.mkdir(parents=True,exist_ok=True)
    frozen=out/'model.py'
    if not frozen.exists():shutil.copyfile(source,frozen)
    write(out/'model-config.json',config)
    command=[sys.executable,'-u',str(HERE/'train.py'),'--data',str(data),'--model',str(frozen),
             '--config',str(out/'model-config.json'),'--output',str(out),'--seed',str(seed),
             '--epochs',str(protocol['epochs']),'--batch-size',str(protocol['batch_size']),
             '--history-limit',str(protocol['history_limit'])]
    if calibration:command+=['--calibration-steps',str(calibration)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    result=bounded(command,out,min(protocol.get('trial_cap_seconds',21600),max(1,deadline-time.time())),env)
    receipt={'seed':seed,'gpu':gpu,'source_sha256':sha(frozen),'config':config,'output':str(out),**result}
    if result['status']=='success' and (out/'result.json').exists():receipt['metrics']=read(out/'result.json')
    else:receipt.update(status='failed',error=read(out/'failure.json') if (out/'failure.json').exists() else 'Inspect process.log')
    write(out/'receipt.json',receipt)
    return receipt


def run_jobs(jobs,gpus):
    # Each lane owns one GPU for the whole job; no accidental variant contention.
    queues=[[] for _ in gpus]
    for index,job in enumerate(jobs):queues[index%len(gpus)].append(job)
    def lane(gpu,queue):return [train_job(gpu=gpu,**job) for job in queue]
    results=[]
    with ThreadPoolExecutor(len(gpus)) as pool:
        for future in as_completed([pool.submit(lane,gpu,queue) for gpu,queue in zip(gpus,queues)]):results.extend(future.result())
    return results


def public_trial(row):
    metrics=row.get('metrics',{})
    return {key:row[key] for key in ['id','parent','role','source_sha256','config','status','wall_seconds'] if key in row}|{
        'validation':metrics.get('validation'),'parameters':metrics.get('parameters'),
        'embedding_parameters':metrics.get('embedding_parameters'),'dense_parameters':metrics.get('dense_parameters'),
        'train_seconds':metrics.get('train_seconds'),'peak_gpu_bytes':metrics.get('peak_gpu_bytes'),
        'candidate_diagnostics':metrics.get('candidate_diagnostics'),
        'error':row.get('error'),'note':row.get('note')}


def phase_contract(phase,mode,state,protocol,remaining):
    selected_ids={t['id'] for t in state['trials'][-12:]}|{t['id'] for t in state['trials'] if t.get('baseline_kind')}|{state.get('incumbent')}
    context={'history':[public_trial(t) for t in state['trials'] if t['id'] in selected_ids],
             'trajectory_index':[{'id':t['id'],'parent':t.get('parent'),'role':t.get('role'),'round':t.get('round'),
                                  'status':t['status'],'auc':t.get('metrics',{}).get('validation',{}).get('auc'),
                                  'wall_seconds':t['wall_seconds']} for t in state['trials']],
             'incumbent':state.get('incumbent'),'remaining_seconds':remaining,'protocol':protocol,
             'method':mode,'budget_scope':'all preparation, calibration, research calls, failures, controls and final assessment count',
             'no_test_feedback':True}
    if phase=='query':
        return {'phase':phase,'expected_files':['retrieval-plan.json'],'context':context,
                'action':'Diagnose a concrete limitation from actual validation evidence. Search evidence with tools.py as useful. Write retrieval-plan.json with query (string), question (string), evidence_refs (real trial IDs), requested_card_ids (array, may be empty), and missing_information (array). Do not edit model code or train.'}
    if phase=='analyze':
        return {'phase':phase,'expected_files':['analysis.json'],'context':context,
                'action':'Review multiple relevant trajectories, latest variants and failures. Write analysis.json with observations, interpretations, alternatives, falsifiers, evidence_refs (existing trial IDs including the latest round), next_action, and limitations. Keep mechanism interpretations provisional. Do not edit model code or train.'}
    schema=read(DESIGN/'agent/study.schema.json')
    return {'phase':'propose','expected_files':['proposal.json'],'context':context,
            'study_schema':schema,'required_study_protocol_ref':protocol['identity'],
            'action':'Read candidate.py and baseline_reference.py, diagnose an observed limitation, check primary nearest work, and implement a feasible candidate. Write proposal.json containing study (matching study_schema), parent_trial (existing trial ID), and variants (1-4 objects with distinct name, file basename ending .py, role candidate/control, config object, and note). At least one candidate is required. Add key ablations or matched controls for the claimed mechanism when feasible; reference existing matched trials as controls where justified. Each variant is a standalone module. Do not run training. New architecture choices come from evidence; industry cards do not mandate them.',
            'candidate_api':{'Model':'Model(meta, config) -> torch.nn.Module; forward(batch) -> finite [B] logits',
                'meta':['fields','buckets','category_field_index','brand_field_index','history_columns'],
                'batch':['static: int64 [B,F]','history_category: int64 [B,L]','history_brand: int64 [B,L]',
                         'history_type: int64 [B,L], 1=pv,2=cart,3=fav,4=buy','history_age_seconds: float32 [B,L]','history_mask: bool [B,L]'],
                'optional':['literal CONFIG dictionary for model dimensions/lr','make_optimizer(model)','Model.training_loss(batch, logits, train_labels) -> differentiable scalar',
                            'Model.diagnostics(batch) -> at most 64 named finite scalars; host calls this on the first 512 validation impressions at the frozen best checkpoint, under no_grad. These are candidate-supplied measurements, requiring source-level scrutiny and controls.'],
                'ownership':'Edit or create only candidate model modules and proposal.json. No filesystem/network access from model code. Numeric imports only: torch,numpy,math,typing,copy,functools,collections,itertools,dataclasses,contextlib.',
                'local_checks':'Torch is not installed in the coding jail. Perform syntax/source checks there; host training and scoring follow automatically. Do not install packages or wait for human review.',
                'training':'from scratch; fixed host epochs, data, sample coverage, supplied history, scorer and seeds; optional optimizer/loss changes require explicit attribution controls'}}


def make_task(protocol):
    task=read(DESIGN/'tasks/taobao_display_ctr/task.json')
    task['status']='active_external_task_contract'
    task['protocol_ref']='runtime contract: protocol'
    task['protocol_hash']=protocol['identity']
    task['knowledge']['industry']['delivery']='Five shared source cards available on demand through tools.py search; evidence is external only.'
    task['knowledge']['agent']['delivery']='Current study receipts and source-linked provisional interpretations, when supplied. Other arms are inaccessible.'
    # Resolved references in this export are descriptions, never host filesystem capabilities.
    task['dataset'].pop('local_manifest',None)
    task['dataset']['acquisition_status']='four raw tables verified by host; private physical paths not exported'
    return task


def researcher(directory,phase,mode,state,protocol,deadline,evidence,retrieval=None):
    runtime=phase_contract(phase,mode,state,protocol,deadline-time.time())
    if retrieval:runtime['retrieved_evidence']=retrieval
    chosen=next((t for t in state['trials'] if t['id']==state.get('incumbent')),None)
    if phase=='propose':
        shutil.copyfile(chosen['model_path'],directory/'candidate.py')
        shutil.copyfile(HERE/'baseline.py',directory/'baseline_reference.py')
    ancestors=[]
    if phase=='propose':
        ids={t['id'] for t in state['trials'][-4:]}|{t['id'] for t in state['trials'] if t.get('baseline_kind')}
        for row in state['trials']:
            if row['id'] in ids and Path(row['model_path']).is_file():
                name='history_'+row['id'].replace('-','_')+'.py'
                shutil.copyfile(row['model_path'],directory/name)
                ancestors.append({'trial_id':row['id'],'file':name,'config':row['config'],'source_sha256':row['source_sha256']})
    workspace={'editable_outputs':runtime['expected_files'],'model_files':['candidate.py','baseline_reference.py'] if phase=='propose' else [],
               'available_evidence':'tools.py search TERMS; source cards and scoped memory','protocol_hash':protocol['identity'],
               'initial_model_config':chosen['config'] if chosen else None,'available_ancestor_sources':ancestors}
    return agent_run(directory,make_task(protocol),runtime,workspace,evidence,min(900,max(1,deadline-time.time())))


def best_trial(state):
    successful=[t for t in state['trials'] if t['status']=='success' and t.get('metrics',{}).get('validation')]
    return max(successful,key=lambda t:t['metrics']['validation']['auc']) if successful else None


def final_assessment(root,arm,state,protocol,data,gpus,deadline):
    selected=best_trial(state)
    if selected is None:raise ValueError('No valid reference; cannot conduct final assessment')
    # Freeze final source identities using validation only, before paired repetitions/test.
    picked=[selected]
    din=next((t for t in state['trials'] if t.get('baseline_kind')=='din'),None)
    if din and din['id']!=selected['id']:picked.append(din)
    for row in state['trials']:
        if row.get('round')==selected.get('round') and row.get('role')=='control' and row['status']=='success' and row['id'] not in {t['id'] for t in picked}:
            picked.append(row)
    picked=picked[:4]
    freeze={'closed_unix':time.time(),'selected_trial':selected['id'],'suite':[{'id':t['id'],'source_sha256':t['source_sha256'],'config':t['config']} for t in picked],
            'paired_seeds':SEEDS,'protocol_hash':protocol['identity'],'reason':'validation selection; final suite cannot be changed from test feedback'}
    write(arm/'selection-freeze.json',freeze)
    state['research_closed']=True;state['selected']=selected['id'];write(arm/'state.json',state)
    status(arm,'paired_repetitions',selected=selected['id'])
    jobs=[];models=[]
    for row in picked:
        models.append((row['id'],42,Path(row['output'])))
        for seed in SEEDS[1:]:
            jobs.append({'data':data,'source':Path(row['model_path']),'config':row['config'],
                         'out':arm/'repeats'/row['id']/str(seed),'seed':seed,'protocol':protocol,'deadline':deadline})
    estimated=math.ceil(len(jobs)/len(gpus))*protocol['estimated_unit_seconds']*1.5+max(300,len(models)*protocol['estimated_validation_seconds']*2)
    if jobs and deadline-time.time()>estimated:
        repeats=run_jobs(jobs,gpus)
        for row in picked:
            for seed in SEEDS[1:]:
                directory=arm/'repeats'/row['id']/str(seed)
                if (directory/'result.json').exists():models.append((row['id'],seed,directory))
        write(arm/'repeat-receipts.json',repeats)
    else:
        write(arm/'repeat-limitations.json',{'reason':'Insufficient remaining budget for all paired repetitions plus assessment','remaining_seconds':deadline-time.time(),'estimated_seconds':estimated})
    assessments=arm/'assessment';assessments.mkdir(exist_ok=True)
    checkpoints=[{'trial':trial,'seed':seed,'checkpoint_sha256':sha(directory/'checkpoint.pt'),'source_sha256':sha(directory/'model.py'),'directory':str(directory)} for trial,seed,directory in models]
    write(assessments/'assessment-freeze.json',{**freeze,'checkpoints':checkpoints})
    status(arm,'closed_test_assessment',selected=selected['id'])
    def assess(item,gpu):
        directory=Path(item['directory']);out=assessments/(item['trial']+'-seed-'+str(item['seed']))
        command=[sys.executable,'-u',str(HERE/'train.py'),'--data',str(data),'--model',str(directory/'model.py'),
                 '--config',str(directory/'model-config.json'),'--output',str(out),'--seed',str(item['seed']),
                 '--history-limit',str(protocol['history_limit']),'--batch-size',str(protocol['batch_size']),
                 '--evaluate','--split','test','--checkpoint',str(directory/'checkpoint.pt')]
        receipt=bounded(command,out,max(1,deadline-time.time()),{**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        return {**item,**receipt,'test':read(out/'result.json') if (out/'result.json').exists() else None}
    queues=[[] for _ in gpus]
    for i,item in enumerate(checkpoints):queues[i%len(gpus)].append(item)
    def lane(gpu,queue):return [assess(item,gpu) for item in queue]
    results=[]
    with ThreadPoolExecutor(len(gpus)) as pool:
        for future in as_completed([pool.submit(lane,gpu,q) for gpu,q in zip(gpus,queues)]):results.extend(future.result())
    write(arm/'final.json',{'selected':selected['id'],'results':results,'limitations':['One research trajectory per method; repeated model seeds are not repeated independent agent trajectories.','Reference papers and winner status do not establish transfer or novelty.','Small-mode screening/calibration scores never compete with full-protocol models.']})
    status(arm,'completed',selected=selected['id'],closed_assessments=len(results))


def worker(root,mode,gpus):
    arm=root/mode;arm.mkdir(exist_ok=True)
    with (arm/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (arm/'state.json').exists():raise ValueError('Existing study state cannot be overwritten; recover from receipts explicitly')
        budget=read(root/'budget.json');deadline=budget['deadline_unix'];protocol=read(root/'protocol.json')
        data=root/'prepared'
        state={'mode':mode,'trials':[],'incumbent':None,'research_closed':False,'agent_calls':[],'memory':[]}
        write(arm/'state.json',state)
        write(arm/'method.json',{'mode':mode,'gpus':gpus,'model':'gpt-6.1-sol','effort':'high','personal':False,
              'same_initial_industry_and_paper_sources':True,
              'difference':'automatic evidence query, conditional retrieval, post-experiment reflection and retrieval of scoped interpretations' if mode=='enhanced' else 'single proposal/edit loop, full trial receipts and optional direct source search; no separate reflection/retrieval phase',
              'attribution':'comparison of workflow bundles; not an isolated estimate of any single knowledge component'})
        try:
            status(arm,'baseline_training')
            jobs=[{'data':data,'source':HERE/'baseline.py','config':{'kind':kind},'out':arm/'trials'/('baseline-'+kind),
                   'seed':42,'protocol':protocol,'deadline':deadline} for kind in ['static','pooled','din']]
            receipts=run_jobs(jobs,gpus)
            for row in receipts:
                kind=row['config']['kind'];row.update(id='baseline-'+kind,parent=None,role='reference',baseline_kind=kind,model_path=str(Path(row['output'])/'model.py'),round=-1)
                state['trials'].append(row)
            selected=best_trial(state)
            if selected is None or any(t['status']!='success' for t in state['trials']):raise RuntimeError('All three shared references must succeed before formal search')
            state['incumbent']=selected['id'];write(arm/'state.json',state)
            cards=[json.loads(line) for line in (DESIGN/'tasks/taobao_display_ctr/industry/cards.jsonl').read_text().splitlines() if line.strip()]
            papers=read(DESIGN/'tasks/taobao_display_ctr/paper_sources.json')
            paper_cards=[{'id':'paper-seed-registry','scientific_status':'external_reference','sources':papers}]
            round_index=0;errors=0
            while True:
                remaining=deadline-time.time()
                reserve=max(3600,protocol['estimated_unit_seconds']*6+protocol['estimated_validation_seconds']*12)
                minimum_round=protocol['estimated_unit_seconds']*1.5+(2700 if mode=='enhanced' else 900)
                if (root/'STOP').exists() or (arm/'STOP').exists() or remaining<reserve+minimum_round:break
                round_root=arm/'rounds'/f'r{round_index:04d}';round_root.mkdir(parents=True,exist_ok=True)
                status(arm,'research',round=round_index,completed_trials=len(state['trials']),remaining_seconds=remaining)
                try:
                    evidence=cards+paper_cards+(state['memory'] if mode=='enhanced' else [])
                    retrieval=None
                    if mode=='enhanced':
                        querydir=round_root/'query';querydir.mkdir(exist_ok=True)
                        receipt=researcher(querydir,'query',mode,state,protocol,deadline-reserve,evidence)
                        state['agent_calls'].append({'phase':'query','round':round_index,**receipt})
                        plan=read(querydir/'retrieval-plan.json')
                        if not isinstance(plan.get('query'),str) or not set(plan.get('evidence_refs',[]))<={t['id'] for t in state['trials']}:raise ValueError('Retrieval plan lacks a query or cites unknown observations')
                        words=set(re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]',plan['query'].lower()))
                        ranked=sorted(evidence,key=lambda card:sum(json.dumps(card,ensure_ascii=False).lower().count(word) for word in words),reverse=True)
                        named={card['id']:card for card in evidence}
                        requested=plan.get('requested_card_ids',[])
                        if not set(requested)<=set(named):raise ValueError('Retrieval names unknown evidence')
                        retrieval={'plan':plan,'cards':[named[name] for name in requested[:3]] or ranked[:3]}
                        write(round_root/'retrieval.json',retrieval)
                    proposedir=round_root/'propose';proposedir.mkdir(exist_ok=True)
                    receipt=researcher(proposedir,'propose',mode,state,protocol,deadline-reserve,evidence,retrieval)
                    state['agent_calls'].append({'phase':'propose','round':round_index,**receipt})
                    proposal=read(proposedir/'proposal.json')
                    Draft202012Validator(read(DESIGN/'agent/study.schema.json')).validate(proposal['study'])
                    if proposal['study']['protocol_ref']!=protocol['identity']:raise ValueError('Study must cite the current protocol hash')
                    if not set(proposal['study']['evidence_refs'])&{t['id'] for t in state['trials']}:raise ValueError('Study must cite at least one actual observed trial')
                    if proposal['parent_trial'] not in {t['id'] for t in state['trials']}:raise ValueError('Unknown parent trajectory')
                    variants=proposal['variants']
                    if not isinstance(variants,list) or not 1<=len(variants)<=4 or not any(v['role']=='candidate' for v in variants):raise ValueError('Require 1-4 variants including a candidate')
                    if len({v['name'] for v in variants})!=len(variants):raise ValueError('Duplicate variants')
                    from train import check_source
                    jobs=[]
                    for variant in variants:
                        if not re.fullmatch('[A-Za-z0-9_-]{1,60}',variant['name']) or Path(variant['file']).name!=variant['file']:raise ValueError('Invalid variant identity')
                        check_source(proposedir/variant['file'])
                        jobs.append({'data':data,'source':proposedir/variant['file'],'config':variant.get('config',{}),
                                     'out':arm/'trials'/(f'r{round_index:04d}-'+variant['name']),'seed':42,'protocol':protocol,'deadline':deadline-reserve})
                    results=run_jobs(jobs,gpus)
                    lookup={str(job['out']):variant for job,variant in zip(jobs,variants)}
                    for row in results:
                        variant=lookup[row['output']]
                        row.update(id=Path(row['output']).name,parent=proposal['parent_trial'],role=variant['role'],round=round_index,
                                   note=variant.get('note'),model_path=str(Path(row['output'])/'model.py'),study_ref=str(proposedir/'proposal.json'))
                        state['trials'].append(row)
                    selected=best_trial(state);state['incumbent']=selected['id'];write(arm/'state.json',state)
                    if mode=='enhanced' and deadline-time.time()>reserve+120:
                        analysisdir=round_root/'analyze';analysisdir.mkdir(exist_ok=True)
                        receipt=researcher(analysisdir,'analyze',mode,state,protocol,deadline-reserve,evidence)
                        state['agent_calls'].append({'phase':'analyze','round':round_index,**receipt})
                        analysis=read(analysisdir/'analysis.json')
                        refs=analysis.get('evidence_refs',[])
                        if not refs or not set(refs)<={t['id'] for t in state['trials']} or not set(refs)&{t['id'] for t in state['trials'] if t.get('round')==round_index}:raise ValueError('Analysis must cite actual latest-round outcomes')
                        record={'id':f'agent-r{round_index:04d}','task_scope':protocol['data_identity'],'protocol_hash':protocol['identity'],
                                'kind':'provisional_interpretation','maturity':'hypothesis','analysis':analysis,'source_trial_refs':refs}
                        state['memory'].append(record)
                        with (arm/'agent-knowledge.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    errors=0
                except Exception as error:
                    errors+=1
                    with (arm/'failures.jsonl').open('a') as stream:stream.write(json.dumps({'round':round_index,'error':repr(error),'traceback':traceback.format_exc(),'at':time.time()})+'\n')
                    if errors>=3:break
                finally:
                    write(arm/'state.json',state);round_index+=1
            final_assessment(root,arm,state,protocol,data,gpus,deadline)
            comparison(root)
        except BaseException as error:
            status(arm,'failed',error=repr(error),traceback=traceback.format_exc())
            raise


def comparison(root):
    rows={}
    for mode in ['auto','enhanced']:
        arm=root/mode
        if not (arm/'state.json').exists():continue
        state=read(arm/'state.json');best=best_trial(state)
        calls=[]
        for path in (arm/'rounds').glob('*/**/agent-receipt.json'):calls.append(read(path))
        usage=[entry for call in calls for entry in call['usage']]
        rows[mode]={'status':read(arm/'status.json') if (arm/'status.json').exists() else None,
                    'trials':len(state['trials']),'failed_trials':sum(t['status']!='success' for t in state['trials']),
                    'best_validation':best['metrics']['validation'] if best else None,
                    'researcher_calls':len(calls),'llm_usage':{key:sum(item.get(key,0) for item in usage) for key in {k for item in usage for k in item if isinstance(item[k],(int,float))}},
                    'usd_cost':None,'final':read(arm/'final.json') if (arm/'final.json').exists() else None}
        if rows[mode]['final']:
            summaries={}
            for outcome in rows[mode]['final']['results']:
                if outcome.get('test'):
                    summaries.setdefault(outcome['trial'],[]).append({'seed':outcome['seed'],'auc':outcome['test']['auc'],'logloss':outcome['test']['logloss']})
            rows[mode]['final_summary']={trial:{'seeds':values,'auc_mean':statistics.mean(v['auc'] for v in values),
                'auc_sample_stddev':statistics.stdev(v['auc'] for v in values) if len(values)>1 else None,
                'logloss_mean':statistics.mean(v['logloss'] for v in values),'evidence':'model-seed variation only; one research trajectory'} for trial,values in summaries.items()}
    setup_calls=[read(path) for path in root.glob('agent-preflight*/agent-receipt.json')]
    write(root/'comparison.json',{'updated_unix':time.time(),'protocol':read(root/'protocol.json') if (root/'protocol.json').exists() else None,'methods':rows,
        'shared_setup':{'elapsed_seconds_so_far':min(time.time(),read(root/'budget.json')['deadline_unix'])-read(root/'budget.json')['start_unix'],
                        'agent_preflight_receipts':setup_calls,'calibration':read(root/'calibration/receipt.json') if (root/'calibration/receipt.json').exists() else None,
                        'preparation':read(root/'preparation-process/process-receipt.json') if (root/'preparation-process/process-receipt.json').exists() else None},
        'cost_scope':'same deadline and four GPUs per arm; all shared preparation elapsed charged to both; auxiliary calls, failures and assessments consume wall budget',
        'interpretation':'One paired method run, not three independent agent trajectories. Report model seed variation separately. No novelty score.'})
    report=['淘宝广告 CTR：auto 与 enhanced 双版本研究记录',
            '两版具有相同数据、基线、初始论文／行业来源、gpt-6.1-sol/high 和四张 GPU 上限。',
            'enhanced 增加显式检索计划、实验后反思和场景知识检索；这是工作流组合对比，不能单独归因某一组件。',
            '个人知识禁用。所有数据准备、校准、失败与评估计入同一 24 小时截止时间。',
            '历史窗口为已冻结协议所示；原始数据无历史商品 ID；静态快照的时点未知假设见 protocol.json。','']
    for mode,item in rows.items():
        state=read(root/mode/'state.json');best=best_trial(state)
        report += [mode+': '+item['status']['phase'],
                   f"完整协议训练记录 {item['trials']}，失败训练 {item['failed_trials']}，研究调用 {item['researcher_calls']}。"]
        if best:
            report += [f"验证选择：{best['id']}；ROC-AUC={best['metrics']['validation']['auc']:.8f}；Logloss={best['metrics']['validation']['logloss']:.8f}。",
                       f"参数量={best['metrics']['parameters']}；训练秒数={best['metrics']['train_seconds']:.2f}。",
                       '模型代码：'+best['model_path'],'模型配置：'+str(Path(best['output'])/'model-config.json'),
                       '训练复现：'+shlex.join(read(Path(best['output'])/'process.json')['argv'])]
            if best.get('study_ref'):
                proposal=read(best['study_ref'])
                report += ['研究问题：'+proposal['study']['question'],'机制假设：'+proposal['study']['hypothesis'],
                           '相近工作差异：'+proposal['study']['proposed_difference'],
                           '完整机制计划与对照：'+best['study_ref']]
            else:report += ['当前最优仍是起点基线，不能称为已验证的新建模方案。']
        if item['final']:
            for outcome in item['final']['results']:
                if outcome.get('test'):
                    report.append(f"封闭测试：{outcome['trial']} seed={outcome['seed']} AUC={outcome['test']['auc']:.8f} Logloss={outcome['test']['logloss']:.8f}")
            report += ['重复结果是模型初始化波动；本次每种方法只有一条 Agent 研究轨迹。',
                       '机制支持需结合关键消融、混杂及失败分支判断；单次涨分和外部名次都不构成机制确认。']
        report.append('')
    report += ['各模型的实际命令、检查点、验证切片、训练与推理成本位于对应 trials/ 目录。',
               '失败与未采纳分支保留于 failures.jsonl 和 rounds/，强化版场景知识位于 agent-knowledge.jsonl。',
               'test 输出只进入 assessment/ 和关闭后的报告，不回流研究上下文。',
               '记录的 token 限于可观测的研究子进程与预检调用；当前交互会话用量不可取得，费用无账单时保持未知。',
               '候选模型静态 I/O 检查是防止意外访问的检查，不能视为任意 Python 程序的安全沙箱。']
    (root/'REPORT.txt').write_text('\n'.join(report)+'\n')


def suite(root):
    with (root/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((root/mode/'state.json').exists() for mode in ['auto','enhanced']):raise ValueError('Study already started; do not reinitialize or reuse its holdout')
        start=read(root/'acquire-process.json')['launched_unix'];deadline=start+BUDGET
        write(root/'budget.json',{'start_unix':start,'deadline_unix':deadline,'seconds_per_arm':BUDGET,
             'gpus':{'auto':[0,1,2,3],'enhanced':[4,5,6,7]},'shared_preparation_charged_to_each':True,
             'single_experiment_upper_fraction':0.25,'model':'gpt-6.1-sol','reasoning_effort':'high','personal_enabled':False})
        try:
            while True:
                if time.time()>=deadline:raise TimeoutError('Budget exhausted during acquisition')
                if (root/'STOP').exists():status(root,'stopped_during_acquisition');return
                if (root/'assets/acquisition-error.json').exists():raise RuntimeError(read(root/'assets/acquisition-error.json'))
                acquisition=read(root/'assets/acquisition.json') if (root/'assets/acquisition.json').exists() else {}
                if acquisition.get('phase')=='complete':break
                status(root,'acquiring_data',downloaded_bytes=acquisition.get('downloaded_bytes'),deadline_unix=deadline)
                time.sleep(10)
            status(root,'data_preparation',deadline_unix=deadline)
            if not (root/'prepared/metadata.json').exists():
                if (root/'prepared').exists():
                    marker=root/'preparation-resume-process.json'
                    if not marker.exists():raise ValueError('Incomplete prior preprocessing requires explicit recovery; not overwriting')
                    recovery=read(marker);pid=recovery['pid']
                    status(root,'resuming_verified_preprocessing',preparation_pid=pid,deadline_unix=deadline)
                    while not (root/'prepared/metadata.json').exists():
                        proc=Path(f'/proc/{pid}/stat')
                        if time.time()>=deadline:
                            stop_children(root);raise TimeoutError('Budget exhausted during preprocessing recovery')
                        if not proc.exists() or proc.read_text().split()[21]!=recovery['start_ticks'] or proc.read_text().split()[2]=='Z':
                            raise RuntimeError('Explicit preprocessing recovery failed; inspect preparation-resume.log')
                        time.sleep(10)
                else:
                    result=bounded([sys.executable,'-u',str(HERE/'data.py'),'--raw',str(root/'assets/raw'),'--output',str(root/'prepared')],root/'preparation-process',max(1,min(14400,deadline-time.time())))
                    if result['status']!='success':raise RuntimeError('Data preparation failed; inspect preparation-process/process.log')
            metadata=read(root/'prepared/metadata.json')
            if metadata['fixture']:raise ValueError('A fixture cannot enter the real campaign')
            prepare_template()
            status(root,'cost_calibration',deadline_unix=deadline)
            protocol={'data_identity':metadata['identity'],'epochs':3,'batch_size':4096,'history_limit':128,
                      'train_seed':42,'final_paired_seeds':SEEDS,'primary':'ROC-AUC','secondary':'unweighted Logloss',
                      'trial_cap_seconds':21600,'test_policy':'one closed phase after source and checkpoint suite freeze',
                      'training':'all first-six-day exposures; natural class ratio; Adam or declared candidate optimizer/loss; scratch initialization',
                      'base_design_protocol_sha256':sha(DESIGN/'tasks/taobao_display_ctr/protocol.draft.json'),
                      'source_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'static_snapshot_assumption':metadata['static_snapshot_assumption']}
            calibration=train_job(root/'prepared',HERE/'baseline.py',{'kind':'din'},root/'calibration',42,0,protocol,deadline,128)
            if calibration['status']!='success':raise RuntimeError('Real-data/GPU calibration failed')
            metrics=calibration['metrics']
            train_count=sum(row['samples'] for row in read(root/'prepared/host-label-audit.json')['daily_counts'][:6])
            estimated_train=metrics['train_seconds']*train_count/metrics['training_examples']*protocol['epochs']
            unit=estimated_train+metrics['validation']['inference_seconds']*protocol['epochs']+60
            if unit*1.5>protocol['trial_cap_seconds']:raise ValueError('Measured complete experiment cannot fit the proposed upper limit; revise protocol before search')
            protocol.update(estimated_unit_seconds=unit,estimated_validation_seconds=metrics['validation']['inference_seconds'],
                            trial_cap_seconds=min(21600,max(900,unit*2.5)),frozen_unix=time.time())
            protocol['identity']=identity(protocol)
            write(root/'protocol.json',protocol)
            snapshot=root/'source-at-launch'
            shutil.copytree(HERE,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(DESIGN,snapshot/'task-design',ignore=shutil.ignore_patterns('source_cache','__pycache__','workspace'))
            status(root,'dual_research_running',deadline_unix=deadline,data_identity=metadata['identity'],protocol_hash=protocol['identity'])
            processes=[]
            for mode,gpus in [('auto','0,1,2,3'),('enhanced','4,5,6,7')]:
                arm=root/mode;arm.mkdir(exist_ok=True)
                with (arm/'worker.log').open('ab') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(snapshot/'controller.py'),'worker','--root',str(root),'--mode',mode,'--gpus',gpus],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'REC_CTR_PROJECT':str(PROJECT)})
                write(arm/'worker-process.json',{'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],'launched_unix':time.time()});processes.append(process)
            while any(process.poll() is None for process in processes):
                comparison(root)
                if time.time()>=deadline:
                    stop_children(root)
                    comparison(root);status(root,'budget_reached');return
                time.sleep(15)
            comparison(root)
            if any(process.poll() not in {0,None} for process in processes):status(root,'completed_with_failures')
            else:status(root,'completed')
        except BaseException as error:
            status(root,'failed',error=repr(error),traceback=traceback.format_exc());raise


def stop_children(root):
    for pattern in ['**/process.json','**/agent-process.json','**/worker-process.json','preparation-resume-process.json']:
        for path in root.glob(pattern):
            item=read(path);pid=item.get('pid');proc=Path(f'/proc/{pid}/stat')
            if pid and proc.exists() and item.get('start_ticks')==proc.read_text().split()[21]:
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['suite','worker','status'])
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--mode',choices=['auto','enhanced'])
    parser.add_argument('--gpus',default='0,1,2,3')
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='suite':suite(root)
    elif args.action=='worker':worker(root,args.mode,[int(value) for value in args.gpus.split(',')])
    else:
        comparison(root)
        print(json.dumps({'suite':read(root/'status.json'),'comparison':str(root/'comparison.json')},ensure_ascii=False,indent=2))
