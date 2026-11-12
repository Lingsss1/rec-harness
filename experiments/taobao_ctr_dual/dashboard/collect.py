"""Collect public validation receipts and resource telemetry, without test scores."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


class Reader:
    def __init__(self):
        self.cache={}
        self.errors=[]

    def json(self,path,default=None):
        path=Path(path)
        try:
            info=path.stat();signature=(info.st_ino,info.st_size,info.st_mtime_ns)
            cached=self.cache.get(str(path))
            if cached and cached[0]==signature:return cached[1]
            value=json.loads(path.read_text())
            self.cache[str(path)]=(signature,value)
            return value
        except FileNotFoundError:return {} if default is None else default
        except (OSError,ValueError) as error:
            self.errors.append(path.name+': '+type(error).__name__)
            return self.cache.get(str(path),(None,{} if default is None else default))[1]


def tail_records(path,limit=30,byte_limit=131072):
    try:
        with Path(path).open('rb') as stream:
            size=stream.seek(0,2);stream.seek(max(0,size-byte_limit))
            lines=stream.read().splitlines()
        values=[]
        for line in lines:
            try:values.append(json.loads(line))
            except ValueError:pass
        return values[-limit:]
    except OSError:return []


def process_alive(record):
    if not record.get('pid'):return False
    try:
        stat=(Path('/proc')/str(record['pid'])/'stat').read_text().split()
        return stat[2]!='Z' and (not record.get('start_ticks') or record['start_ticks']==stat[21])
    except (OSError,IndexError):return False


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def eligible_trial(row,data_identity=None,selection_policy=None):
    """Completed full-protocol validation only; never mix in calibration or test."""
    metrics=row.get('metrics',{})
    score=metrics.get('validation',{})
    if row.get('status')!='success' or metrics.get('calibration_only') or metrics.get('split')=='test':return False
    if data_identity and metrics.get('data_identity')!=data_identity:return False
    if selection_policy and score.get('selection_policy_id')!=selection_policy['id']:return False
    return finite(score.get('auc')) and 0<=score['auc']<=1 and finite(score.get('logloss'))


def incumbent_curve(trials):
    best=None;points=[]
    for row in sorted(trials,key=lambda t:(t['finished_unix'],t['id'])):
        if best is None or row['auc']>best:best=row['auc']
        points.append({'time':row['finished_unix'],'auc':best,'trial':row['id']})
    return points


def safe_text(value,limit=1000):
    if not isinstance(value,str):value=json.dumps(value,ensure_ascii=False)
    value=re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b','[redacted]',value)
    value=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._-]+',r'\1[redacted]',value)
    return value[:limit]


def usage_summary(receipts):
    entries=[item for receipt in receipts for item in receipt.get('usage',[])]
    sums={key:sum(item.get(key,0) or 0 for item in entries) for key in
          ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']}
    sums.update(calls=len(receipts),failed_calls=sum(receipt.get('status')!='success' for receipt in receipts),
                seconds=sum(receipt.get('wall_seconds',0) for receipt in receipts),
                uncached_input_tokens=max(0,sums['input_tokens']-sums['cached_input_tokens']))
    return sums


def normalized_trial(row,reader):
    metrics=row['metrics'];validation=metrics['validation']
    out=Path(row['output'])
    process=reader.json(out/'process.json')
    fallback=(out/'result.json').stat().st_mtime if (out/'result.json').exists() else time.time()
    finish=process.get('started_unix',fallback-row.get('wall_seconds',0))+row.get('wall_seconds',0)
    return {'id':row['id'],'parent':row.get('parent'),'role':row.get('role','unknown'),'seed':row.get('seed'),
            'round':row.get('round',-1),'baseline_kind':row.get('baseline_kind'),'auc':validation['auc'],
            'logloss':validation['logloss'],'samples':validation.get('samples'),'slices':validation.get('slices',{}),
            'per_task':validation.get('per_task',{}),'mean_task_auc':validation.get('mean_task_auc'),
            'selection_metric':validation.get('selection_metric','click_auc'),
            'selection_policy_id':validation.get('selection_policy_id'),
            'click_auc':validation.get('click_auc',validation.get('per_task',{}).get('is_click',{}).get('auc',validation['auc'])),
            'wall_seconds':row.get('wall_seconds',0),'train_seconds':metrics.get('train_seconds',0),
            'gpu_seconds':metrics.get('allocated_gpu_seconds',0),'parameters':metrics.get('parameters'),
            'dense_parameters':metrics.get('dense_parameters'),'embedding_parameters':metrics.get('embedding_parameters'),
            'peak_gpu_bytes':metrics.get('peak_gpu_bytes'),'inference_seconds':validation.get('inference_seconds'),
            'throughput':validation.get('examples_per_second'),'latency_p50':validation.get('batch_latency_ms_p50'),
            'latency_p95':validation.get('batch_latency_ms_p95'),'finished_unix':finish,
            'diagnostics_status':metrics.get('candidate_diagnostics',{}).get('status'),'output':str(out),
            'epoch_curve':[{'epoch':entry.get('epoch'),'training_loss':entry.get('training_loss'),
                            'auc':entry.get('validation',{}).get('auc')} for entry in metrics.get('epochs',[])]}


class Collector:
    def __init__(self,root):
        self.root=Path(root).resolve();self.reader=Reader();self.gpu_cache=[];self.gpu_at=0
        self.cpu_previous=None

    def resources(self):
        now=time.time();resource={'gpu_error':None}
        if now-self.gpu_at>4:
            try:
                output=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True).stdout
                values=[]
                for line in output.splitlines():
                    cells=[cell.strip() for cell in line.split(',')]
                    def number(index):
                        try:return float(cells[index])
                        except ValueError:return None
                    values.append({'index':int(cells[0]),'name':cells[1],'utilization':number(2),'memory_used_mib':number(3),
                                   'memory_total_mib':number(4),'power_w':number(5),'temperature':number(6)})
                self.gpu_cache=values;self.gpu_at=now
            except (OSError,subprocess.SubprocessError,ValueError) as error:resource['gpu_error']=type(error).__name__
        resource['gpus']=self.gpu_cache
        try:
            values=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
            total=sum(values);idle=values[3]+values[4]
            resource['cpu_percent']=None
            if self.cpu_previous and total>self.cpu_previous[0]:
                resource['cpu_percent']=100*(1-(idle-self.cpu_previous[1])/(total-self.cpu_previous[0]))
            self.cpu_previous=(total,idle)
            memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
            resource.update(memory_total=memory['MemTotal'],memory_used=memory['MemTotal']-memory['MemAvailable'],
                            load_average=list(os.getloadavg()),cpu_count=os.cpu_count())
        except (OSError,KeyError,ValueError):pass
        return resource

    def arm(self,mode,campaign):
        root=self.root;arm=root/mode;reader=self.reader
        state=reader.json(arm/'state.json');status=reader.json(arm/'status.json')
        raw=state.get('trials',[]);trials=[normalized_trial(row,reader) for row in raw if eligible_trial(row,campaign.get('data_identity'))]
        trials.sort(key=lambda row:row['finished_unix'])
        best=max(trials,key=lambda row:row['auc'],default=None)
        baseline=next((row for row in trials if row.get('baseline_kind')=='din'),None)
        lookup={row['id']:row for row in trials}
        for row in trials:
            parent=lookup.get(row.get('parent'))
            row['parent_delta']=row['auc']-parent['auc'] if parent else None
        completed_ids={row['id'] for row in raw}
        active=[]
        for path in [*arm.glob('trials/*/process.json'),*arm.glob('repeats/*/*/process.json')]:
            process=reader.json(path)
            if not process_alive(process):continue
            progress=reader.json(path.parent/'progress.json')
            receipt=reader.json(path.parent/'receipt.json')
            train_samples=campaign.get('train_samples',0)
            total_steps=math.ceil(train_samples/campaign.get('batch_size',4096))*campaign.get('epochs',3) if train_samples else 0
            steps=progress.get('steps',0)
            active.append({'id':path.parent.name,'gpu':receipt.get('gpu'),'phase':progress.get('phase','starting'),
                           'steps':steps,'total_steps':total_steps,'fraction':min(1,steps/total_steps) if total_steps else 0,
                           'epoch':progress.get('epoch'),'elapsed':time.time()-process.get('started_unix',time.time()),
                           'last_update_age':time.time()-path.parent.joinpath('progress.json').stat().st_mtime if (path.parent/'progress.json').exists() else None})
        rounds=sorted(arm.glob('rounds/r*'))
        current=rounds[-1] if rounds else None
        study={};variants=[];retrieval={};analysis={};agent_phase=None;agent_age=None;agent_receipts=[];events=[]
        for path in arm.glob('rounds/*/*/agent-receipt.json'):
            receipt=reader.json(path);agent_receipts.append(receipt)
            events.append({'at':path.stat().st_mtime,'mode':mode,'type':'agent','text':path.parent.parent.name+' '+path.parent.name+' '+receipt.get('status','?')})
        if current:
            proposal=reader.json(current/'propose/proposal.json');study=proposal.get('study',{})
            variants=[{'name':row['name'],'role':row['role']} for row in proposal.get('variants',[])]
            retrieval=reader.json(current/'retrieval.json')
            analysis=reader.json(current/'analyze/analysis.json')
            for stage in ['analyze','propose','query']:
                process=reader.json(current/stage/'agent-process.json')
                if process_alive(process):
                    agent_phase=stage
                    log=current/stage/'codex.jsonl'
                    agent_age=time.time()-log.stat().st_mtime if log.exists() else None
                    break
        if not analysis:
            candidates=sorted(arm.glob('rounds/*/analyze/analysis.json'))
            if candidates:analysis=reader.json(candidates[-1])
        failures=tail_records(arm/'failures.jsonl',30)
        for item in failures:events.append({'at':item.get('at',0),'mode':mode,'type':'failure','text':safe_text(item.get('error','unknown'),240)})
        for row in trials:events.append({'at':row['finished_unix'],'mode':mode,'type':'trial','text':row['id']+f" AUC={row['auc']:.6f}"})
        memories=state.get('memory',[])
        usage=usage_summary(agent_receipts)
        counts=Counter(row.get('role','unknown') for row in raw)
        recent_round=max((row.get('round',-1) for row in trials),default=-1)
        group=[row for row in trials if row.get('round')==recent_round]
        phase=agent_phase or ('train' if active else status.get('phase','unknown'))
        pending_ids=[path.parent.name for path in arm.glob('trials/*/result.json') if path.parent.name not in completed_ids]
        return {'mode':mode,'status':status,'worker_alive':process_alive(reader.json(arm/'worker-process.json')),
                'phase':phase,'agent_log_age':agent_age,'round':current.name if current else None,
                'trials':trials,'best':best,'baseline':baseline,'curve':incumbent_curve(trials),'active_jobs':active,
                'counts':{'total':len(raw),'success':len(trials),'failed_training':sum(row.get('status')!='success' for row in raw),
                          'reference':counts['reference'],'candidate':counts['candidate'],'control':counts['control'],
                          'admission_errors':len(failures),'completed_rounds':len({row.get('round') for row in raw if row.get('round',-1)>=0})},
                'pending_commit':len(pending_ids),'study':{key:safe_text(study.get(key,''),2400) for key in ['question','hypothesis','proposed_difference','falsifiers','predictions']},
                'nearest_work_count':len(study.get('nearest_work',[])),'contrast_count':len(study.get('contrasts',[])),
                'variants':variants,'round_results':group,'analysis':{key:safe_text(analysis.get(key,''),1200) for key in ['observations','interpretations','alternatives','next_action']},
                'knowledge':{'memory_records':len(memories),'maturity':dict(Counter(row.get('maturity','unknown') for row in memories)),
                             'retrieved_ids':[card.get('id') for card in retrieval.get('cards',[])],'query':safe_text(retrieval.get('plan',{}).get('query',''),400)},
                'usage':usage,'failures':failures[-3:],'events':sorted(events,key=lambda row:row['at'])[-12:]}

    def snapshot(self):
        self.reader.errors=[];reader=self.reader;root=self.root
        budget=reader.json(root/'budget.json');protocol=reader.json(root/'protocol.json')
        meta=reader.json(root/'prepared/metadata.json');audit=reader.json(root/'prepared/host-label-audit.json')
        recovery=reader.json(root/'recovery.json')
        campaign={'start':budget.get('start_unix',time.time()),'deadline':budget.get('deadline_unix',time.time()),
                  'status':reader.json(root/'status.json'),'gpus':budget.get('gpus',{}),'model':budget.get('model','?'),
                  'effort':budget.get('reasoning_effort','?'),'protocol':protocol.get('identity','?'),'data_identity':protocol.get('data_identity'),
                  'rows':meta.get('rows'),'events':meta.get('events'),'train_samples':sum(row['samples'] for row in audit.get('daily_counts',[])[:6]),
                  'epochs':protocol.get('epochs',3),'batch_size':protocol.get('batch_size',4096),'history_limit':protocol.get('history_limit'),
                  'recovery':bool(recovery),'prior_auto_costs':recovery.get('prior_attempt_costs',{}),
                  'auto_test_disabled':reader.json(root/'auto-validation-only.json').get('enabled',False),
                  'personal_enabled':budget.get('personal_enabled',False),'dataset_dates':meta.get('split',{}).get('dates',[])}
        arms={mode:self.arm(mode,campaign) for mode in ['auto','enhanced']}
        events=sorted([event for arm in arms.values() for event in arm['events']],key=lambda row:row['at'])[-24:]
        return {'schema':'ctr-dashboard-observer-v1','updated_unix':time.time(),'root':str(root),
                'campaign':campaign,'arms':arms,'resources':self.resources(),'events':events,
                'read_errors':reader.errors,'measurement_scope':'Completed full-protocol validation metrics only. Test/assessment score files are never read.'}
