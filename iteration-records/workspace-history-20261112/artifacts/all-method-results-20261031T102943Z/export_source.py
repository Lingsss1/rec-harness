"""Read-only evidence aggregation; writes only a new report directory."""
from pathlib import Path
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import os
import time
import zipfile

W=Path('/home/liuzhishan/lhr/rec-harness')
NOW=time.time()
STAMP=dt.datetime.fromtimestamp(NOW,dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=W/'artifacts'/('all-method-results-'+STAMP)
OUT.mkdir(exist_ok=False)


def read(p, default=None):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return {} if default is None else default


def rel(p):
    try:return str(Path(p).relative_to(W))
    except ValueError:return str(p)


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def csvwrite(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


NAMES=['amazon-aligned-20261022','amazon-repeats-20261025','amazon-reruns-20261026',
       'self-evorec-resilient-20261027','amazon-tpe-20261028','amazon-tpe-sasrec-20261028',
       'mlevolve-single-repeat-20261029','recharness-gru-jump-20261028',
       'recharness-gru-jump-v3-20261028','recharness-gru-jump-129600-20261028',
       'recharness-sasrec-default-jump-43200-20261029',
       'recharness-sasrec-default-prompt-v2-43200-20261029',
       'recharness-sasrec-tpebest-jump-43200-20261028',
       'recharness-sasrec-jump-v4-43200-20261029',
       'model-family-comparison-20261029','recharness-family-comparison-20261031']

runs={}
for name in NAMES:
    base=W/'artifacts'/name
    for pattern in ['config.json','*/config.json','*/*/config.json','*/*/*/config.json']:
        for path in base.glob(pattern):
            parts=path.relative_to(base).parts
            if set(parts)&{'preflight','smoke','before','source','trials','calls'}:continue
            c=read(path)
            if not c.get('root') or not c.get('methods'):continue
            for method in c['methods']:
                if method not in ['mlestar','mlevolve','self_evolverec','recharness','tpe']:continue
                root=Path(c['root'])/method
                if root in runs:continue
                runs[root]={'config':c,'config_path':path,'method':method,'cohort':name}

reselected={}
for path in (W/'artifacts/amazon-single-model-20261026').glob('*/summary.json'):
    d=read(path)
    if d.get('job',{}).get('run'):reselected[Path(d['job']['run'])]=(path,d)

attempts=[];token_events=[];errors=[];rows=[];per_dataset=[];trials=[]
global_events=set()


def scan_usage(root,run_id,c):
    total=collections.Counter();count=collections.Counter();efforts=collections.Counter();models=collections.Counter()
    calls=[p.parent for p in (root/'calls').glob('*/request.json')]
    for call in sorted(calls):
        receipt=read(call/'receipt.json');started=read(call/'started.json')
        model=receipt.get('model',started.get('model',c.get('llm_model')))
        effort=receipt.get('effort',started.get('effort',c.get('llm_effort')))
        request=read(call/'request.json')
        cmd=started.get('command',[])
        if not cmd:
            logs=sorted(call.glob('attempts/*/started.json'))
            if logs:cmd=read(logs[0]).get('command',[])
        for arg in cmd:
            if isinstance(arg,str) and arg.startswith('model_reasoning_effort='):
                effort=arg.split('=',1)[1].strip('"')
        if model:models[model]+=1
        if effort:efforts[effort]+=1
        count['llm_calls']+=1
        count['llm_calls_with_receipt']+=int(bool(receipt))
        events_files=sorted(call.rglob('events.jsonl'))
        if not events_files:
            count['calls_without_event_log']+=1
        call_has_usage=False
        for path in events_files:
            count['llm_attempt_event_logs']+=1
            usage=collections.Counter();events_seen=0;thread=None;turn_index=0;bad_lines=0
            thread_started=False;seen_turns=[]
            with path.open(errors='replace') as f:
                for lineno,line in enumerate(f,1):
                    if not any(k in line for k in ['thread.started','"usage"','"token_usage"']):continue
                    try:event=json.loads(line)
                    except ValueError:
                        bad_lines+=1;continue
                    if event.get('type')=='thread.started':
                        thread=event.get('thread_id');thread_started=True;turn_index=0
                    u=event.get('usage')
                    if event.get('type') not in ['turn.completed','turn.failed'] or not isinstance(u,dict):continue
                    if not isinstance(u.get('input_tokens'),int) or not isinstance(u.get('output_tokens'),int):
                        errors.append({'file':rel(path),'line':lineno,'reason':'usage missing input/output counts'});continue
                    identity=(thread,turn_index) if thread else (str(path),lineno)
                    turn_index+=1
                    if identity in global_events:
                        count['duplicate_usage_events_skipped']+=1;continue
                    global_events.add(identity)
                    values={key:int(u.get(key,0)) for key in ['input_tokens','cached_input_tokens',
                        'cache_write_input_tokens','output_tokens','reasoning_output_tokens']}
                    values['total_tokens']=values['input_tokens']+values['output_tokens']
                    if values['cached_input_tokens']>values['input_tokens'] or values['reasoning_output_tokens']>values['output_tokens']:
                        errors.append({'file':rel(path),'line':lineno,'reason':'subset count exceeds corresponding total'})
                    usage.update(values);events_seen+=1;call_has_usage=True
                    token_events.append({'run_id':run_id,'method':c.get('_method'),
                        'call':call.name,'thread_id':thread,'turn_index':turn_index-1,
                        'event_type':event['type'],'event_file':rel(path),'line':lineno,**values})
                    seen_turns.append(lineno)
            complete=(path.parent/'answer.json').exists()
            failure=read(path.parent/'failure.json')
            if not failure and not complete:
                failure=read(call/'failure.json')
            row={'run_id':run_id,'call':call.name,'attempt':rel(path.parent.relative_to(call)),
                 'event_file':rel(path),'thread_started':thread_started,'reported_turns':events_seen,
                 'call_receipt':bool(receipt),'answer_exists':complete,'failure_type':failure.get('type'),
                 'model':model,'effort':effort,'malformed_candidate_lines':bad_lines,
                 'usage_available':bool(events_seen),**dict(usage)}
            attempts.append(row)
            total.update(usage)
            count['llm_attempts_with_usage']+=int(bool(events_seen))
            count['llm_attempts_without_usage']+=int(not events_seen)
            count['cli_started_without_usage']+=int(thread_started and not events_seen)
            count['reported_turns']+=events_seen
        count['llm_calls_with_usage']+=int(call_has_usage)
    return {**{k:total[k] for k in ['input_tokens','cached_input_tokens','cache_write_input_tokens',
                                  'output_tokens','reasoning_output_tokens','total_tokens']},
            **dict(count),'effort_distribution':dict(efforts),'model_distribution':dict(models)}


for root,info in sorted(runs.items(),key=lambda pair:str(pair[0])):
    c=info['config'];method=info['method'];c['_method']=method
    run_id=rel(root).removeprefix('artifacts/')
    config_path=info['config_path'];budget=read(root/'budget.json');summary=read(root/'summary.json')
    status=read(config_path.parent/'status.json') or read(root/'status.json')
    failure=read(config_path.parent/'failure.json') or read(root/'failure.json')
    receipts=[]
    for path in sorted(root.glob('trials/*/receipt.json')):
        d=read(path);receipts.append(d)
        trials.append({'run_id':run_id,'method':method,'trial':d.get('id',path.parent.name),
            'status':d.get('status'),'validation_hr10':d.get('score'),
            'gpu_seconds':d.get('allocated_gpu_seconds',0),'source_sha256':d.get('source_sha256'),
            'output':d.get('output'),'receipt':rel(path)})
    successful=[r for r in receipts if r.get('status')=='success' and isinstance(r.get('score'),(int,float))]
    best=max(successful,key=lambda r:r['score'],default={})
    selected=summary.get('selected',{})
    test=summary.get('test',{})
    original_selected=selected;original_test=test
    reselection_source=None;posthoc=False
    if root in reselected:
        reselection_source,rs=reselected[root]
        selected=rs.get('selection',{}).get('selected',{});test=rs.get('test',{})
        posthoc=True
    if summary.get('status')=='complete':search_status='complete'
    elif status.get('status'):search_status=status['status']
    elif budget:
        search_status='interrupted' if failure else 'incomplete'
    else:search_status='prepared_not_started'
    current=info['cohort'] in ['model-family-comparison-20261029','recharness-family-comparison-20261031']
    if current:family=c['model_family']
    elif 'gru-jump' in info['cohort']:family='gru4rec'
    elif 'sasrec' in info['cohort']:family='sasrec'
    elif method=='tpe':family='multi_family_hpo'
    else:family='sasrec_start_free_family'
    active=[]
    active_elapsed=0
    for key,cap in budget.get('reservations',{}).items():
        pid=read(root/'trials'/key/'pid.json');started=pid.get('started_at')
        age=max(0,NOW-started) if isinstance(started,(int,float)) else None
        if age is not None:active_elapsed+=age
        active.append({'trial':key,'pid':pid.get('pid'),'elapsed_gpu_seconds_estimate':age,'reserved_cap':cap})
    spent=budget.get('spent_gpu_seconds',0)
    imported=sum(v.get('allocated_gpu_seconds',0) for k,v in budget.get('settled',{}).items()
                 if k=='prior-protocol-charge')
    usage=scan_usage(root,run_id,c)
    if not current and search_status=='queued' and not budget and not usage.get('llm_calls',0):
        search_status='prepared_not_started'
    performance_mode=('posthoc_single_model_selection' if posthoc else
                      'single_model_search' if c.get('single_model_search') else
                      'unconstrained_or_native_search')
    if method=='tpe':usage['token_status']='no_llm_used'
    elif not usage.get('llm_calls',0):usage['token_status']='no_llm_calls_yet'
    elif usage.get('llm_attempts_without_usage',0) or usage.get('calls_without_event_log',0):
        usage['token_status']='reported_usage_only_some_attempts_missing'
    else:usage['token_status']='all_attempt_logs_have_usage'
    row={'run_id':run_id,'cohort':info['cohort'],'current_family_comparison':current,
         'method':method,'family':family,'status':search_status,'status_reported':status.get('status'),'performance_mode':performance_mode,
         'baseline_validation_hr10':next((r.get('score') for r in receipts if r.get('id')=='baseline' or str(r.get('id','')).startswith('round-0000-')),None),
         'best_observed_validation_hr10':best.get('score'),
         'selected_validation_hr10':selected.get('score'),'selected_trial':selected.get('id'),
         'test_hr10':test.get('score'),'test_ndcg10':test.get('aggregate',{}).get('NDCG@10'),
         'test_hr5':test.get('aggregate',{}).get('HR@5'),'test_hr20':test.get('aggregate',{}).get('HR@20'),
         'test_ndcg5':test.get('aggregate',{}).get('NDCG@5'),'test_ndcg20':test.get('aggregate',{}).get('NDCG@20'),
         'raw_original_validation_hr10':original_selected.get('score'),
         'raw_original_test_hr10':original_test.get('score'),
         'raw_original_test_ndcg10':original_test.get('aggregate',{}).get('NDCG@10'),
         'budget_gpu_seconds':budget.get('total_gpu_seconds',c.get('search_budget_gpu_seconds')),
         'settled_gpu_seconds':spent,'imported_prior_gpu_seconds':imported,
         'fresh_settled_gpu_seconds':spent-imported,
         'active_gpu_seconds_estimate':active_elapsed,
         'settled_plus_active_gpu_seconds_estimate':spent+active_elapsed,
         'active_trials':active,'trial_count':len(receipts),'successful_trials':len(successful),
         'trial_status_counts':dict(collections.Counter(r.get('status') for r in receipts)),
         'final_inference_seconds':summary.get('final_inference',{}).get('seconds'),
         'llm_model':c.get('llm_model'),'llm_effort':c.get('llm_effort'),**usage,
         'config_file':rel(config_path),'method_root':rel(root),
         'summary_file':rel(root/'summary.json') if summary else None,
         'single_model_reselection_file':rel(reselection_source) if reselection_source else None,
         'failure_tail':str(failure.get('error',''))[-1200:]}
    for phase,value in [('selected_validation',selected),('test',test)]:
        for name,entry in value.get('per_dataset',{}).items():
            per_dataset.append({'run_id':run_id,'cohort':info['cohort'],'method':method,'family':family,
                                'phase':phase,'dataset':name,'n_users':entry.get('n_users'),
                                **entry.get('metrics',{})})
    rows.append(row)
    print('SCANNED',run_id,'tokens',usage['total_tokens'],flush=True)

current=[r for r in rows if r['current_family_comparison']]
current.sort(key=lambda r:(['gru4rec','bert4rec','nextitnet','hstu'].index(r['family']),
                          ['mlestar','self_evolverec','recharness'].index(r['method'])))
totals=[]
for scope,subset in [('all_formal_runs',rows),('current_fixed_family_comparison',current)]:
    for method in ['mlevolve','mlestar','self_evolverec','recharness','tpe']:
        rs=[r for r in subset if r['method']==method]
        if not rs:continue
        keys=['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens',
              'settled_gpu_seconds','fresh_settled_gpu_seconds','active_gpu_seconds_estimate','llm_calls',
              'llm_calls_with_receipt','llm_attempt_event_logs','llm_attempts_with_usage','llm_attempts_without_usage']
        totals.append({'scope':scope,'method':method,'runs':len(rs),
                       'started_runs':sum(bool(r['trial_count'] or r.get('llm_calls')) for r in rs),
                       'completed_searches':sum(r['status']=='complete' for r in rs),
                       **{k:sum(r.get(k,0) for r in rs) for k in keys}})

meta={'snapshot_utc':dt.datetime.fromtimestamp(NOW,dt.timezone.utc).isoformat(),
      'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'scope':'Amazon task experiments requested in this conversation; excludes unrelated Movies-only/KuaiRand projects, preflight, copied archives and this assistant conversation',
      'run_count':len(rows),'current_family_cells':len(current),'source_cohorts':NAMES,
      'token_accounting':'sum usage.input_tokens + usage.output_tokens from CLI turn.completed/turn.failed events; cached_input_tokens and reasoning_output_tokens are displayed separately and never added a second time',
      'token_coverage':'all attempts with reported usage count, including later failed schema checks; missing usage is not estimated and totals may be lower bounds; live calls are incomplete at snapshot',
      'usage_deduplication':'canonical actual run roots only; (thread_id,turn_index) deduplicates repeated usage events across all roots',
      'single_model_policy':'historical posthoc single-model results are displayed in selected/test columns when available; raw ensemble-allowed search results remain in raw_original columns, without counting their tokens twice',
      'gpu_accounting':'settled search ledger plus separately labelled active elapsed estimates; preflight and final inference excluded from search; imported prior-protocol charges removed only in fresh_settled totals',
      'results_are_snapshots':True,'errors':errors}
dump('metadata.json',meta);dump('all_runs.json',rows);dump('current_comparison.json',current)
dump('method_totals.json',totals);dump('token_events.json',token_events)
csvwrite('all_runs.csv',rows);csvwrite('current_comparison.csv',current)
csvwrite('method_totals.csv',totals);csvwrite('per_dataset_metrics.csv',per_dataset)
csvwrite('llm_attempts.csv',attempts);csvwrite('token_events.csv',token_events);csvwrite('trials.csv',trials)


def fmt(value):
    if value is None:return '—'
    if isinstance(value,float):return f'{value:,.4f}'
    if isinstance(value,(dict,list)):return json.dumps(value,ensure_ascii=False)
    return str(value)


def table(data,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(row.get(key)))+'</td>'
        for key,_ in columns)+'</tr>' for row in data)+'</tbody></table>'

columns=[('family','模型/范围'),('method','方法'),('status','搜索状态'),
    ('best_observed_validation_hr10','最佳已观察验证 HR@10'),('selected_validation_hr10','选定验证 HR@10'),('test_hr10','测试 HR@10'),('test_ndcg10','测试 NDCG@10'),
    ('settled_gpu_seconds','已结算 GPU 秒'),('active_gpu_seconds_estimate','运行中 GPU 秒估算'),
    ('trial_count','已结束候选'),('successful_trials','成功候选'),('llm_calls','LLM 请求数'),
    ('input_tokens','输入 token'),('cached_input_tokens','其中缓存输入'),('output_tokens','输出 token'),
    ('reasoning_output_tokens','其中推理输出'),('total_tokens','已报告总 token'),('token_status','token 覆盖')]
historical=[r for r in rows if not r['current_family_comparison']]
document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Amazon 全方法实验与 token 明细</title>
<style>body{font:14px/1.6 system-ui;margin:24px;color:#17243b}h1{font-size:25px}h2{margin-top:32px}table{border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}td,th{border:1px solid #d9e0ea;padding:7px 11px;text-align:right}td:first-child,td:nth-child(2),th{text-align:left}thead{background:#eef3fa;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}.scroll{overflow:auto}input{padding:8px;width:380px}p{max-width:1200px}a{color:#2255a4}</style>
<h1>Amazon 全方法实验结果与 token 消耗</h1>'''
document+='<p>快照：'+html.escape(meta['snapshot_utc'])+'。本轮 12 项固定模型对照；历史实验单列。</p>'
document+='<p>总 token = 输入 + 输出。缓存输入、推理输出为单列，不重复相加。包含能读到 usage 的失败/重试；缺失 usage 未估算，运行中记录持续变化。预检和本次助理对话不计入。</p>'
document+='<p>历史“单模型重选”是在原搜索候选中筛选并重评；它保留了原搜索使用集成候选的消耗，不能视为从头禁止集成的独立搜索。129,600 GPU 秒和 TPE-best 起点实验也不与本轮直接合并。</p>'
document+='<p><a href="all_runs.csv">全部实验 CSV</a> · <a href="current_comparison.csv">本轮对照 CSV</a> · <a href="per_dataset_metrics.csv">四类别指标 CSV</a> · <a href="llm_attempts.csv">逐次调用 CSV</a> · <a href="trials.csv">逐候选 CSV</a></p>'
document+='<input id="filter" placeholder="筛选模型、方法、状态或实验编号" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
document+='<h2>本轮固定模型对照</h2><div class="scroll">'+table(current,columns)+'</div>'
document+='<h2>历史实验（含中断、停止和未启动计划）</h2><div class="scroll">'+table(historical,[('run_id','实验编号'),('performance_mode','结果口径')]+columns)+'</div>'
document+='<h2>按方法累计消耗</h2><div class="scroll">'+table(totals,[('scope','汇总范围'),('method','方法'),('runs','计划数'),('completed_searches','完成数'),('input_tokens','输入'),('cached_input_tokens','缓存输入'),('output_tokens','输出'),('reasoning_output_tokens','推理输出'),('total_tokens','总 token'),('llm_attempts_without_usage','无 usage 日志尝试')])+'</div>'
document+='<p>完整字段、来源路径、预算规则、token 缺失情况及去重规则见 metadata.json 和各 CSV。没有测试结果的运行保持空值。</p></html>'
(OUT/'report.html').write_text(document)
(OUT/'export_source.py').write_text(Path(__file__).read_text())
with zipfile.ZipFile(OUT/'results-and-tokens.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.suffix in ['.csv','.json','.html','.py']:z.write(p,p.name)
manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in OUT.iterdir() if p.is_file()}
dump('manifest.json',manifest)
print('REPORT_DIR',OUT,flush=True)
print('CURRENT_ROWS',len(current),flush=True)
print('TOTALS',json.dumps(totals,ensure_ascii=False),flush=True)
print('ERRORS',json.dumps(errors,ensure_ascii=False),flush=True)
