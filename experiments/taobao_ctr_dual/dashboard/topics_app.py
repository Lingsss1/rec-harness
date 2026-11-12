"""Read-only terminal observer for the GPU topic campaign."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app import atomic
from collect import Collector, eligible_trial, incumbent_curve, normalized_trial, process_alive, safe_text, tail_records, usage_summary
from render import COLORS, PHASES, chart, delta, duration, meter, number, paragraph, utc

PALETTE=['cyan','magenta','yellow','green','blue','red','bright_cyan','bright_magenta']
PHASES.update(topic_decision='课题决策',ready='就绪',training='训练',research_stopped='已停止',cancelled='已取消',
              budget_insufficient='预算不足',topic_research='课题研究中',topic_research_stopped='研究结束',
              visibility_choice_pending='待选择历史反馈口径',waiting_for_dataset='下载数据中',
              data_audit='数据核查',din_calibration='DIN 成本校准',din_baseline='DIN 基线训练',setup_failed='准备失败')


def metric_label(s):
    return '四任务平均 AUC' if s['protocol'].get('selection_policy',{}).get('primary')=='macro_auc' else '点击 AUC'


def best_row(rows):
    return min(rows,key=lambda row:(-row['auc'],row['logloss'],row.get('parameters') if row.get('parameters') is not None else math.inf,row['id']),default=None)


class TopicCollector(Collector):
    def snapshot(self):
        self.reader.errors=[];reader=self.reader;root=self.root;now=time.time()
        budget=reader.json(root/'budget.json');protocol=reader.json(root/'protocol.json')
        manifest=reader.json(root/'topic-runtime-manifest.json')
        reference_rows=reader.json(root/'references.json').get('trials',[])
        references=[normalized_trial(row,reader) for row in reference_rows if eligible_trial(row,protocol.get('data_identity'),protocol.get('selection_policy'))]
        baseline=best_row(references)
        anchor=baseline['auc'] if baseline else None
        audit=reader.json(root/'prepared/host-label-audit.json')
        train_samples=sum(row['samples'] for row in audit.get('daily_counts',[])[:6])
        if not train_samples:train_samples=reader.json(root/'data/Taobao_Action/meta_data.json').get('sample_size',{}).get('train',0)
        steps_total=math.ceil(train_samples/protocol.get('batch_size',4096))*protocol.get('epochs',3)
        lanes=[];events=[]
        planned_topics=reader.json(root/'topics.json').get('topics',[])
        board=reader.json(root/'topic-board.json')
        for lane_index,gpu in enumerate(budget.get('gpus',[])):
            folder=root/'topics'/f'gpu-{gpu}';state=reader.json(folder/'state.json')
            topic=next((t for t in state.get('topics',[]) if t['id']==state.get('active_topic_id')), {})
            if not topic and lane_index<len(planned_topics):topic={**planned_topics[lane_index],'status':'planned'}
            trials=[]
            for row in state.get('trials',[]):
                if eligible_trial(row,protocol.get('data_identity'),protocol.get('selection_policy')):
                    row={**normalized_trial(row,reader),'topic_id':row.get('topic_id'),'gpu':gpu}
                    trials.append(row)
            own=[row for row in trials if row['topic_id']==topic.get('id')]
            best=best_row(own)
            rounds=sorted(folder.glob('rounds/r*'));current=rounds[-1] if rounds else folder/'rounds/r0000'
            proposal=reader.json(current/'propose/proposal.json');study=proposal.get('study',{})
            live_agent=None
            for stage in ['decision','propose']:
                path=current/stage
                if process_alive(reader.json(path/'agent-process.json')):
                    log=path/'codex.jsonl';items=tail_records(log,4,32768)
                    item=items[-1] if items else {}
                    live_agent={'stage':stage,'log_age':now-log.stat().st_mtime if log.exists() else None,
                                'event':str(item.get('type','starting'))+' / '+str(item.get('item',{}).get('type',''))}
                    break
            active=[]
            for path in folder.glob('trials/*/process.json'):
                process=reader.json(path)
                if process_alive(process):
                    progress=reader.json(path.parent/'progress.json');steps=progress.get('steps',0)
                    job_steps=progress.get('total_steps',steps_total)
                    active.append({'id':path.parent.name,'phase':progress.get('phase','starting'),
                        'epoch':progress.get('epoch'),'steps':steps,'steps_total':job_steps,
                        'fraction':min(1,steps/job_steps) if job_steps else 0,
                        'elapsed':now-process.get('started_unix',now),'auc':progress.get('best_auc')})
            failures=tail_records(folder/'failures.jsonl',8)+tail_records(folder/'decision-failures.jsonl',8)
            for item in failures:
                events.append({'at':item.get('at',0),'gpu':gpu,'type':'失败','text':safe_text(item.get('error',''),300)})
            for decision in state.get('decisions',[]):
                events.append({'at':decision.get('at_unix',0),'gpu':gpu,'type':decision.get('action','?'),
                               'text':safe_text(decision.get('reason',''),600),'refs':decision.get('evidence_refs',[]),
                               'topic_id':decision.get('resulting_topic_id'),'sources':decision.get('source_topic_ids',[])})
            for row in trials[-10:]:
                events.append({'at':row['finished_unix'],'gpu':gpu,'type':row['role'],
                               'text':row['id']+f" AUC={row['auc']:.6f}"})
            completed=state.get('completed_research_rounds',0);interval=manifest.get('decision_every',budget.get('decision_every',6))
            usage=usage_summary(state.get('agent_calls',[]))
            curve=incumbent_curve(trials)
            if baseline:
                high=anchor;curve=[{'time':budget.get('start_unix') or now,'auc':anchor,'trial':baseline['id']}]+curve
                for point in curve:high=max(high,point['auc']);point['auc']=high
            lanes.append({'gpu':gpu,'phase':state.get('phase','starting'),'topic':topic,
                          'completed_rounds':completed,'next_decision_round':(completed//interval+1)*interval,
                          'decision_every':interval,'trials':trials,'best':best,'curve':curve,
                          'success':len(trials),'failed':sum(r.get('status')=='failed' for r in state.get('trials',[])),
                          'attempts':len(state.get('trials',[])),'study':{k:safe_text(study.get(k,''),1600) for k in
                              ['question','hypothesis','proposed_difference','predictions','falsifiers']},
                          'variants':[{k:row.get(k) for k in ['name','role','note']} for row in proposal.get('variants',[])],
                          'active_jobs':active,'live_agent':live_agent,'usage':usage,'failures':failures[-3:],
                          'job_seconds':sum(r.get('wall_seconds',0) for r in state.get('trials',[])),
                          'knowledge_usage':reader.json(current/'knowledge-usage.json'),
                          'decisions':state.get('decisions',[])})
        preparation=[]
        for folder in [root/'calibration',root/'baselines/din']:
            process=reader.json(folder/'process.json')
            if process_alive(process):preparation.append({'id':folder.name,'progress':reader.json(folder/'progress.json'),
                'elapsed':now-process.get('started_unix',now)})
        return {'schema':'ctr-topic-dashboard-v1','updated_unix':now,'root':str(root),
                'budget':budget,'protocol':protocol,'status':reader.json(root/'status.json'),
                'setup':reader.json(root/'setup.json'),'acquisition':reader.json(root/'acquisition-status.json'),
                'selection_rule_update':reader.json(root/'selection-rule-update.json'),
                'preparation_jobs':preparation,
                'runner_alive':process_alive(reader.json(root/'runner-process.json')),
                'references':references,'baseline':baseline,'lanes':lanes,'topics':board.get('topics',[]),
                'events':sorted(events,key=lambda item:item['at'])[-40:],
                'resources':self.resources(),'read_errors':reader.errors,
                'measurement_scope':'Committed full-protocol validation only; no assessment or test score files read.'}


def table(*columns):
    result=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    for column in columns:result.add_column(column)
    return result


def text(value,limit=300):return Text(safe_text(value,limit))
def score(value):return '—' if value is None else f'{value:.6f}'
def phase(value):return PHASES.get(value,value)


def overview(s,width,height):
    budget=s['budget'];now=s['updated_unix'];start=budget.get('start_unix');deadline=budget.get('deadline_unix')
    hours=budget.get('seconds',86400)/3600
    waiting=start is None or deadline is None
    if waiting:
        header=Text(f"{phase(s['status'].get('phase','starting'))} · {hours:g}h 预算待数据下载校验完成后开始\n")
    else:header=Text(f"{phase(s['status'].get('phase','starting'))} · 已用 {duration(now-start)} / 剩余 {duration(deadline-now)}\n")
    header.append(f"{budget.get('model')} / {budget.get('reasoning_effort')} · 每 {budget.get('decision_every',6)} 轮决策 · 验证集 · 个人知识禁用\n",style='dim')
    header.append(f"截止 {'待数据就绪' if waiting else utc(deadline,True)} · 运行器 {'存活' if s['runner_alive'] else '未运行'}")
    setup=s.get('setup',{})
    if setup.get('baseline')=='din':
        header.append('\nUniRank Taobao · 统一建模研究 · 初始参照 DIN · '+metric_label(s),style='cyan')
        acquisition=s.get('acquisition',{})
        if acquisition.get('completed_lfs_files') is not None:
            header.append(f"\n官方数据文件 {acquisition['completed_lfs_files']}/{acquisition.get('total_lfs_files','?')}",style='dim')
        elif acquisition.get('phase')=='downloaded_and_verified':header.append('\n官方数据已下载并校验',style='dim')
    if s['baseline']:header.append(f"\n公共参考 {s['baseline']['id']} · AUC {score(s['baseline']['auc'])}",style='dim')
    if s.get('selection_rule_update') and not s['protocol'].get('selection_policy'):
        header.append('\n新规则已同步四任务平均 AUC；本页保留旧点击目标轨迹，见「多任务审计」窗口。',style='yellow')
    rows=table('GPU / 阶段','当前课题','完成轮','最佳 '+metric_label(s),'Δ公共','Δ进入')
    anchor=s['baseline']['auc'] if s['baseline'] else None
    for lane in s['lanes']:
        best=lane['best'];auc=best['auc'] if best else None;entry=lane['topic'].get('entry_auc')
        rows.add_row(str(lane['gpu'])+' / '+phase(lane['phase']),text(lane['topic'].get('title','初始化'),70),
            f"{lane['completed_rounds']} / 下次{lane['next_decision_round']}",score(auc),
            delta(auc-anchor if auc is not None and anchor is not None else None),
            delta(auc-entry if auc is not None and entry is not None else None))
    curves={f"GPU {lane['gpu']}":lane['curve'] for lane in s['lanes']}
    for lane in s['lanes']:COLORS[f"GPU {lane['gpu']}"]=PALETTE[lane['gpu']%len(PALETTE)]
    legend=Text('历史最佳曲线：')
    for lane in s['lanes']:legend.append(f" GPU {lane['gpu']} ",style=PALETTE[lane['gpu']%len(PALETTE)])
    curve_start=start if start is not None else now
    return Panel(Group(header,rows,legend,chart(curves,width-5,max(5,height-22),curve_start,max(curve_start+1,now)),
        Text('表格为当前课题；曲线为各 GPU 的历史最佳。单种子增量仍是初步证据。',style='dim')),
        title=f'多课题 AUTO · {hours:g}h',border_style='cyan')


def resources(s,width,height):
    rows=table('GPU','利用率','显存 GiB','温度','状态')
    selected={lane['gpu']:lane for lane in s['lanes']}
    for gpu in s['resources'].get('gpus',[]):
        if gpu['index'] not in selected:continue
        lane=selected[gpu['index']]
        rows.add_row(str(gpu['index']),f"{gpu.get('utilization') or 0:.0f}%",f"{(gpu.get('memory_used_mib') or 0)/1024:.1f}",
                     str(gpu.get('temperature') or '—'),phase(lane['phase']))
    jobs=table('GPU','当前进程','进度 / 活动')
    for lane in s['lanes']:
        for job in lane['active_jobs']:
            jobs.add_row(str(lane['gpu']),text(job['id'],50),f"{job['fraction']:.0%} · {job['steps']}/{job['steps_total']} · {duration(job['elapsed'])}")
        agent=lane['live_agent']
        if agent:jobs.add_row(str(lane['gpu']),'LLM '+agent['stage'],text(agent['event']+' · 日志距今 '+duration(agent['log_age']),85))
    for job in s.get('preparation_jobs',[]):
        progress=job['progress']
        jobs.add_row('0',job['id'],f"{progress.get('phase','starting')} · {progress.get('steps',0)} 步 · {duration(job['elapsed'])}")
    resource=s['resources'];cpu=resource.get('cpu_percent')
    detail=Text(f"CPU {'—' if cpu is None else f'{cpu:.0f}%'} · 内存 {(resource.get('memory_used',0)/2**30):.1f} GiB\n")
    detail.append('GPU 在方案生成时空闲属正常阶段；进度仅来自实测训练步骤。',style='dim')
    if s['read_errors']:detail.append('\n采样异常：'+', '.join(s['read_errors']),style='red')
    return Panel(Group(rows,jobs,detail),title='资源与实时活动',border_style='green')


def lane_view(s,gpu,width,height):
    lane=next((lane for lane in s['lanes'] if lane['gpu']==gpu),None)
    if lane is None:return Panel('等待课题状态')
    topic=lane['topic'];study=lane['study'];pieces=[]
    pieces.append(Text(f"{phase(lane['phase'])} · 已完成 {lane['completed_rounds']} 轮 · 下次调度第 {lane['next_decision_round']} 轮\n"
                       f"成功 {lane['success']} / 失败 {lane['failed']} · LLM 已完成 {lane['usage']['calls']} 次"))
    for label,value in [('课题',topic.get('question')),('初始假设',topic.get('hypothesis')),
                         ('当前问题',study.get('question')),('本轮改动',study.get('proposed_difference')),
                         ('预测',study.get('predictions')),('否定条件',study.get('falsifiers'))]:
        if value:pieces.extend([Text(label,style='bold'),paragraph(value,max(16,width-6),2)])
    if lane['variants']:
        pieces.append(Text('计划变体：'+', '.join(f"{v['name']}({v['role']})" for v in lane['variants']),style='yellow'))
    recent=table('最近完成',metric_label(s),'Logloss','耗时')
    for row in lane['trials'][-3:]:recent.add_row(text(row['id'],40),score(row['auc']),score(row['logloss']),duration(row['wall_seconds']))
    pieces.append(recent)
    for job in lane['active_jobs']:pieces.extend([text(job['id']),meter(job['fraction'],max(10,width-16)),Text(f"{job['steps']}/{job['steps_total']} 步 · {duration(job['elapsed'])}")])
    if lane['live_agent']:pieces.append(text('LLM：'+lane['live_agent']['event']))
    if lane['failures']:pieces.append(Text('最近错误：'+safe_text(lane['failures'][-1].get('error',''),180),style='red'))
    return Panel(Group(*pieces),title=f"GPU {gpu} · {topic.get('title','初始化')}",border_style=PALETTE[gpu%len(PALETTE)])


def experiments(s,width,height):
    rows=table('时间 UTC','GPU / 课题','角色 / ID',metric_label(s),'Logloss','训练墙钟','参数')
    trials=sorted([row for lane in s['lanes'] for row in lane['trials']],key=lambda r:r['finished_unix'])
    for row in trials[-max(5,height-7):]:
        rows.add_row(utc(row['finished_unix']),str(row['gpu'])+' / '+str(row['topic_id']),text(row['role']+' '+row['id'],80),
                     score(row['auc']),score(row['logloss']),duration(row['wall_seconds']),number(row['parameters']))
    return Panel(rows,title='已提交全量验证结果 · 训练中指标不参与排名',border_style='cyan')


def decisions(s,width,height):
    rows=table('时间 UTC','GPU / 动作','理由与证据')
    for event in [e for e in s['events'] if e['type'] in ['continue','switch','merge','失败']][-max(4,height//4):]:
        detail=event['text']
        if event.get('refs'):detail+='\n证据：'+', '.join(event['refs'])
        rows.add_row(utc(event['at']),str(event['gpu'])+' / '+event['type'],text(detail,850))
    note=Text('默认每完成 6 轮才检查一次。未发生决策时此处为空。\n',style='dim')
    note.append('组合收益与较强来源比较；来源单次涨分不代表机制已证实。',style='dim')
    return Panel(Group(note,rows),title='继续 / 切换 / 合并 · 决策及失败记录',border_style='yellow')


def costs(s,width,height):
    rows=table('GPU','候选/对照尝试','训练作业墙钟','LLM 次/失败','LLM 耗时','输入/缓存/输出 token')
    for lane in s['lanes']:
        usage=lane['usage']
        rows.add_row(str(lane['gpu']),str(lane['attempts']),duration(lane['job_seconds']),f"{usage['calls']}/{usage['failed_calls']}",
                     duration(usage['seconds']),f"{number(usage['input_tokens'])} / {number(usage['cached_input_tokens'])} / {number(usage['output_tokens'])}")
    models=table('课题',metric_label(s),'Logloss','参数量','训练墙钟','验证吞吐/s','整批 p95 ms')
    for lane in s['lanes']:
        row=lane['best']
        if row:models.add_row(text(lane['topic'].get('title'),40),score(row['auc']),score(row['logloss']),number(row['parameters']),
            duration(row['wall_seconds']),number(row['throughput']),score(row['latency_p95']))
    refs=table('公共参考',metric_label(s),'Logloss','墙钟','参数')
    for row in s['references']:refs.add_row(row['id'],score(row['auc']),score(row['logloss']),duration(row['wall_seconds']),number(row['parameters']))
    budget=s['budget'];hours=budget.get('seconds',86400)/3600
    scope=f'复用成本单列；本次 {hours:g}h 覆盖全部新调用、训练、对照与失败。'
    if budget.get('start_event')=='dataset_download_and_checksum_verification_completed':
        before=budget.get('pre_budget',{}).get('wall_seconds')
        scope+=f"\n从数据校验完成时计时；此前准备耗时 {duration(before) if before is not None else '待记录'}，单独记账。"
    note=Text(scope+'\nLLM 成本统计已结束调用；金额未计价。输入已含缓存，输出已含推理，不重复相加。单种子结果不表示显著性。',style='dim')
    tasks=table('模型','任务','AUC','Δ参考 AUC','Logloss','Δ参考 Logloss')
    reference=s['baseline'].get('per_task',{}) if s['baseline'] else {}
    for row in s['references']+[lane['best'] for lane in s['lanes'] if lane['best']]:
        for label,metrics in row.get('per_task',{}).items():
            base=reference.get(label,{})
            d_auc=metrics['auc']-base['auc'] if metrics.get('auc') is not None and base.get('auc') is not None else None
            d_loss=metrics['logloss']-base['logloss'] if metrics.get('logloss') is not None and base.get('logloss') is not None else None
            tasks.add_row(text(row['id'],45),label,score(metrics.get('auc')),delta(d_auc),score(metrics.get('logloss')),delta(d_loss))
    return Panel(Group(rows,models,refs,tasks,note),title='效果与实测成本',border_style='magenta')


def slices(s,width,height):
    groups=['empty_history','history_1_16','history_17_64','history_gt64','past_buy_present','past_buy_absent','category_match_present','category_match_absent']
    rows=table('验证切片','样本数','参考 AUC',*[f"GPU {lane['gpu']} ΔAUC" for lane in s['lanes']])
    base=s['baseline'].get('slices',{}) if s['baseline'] else {}
    for name in groups:
        reference=base.get(name,{}).get('auc');values=[]
        for lane in s['lanes']:
            value=lane['best'].get('slices',{}).get(name,{}).get('auc') if lane['best'] else None
            values.append(delta(value-reference if value is not None and reference is not None else None))
        rows.add_row(name,number(base.get(name,{}).get('samples')),score(reference),*values)
    return Panel(Group(rows,Text('各 GPU 当前课题最佳 vs 公共参考；仅验证集观察，未做多重比较校正。',style='dim')),
                 title='历史长度 / 行为 / 匹配条件切片',border_style='cyan')


def render(name,s,width,height):
    if name.startswith('lane-'):return lane_view(s,int(name.split('-')[1]),width,height)
    return {'overview':overview,'resources':resources,'experiments':experiments,'decisions':decisions,
            'costs':costs,'slices':slices}.get(name,overview)(s,width,height)


def sample(root,interval):
    output=root/'dashboard';output.mkdir(exist_ok=True)
    with (output/'sampler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(output/'sampler.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                                    'started_unix':time.time(),'read_only_experiment_observer':True})
        collector=TopicCollector(root);last_history=0
        while True:
            started=time.monotonic()
            try:
                snapshot=collector.snapshot();atomic(output/'snapshot.json',snapshot)
                if time.time()-last_history>=5:
                    record={'at':snapshot['updated_unix'],'lanes':{str(lane['gpu']):{'phase':lane['phase'],
                        'rounds':lane['completed_rounds'],'auc':lane['best']['auc'] if lane['best'] else None} for lane in snapshot['lanes']}}
                    with (output/'telemetry.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    last_history=time.time()
            except Exception as error:
                atomic(output/'sampler-error.json',{'at':time.time(),'error':repr(error)});traceback.print_exc()
            time.sleep(max(.1,interval-(time.monotonic()-started)))


def view(root,name,once=False,width=None,height=None):
    console=Console(width=width,height=height,force_terminal=False if once else None)
    def content():
        path=root/'dashboard/snapshot.json'
        if not path.exists():return Panel('等待多课题采样器…')
        return render(name,json.loads(path.read_text()),console.width,console.height)
    if once:console.print(content());return
    with Live(content(),console=console,screen=True,auto_refresh=False,vertical_overflow='crop') as live:
        while True:
            try:live.update(content(),refresh=True)
            except (OSError,ValueError):pass
            time.sleep(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['sample','view'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--interval',type=float,default=2)
    parser.add_argument('--view',default='overview');parser.add_argument('--once',action='store_true')
    parser.add_argument('--width',type=int);parser.add_argument('--height',type=int)
    args=parser.parse_args()
    try:
        if args.action=='sample':sample(args.root.resolve(),args.interval)
        else:view(args.root.resolve(),args.view,args.once,args.width,args.height)
    except KeyboardInterrupt:pass
