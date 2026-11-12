"""Terminal-native charts: completed validation observations, never test metrics."""
from datetime import datetime,timezone
import math
import time

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console,Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

COLORS={'auto':'cyan','enhanced':'magenta'}
LABELS={'auto':'AUTO','enhanced':'强化版'}
PHASES={'baseline_training':'基线训练','train':'训练','validation':'验证','query':'检索规划','propose':'提出方案 / 编码',
        'analyze':'多轨迹归因','research':'研究调度','completed':'已结束','completed_validation_only':'验证研究已结束',
        'paired_validation_repetitions':'验证集种子重复','paired_repetitions':'种子重复','closed_test_assessment':'封闭评估（分数隐藏）',
        'failed':'异常退出','failed_before_assessment':'准入故障退出','initializing_restart':'重启准备'}
ROLES={'reference':'基线','candidate':'候选','control':'对照'}
WRAP_CONSOLE=Console(force_terminal=False)


def number(value,digits=2):
    if value is None:return '—'
    for unit,scale in [('G',1e9),('M',1e6),('K',1e3)]:
        if abs(value)>=scale:return f'{value/scale:.{digits}f}{unit}'
    return f'{value:.0f}'


def duration(seconds):
    if seconds is None:return '—'
    seconds=max(0,int(seconds));hours,rest=divmod(seconds,3600);minutes,seconds=divmod(rest,60)
    if hours:return f'{hours}h{minutes:02d}m'
    return f'{minutes}m{seconds:02d}s'


def utc(timestamp,full=False):
    return datetime.fromtimestamp(timestamp,timezone.utc).strftime('%m-%d %H:%M UTC' if full else '%H:%M:%S')


def short(value,length=100):
    text=str(value or '暂无')
    return text if len(text)<=length else text[:max(0,length-1)]+'…'


def paragraph(value,width,lines=3,style=None):
    text=Text(str(value or '尚未生成；正在执行当前阶段。'),style=style)
    wrapped=text.wrap(WRAP_CONSOLE,max(12,width),overflow='fold',no_wrap=False)
    chosen=list(wrapped[:lines])
    if len(wrapped)>lines and chosen:chosen[-1].append(' …',style='dim')
    return Group(*chosen)


def delta(value):
    if value is None:return Text('—',style='dim')
    return Text(f'{value:+.6f}',style='green' if value>0 else 'red' if value<0 else 'dim')


def meter(value,width=12,color='cyan'):
    value=max(0,min(1,value or 0));filled=round(width*value)
    text=Text('━'*filled,style=color);text.append('━'*(width-filled),style='grey30')
    return text


def chart(curves,width,height,start,end):
    """A staircase incumbent curve; blank before the first completed trial."""
    width=max(16,width-12);height=max(5,height)
    values=[point['auc'] for series in curves.values() for point in series]
    if not values:return Align.center(Text('等待第一条完成的全量验证结果',style='dim'),vertical='middle',height=height+2)
    low=min(values);high=max(values)
    padding=max(0.0002,(high-low)*0.1);low-=padding;high+=padding
    if high-low<0.001:low-=0.0005;high+=0.0005
    grid=[[(' ',None) for _ in range(width)] for _ in range(height)]
    for mode,series in curves.items():
        series=sorted(series,key=lambda point:point['time']);cursor=-1;previous_y=None
        for x in range(width):
            now=start+(end-start)*x/max(1,width-1)
            while cursor+1<len(series) and series[cursor+1]['time']<=now:cursor+=1
            if cursor<0:continue
            y=min(height-1,max(0,round((high-series[cursor]['auc'])/(high-low)*(height-1))))
            if previous_y is not None and previous_y!=y:
                for middle in range(min(y,previous_y)+1,max(y,previous_y)):
                    grid[middle][x]=('│',COLORS[mode])
            grid[y][x]=('◆','white') if grid[y][x][0]!=' ' else ('━',COLORS[mode])
            previous_y=y
    lines=[]
    for y,row in enumerate(grid):
        value=high-(high-low)*y/max(1,height-1)
        line=Text(f'{value:.5f} ┤' if y in {0,height//2,height-1} else '        │',style='dim')
        for char,style in row:line.append(char,style=style)
        lines.append(line)
    hours=max(0,(end-start)/3600)
    lines.append(Text('        └'+'─'*width,style='grey50'))
    suffix=f'{hours:.2f}h'
    lines.append(Text('         0h'+' '*max(1,width-len(suffix)-3)+suffix,style='dim'))
    return Group(*lines)


def scatter(arms,width,height=12,parameter_axis=False):
    points=[(mode,row) for mode,arm in arms.items() for row in arm['trials'] if row.get('dense_parameters')]
    if not points:return Text('等待完成的全量训练。',style='dim')
    columns=max(20,width-13);rows=max(6,height)
    def xpos(row):return math.log10(max(1,row['dense_parameters'])) if parameter_axis else row['wall_seconds']/60
    xs=[xpos(row) for _,row in points];ys=[row['auc'] for _,row in points]
    lo_x=min(xs);hi_x=max(xs);lo_y=min(ys);hi_y=max(ys)
    dx=max(0.15,(hi_x-lo_x)*0.08);dy=max(0.0003,(hi_y-lo_y)*0.08)
    lo_x-=dx;hi_x+=dx;lo_y-=dy;hi_y+=dy
    cells=[[(' ',None) for _ in range(columns)] for _ in range(rows)]
    for mode,row in points:
        x=round((xpos(row)-lo_x)/(hi_x-lo_x)*(columns-1));y=round((hi_y-row['auc'])/(hi_y-lo_y)*(rows-1))
        marker={'candidate':'●','control':'+','reference':'□'}.get(row['role'],'·')
        cells[y][x]=('◆','white') if cells[y][x][0]!=' ' else (marker,COLORS[mode])
    lines=[]
    for y,row in enumerate(cells):
        line=Text(f'{hi_y-(hi_y-lo_y)*y/(rows-1):.5f} ┤' if y in {0,rows//2,rows-1} else '        │',style='dim')
        for char,style in row:line.append(char,style=style)
        lines.append(line)
    left=number(10**lo_x) if parameter_axis else f'{max(0,lo_x):.1f}m'
    right=number(10**hi_x) if parameter_axis else f'{hi_x:.1f}m'
    lines += [Text('        └'+'─'*columns,style='grey50'),Text('         '+left+' '*max(1,columns-len(left)-len(right))+right,style='dim')]
    return Group(*lines)


def campaign_banner(snapshot):
    campaign=snapshot['campaign'];now=time.time()
    parts=Text('CTR 双版本自主研究',style='bold white')
    parts.append('  '+utc(now)+' UTC',style='dim')
    age=now-snapshot['updated_unix']
    if age>15:parts.append(f'  采样延迟 {age:.0f}s',style='bold red')
    if campaign['recovery']:parts.append('  AUTO 已重跑 · 独立测试比较受影响',style='bold yellow')
    return parts


def legend():
    text=Text('AUTO',style='cyan');text.append('  强化版',style='magenta')
    text.append('  ◆ 重叠点  · 只含已提交的全量验证结果',style='dim')
    return text


def overview(snapshot,width,height):
    campaign=snapshot['campaign'];now=time.time();elapsed=max(0,min(now,campaign['deadline'])-campaign['start'])
    total=max(1,campaign['deadline']-campaign['start']);fraction=min(1,elapsed/total)
    budget=Table.grid(expand=True);budget.add_column(ratio=1);budget.add_column(justify='right')
    budget.add_row(ProgressBar(total=1,completed=fraction,width=None,complete_style='blue',finished_style='blue'),
                   f' {fraction:.1%}  剩余 {duration(campaign["deadline"]-now)}')
    table=Table(box=box.SIMPLE_HEAVY,expand=True,padding=(0,1))
    for name in ['版本 / 阶段','完成','最佳验证 AUC','较 DIN']:
        table.add_column(name,justify='left' if name.startswith('版本') else 'right',no_wrap=True)
    for mode,arm in snapshot['arms'].items():
        best=arm['best'];baseline=arm['baseline']
        table.add_row(Text(LABELS[mode]+' · '+PHASES.get(arm['phase'],arm['phase']),style=COLORS[mode]),
                      str(arm['counts']['success']),f"{best['auc']:.6f}" if best else '训练中',
                      delta(best['auc']-baseline['auc']) if best and baseline else Text('—'))
    curves={mode:arm['curve'] for mode,arm in snapshot['arms'].items()}
    end=max(campaign['start']+300,min(now,campaign['deadline']))
    plot_height=max(5,min(12,height-18))
    items=[campaign_banner(snapshot),budget,Text('截止 '+utc(campaign['deadline'],True)+'  · '+campaign['model']+' / '+campaign['effort'],style='dim'),
           table,Text('最佳验证 AUC 随累计时间变化（纵轴缩放）',style='bold'),
           chart(curves,width-4,plot_height,campaign['start'],end),legend(),
           Text('单种子观察；时间含准备、失败及重启；两版各 4 GPU。',style='dim')]
    return Panel(Group(*items),title='01 · 研究总览',border_style='blue',padding=(0,1))


def resources(snapshot,width,height,history):
    resource=snapshot['resources'];cpu=resource.get('cpu_percent');memory=resource.get('memory_used',0);total=resource.get('memory_total',1)
    header=Text(f"主机 CPU {cpu:.1f}%" if cpu is not None else '主机 CPU 采样中',style='bold')
    header.append(f'  RAM {memory/2**30:.0f}/{total/2**30:.0f} GiB',style='dim')
    gpu=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    labels=['GPU','版本','利用率','显存 GiB','功耗 / 温度']
    if width>=90:labels.append('近 5 分钟')
    for label in labels:gpu.add_column(label,no_wrap=True)
    for row in resource.get('gpus',[]):
        mode=next((name for name,indices in snapshot['campaign']['gpus'].items() if row['index'] in indices),None)
        util=row.get('utilization') or 0
        usage=Text.assemble(meter(util/100,8,COLORS.get(mode,'grey50')),f' {util:.0f}%')
        cells=[str(row['index']),Text(LABELS.get(mode,'未分配'),style=COLORS.get(mode,'grey50')),usage,
               f"{(row.get('memory_used_mib') or 0)/1024:.1f}/{(row.get('memory_total_mib') or 0)/1024:.0f}",
               f"{row.get('power_w') or 0:.0f}W / {row.get('temperature') or 0:.0f}°C"]
        if width>=90:
            samples=[item.get('gpu_utilization',{}).get(str(row['index'])) for item in history[-60:]]
            samples=[value for value in samples if value is not None]
            step=max(1,math.ceil(len(samples)/12));levels='▁▂▃▄▅▆▇█'
            cells.append(Text(''.join(levels[min(7,max(0,round(sum(samples[i:i+step])/len(samples[i:i+step])/100*7)))] for i in range(0,len(samples),step)) or '采样中',style=COLORS.get(mode,'grey50')))
        gpu.add_row(*cells)
    llm=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    for label in ['LLM','调用','输入 / 缓存','输出']:llm.add_column(label,no_wrap=True)
    for mode,arm in snapshot['arms'].items():
        u=arm['usage'];llm.add_row(Text(LABELS[mode],style=COLORS[mode]),str(u['calls']),
                                 number(u['input_tokens'])+' / '+number(u['cached_input_tokens']),number(u['output_tokens']))
    prior=snapshot['campaign']['prior_auto_costs']
    note=Text(f"旧 AUTO：{prior.get('researcher_calls',0)} 次调用保留计费；缓存已包含在输入中。",style='dim')
    health=[]
    for mode,arm in snapshot['arms'].items():
        terminal=arm['status'].get('phase','').startswith(('completed','failed'))
        health.append(Text(LABELS[mode]+(' ● 存活' if arm['worker_alive'] else ' ○ 已结束' if terminal else ' ! 进程缺失'),style=COLORS[mode] if arm['worker_alive'] else 'dim' if terminal else 'red'))
    return Panel(Group(header,gpu,llm,note,Text('费用未计价；进行中的调用用量待结算。',style='dim'),Columns(health)),
                 title='02 · 资源与调用成本',border_style='cyan',padding=(0,1))


def active_table(arm):
    table=Table(box=None,expand=True,padding=(0,1));table.add_column('模型',ratio=2,overflow='ellipsis');table.add_column('训练进度',ratio=2);table.add_column('已用',justify='right')
    for row in arm['active_jobs'][:4]:
        progress=Text.assemble(meter(row['fraction'],12,COLORS[arm['mode']]),f" {row['fraction']:.0%}")
        table.add_row(row['id'],progress,duration(row['elapsed']))
    if not arm['active_jobs']:
        message='模型调用中 · 日志 '+duration(arm['agent_log_age'])+' 前更新' if arm['agent_log_age'] is not None else '等待当前阶段产物'
        table.add_row(message,'','')
    return table


def experiments(snapshot,width,height):
    items=[]
    for mode,arm in snapshot['arms'].items():
        counts=arm['counts'];title=Text(LABELS[mode],style='bold '+COLORS[mode])
        title.append('  '+(arm['round'] or '起点')+' · '+PHASES.get(arm['phase'],arm['phase']),style='white')
        items += [title,active_table(arm)]
        items.append(Text(f"候选 {counts['candidate']} / 对照 {counts['control']} / 基线 {counts['reference']}  · 训练错 {counts['failed_training']} / 准入错 {counts['admission_errors']}",style='dim'))
    latest=sorted([(mode,row) for mode,arm in snapshot['arms'].items() for row in arm['trials']],key=lambda pair:pair[1]['finished_unix'],reverse=True)
    table=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    for name in ['最近完成','角色','验证 AUC','相对父模型']:table.add_column(name,no_wrap=name!='最近完成',overflow='ellipsis')
    for mode,row in latest[:max(3,min(6,height-18))]:
        table.add_row(Text(row['id'],style=COLORS[mode]),ROLES.get(row['role'],row['role']),f"{row['auc']:.6f}",delta(row['parent_delta']))
    items.append(table)
    items.append(Text('进度按冻结的最大训练步数估计；准入后才计入效果曲线。',style='dim'))
    return Panel(Group(*items),title='03 · 正在做什么 / 实验队列',border_style='green',padding=(0,1))


def events(snapshot,width,height):
    items=[]
    if snapshot['campaign']['recovery']:
        items.append(Text('协议事件：AUTO 已修复重跑；后续仅验证集，测试分数隐藏。',style='bold yellow'))
    for mode,arm in snapshot['arms'].items():
        items.append(Text(f"{LABELS[mode]}  场景记忆 {arm['knowledge']['memory_records']} 条  · 最近工作 {arm['nearest_work_count']} 项",style=COLORS[mode]))
    items.append(Text('行业初始 5 项 · 个人知识关闭 · 记忆不等于已确认机制',style='dim'))
    table=Table(box=None,expand=True,padding=(0,1));table.add_column('UTC',width=8);table.add_column('版本',width=7);table.add_column('事件',ratio=1,overflow='ellipsis',no_wrap=True)
    for event in snapshot['events'][-max(4,height-11):]:
        table.add_row(utc(event['at']),Text(LABELS[event['mode']],style=COLORS[event['mode']]),
                      Text(event['text'],style='red' if event['type']=='failure' else 'white'))
    items.append(table)
    if snapshot['read_errors']:items.append(Text('部分日志读取延迟：'+', '.join(snapshot['read_errors'][:2]),style='yellow'))
    items.append(Text('切换窗口 C-b n / p · 放大当前面板 C-b z · 仅观察，不控制训练',style='dim'))
    return Panel(Group(*items),title='04 · 证据与事件时间线',border_style='yellow',padding=(0,1))


def research(snapshot,width,height,mode):
    arm=snapshot['arms'][mode];study=arm['study'];counts=arm['counts']
    items=[Text(LABELS[mode]+' · '+PHASES.get(arm['phase'],arm['phase'])+' · '+(arm['round'] or '起点'),style='bold '+COLORS[mode]),
           active_table(arm),Text('当前研究问题',style='bold'),paragraph(study['question'],width-6,5),
           Text('机制假设',style='bold'),paragraph(study['hypothesis'],width-6,4),
           Text('与相近工作的差异',style='bold'),paragraph(study['proposed_difference'],width-6,4),
           Text('什么结果会否定它',style='bold'),paragraph(study['falsifiers'],width-6,3)]
    table=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    for label in ['最新完整实验组','角色','AUC','较父模型','稠密参数']:table.add_column(label,overflow='ellipsis',no_wrap=label!='最新完整实验组')
    for row in arm['round_results'][-5:]:table.add_row(row['id'],ROLES.get(row['role'],row['role']),f"{row['auc']:.6f}",delta(row['parent_delta']),number(row['dense_parameters']))
    items += [table,Text('最近归因 / 下一步',style='bold'),paragraph(arm['analysis'].get('next_action'),width-6,4),
              Text(f"已提交 {counts['success']}  · 候选 {counts['candidate']} / 对照 {counts['control']}  · 前瞻对照 {arm['contrast_count']} 项",style='dim'),
              Text('检索 '+short(', '.join(str(x) for x in arm['knowledge']['retrieved_ids']),max(25,width-10)),style='dim'),
              Text('机制解释按证据成熟度保留；单种子差异不构成确认。',style='yellow')]
    return Panel(Group(*items),title='研究轨迹 · '+LABELS[mode],border_style=COLORS[mode],padding=(0,1))


def costs(snapshot,width,height):
    half=max(36,width//2-4)
    plots=Columns([Panel(scatter(snapshot['arms'],half-4,12),title='验证 AUC × 单次作业分钟',width=half,border_style='blue'),
                   Panel(scatter(snapshot['arms'],half-4,12,True),title='验证 AUC × 稠密参数（log 横轴）',width=half,border_style='magenta')],expand=True,equal=True)
    table=Table(box=box.SIMPLE_HEAVY,expand=True,padding=(0,1))
    for name in ['版本','已提交作业 GPU-min','LLM 调用时长','未缓存输入','缓存输入','输出','历史准入失败']:table.add_column(name,justify='right')
    for mode,arm in snapshot['arms'].items():
        usage=arm['usage'];gpu_seconds=sum(row['gpu_seconds'] for row in arm['trials'])
        table.add_row(Text(LABELS[mode],style=COLORS[mode]),f'{gpu_seconds/60:.1f}',duration(usage['seconds']),number(usage['uncached_input_tokens']),
                      number(usage['cached_input_tokens']),number(usage['output_tokens']),str(arm['counts']['admission_errors']))
    frontier=[]
    for mode,arm in snapshot['arms'].items():
        for row in arm['trials']:
            dominated=any(other['wall_seconds']<=row['wall_seconds'] and other['auc']>=row['auc'] and
                          (other['wall_seconds']<row['wall_seconds'] or other['auc']>row['auc']) for other in arm['trials'])
            if not dominated:frontier.append((mode,row))
    models=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    for name in ['各版观察到的效果—成本前沿','角色','AUC','Logloss','总 / 稠密参数','训练','推理吞吐/s','批 p95 ms']:models.add_column(name,overflow='ellipsis',no_wrap=True)
    for mode,row in sorted(frontier,key=lambda pair:pair[1]['auc'],reverse=True)[:max(4,min(10,height-33))]:
        models.add_row(Text(row['id'],style=COLORS[mode]),ROLES.get(row['role'],row['role']),f"{row['auc']:.6f}",f"{row['logloss']:.6f}",
                       number(row['parameters'])+' / '+number(row['dense_parameters']),duration(row['train_seconds']),number(row['throughput']),f"{row['latency_p95']:.1f}" if row['latency_p95'] is not None else '—')
    prior=snapshot['campaign']['prior_auto_costs']
    notes=Text(f"旧 AUTO 开销另存：{duration(prior.get('training_wall_seconds'))} 基线作业，{prior.get('researcher_calls',0)} 次调用；未从 24h 时钟扣除。\n",style='dim')
    notes.append('□ 基线  ● 候选  + 对照  ◆ 重叠。前沿仅是当前已观察结果，不是统计优势。\n',style='dim')
    notes.append('吞吐含现有特征构造与评分流程；p95 是 4096 样本整批耗时。缓存和推理 token 不重复加总，费用未知。',style='dim')
    return Panel(Group(campaign_banner(snapshot),plots,legend(),table,models,notes),title='效果 / 计算成本 / 容量',border_style='cyan',padding=(0,1))


def slices(snapshot,width,height,mode):
    arm=snapshot['arms'][mode];best=arm['best'];baseline=arm['baseline']
    if not best or not baseline:return Panel('完成 DIN 基线后展示探索性切片。',title=LABELS[mode],border_style=COLORS[mode])
    keys=[('empty_history','空历史'),('history_1_16','历史 1–16'),('history_17_64','历史 17–64'),('history_gt64','历史 >64'),
          ('category_match_present','有类目匹配'),('category_match_absent','无类目匹配'),
          ('past_cart_present','有加购'),('past_fav_present','有收藏'),('past_buy_present','有购买'),
          ('user_train_frequency_0_1','训练未见用户'),('user_train_frequency_2_10','低频用户 2–9'),
          ('user_train_frequency_100_4294967296','高频用户 ≥100')]
    rows=[]
    for key,label in keys:
        a=best['slices'].get(key,{});b=baseline['slices'].get(key,{})
        if a.get('auc') is not None and b.get('auc') is not None:rows.append((label,a['auc']-b['auc'],a))
    scale=max([abs(value) for _,value,_ in rows]+[0.0001]);half=max(5,min(15,(width-51)//2))
    table=Table(box=box.SIMPLE,expand=True,padding=(0,1))
    for label in ['群体','样本 / 正例','相对 DIN 的 AUC 变化','Δ AUC']:table.add_column(label,no_wrap=True)
    for label,value,row in rows:
        count=round(abs(value)/scale*half)
        bar=Text(' '*(half-count)+'━'*count if value<0 else ' '*half,style='red')
        bar.append('│',style='grey50');bar.append('━'*count+' '*(half-count) if value>=0 else ' '*half,style='green')
        table.add_row(label,number(row['samples'])+' / '+number(row['positives']),bar,delta(value))
    items=[Text('当前最佳：'+best['id'],style='bold '+COLORS[mode]),
           Text(f"验证 AUC {best['auc']:.6f}  · DIN {baseline['auc']:.6f}  · seed {best['seed']}",style='dim'),table,
           Text('同协议、同种子的探索性切片；群体并不互斥。',style='yellow'),
           Text('没有误差条时不声明显著性；切片分数不替代机制对照。',style='yellow'),
           Text('当前最佳模型已完成的 epoch（训练损失按 epoch 记录）',style='bold')]
    for epoch in best['epoch_curve']:
        if epoch.get('auc') is not None:items.append(Text(f"epoch {epoch['epoch']}  train loss {epoch['training_loss']:.6f}  validation AUC {epoch['auc']:.6f}",style='dim'))
    return Panel(Group(*items),title=LABELS[mode]+' · 哪些群体受益 / 受损',border_style=COLORS[mode],padding=(0,1))


def render(view,snapshot,width,height,history=None):
    history=history or []
    if view=='overview':return overview(snapshot,width,height)
    if view=='resources':return resources(snapshot,width,height,history)
    if view=='experiments':return experiments(snapshot,width,height)
    if view=='events':return events(snapshot,width,height)
    if view=='costs':return costs(snapshot,width,height)
    if view.startswith('research-'):return research(snapshot,width,height,view.split('-',1)[1])
    if view.startswith('slices-'):return slices(snapshot,width,height,view.split('-',1)[1])
    raise ValueError('Unknown dashboard view')
