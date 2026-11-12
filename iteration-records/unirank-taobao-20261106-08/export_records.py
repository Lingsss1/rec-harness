"""Export the completed, frozen UniRank run without touching live research code."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


def read(path):return json.loads(path.read_text())
def digest(data):return hashlib.sha256(data).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def export(campaign,output):
    campaign=campaign.resolve();output=output.resolve()
    if (output/'manifest.json').exists():raise ValueError('Export already exists')
    output.mkdir(parents=True,exist_ok=True)
    copies=[];derived=[];omitted=[]
    secret=re.compile(rb'(?:\bsk-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{25,}|\bgithub_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')

    def copy(source,target):
        if not source.is_file():return
        raw=source.read_bytes()
        if secret.search(raw):raise ValueError('Potential secret in allowed export file: '+str(source.relative_to(campaign)))
        dest=output/target;dest.parent.mkdir(parents=True,exist_ok=True)
        if b'\x00' in raw and source.suffix=='.log':
            dest.write_bytes(raw.replace(b'\x00',b''));derived.append({'path':str(target),'source':str(source.relative_to(campaign)),
                'source_sha256':digest(raw),'transformation':'Remove NUL bytes from a shared-memory error traceback; other bytes unchanged.'})
        else:
            dest.write_bytes(raw);copies.append({'path':str(target),'source':str(source.relative_to(campaign)),
                'sha256':digest(raw),'bytes':len(raw)})

    status=read(campaign/'status.json')
    if status['phase'] not in ['topic_research_stopped','topic_research_with_failures','terminated_by_user']:
        raise ValueError('Only completed/stopped experiments can be exported')
    snapshot=read(campaign/'source-snapshot.json')
    for name,expected in snapshot['sources'].items():
        if digest((campaign/'source'/name).read_bytes())!=expected:raise ValueError('Frozen source changed: '+name)

    roots=['setup.json','status.json','budget.json','budget-policy.json','budget-changes.jsonl','protocol.json',
        'references.json','baseline-summary.json','topic-board.json','topic-runtime-manifest.json',
        'source-snapshot.json','dataset-source-manifest.json','data-audit.json','environment.json','environment-freeze.txt',
        'research-scope.json','research-scope-update.json','user-protocol-choice.json','visibility-choice.json',
        'experiment-launch.json','setup-costs.json','readiness-checks.json','knowledge-update.json','knowledge-updates.jsonl','campaign.log']
    for name in roots:copy(campaign/name,Path('protocol')/name)
    copy(campaign/'data/Taobao_Action/meta_data.json',Path('protocol/dataset-meta-data.json'))
    for path in sorted((campaign/'config').glob('*.yaml')):copy(path,Path('protocol/config')/path.name)
    for path in sorted((campaign/'source').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        relative=path.relative_to(campaign/'source')
        if 'source_cache' in relative.parts or path.suffix in ['.pdf','.png','.jpg','.pt']:continue
        if path.suffix not in ['.py','.json','.jsonl','.txt','.yaml','.yml']:continue
        copy(path,Path('source-at-freeze')/relative)
    copy(campaign/'knowledge-assets-v2/manifest.json',Path('knowledge/source-manifest.json'))
    copy(campaign/'upstream/LICENSE',Path('protocol/licenses/UniRank-LICENSE'))
    for folder in [campaign/'baselines',campaign/'calibration',campaign/'setup-attempts']:
        if not folder.exists():continue
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.py','.json','.jsonl','.log'] and '__pycache__' not in path.parts:
                copy(path,Path('evidence')/path.relative_to(campaign))
    for path in sorted((campaign/'final-assessment').rglob('*')):
        if path.is_file() and path.suffix in ['.json','.csv','.txt','.log']:
            copy(path,Path('final-assessment')/path.relative_to(campaign/'final-assessment'))
    for path in sorted((campaign/'best-model').glob('*')):
        if path.is_file() and path.suffix in ['.py','.json','.txt']:
            copy(path,Path('selected-model')/path.name)

    states=[read(campaign/f'topics/gpu-{gpu}/state.json') for gpu in range(4)]
    references=read(campaign/'references.json')['trials']
    all_trials=references+[t for s in states for t in s['trials']]
    catalog={t['id']:t for t in all_trials};iterations=[];calls=[];events=[];lanes=[]
    for state in states:
        gpu=state['gpu'];folder=campaign/f'topics/gpu-{gpu}'
        for path in sorted(folder.glob('*.json*')):copy(path,Path('topics')/f'gpu-{gpu}'/path.name)
        for trial in state['trials']:
            source_dir=Path(trial['output']);target=Path('topics')/f'gpu-{gpu}'/'trials'/trial['id']
            for name in ['model.py','model-config.json','result.json','receipt.json','progress.json',
                         'process.json','process-receipt.json','process.log','failure.json']:
                copy(source_dir/name,target/name)
            source=(source_dir/'model.py').read_text();parent=catalog.get(trial.get('parent'))
            if parent and Path(parent['model_path']).is_file():
                parent_source=Path(parent['model_path']).read_text()
                patch=''.join(difflib.unified_diff(parent_source.splitlines(True),source.splitlines(True),
                    fromfile=parent['id']+'/model.py',tofile=trial['id']+'/model.py',n=2))
                (output/target/'changes-from-parent.patch').write_text(patch)
                derived.append({'path':str(target/'changes-from-parent.patch'),'source_trial':trial['id'],
                    'source_sha256':trial['source_sha256'],'parent_trial':parent['id'],
                    'parent_sha256':parent['source_sha256'],'transformation':'Unified source diff against the actual recorded parent;2 context lines.'})
            metrics=trial.get('metrics',{});validation=metrics.get('validation',{});tasks=validation.get('per_task',{})
            row={k:trial.get(k) for k in ['id','parent','role','round','variant_name','topic_id','seed','status','wall_seconds','source_sha256']}
            row.update(gpu=gpu,model_path=str(target/'model.py'),config_path=str(target/'model-config.json'),
                result_path=str(target/'result.json'),parameters=metrics.get('parameters'),train_seconds=metrics.get('train_seconds'),
                historical_selection_metric='validation click AUC',validation_click_auc=validation.get('auc'),
                validation_click_logloss=validation.get('logloss'),validation_mean_task_auc=validation.get('mean_task_auc'))
            for name,values in tasks.items():
                row['validation_'+name+'_auc']=values['auc'];row['validation_'+name+'_logloss']=values['logloss']
            iterations.append(row)

        for round_dir in sorted((folder/'rounds').glob('r*')):
            for path in sorted(round_dir.rglob('*')):
                if not path.is_file() or path.suffix not in ['.json','.jsonl','.log']:continue
                if path.name=='codex.jsonl':continue
                copy(path,Path('topics')/f'gpu-{gpu}'/'rounds'/path.relative_to(folder/'rounds'))
            for phase in ['propose','decision']:
                call_dir=round_dir/phase;event_path=call_dir/'codex.jsonl'
                if not event_path.is_file():continue
                raw=event_path.read_bytes();counts=Counter();usage=[];bad=0
                for number,line in enumerate(raw.splitlines(),1):
                    try:event=json.loads(line)
                    except ValueError:bad+=1;continue
                    counts[event.get('type','unknown')]+=1
                    if event.get('item',{}).get('type'):counts['item:'+event['item']['type']]+=1
                    if event.get('type')=='turn.completed':
                        item={'gpu':gpu,'round':int(round_dir.name[1:]),'phase':phase,'line':number,
                              'usage':event.get('usage',{}),'source_sha256':digest(raw)}
                        usage.append(item);events.append(item)
                summary={'source':str(event_path.relative_to(campaign)),'source_sha256':digest(raw),
                    'source_bytes':len(raw),'event_counts':dict(counts),'unparsed_lines':bad,'usage_events':usage,
                    'raw_event_stream_exported':False,'note':'Raw CLI streams omitted; plans, runtime context, tool-access records, code and execution receipts are exported separately.'}
                dest=Path('topics')/f'gpu-{gpu}'/'rounds'/round_dir.name/phase/'event-summary.json'
                dump(output/dest,summary);derived.append({'path':str(dest),'source':summary['source'],
                    'source_sha256':summary['source_sha256'],'transformation':'Count CLI event types and extract exact reported usage events; no prompt or raw tool output is copied.'})
                omitted.append({'source':summary['source'],'sha256':summary['source_sha256'],'bytes':len(raw),'reason':'Raw CLI event stream replaced by event/usage summary.'})
        for call in state['agent_calls']:
            calls.append({'gpu':gpu,**{k:call.get(k) for k in ['phase','round','topic_id','status','wall_seconds','model','reasoning_effort','fixed_instructions_sha256']},
                'usage':call.get('usage',[]),'input_tokens':sum(u.get('input_tokens',0) or 0 for u in call.get('usage',[])),
                'cached_input_tokens':sum(u.get('cached_input_tokens',0) or 0 for u in call.get('usage',[])),
                'output_tokens':sum(u.get('output_tokens',0) or 0 for u in call.get('usage',[]))})
        lanes.append({'gpu':gpu,'rounds':state['completed_research_rounds'],'attempts':len(state['trials']),
            'trial_statuses':dict(Counter(t['status'] for t in state['trials'])),
            'topics':len(state['topics']),'decisions':dict(Counter(x['action'] for x in state['decisions'])),
            'agent_calls':len(state['agent_calls']),'job_wall_seconds':sum(t['wall_seconds'] for t in state['trials'])})

    iterations.sort(key=lambda x:(x['gpu'],x['round'],x['id']))
    dump(output/'iterations.json',iterations);dump(output/'calls.json',calls);dump(output/'token-events.json',events)
    for name,rows in [('iterations',iterations),('calls',calls)]:
        fields=list(rows[0]);
        with (output/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    final=read(campaign/'final-assessment/report.json')
    summary={'campaign':campaign.name,'exported_at_utc':datetime.now(timezone.utc).isoformat(),
        'historical_objective':'Four-task training with click-only validation selection; not a multi-task-average search.',
        'original_protocol_unchanged':True,'scope':'Frozen2026-11-06..08 run; excludes subsequent multi-task selection-rule changes.',
        'rounds':sum(s['rounds'] for s in lanes),'training_attempts':len(iterations),'agent_calls':len(calls),'lanes':lanes,
        'observed_training_seeds':sorted({x['seed'] for x in iterations}),
        'input_tokens':sum(x['input_tokens'] for x in calls),'cached_input_tokens':sum(x['cached_input_tokens'] for x in calls),
        'output_tokens':sum(x['output_tokens'] for x in calls),'calls_missing_usage':sum(not x['usage'] for x in calls),
        'selected_trial':next(x['id'] for x in final['selection_freeze']['suite'] if x['key']=='selected'),
        'heldout_test_samples':final['test']['selected']['metrics']['samples'],
        'heldout':{key:{'id':x['id'],'per_task':x['metrics']['per_task'],'mean_task_auc':x['metrics']['mean_task_auc']} for key,x in final['test'].items()},
        'assessment_outside_original48h':True,'excluded':['raw data','model weights','virtual environments','third-party full-paper caches','authentication files','raw CLI event streams','unrelated/current-main-thread changes']}
    dump(output/'summary.json',summary);dump(output/'copy-sources.json',copies);dump(output/'derived-sources.json',derived);dump(output/'omitted-cli-streams.json',omitted)
    text=['UniRank Taobao：48小时历史迭代记录（2026-11-06至2026-11-08）','',
        '这是已完成实验的冻结归档。原实验为四任务训练，但 checkpoint、父模型、课题收益与最终选模均以点击验证AUC为主；不是按四任务平均AUC开展的搜索。后续多任务选模规则修改不属于本归档。','',
        f'范围：{summary["rounds"]}轮研究、{summary["training_attempts"]}次训练、{summary["agent_calls"]}次Agent调用；4个GPU课题，每完成6轮决策。模型gpt-6.1-sol/high；全部训练仅一个seed20262027。',
        '初始参照DIN；允许整网重写。行业参考为UniRank中的15种对比方法。使用论文官方离线历史反馈口径，长度100。具体偏差与假设保留在protocol/protocol.json。','',
        '独立测试共2907238条曝光；选模在测试前冻结。DIN点击AUC0.652227；验证集选定版本点击AUC0.666318。选定版本四任务平均AUC0.782178，低于DIN的0.807348，其他三个反馈任务退步。该结果支持本次点击指标改善，不能解释为整体多任务提升或充分验证的新统一架构。',
        '同轮完整候选测试点击AUC0.666459，仅作为预先冻结的对照展示，没有据此重新选模。最终验证重放和测试在原48h之外补做并单列成本。','',
        '文件入口：',
        '  iterations.csv / iterations.json：每次训练的父模型、课题、角色、四任务验证指标、成本与代码路径。',
        '  topics/gpu-N/state.json：原始完整轨迹、课题变化和决策。',
        '  topics/gpu-N/rounds/rXXXX/propose/proposal.json：问题、假设、相近工作、方案、对照、知识采用/拒绝依据。',
        '  topics/gpu-N/trials/TRIAL/model.py、model-config.json、result.json、changes-from-parent.patch：实际执行代码、配置、结果及父子源码差异。',
        '  各轮runtime/task/workspace导出与knowledge-access/usage：当时上下文及知识访问证据。',
        '  calls.* / token-events.json / event-summary.json：调用、Token与原始CLI流的来源SHA；输入包含缓存，不能再次相加。',
        '  evidence/：DIN、成本校准及共享内存故障/修复记录。',
        '  final-assessment/：最终冻结、验证重放、独立测试、四任务取舍和补测成本。',
        '  source-at-freeze/：当时运行器、训练器、任务和通用Agent指令；不是当前主线程代码。',
        '  selected-model/：按原点击验证指标选定的代码和配置，不含权重。','',
        '不含原始样本、模型权重、认证文件、虚拟环境或第三方论文全文缓存。没有这些资产时本目录用于审计和重建，不可直接复现已训练权重的推理。原始绝对路径作为来源证据保留，迁移时需重新配置。',
        'copy-sources.json记录原样复制文件；derived-sources.json记录派生差异和事件摘要；manifest.json可逐文件核验。',
        '本归档不会启动训练、调用LLM或重新打开测试集。']
    (output/'README.txt').write_text('\n'.join(text)+'\n')
    (output/'verify_records.py').write_text('''"""Verify archive hashes and historical record counts without original data."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for name,item in manifest['files'].items():
    raw=(root/name).read_bytes()
    assert len(raw)==item['bytes'],name
    assert hashlib.sha256(raw).hexdigest()==item['sha256'],name
rows=json.loads((root/'iterations.json').read_text());summary=json.loads((root/'summary.json').read_text())
assert len(rows)==summary['training_attempts']==1386
assert sum(x['rounds'] for x in summary['lanes'])==summary['rounds']==356
assert len(json.loads((root/'calls.json').read_text()))==summary['agent_calls']==413
for row in rows:
    assert hashlib.sha256((root/row['model_path']).read_bytes()).hexdigest()==row['source_sha256'],row['id']
print(json.dumps({'verified_files':len(manifest['files']),'rounds':356,'training_attempts':1386,'agent_calls':413}))
''')
    files={}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path!=output/'manifest.json':
            raw=path.read_bytes()
            if secret.search(raw):raise ValueError('Potential secret found in export')
            files[str(path.relative_to(output))]={'sha256':digest(raw),'bytes':len(raw)}
    dump(output/'manifest.json',{'exported_at_utc':summary['exported_at_utc'],'campaign':campaign.name,
        'historical_primary':'validation click AUC','files':files,'total_bytes':sum(v['bytes'] for v in files.values()),
        'source_protocol_sha256':digest((campaign/'protocol.json').read_bytes()),
        'source_snapshot_sha256':digest((campaign/'source-snapshot.json').read_bytes())})
    print(json.dumps({'export':str(output),'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),
        'rounds':summary['rounds'],'training_attempts':len(iterations),'agent_calls':len(calls)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.campaign,args.output)
