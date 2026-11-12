"""Validation-only trajectory audit against immutable captured states."""
import csv
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
AUDIT=Path(__file__).absolute().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/taobao_ctr_dual'))
from common import read,write,sha

snapshot=read(AUDIT/'snapshot.json')
stop=read(AUDIT/'enhanced-state.json')['last_runtime_error']
failures=[json.loads(line) for line in (ROOT/'enhanced/failures.jsonl').read_text().splitlines()]
stopped=max(f['at'] for f in failures if f['round']==37)
summary={'snapshot':snapshot,'enhanced_stopped_unix':stopped,'scope':'validation only; no closed test artifacts read','methods':{}}
experiments=[];rounds=[]
for mode in ['auto','enhanced']:
    state=read(AUDIT/(mode+'-state.json'));trials=state['trials'];max_round=max(t.get('round',-1) for t in trials)
    current=[]
    for t in trials:
        m=t.get('metrics',{});v=m.get('validation',{});p=read(Path(t['output'])/'process.json')
        row={'method':mode,'id':t['id'],'round':t.get('round',-1),'role':t.get('role'),'parent':t.get('parent'),'seed':t.get('seed'),
            'status':t['status'],'auc':v.get('auc'),'logloss':v.get('logloss'),'parameters':m.get('parameters'),
            'dense_parameters':m.get('dense_parameters'),'training_seconds':m.get('train_seconds'),
            'inference_seconds':v.get('inference_seconds'),'job_wall_seconds':t['wall_seconds'],
            'started_unix':p['started_unix'],'finished_unix':p['started_unix']+t['wall_seconds'],
            'source_sha256':t['source_sha256'],'knowledge_revision':t.get('knowledge_revision','launch')}
        current.append(row)
    experiments+=current
    ordered=sorted(current,key=lambda r:(r['finished_unix'],r['id']))
    best=-1;curve=[]
    for i,row in enumerate(ordered,1):
        if row['auc'] is not None:best=max(best,row['auc'])
        curve.append({'completed_attempts':i,'best_auc':best,'elapsed_hours':(row['finished_unix']-read(ROOT/'budget.json')['start_unix'])/3600,'trial_id':row['id']})
    all_good=[row for row in current if row['auc'] is not None];best_row=max(all_good,key=lambda r:r['auc'])
    same_wall=max([row for row in all_good if row['finished_unix']<=stopped],key=lambda r:r['auc'])
    phase_costs={phase:{'calls':sum(c['phase']==phase for c in state['agent_calls']),
                       'seconds':sum(c['wall_seconds'] for c in state['agent_calls'] if c['phase']==phase),
                       'input_tokens':sum(u.get('input_tokens',0) or 0 for c in state['agent_calls'] if c['phase']==phase for u in c.get('usage',[])),
                       'output_tokens':sum(u.get('output_tokens',0) or 0 for c in state['agent_calls'] if c['phase']==phase for u in c.get('usage',[]))}
                 for phase in ['query','propose','analyze']}
    declarations=[];industry_decisions=[];read_events=[];memory_recall=[]
    model_rounds=0;branch_jobs=0
    for index in range(max_round+1):
        directory=ROOT/mode/'rounds'/f'r{index:04d}';proposal_path=directory/'propose/proposal.json'
        proposal=read(proposal_path) if proposal_path.exists() else {};study=proposal.get('study',{})
        rs=[row for row in current if row['round']==index]
        if rs:model_rounds+=1;branch_jobs+=len(rs)
        retrieval_path=directory/'retrieval.json';retrieval=read(retrieval_path) if retrieval_path.exists() else {}
        usage_path=directory/'knowledge-usage.json';usage=read(usage_path) if usage_path.exists() else {}
        ds=[{'round':index,**item} for item in usage.get('assessments',[])];declarations+=ds
        industry_decisions += [d for d in ds if str(d.get('knowledge_id','')).startswith('taac2026-')]
        kinds={card['id']:card.get('knowledge_type','legacy') for card in retrieval.get('cards',[])}
        memory_recall.append({'round':index,'ids':retrieval.get('selected_ids',[c['id'] for c in retrieval.get('cards',[])]),'types':kinds})
        for phase in ['query','propose','analyze']:
            log=directory/phase/'knowledge-access.jsonl'
            if log.exists():
                read_events += [{'round':index,'phase':phase,**event} for event in (json.loads(line) for line in log.read_text().splitlines()) if event['operation']=='read']
        rounds.append({'method':mode,'round':index,'jobs':len(rs),'best_round_auc':max((r['auc'] for r in rs if r['auc'] is not None),default=None),
            'parent':proposal.get('parent_trial'),'variants':';'.join(v['name'] for v in proposal.get('variants',[])),
            'question':str(study.get('question','')),'hypothesis':str(study.get('hypothesis','')),
            'nearest_work':json.dumps(study.get('nearest_work',study.get('nearest_existing_work',[])),ensure_ascii=False),
            'proposed_difference':str(study.get('proposed_difference','')),
            'retrieved_ids':';'.join(memory_recall[-1]['ids']),
            'knowledge_usage_status':usage.get('status'),'industry_used':';'.join(d['knowledge_id'] for d in ds if d.get('decision')=='used' and d['knowledge_id'].startswith('taac2026-'))})
    summary['methods'][mode]={'training_attempts':len(current),'successful':len(all_good),'failed':len(current)-len(all_good),
        'research_rounds_with_training':model_rounds,'research_training_attempts':branch_jobs,'average_models_per_training_round':branch_jobs/model_rounds,
        'best':best_row,'best_at_enhanced_failure':same_wall,'best_after_equal_attempts':{str(n):curve[n-1]['best_auc'] for n in [7,15,23,35,67,96] if len(curve)>=n},
        'phase_costs':phase_costs,'llm_blocking_seconds':sum(v['seconds'] for v in phase_costs.values()),
        'curve':curve,'industry_decisions':industry_decisions,
        'knowledge_declaration_counts':{name:sum(d.get('decision')==name for d in declarations) for name in ['used','considered','rejected','unresolved']},
        'industry_declared_used':[{'round':d['round'],'id':d['knowledge_id']} for d in industry_decisions if d.get('decision')=='used'],
        'primary_text_reads':{source:sum(e.get('source_id')==source for e in read_events) for source in sorted(set(e['source_id'] for e in read_events if e.get('source_id')))},
        'retrievals':memory_recall}
write(AUDIT/'summary.json',summary)
for filename,rows in [('experiments.csv',experiments),('rounds.csv',rounds)]:
    with (AUDIT/filename).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({mode:{k:v for k,v in data.items() if k in ['training_attempts','successful','failed','average_models_per_training_round','llm_blocking_seconds','best_after_equal_attempts','industry_declared_used']} for mode,data in summary['methods'].items()},ensure_ascii=False))
