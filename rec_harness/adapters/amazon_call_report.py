"""Read-only, event-bound role attempt counts; never infer internal API usage."""
from pathlib import Path
import re
from datetime import datetime
import statistics

from rec_harness.core import ContractError,digest,read_json
from rec_harness.cost_audit import audit_cost
from rec_harness.store import Store

ROLE_ORDER={role:index for index,role in enumerate(['evidence','measurement','hypothesis','experiment','evaluation','lead','knowledge'])}


def submission_cadence(attempts):
    result={'submissions':len(attempts),'span_seconds':None,'minimum_interval_seconds':None,
            'median_interval_seconds':None,'peak_submissions_in_60_seconds':None,
            'qualification':'Framework submission timestamps only; not internal API rate, execution concurrency or service limits.'}
    times=[]
    try:
        for attempt in attempts:
            value=datetime.fromisoformat(attempt['submitted_at'])
            if value.utcoffset() is None:raise ValueError('timezone missing')
            times.append(value.timestamp())
    except (ValueError,TypeError,KeyError):
        return {**result,'status':'unknown','reason':'invalid or timezone-free submission timestamp'}
    if any(b<a for a,b in zip(times,times[1:])):
        return {**result,'status':'unknown','reason':'submission clock moved backwards'}
    result['status']='observed'
    if not times:
        result['peak_submissions_in_60_seconds']=0
        return result
    intervals=[b-a for a,b in zip(times,times[1:])]
    result['span_seconds']=times[-1]-times[0]
    if intervals:
        result['minimum_interval_seconds']=min(intervals)
        result['median_interval_seconds']=statistics.median(intervals)
    left=0;peak=0
    for right,current in enumerate(times):
        while current-times[left]>60:left+=1
        peak=max(peak,right-left+1)
    result['peak_submissions_in_60_seconds']=peak
    return result


def markdown_report(report):
    lines=['| Round | Role | Submitted | Completed | Rejected | Pending | Unresolved | Reused | Charged |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=['round','role','submitted','completed','rejected','pending','unresolved','reused','charged_or_reserved']
    for row in report['rows']:
        lines.append('| '+' | '.join(str(row[k]).replace('|','\\|').replace('\n',' ') for k in keys)+' |')
    lines.extend(['',f"Local submissions: {report['local_submissions']}; inherited charges: {report['retained_cost']}; "
                  f"unattributed charges: {report['unattributed_cost']}; total charged/reserved: {report['budget']['charged_or_reserved']}.",
                  '',report['qualification']])
    cadence=report['submission_cadence']
    lines.extend(['',f"Submission cadence: {cadence['status']}; span seconds: {cadence['span_seconds']}; "
                  f"minimum interval seconds: {cadence['minimum_interval_seconds']}; "
                  f"median interval seconds: {cadence['median_interval_seconds']}; "
                  f"peak submissions in an inclusive 60-second window: {cadence['peak_submissions_in_60_seconds']}.",
                  cadence['qualification']])
    return '\n'.join(lines)

def call_report(directory):
    root=Path(directory).resolve()
    if not (root/'amazon-auto.json').is_file():raise ContractError('Amazon role controller is not initialized')
    config=read_json(root/'amazon-auto.json')
    if digest(config)!=read_json(root/'auto-lock.json')['config_hash']:raise ContractError('autonomous configuration changed')
    store=Store(root/'memory',read_only=True)
    try:
        store.db.execute('BEGIN')
        cost=audit_cost(store,config['role_call_limit']);events=store.events()
        ledger={r['id']:dict(r) for r in store.db.execute('SELECT * FROM budget')}
        submissions={};completed={};rejected={}
        for event in events:
            value=event['payload'];kind=event['kind']
            if kind=='agent_submitted':
                task=value['task_id']
                if task in submissions:raise ContractError('duplicate role submission')
                if task not in ledger:raise ContractError('role submission lacks reservation')
                submissions[task]={**value,'submitted_at':event['created_at']}
            elif kind in {'role_completed','role_rejected'}:
                task=value['task'];target=completed if kind=='role_completed' else rejected
                if task in target:raise ContractError('duplicate role outcome')
                target[task]=value
        if set(completed)&set(rejected):raise ContractError('role attempt has conflicting outcomes')
        if (set(completed)|set(rejected))-set(submissions):raise ContractError('role outcome lacks submission')
        rows={};attempts=[]
        def group(round_name,role):
            return rows.setdefault((round_name,role),{'round':round_name,'role':role,'submitted':0,
                'completed':0,'rejected':0,'pending':0,'unresolved':0,'reused':0,'charged_or_reserved':0})
        def charge(row):return row['reserved'] if row['actual'] is None else row['actual']
        for task,submission in submissions.items():
            match=re.match(r'^(round-\d+)-',task);round_name=match[1] if match else 'unclassified'
            role=submission['role'];row=group(round_name,role);budget=ledger[task]
            outcome=completed.get(task) or rejected.get(task)
            if outcome and outcome['role']!=role:raise ContractError('role outcome identity differs')
            status=('completed' if task in completed else 'rejected' if task in rejected else
                    'pending' if budget['status']=='running' else 'unresolved')
            row['submitted']+=1;row[status]+=1;row['charged_or_reserved']+=charge(budget)
            attempts.append({'task':task,'round':round_name,'role':role,'status':status,
                'submitted_at':submission['submitted_at'],'charged_or_reserved':charge(budget),
                'ledger_status':budget['status'],
                'completed_wall_seconds':completed.get(task,{}).get('wall_seconds')})
        from .amazon_auto_audit import verify_reused_role
        for marker in sorted((root/'rounds').glob('round-*/*/reuse.json')):
            accepted=read_json(marker.parent/'accepted.json')
            verify_reused_role(marker.parent,accepted)
            group(marker.parent.parent.name,marker.parent.name)['reused']+=1
        retained=[{'task':task,'charged_or_reserved':charge(row)} for task,row in ledger.items()
                  if task not in submissions and row['status']=='retained']
        unattributed=[{'task':task,'charged_or_reserved':charge(row),'status':row['status']} for task,row in ledger.items()
                      if task not in submissions and row['status']!='retained']
        submitted_cost=sum(r['charged_or_reserved'] for r in attempts)
        retained_cost=sum(r['charged_or_reserved'] for r in retained)
        unattributed_cost=sum(r['charged_or_reserved'] for r in unattributed)
        if abs(submitted_cost+retained_cost+unattributed_cost-cost['charged_or_reserved'])>1e-9:
            raise ContractError('role report does not reconcile ledger')
        return {'schema':'amazon-role-call-report-v1','campaign':str(root),'source_integrity':cost['integrity'],
            'budget':{k:cost[k] for k in ['limit','charged_or_reserved','remaining','pending']},
            'rows':sorted(rows.values(),key=lambda r:(int(r['round'].split('-')[1]) if r['round']!='unclassified' else -1,ROLE_ORDER.get(r['role'],99),r['role'])),
            'attempts':attempts,'local_submissions':len(submissions),'local_submission_cost':submitted_cost,
            'retained_cost':retained_cost,'retained_reservations':retained,
            'unattributed_cost':unattributed_cost,'unattributed_reservations':unattributed,
            'internal_api_requests':None,'submission_cadence':submission_cadence(attempts),
            'qualification':'Role submissions are recorded before transport and may fail before CLI startup. Reused roles are not new calls. Internal CLI reconnect/API counts are unknown; inherited charges are separate.'}
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()
