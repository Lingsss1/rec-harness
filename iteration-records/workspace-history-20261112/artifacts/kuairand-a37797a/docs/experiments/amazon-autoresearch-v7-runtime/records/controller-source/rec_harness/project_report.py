"""Read-only project ledger inventory, including nested validation work."""
from pathlib import Path
from .core import ContractError,digest,read_json
from .cost_audit import audit_cost
from .store import Store
from .adapters.amazon_call_report import submission_cadence


def project_report(directory):
    root=Path(directory).resolve()
    config=read_json(root/'project.json')
    if digest(config)!=read_json(root/'project-lock.json')['config_hash']:
        raise ContractError('frozen project changed')
    rows=[];attempts=[];seen=set();state=None;committed=[]
    paths=[root/'memory/memory.sqlite',*sorted((root/'episodes').rglob('memory.sqlite'))]
    for path in paths:
        resolved=path.resolve()
        if not resolved.is_relative_to(root):raise ContractError('project ledger outside project')
        identity=(path.stat().st_dev,path.stat().st_ino)
        if identity in seen:raise ContractError('duplicate project ledger alias')
        seen.add(identity)
        relative=path.parent.relative_to(root)
        episode=relative.parts[1] if relative.parts[0]=='episodes' else 'project'
        store=Store(path.parent,read_only=True)
        try:
            store.db.execute('BEGIN')
            cost=audit_cost(store,store.budget()['limit']);events=store.events()
            submissions=[e for e in events if e['kind'] in {'agent_submitted','candidate_generation_submitted'}]
            prefix=sum(e['kind']=='prefix_replay_completed' for e in events)
            if episode=='project':
                transitions=[e for e in events if e['kind']=='project_transition']
                if not transitions:raise ContractError('project has no committed state')
                state=transitions[-1]['payload']['state']
                committed=[e['payload']['episode'] for e in transitions if 'episode' in e['payload']]
            for event in submissions:
                attempts.append({'ledger':str(relative),'event_hash':event['hash'],
                    'kind':event['kind'],'submitted_at':event['created_at']})
            rows.append({'episode':episode,'ledger':str(relative),
                'role_submissions':sum(e['kind']=='agent_submitted' for e in submissions),
                'generation_submissions':sum(e['kind']=='candidate_generation_submitted' for e in submissions),
                'prefix_completions':prefix,'cost':cost})
        finally:
            if store.db.in_transaction:store.db.execute('ROLLBACK')
            store.close()
    attempts.sort(key=lambda row:row['submitted_at'])
    return {'schema':'project-report-v1','task_id':config['task_id'],'state':state,
        'committed_episodes':committed,'ledgers':rows,
        'recorded_submissions':len(attempts),
        'prefix_completions':sum(row['prefix_completions'] for row in rows),
        'pending_reservations':sum(row['cost']['pending'] for row in rows),
        'submission_cadence':submission_cadence(attempts),'submissions':attempts,
        'qualification':'Each distinct local ledger is audited once, including uncommitted child work. Costs remain per ledger: parent validator fees and child execution charges are not equivalent units. External Amazon study/GPU ledgers are excluded. Prefix completions are not submission timestamps or a complete attempt count. Independent database snapshots are not an atomic cross-project snapshot. No agent execution, recovery, internal API count or process liveness is inferred.'}


def markdown_report(report):
    lines=['| Episode | Ledger | Role submissions | Generation submissions | Prefix completions | Charged/reserved | Pending |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in report['ledgers']:
        values=[row['episode'],row['ledger'],row['role_submissions'],row['generation_submissions'],
                row['prefix_completions'],row['cost']['charged_or_reserved'],row['cost']['pending']]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in values)+' |')
    cadence=report['submission_cadence']
    lines.extend(['',f"Recorded submissions: {report['recorded_submissions']}; prefix completions: {report['prefix_completions']}; pending reservations: {report['pending_reservations']}.",
        f"Submission cadence: {cadence['status']}; peak in inclusive 60 seconds: {cadence['peak_submissions_in_60_seconds']}; minimum interval: {cadence['minimum_interval_seconds']} seconds.",
        '',report['qualification']])
    return '\n'.join(lines)
