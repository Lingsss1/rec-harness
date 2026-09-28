"""Read-only workflow status; never recover a ledger while inspecting it."""
from pathlib import Path
from .core import read_json
from .store import Store


def inspect_status(directory):
    root=Path(directory)
    store=Store(root/'memory',read_only=True)
    try:
        store.db.execute('BEGIN')
        integrity=store.verify()
        events=store.events()
        published=read_json(root/'state.json') if (root/'state.json').exists() else None
        kinds=set()
        if (root/'project.json').exists():kinds={'project_transition'}
        elif (root/'amazon-auto.json').exists():kinds={'auto_round_completed','auto_round_reopened'}
        transitions=[event for event in events if event['kind'] in kinds]
        state=published;source='state.json';projection='not_event_bound'
        event_hash=None
        if transitions:
            latest=transitions[-1];committed=latest['payload']['state'];event_hash=latest['hash']
            state=committed;source='committed_event'
            if published==committed:
                projection='matches'
            elif (latest['kind']=='auto_round_completed' and isinstance(published,dict)
                  and published.get('status')=='stopped'
                  and {k:v for k,v in published.items() if k not in {'status','stop_reason'}}==
                      {k:v for k,v in committed.items() if k not in {'status','stop_reason'}}):
                # Existing controllers publish their terminal stop after the last
                # committed round. Preserve that explicitly file-derived overlay.
                state=published;source='committed_round_with_published_stop';projection='terminal_overlay'
            else:
                projection='missing' if published is None else 'differs_from_event_snapshot'
        return {'state':state,'budget':store.budget(),'graph_version':store.version,
                'state_source':source,'state_projection':projection,'state_event_hash':event_hash,
                'integrity':integrity,'process_status':'not_inspected',
                'qualification':'Read-only committed snapshot. A differing state file may reflect interrupted publication or a concurrent writer; no repair performed. Published stop labels are not event-bound. Workflow state does not establish process liveness.'}
    finally:
        if store.db.in_transaction:store.db.execute('ROLLBACK')
        store.close()
