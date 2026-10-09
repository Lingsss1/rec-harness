"""Recompute charges from committed events and check the budget projection."""
from .core import ContractError,number


def audit_cost(store,limit):
    number(limit,'frozen budget',0);integrity=store.verify();rows={}
    for event in store.events():
        value=event['payload'];kind=event['kind']
        if kind=='budget_reserved':
            task=value['task_id'];amount=number(value['amount'],'reservation',0)
            if task in rows:raise ContractError('duplicate budget reservation event')
            rows[task]={'id':task,'reserved':amount,'actual':None,'status':'running'}
        elif kind=='budget_settled':
            task=value['task_id'];row=rows.get(task)
            if row is None or row['status']!='running':raise ContractError('settlement without running reservation')
            if value['reserved']!=row['reserved']:raise ContractError('settlement changed reservation')
            if value['actual'] is not None:number(value['actual'],'actual cost',0)
            row.update(actual=value['actual'],status=value['status'])
    projected={row['id']:dict(row) for row in store.db.execute('SELECT * FROM budget')}
    if rows!=projected:raise ContractError('budget projection differs from committed events')
    total=sum(row['reserved'] if row['actual'] is None else row['actual'] for row in rows.values())
    expected={'limit':limit,'charged_or_reserved':total,'remaining':limit-total}
    if store.budget()!=expected:raise ContractError('budget summary differs from frozen limit and events')
    if store.verify()!=integrity:raise ContractError('cost source changed during audit')
    return {**expected,'reservations':len(rows),'pending':sum(row['status']=='running' for row in rows.values()),'integrity':integrity}


def freeze_cost_prefix(store,limit):
    """Snapshot all incurred/reserved controller costs without settling a live job."""
    cost=audit_cost(store,limit)
    return {'schema':'controller-cost-prefix-v1','memory':str(store.root.resolve()),
        'cutoff_seq':cost['integrity']['events'],'event_head':cost['integrity']['head'],
        'limit':limit,'charged_or_reserved':cost['charged_or_reserved'],
        'pending':cost['pending'],'qualification':'Includes current validator reservation; later events are outside this frozen prefix'}


def verify_cost_prefix(store,prefix):
    if prefix.get('schema')!='controller-cost-prefix-v1' or str(store.root.resolve())!=prefix['memory']:
        raise ContractError('cost prefix source changed')
    integrity=store.verify();events=store.events();cutoff=prefix['cutoff_seq']
    if type(cutoff) is not int or not 0<=cutoff<=len(events):raise ContractError('invalid cost prefix cutoff')
    head=events[cutoff-1]['hash'] if cutoff else '0'*64
    if head!=prefix['event_head']:raise ContractError('cost prefix event head changed')
    rows={}
    for event in events[:cutoff]:
        value=event['payload']
        if event['kind']=='budget_reserved':
            if value['task_id'] in rows:raise ContractError('duplicate prefix reservation')
            rows[value['task_id']]={'reserved':number(value['amount'],'reservation',0),'actual':None,'status':'running'}
        elif event['kind']=='budget_settled':
            row=rows.get(value['task_id'])
            if row is None or row['status']!='running' or row['reserved']!=value['reserved']:
                raise ContractError('invalid prefix settlement')
            if value['actual'] is not None:number(value['actual'],'cost',0)
            row.update(actual=value['actual'],status=value['status'])
    total=sum(row['reserved'] if row['actual'] is None else row['actual'] for row in rows.values())
    pending=sum(row['status']=='running' for row in rows.values())
    if total!=prefix['charged_or_reserved'] or pending!=prefix['pending'] or store.budget()['limit']!=prefix['limit']:
        raise ContractError('cost prefix amount changed')
    if store.verify()!=integrity:raise ContractError('cost source changed during prefix audit')
    return total
