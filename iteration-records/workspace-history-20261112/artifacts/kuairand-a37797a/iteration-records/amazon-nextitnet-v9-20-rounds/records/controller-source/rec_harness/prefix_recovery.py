"""Durable prefix replay admission, without repeating unknown role calls."""
from .core import Artifacts,ContractError,digest
from .store import Store


def prepare(store,objects,folder,key,binding):
    submissions=[e['payload'] for e in store.events() if e['kind']=='prefix_replay_submitted' and e['payload']['key']==key]
    task='prefix-'+digest([key,1])[:32]
    expected={'key':key,'task_id':task,'binding_hash':digest(binding)}
    if submissions:
        if submissions!=[expected]:raise ContractError('prefix replay recovery inputs changed')
        return task,recover_result(store,objects,expected)
    if list(folder.glob('attempt-*')) or store.db.execute('SELECT 1 FROM budget WHERE id=?',(task,)).fetchone():
        raise ContractError('unbound prefix replay requires review; refusing automatic resubmission')
    return task,None


def recover_result(store,objects,submission):
    records=[e['payload'] for e in store.events() if e['kind']=='prefix_replay_result_recorded' and e['payload']['key']==submission['key']]
    if len(records)!=1 or records[0]['request_hash']!=digest(submission):
        raise ContractError('prefix replay outcome is not committed; refusing automatic resubmission')
    result=records[0]['result']
    if result['key']!=submission['key']:raise ContractError('prefix replay result identity changed')
    for ref in result['artifacts'].values():objects.verify(ref)
    return result


def preflight(target):
    for name in ['prefix-replay','memory-prefix-replay']:_preflight(target/name)


def _preflight(root):
    if not root.exists():return
    store=Store(root/'memory',read_only=True);objects=Artifacts(root/'objects')
    try:
        store.verify();events=store.events()
        complete={e['payload']['key'] for e in events if e['kind']=='prefix_replay_completed'}
        submissions={e['payload']['key']:e['payload'] for e in events if e['kind']=='prefix_replay_submitted'}
        for folder in (root/'cases').glob('*'):
            if folder.name not in complete and folder.name not in submissions:
                raise ContractError('unbound prefix replay requires review')
        for key,submission in submissions.items():
            if key not in complete:recover_result(store,objects,submission)
    finally:store.close()
