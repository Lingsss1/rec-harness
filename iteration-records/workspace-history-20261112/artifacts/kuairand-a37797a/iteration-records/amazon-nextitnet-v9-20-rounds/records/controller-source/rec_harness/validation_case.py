"""Resume an input-bound validator CPU case without relaunching unknown work."""
from .core import ContractError,digest
from .cpu_job import recover_completed_cpu_job


def prepare(store,folder,key,files,suite,artifacts):
    binding=digest({'files':files,'execution':suite['execution'],'timeout':suite['timeout'],'cost':suite['case_cost']})
    submitted=[e['payload'] for e in store.events() if e['kind']=='validation_case_submitted' and e['payload']['key']==key]
    task=key+'-1';job=folder/'job-1'
    if submitted:
        if len(submitted)!=1 or submitted[0]!={'key':key,'task_id':task,'binding_hash':binding}:
            raise ContractError('validation case recovery inputs changed')
        receipt=recover_completed_cpu_job(job,suite['execution']['recipe'],files,'worker.py',[],artifacts=artifacts)
        return task,job,receipt
    if list(folder.glob('job-*')) or store.db.execute('SELECT 1 FROM budget WHERE id=?',(task,)).fetchone():
        raise ContractError('uncommitted validation case requires review; refusing automatic relaunch')
    store.reserve(task,suite['case_cost'])
    store.event('validation_case_submitted',{'key':key,'task_id':task,'binding_hash':binding})
    return task,job,None


def settle_once(store,task,cost,status):
    if not any(e['kind']=='budget_settled' and e['payload']['task_id']==task for e in store.events()):
        store.settle(task,cost,status)
