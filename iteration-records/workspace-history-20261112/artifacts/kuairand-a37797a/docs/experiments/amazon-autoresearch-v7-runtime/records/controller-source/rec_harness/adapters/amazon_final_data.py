"""Host-only final evaluation requests shared by every frozen model and seed."""
import fcntl
import random
from pathlib import Path

from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from .amazon_final_selection import load_selection
from .independent_scoring import prepare_request


def read_interactions(path,itemnum):
    users={}
    with Path(path).open() as stream:
        for line in stream:
            try:user,item=map(int,line.split())
            except ValueError as error:raise ContractError('invalid final interaction row') from error
            if user<1 or not 1<=item<=itemnum:
                raise ContractError('final interaction outside frozen vocabulary')
            users.setdefault(user,[]).append(item)
    return users


def dataset_rows(train,valid,test,*,itemnum,count,seed):
    """Fail on ineligible test users before sampling; never silently drop them."""
    if not test:raise ContractError('empty final test population')
    for user,items in test.items():
        if user not in train or not train[user] or user not in valid or not valid[user] or not items:
            raise ContractError('final test user lacks train/validation history')
        seen=set(train[user]+valid[user]+[items[0]])
        if itemnum-len(seen)<99:raise ContractError('insufficient final negative candidate pool')
    rng=random.Random(seed);users=sorted(test)
    if len(users)>count:users=sorted(rng.sample(users,count))
    rows=[]
    for user in users:
        history=train[user]+valid[user];positive=test[user][0]
        seen=set(history+[positive]);negatives=[]
        while len(negatives)<99:
            item=rng.randint(1,itemnum)
            if item not in seen:negatives.append(item);seen.add(item)
        rows.append({'user_id':user,'history':history,'candidates':[positive]+negatives})
    return rows


def prepare_final_data(output):
    """First test access happens here, only after selection has been sealed."""
    output=Path(output).resolve()
    plan=load_selection(output)
    study=Path(plan['request']['study'])
    with (study/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=load_selection(output)
        target=output/'private-evaluation.json'
        if target.exists():
            record=read_json(target)
            if record.get('selection_hash')!=plan['hash'] or record.get('hash')!=digest({k:v for k,v in record.items() if k!='hash'}):
                raise ContractError('final evaluation request changed')
            for ref in record['inputs']:
                if file_hash(ref['path'])!=ref['sha256']:raise ContractError('final evaluation data changed')
            return record
        inputs=[];datasets={}
        for dataset in plan['datasets']:
            sizes={job['itemnum'] for job in plan['jobs'] if job['dataset']==dataset}
            if len(sizes)!=1:raise ContractError('inconsistent final vocabulary')
            itemnum=next(iter(sizes));data={}
            for split in ['train','valid','test']:
                directory=plan['request']['test_directory'] if split=='test' else plan['data']
                path=Path(directory)/(dataset+'_'+split+'.txt');before=file_hash(path)
                if split!='test' and before!=plan['data_hashes'][path.name]:
                    raise ContractError('frozen history changed before final evaluation')
                data[split]=read_interactions(path,itemnum)
                if file_hash(path)!=before:raise ContractError('final input changed while reading')
                inputs.append({'path':str(path),'sha256':before})
            rows=dataset_rows(**data,itemnum=itemnum,count=plan['request']['eval_users'],
                seed=int(digest({'dataset':dataset,'seed':plan['request']['eval_seed']}),16))
            request,key=prepare_request(rows)
            datasets[dataset]={'request':request,'private_key':key,'test_population':len(data['test']),'users':len(rows)}
        load_selection(output)
        record={'schema':'amazon-final-data-v1','selection_hash':plan['hash'],'inputs':inputs,'datasets':datasets,
            'scope':'Host-only test material; never export to research roles or research memory.'}
        record['hash']=digest(record);write_json(target,record);target.chmod(0o600)
        return record
