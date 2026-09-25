"""Controller-owned synthetic data and scoring, separate from generated code."""
import math
import random
import time
from pathlib import Path

from .core import ContractError, file_hash, number, read_json, write_json
from .cpu_job import run_cpu_job


def samples(seed,n):
    rng=random.Random(seed);rows=[]
    for _ in range(n):
        a,b=rng.uniform(-1,1),rng.uniform(-1,1)
        rows.append(([a,b],.25*a+.15*b+1.5*a*b))
    return rows


def score_response(response,targets):
    if not isinstance(response,dict) or set(response)!={'weights','predictions'}:
        raise ContractError('worker may return only weights and predictions, never metrics')
    weights=response['weights'];predictions=response['predictions']
    if not isinstance(weights,list) or not weights or len(weights)>1024:
        raise ContractError('invalid serialized coefficients')
    if not isinstance(predictions,list) or len(predictions)!=len(targets):
        raise ContractError('prediction count mismatch')
    for value in weights+predictions:number(value,'finite worker value')
    try:mse=sum((a-b)**2 for a,b in zip(predictions,targets))/len(targets)
    except OverflowError as error:raise ContractError('prediction overflow') from error
    if not math.isfinite(mse):raise ContractError('nonfinite host error')
    return {'quality':1/(1+mse),'mse':mse,'serialized_coefficients':len(weights)}


def evaluate(model,output,seed,execution,artifacts,timeout):
    model,output=Path(model),Path(output);directory=output.parent/'isolated'
    directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    validation=samples(seed+10000,120)
    request={'seed':seed,'training':samples(seed,180),'queries':[row for row,_ in validation]}
    write_json(directory/'request.json',request)
    worker=Path(__file__).with_name('toy_worker.py')
    files={'worker.py':{'path':str(worker),'sha256':file_hash(worker)},
           'request.json':{'path':str((directory/'request.json').resolve()),'sha256':file_hash(directory/'request.json')}}
    for path in model.rglob('*'):
        if path.is_file():
            if path.is_symlink():raise ContractError('candidate symlinks prohibited')
            files['model/'+str(path.relative_to(model))]={'path':str(path.resolve()),'sha256':file_hash(path)}
    responses=[];receipts=[]
    for index in range(2):
        remaining=timeout-(time.monotonic()-started)
        if remaining<=0:raise ContractError('isolated evaluation deadline exhausted')
        receipt=run_cpu_job(directory/str(index),execution['recipe'],files,'worker.py',
            ['--request','/inputs/request.json','--output','/output/predictions.json'],
            python=execution['python'],uid_registry=execution['uid_registry'],artifacts=artifacts,timeout=remaining)
        receipts.append(receipt)
        if receipt['process']['status']!='success':raise ContractError('isolated candidate process failed')
        response=read_json(directory/str(index)/'results/predictions.json')
        score_response(response,[target for _,target in validation]);responses.append(response)
    if responses[0]!=responses[1]:raise ContractError('fresh-process reproducibility check failed')
    metrics=score_response(responses[1],[target for _,target in validation]);write_json(output,metrics)
    return {'status':'success','wall_seconds':time.monotonic()-started,'gpu_seconds':0,
            'scoring':'host-only, validation targets absent from candidate processes',
            'reproducibility':'two fresh isolated processes','receipts':receipts}
