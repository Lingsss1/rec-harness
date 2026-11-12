"""Host training/scoring worker. Coding agents never receive held-out labels."""
import argparse
import ast
import importlib.util
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import torch

from common import read, write, sha
from data import Data


def check_source(path):
    tree=ast.parse(Path(path).read_text())
    allowed={'torch','math','numpy','typing','copy','functools','collections','itertools','dataclasses','contextlib'}
    forbidden={'open','exec','eval','compile','__import__','globals','locals','vars','breakpoint',
               'load','save','savez','savez_compressed','fromfile','tofile','memmap','read_text','read_bytes',
               'write_text','write_bytes','system','popen','getenv','import_module','load_library'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(module.split('.')[0] not in allowed for module in modules):
                raise ValueError('Model imports outside numeric-module allowlist')
        if isinstance(node,ast.Name) and node.id in forbidden:
            raise ValueError('Model uses forbidden I/O or dynamic execution: '+node.id)
        if isinstance(node,ast.Attribute) and (node.attr in forbidden or (node.attr.startswith('__') and node.attr not in {'__init__','__class__'})):
            raise ValueError('Model uses forbidden attribute: '+node.attr)
    # This reduces accidental I/O; it is not a general-purpose Python sandbox.
    return tree


def load_model(path,meta,config,device):
    check_source(path)
    spec=importlib.util.spec_from_file_location('registered_candidate_'+sha(path)[:16],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    public_meta={key:meta[key] for key in ['fields','buckets','category_field_index','brand_field_index','history_columns']}
    model=module.Model(public_meta,config).to(device)
    return model,module


def score(model,data,indices,device,batch_size,output=None):
    model.eval()
    predictions=np.empty(len(indices),dtype=np.float32)
    action_present=np.zeros((len(indices),4),dtype=bool)
    category_match=np.zeros(len(indices),dtype=bool)
    latencies=[]
    started=time.monotonic()
    with torch.no_grad():
        for offset in range(0,len(indices),batch_size):
            start=time.monotonic()
            chunk=indices[offset:offset+batch_size]
            batch=data.batch(chunk,device)
            present=((batch['history_type'][:,:,None]==torch.arange(1,5,device=device)[None,None,:])&batch['history_mask'][:,:,None]).any(1)
            action_present[offset:offset+len(chunk)]=present.cpu().numpy()
            target_category=batch['static'][:,data.meta['category_field_index']]
            category_match[offset:offset+len(chunk)]=((batch['history_category']==target_category[:,None])&batch['history_mask']&(target_category[:,None]!=0)).any(1).cpu().numpy()
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=model(batch)
            if logits.shape!=(len(chunk),) or not torch.isfinite(logits).all():
                raise ValueError('Invalid model output shape or nonfinite logits')
            predictions[offset:offset+len(chunk)]=logits.float().sigmoid().cpu().numpy()
            latencies.append(time.monotonic()-start)
    labels=np.asarray(data.labels[indices])
    if len(np.unique(labels))!=2:
        raise ValueError('Scored split must contain both labels')
    if not np.isfinite(predictions).all() or np.any((predictions<0)|(predictions>1)):
        raise ValueError('Invalid probabilities')
    auc=float(roc_auc_score(labels,predictions))
    clipped=np.clip(predictions.astype(np.float64),1e-7,1-1e-7)
    loss=float(log_loss(labels,clipped,labels=[0,1]))
    elapsed=time.monotonic()-started
    result={'auc':auc,'logloss':loss,'samples':len(indices),'inference_seconds':elapsed,
            'examples_per_second':len(indices)/elapsed,'latency_unit':'whole batch, including causal feature gathering',
            'inference_batch_size':batch_size,'batch_latency_ms_p50':float(np.percentile(latencies[3:] or latencies,50)*1000),
            'batch_latency_ms_p95':float(np.percentile(latencies[3:] or latencies,95)*1000)}
    if output:
        np.save(output/'predictions.npy',predictions)
        np.save(output/'sample_ids.npy',indices)
    # Slices are host-created and validation-only during search.
    slices={}
    lengths=np.minimum(data.history_end[indices]-data.history_start[indices],data.history_limit)
    groups={'empty_history':lengths==0,'history_1_16':(lengths>0)&(lengths<=16),
            'history_17_64':(lengths>16)&(lengths<=64),'history_gt64':lengths>64}
    hours=np.asarray(data.static[indices,data.meta['fields'].index('hour')])-1
    for lo in [0,6,12,18]:groups[f'hour_{lo}_{lo+6}']=(hours>=lo)&(hours<lo+6)
    for i,name in enumerate(['pv','cart','fav','buy']):
        groups['past_'+name+'_present']=action_present[:,i]
        groups['past_'+name+'_absent']=~action_present[:,i]
    groups['category_match_present']=category_match
    groups['category_match_absent']=~category_match
    frequencies=data.user_train_frequency[data.users[indices]]
    for lo,hi in [(0,1),(1,2),(2,10),(10,100),(100,2**32)]:groups[f'user_train_frequency_{lo}_{hi}']=(frequencies>=lo)&(frequencies<hi)
    for name,mask in groups.items():
        target=labels[mask]
        row={'samples':int(mask.sum()),'positives':int(target.sum()),'negatives':int(len(target)-target.sum())}
        if len(target)>100 and len(np.unique(target))==2:
            row.update(auc=float(roc_auc_score(target,predictions[mask])),logloss=float(log_loss(target,clipped[mask],labels=[0,1])))
        slices[name]=row
    result['slices']=slices
    return result


def run(args):
    started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False
    data=Data(args.data,args.history_limit)
    override=read(args.config) if args.config else {}
    model,module=load_model(args.model,data.meta,override,device)
    effective={**getattr(module,'CONFIG',{}),**override}
    learning_rate=float(effective.get('lr',0.001))
    if not math.isfinite(learning_rate) or not 0<learning_rate<=0.1:
        raise ValueError('Invalid candidate learning rate')
    params=sum(p.numel() for p in model.parameters())
    embedding_params=sum(m.weight.numel() for m in model.modules() if isinstance(m,torch.nn.Embedding))
    validation=data.indices('validation')
    if args.evaluate:
        if args.split=='test' and not (args.output.parent/'assessment-freeze.json').is_file():
            raise ValueError('Test evaluation requires the closed assessment manifest')
        checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(checkpoint)
        result=score(model,data,data.indices(args.split),device,args.batch_size,args.output)
        result.update(status='success',split=args.split,source_sha256=sha(args.model),checkpoint_sha256=sha(args.checkpoint),seed=args.seed,wall_seconds=time.monotonic()-started,parameters=params,embedding_parameters=embedding_params)
        write(args.output/'result.json',result)
        return result
    optimizer=module.make_optimizer(model) if hasattr(module,'make_optimizer') else torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_indices=data.indices('train')
    if args.calibration_steps:
        train_indices=train_indices[:args.calibration_steps*args.batch_size]
    epochs=1 if args.calibration_steps else args.epochs
    best=-math.inf;best_result=None;stale=0;steps=0;examples=0;training_seconds=0;epoch_rows=[]
    for epoch in range(epochs):
        order=np.random.default_rng(args.seed+epoch).permutation(train_indices)
        model.train();train_start=time.monotonic();loss_sum=0
        for offset in range(0,len(order),args.batch_size):
            selected=order[offset:offset+args.batch_size]
            batch=data.batch(selected,device)
            labels=torch.as_tensor(np.asarray(data.labels[selected],dtype=np.float32),device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=model(batch)
                loss=model.training_loss(batch,logits,labels) if hasattr(model,'training_loss') else torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),labels)
            if loss.ndim!=0 or not torch.isfinite(loss):raise ValueError('Invalid training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            steps+=1;examples+=len(selected);loss_sum+=float(loss.detach())*len(selected)
            if steps%100==0:
                write(args.output/'progress.json',{'phase':'train','epoch':epoch,'steps':steps,'examples':examples,'wall_seconds':time.monotonic()-started})
        epoch_train=time.monotonic()-train_start;training_seconds+=epoch_train
        metrics=score(model,data,validation,device,args.batch_size)
        epoch_rows.append({'epoch':epoch+1,'train_seconds':epoch_train,'training_loss':loss_sum/len(order),'validation':metrics})
        if metrics['auc']>best:
            best=metrics['auc'];best_result=metrics;stale=0
            torch.save(model.state_dict(),args.output/'checkpoint.pt')
        else:
            stale+=1
        write(args.output/'progress.json',{'phase':'validation','epoch':epoch+1,'steps':steps,'auc':metrics['auc'],'best_auc':best,'wall_seconds':time.monotonic()-started})
        if stale>=2:break
    diagnostics={'status':'not_requested'}
    if hasattr(model,'diagnostics'):
        try:
            model.load_state_dict(torch.load(args.output/'checkpoint.pt',map_location=device,weights_only=True))
            model.eval()
            with torch.no_grad():
                values=model.diagnostics(data.batch(validation[:512],device))
            if not isinstance(values,dict) or len(values)>64:raise ValueError('At most 64 named scalar diagnostics allowed')
            scalars={}
            for name,value in values.items():
                scalar=float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                if not math.isfinite(scalar):raise ValueError('Nonfinite candidate diagnostic')
                scalars[str(name)]=scalar
            diagnostics={'status':'success','origin':'candidate-supplied code; not an independently verified measurement','scope':'first 512 validation impressions, frozen best checkpoint','values':scalars}
        except Exception as error:diagnostics={'status':'failed','error':repr(error)}
    result={'status':'success','seed':args.seed,'source_sha256':sha(args.model),'effective_config':effective,
            'checkpoint_sha256':sha(args.output/'checkpoint.pt'),'data_identity':data.meta['identity'],
            'validation':best_result,'epochs':epoch_rows,'optimizer_steps':steps,'training_examples':examples,
            'train_seconds':training_seconds,'wall_seconds':time.monotonic()-started,'allocated_gpu_seconds':time.monotonic()-started if device.type=='cuda' else 0,
            'parameters':params,'embedding_parameters':embedding_params,'dense_parameters':params-embedding_params,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            'precision':'bf16 autocast' if device.type=='cuda' else 'fp32','hardware':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
            'batch_size':args.batch_size,'history_limit':args.history_limit,'calibration_only':bool(args.calibration_steps),
            'no_test_feedback':True,'initialization':'from_scratch'}
    result['candidate_diagnostics']=diagnostics
    write(args.output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=4096)
    parser.add_argument('--history-limit',type=int,default=128)
    parser.add_argument('--calibration-steps',type=int,default=0)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        write(args.output/'failure.json',{'status':'failed','error':repr(error),'source_sha256':sha(args.model),'seed':args.seed,'at':time.time()})
        raise
