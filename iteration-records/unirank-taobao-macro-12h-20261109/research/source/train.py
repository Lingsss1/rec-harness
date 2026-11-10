"""Fixed Taobao host trainer/scorer using UniRank's released feature and loader APIs."""
import argparse
import ast
import copy
import importlib.util
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
import torch

from common import read,write,sha
from selection import POLICY,aggregate,validate_policy,checkpoint_key


def check_source(path):
    tree=ast.parse(Path(path).read_text())
    allowed={'torch','numpy','math','typing','copy','functools','collections','itertools',
             'dataclasses','contextlib','unirank','model_zoo'}
    forbidden={'open','exec','eval','compile','__import__','globals','locals','vars','breakpoint',
               'load','save','savez','savez_compressed','fromfile','tofile','memmap','read_text','read_bytes',
               'write_text','write_bytes','system','popen','getenv','import_module','load_library'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(module.split('.')[0] not in allowed for module in modules):raise ValueError('Model imports outside approved numeric/framework modules')
        if isinstance(node,ast.Name) and node.id in forbidden:raise ValueError('Forbidden model operation: '+node.id)
        if isinstance(node,ast.Attribute) and (node.attr in forbidden or node.attr.startswith('__') and node.attr not in ['__init__','__class__']):
            raise ValueError('Forbidden model attribute: '+node.attr)
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['fit','evaluate','get_labels','get_loss','get_group_id']:
            raise ValueError('Host owns training/scoring methods: '+node.name)
    return tree


def framework(root):
    upstream=Path(read(root/'setup.json')['upstream_path'])
    if str(upstream) not in sys.path:sys.path.insert(0,str(upstream))
    return upstream


def parameters(root,output=None):
    framework(root)
    from unirank.utils import load_config
    params=load_config(str(root/'config'),'DIN_Taobao_Action')
    params.update(gpu=0,enable_torch_compile=False,enable_bf16=True,
                  distributed=False,rank=0,world_size=1,test_data=None,
                  model_root=str(output or root/'checkpoints'),model_id='DIN_Taobao_Action')
    return params


def prepare(root):
    params=parameters(root)
    from unirank.preprocess import FeatureProcessor,build_dataset
    encoder=FeatureProcessor(**params);build_dataset(encoder,**params)
    path=Path(params['data_root'])/params['dataset_id']/'feature_map.json'
    if not path.is_file():raise ValueError('Feature map was not generated')
    return path


def make_model(root,source,config,output,seed):
    framework(root)
    from unirank.features import FeatureMap
    params=parameters(root,output);protocol=read(root/'protocol.json')
    effective={**config}
    check_source(source)
    spec=importlib.util.spec_from_file_location('candidate_'+sha(source)[:16],source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    effective={**getattr(module,'CONFIG',{}),**effective}
    for key in ['batch_size','epochs','max_len','seed','data_root','train_data','valid_data','test_data','num_workers',
                'dense_optimizer','sparse_optimizer','dense_learning_rate','sparse_learning_rate','loss','num_tasks',
                'selection_policy','selection_metric','primary','selection_weights']:
        if key in effective:raise ValueError('Frozen host parameter cannot be supplied as model config: '+key)
    params.update(effective)
    fmap=FeatureMap(params['dataset_id'],str(Path(params['data_root'])/params['dataset_id']))
    fmap.load(str(Path(fmap.data_dir)/'feature_map.json'),params)
    constructor=dict(params);constructor.pop('model_id',None);constructor.pop('gpu',None)
    model=module.Model(fmap,model_id=output.name,gpu=0,**constructor)
    model.to(torch.device('cuda:0'))
    return model,fmap,params,effective


def loader(root,fmap,params,split):
    from unirank.pytorch.dataloaders.unirank_dataloader import UniRankDataloader
    from visibility import select_policy
    protocol=read(root/'protocol.json');select_policy(protocol['history_policy'])
    kwargs={key:value for key,value in params.items() if key not in
            ['batch_size','max_len','num_workers','shuffle','split','data_path','drop_last']}
    return UniRankDataloader(fmap,data_path=params[split+'_data'],split=split,batch_size=protocol['batch_size'],
        num_workers=protocol.get('num_workers',4),max_len=protocol['history_limit'],
        shuffle=split=='train',drop_last=split=='train',**kwargs)


def without_labels(inputs,labels,device):
    batch,items,mask,multi=inputs
    batch={name:tensor.to(device,non_blocking=True) for name,tensor in batch.items()
           if name not in labels and name not in ['user_index','item_index','seq_len']}
    items={name:tensor.to(device,non_blocking=True) for name,tensor in items.items()}
    return batch,items,mask.to(device,non_blocking=True),[value.to(device,non_blocking=True) for value in multi]


def predict(model,features,labels):
    result=model(features)
    if not isinstance(result,dict) or set(result)!={label+'_pred' for label in labels}:raise ValueError('Return one probability per registered task')
    values=torch.stack([result[label+'_pred'].reshape(-1) for label in labels],dim=-1).float()
    expected=len(features[2])
    if values.shape!=(expected,len(labels)) or not torch.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Invalid prediction shape, probability range or finiteness')
    return values


def score(model,iterator,labels,device,output):
    started=time.monotonic();model.eval();truth=[];probabilities=[];lengths=[];latencies=[]
    with torch.no_grad():
        for index,inputs in enumerate(iterator):
            target=np.stack([inputs[0][label].numpy() for label in labels],axis=-1).reshape(-1,len(labels))
            features=without_labels(inputs,labels,device)
            torch.cuda.synchronize();stamp=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):values=predict(model,features,labels)
            torch.cuda.synchronize();latencies.append(time.monotonic()-stamp)
            truth.append(target);probabilities.append(values.cpu().numpy());lengths.append(inputs[2].sum(-1).numpy())
            if index%25==0:write(output/'progress.json',{'phase':'validation','batches':index+1,'wall_seconds':time.monotonic()-started})
    y=np.concatenate(truth);p=np.concatenate(probabilities);length=np.concatenate(lengths)
    tasks={}
    for index,label in enumerate(labels):
        if len(np.unique(y[:,index]))!=2:raise ValueError('Both classes required for '+label)
        tasks[label]={'auc':float(roc_auc_score(y[:,index],p[:,index])),
                      'logloss':float(log_loss(y[:,index],np.clip(p[:,index].astype(np.float64),1e-7,1-1e-7),labels=[0,1])),
                      'positives':int(y[:,index].sum()),'samples':len(y)}
    primary=aggregate(tasks);elapsed=time.monotonic()-started
    slices={}
    for name,selected in [('empty_history',length==0),('history_1_16',(length>0)&(length<=16)),
                          ('history_17_64',(length>16)&(length<=64)),('history_gt64',length>64)]:
        item={'samples':int(selected.sum()),'per_task':{}}
        if selected.sum():
            for index,label in enumerate(labels):
                positive=int(y[selected,index].sum());both=len(np.unique(y[selected,index]))==2
                item['per_task'][label]={'positives':positive,'negatives':int(selected.sum())-positive,
                    'auc':float(roc_auc_score(y[selected,index],p[selected,index])) if selected.sum()>100 and both else None,
                    'logloss':float(log_loss(y[selected,index],np.clip(p[selected,index].astype(np.float64),1e-7,1-1e-7),labels=[0,1]))}
            item.update(aggregate(item['per_task'],allow_undefined=True))
            item['status']='complete' if item['auc'] is not None else 'insufficient_classes_or_samples_for_four_task_auc'
        slices[name]=item
    return {**primary,'per_task':tasks,'samples':len(y),
            'inference_seconds':elapsed,'examples_per_second':len(y)/elapsed,
            'batch_latency_ms_p50':float(np.percentile(latencies[3:] or latencies,50)*1000),
            'batch_latency_ms_p95':float(np.percentile(latencies[3:] or latencies,95)*1000),
            'inference_batch_size':iterator._configured_batch_size,'latency_unit':'model only, whole batch; wall time includes loading',
            'slices':slices}


def run(args):
    root=args.data.parent.parent if args.data.name=='Taobao_Action' else args.data.parent
    # The campaign is explicit: prepared is a symlink, so never resolve it to infer ownership.
    if args.campaign:root=args.campaign
    protocol=read(root/'protocol.json');args.output.mkdir(parents=True,exist_ok=True)
    validate_policy(protocol['selection_policy'])
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic();config=read(args.config) if args.config else {}
    model,fmap,params,effective=make_model(root,args.model,config,args.output,args.seed)
    labels=list(fmap.labels);train=loader(root,fmap,params,'train');valid=loader(root,fmap,params,'valid')
    embeddings={id(p) for module in model.modules() if isinstance(module,torch.nn.Embedding) for p in module.parameters(recurse=False)}
    sparse=[p for p in model.parameters() if id(p) in embeddings and p.requires_grad]
    dense=[p for p in model.parameters() if id(p) not in embeddings and p.requires_grad]
    optimizers=[torch.optim.AdamW(dense,lr=protocol['dense_learning_rate'],weight_decay=0),
                torch.optim.Adagrad(sparse,lr=protocol['sparse_learning_rate'])]
    steps=0;examples=0;epochs=[];best_key=None;best=None;training_seconds=0
    total_batches=len(train)
    for epoch in range(protocol['epochs']):
        model.train();train.set_epoch(epoch);epoch_start=time.monotonic();loss_sum=0;epoch_examples=0
        for index,inputs in enumerate(train):
            targets=torch.stack([inputs[0][label].to(device).reshape(-1) for label in labels],dim=-1).float()
            features=without_labels(inputs,labels,device)
            for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):values=predict(model,features,labels)
            loss=torch.nn.functional.binary_cross_entropy(values,targets,reduction='none').mean(0).sum()
            if hasattr(model,'training_loss'):loss=model.training_loss(features,values,targets)
            if loss.ndim!=0 or not torch.isfinite(loss):raise ValueError('Training loss must be one finite scalar')
            loss.backward();torch.nn.utils.clip_grad_norm_(dense,protocol['gradient_clip'])
            for optimizer in optimizers:optimizer.step()
            steps+=1;examples+=len(targets);epoch_examples+=len(targets);loss_sum+=float(loss.detach())*len(targets)
            if steps%10==0:
                write(args.output/'progress.json',{'phase':'train','epoch':epoch+1,'steps':steps,
                    'total_steps':total_batches*protocol['epochs'],'examples':examples,'loss':float(loss.detach()),
                    'wall_seconds':time.monotonic()-started})
            if args.calibration_steps and steps>=args.calibration_steps:break
        seconds=time.monotonic()-epoch_start;training_seconds+=seconds
        if args.calibration_steps:
            result={'status':'success','calibration_only':True,'steps':steps,'training_examples':examples,
                    'wall_seconds':time.monotonic()-started,'train_seconds':training_seconds,'parameters':sum(p.numel() for p in model.parameters())}
            write(args.output/'result.json',result);return
        validation=score(model,valid,labels,device,args.output)
        if validation['samples']!=read(root/'data/Taobao_Action/meta_data.json')['sample_size']['valid']:
            raise ValueError('Validation coverage differs from the frozen released split')
        epochs.append({'epoch':epoch+1,'train_seconds':seconds,'training_loss':loss_sum/max(1,epoch_examples),'validation':validation})
        selection_key=checkpoint_key(validation)
        if best_key is None or selection_key<best_key:
            best_key=selection_key;best=copy.deepcopy(validation);torch.save(model.state_dict(),args.output/'checkpoint.pt')
    params_total=sum(p.numel() for p in model.parameters());embedding_params=sum(p.numel() for p in sparse)
    result={'status':'success','seed':args.seed,'source_sha256':sha(args.model),'effective_config':effective,
        'data_identity':protocol['data_identity'],'protocol_hash':protocol['identity'],'validation':best,'epochs':epochs,'optimizer_steps':steps,
        'training_examples':examples,'train_seconds':training_seconds,'wall_seconds':time.monotonic()-started,
        'allocated_gpu_seconds':time.monotonic()-started,'parameters':params_total,'embedding_parameters':embedding_params,
        'dense_parameters':params_total-embedding_params,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'precision':'bf16 autocast','hardware':torch.cuda.get_device_name(),'batch_size':protocol['batch_size'],
        'history_limit':protocol['history_limit'],'history_policy':protocol['history_policy'],'initialization':'from_scratch',
        'selection_policy':POLICY,
        'no_test_feedback':True,'candidate_diagnostics':{'status':'not_requested'},'checkpoint_sha256':sha(args.output/'checkpoint.pt')}
    write(args.output/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--campaign',type=Path);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20262027);parser.add_argument('--calibration-steps',type=int,default=0)
    # Preserve the existing topic runner interface; actual values always come from protocol.json.
    parser.add_argument('--epochs',type=int);parser.add_argument('--batch-size',type=int);parser.add_argument('--history-limit',type=int)
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        write(args.output/'failure.json',{'status':'failed','error':repr(error),'at':time.time()});raise
