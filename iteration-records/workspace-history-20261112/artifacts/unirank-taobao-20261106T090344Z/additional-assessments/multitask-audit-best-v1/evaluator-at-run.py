"""Explicit supplementary evaluation of a frozen validation-audit winner."""
import argparse
import fcntl
import os
from pathlib import Path
import random
import shutil
import sys
import time

from common import read,write,sha,identity
from selection import POLICY,aggregate


def run(root,output):
    started=time.monotonic()
    audit_path=root/'audits/multitask-selection-v1/report.json';audit=read(audit_path)
    best=audit['audit_validation_best'];trial=Path(best['model_path']).parent
    recorded=read(trial/'result.json');config=read(trial/'model-config.json');protocol=read(root/'protocol.json')
    source=trial/'model.py';checkpoint=trial/'checkpoint.pt'
    if sha(source)!=best['source_sha256'] or sha(source)!=recorded['source_sha256']:raise ValueError('Source identity mismatch')
    if sha(checkpoint)!=recorded['checkpoint_sha256']:raise ValueError('Checkpoint identity mismatch')
    for path,digest in audit['input_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Historical validation receipt changed')
    for name,digest in protocol['source_sha256'].items():
        if sha(root/'source'/name)!=digest:raise ValueError('Historical host source changed')
    if recorded['protocol_hash']!=protocol['identity'] or recorded['data_identity']!=protocol['data_identity']:
        raise ValueError('Historical protocol/data mismatch')
    freeze={'frozen_unix':time.time(),'id':best['id'],'role':best['role'],
        'selection':'Maximum four-task macro validation AUC in historical validation audit',
        'selection_policy':POLICY,'validation':best,'source_path':str(source),'source_sha256':sha(source),
        'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'config':config,
        'seed':recorded['seed'],'historical_protocol_hash':protocol['identity'],'data_identity':protocol['data_identity'],
        'audit_report_sha256':sha(audit_path),'evaluator_sha256':sha(Path(__file__)),
        'evaluation_type':'supplementary evaluation after a prior test suite has been observed; not a new unseen-holdout confirmation',
        'user_request':'测试一下该最佳模型','no_retraining':True,'no_further_model_selection_from_test':True}
    freeze['identity']=identity(freeze);write(output/'selection-freeze.json',freeze)
    shutil.copyfile(Path(__file__),output/'evaluator-at-run.py')
    write(output/'evaluation-start.json',{'at_unix':time.time(),'pid':os.getpid(),
        'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'freeze_identity':freeze['identity']})
    for row in read(root/'dataset-source-manifest.json')['files']:
        if sha(root/'data/Taobao_Action'/row['path'])!=row['local_sha256']:raise ValueError('Released data digest mismatch')

    # Restore using the exact historical trainer/model interface. The new policy
    # aggregates the returned per-task values without changing model inference.
    sys.path.insert(0,str(root/'source'))
    import train as host
    import torch
    import numpy as np
    torch.set_num_threads(4);seed=recorded['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    model,fmap,params,effective=host.make_model(root,source,config,output,seed)
    if effective!=recorded['effective_config']:raise ValueError('Restored configuration mismatch')
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'),strict=True)
    evaluations={}
    for split in ['valid','test']:
        folder=output/split;folder.mkdir()
        write(output/'status.json',{'phase':'validation_replay' if split=='valid' else 'supplementary_test',
            'at_unix':time.time(),'freeze_identity':freeze['identity']})
        if split=='test':params['test_data']=str(root/'data/Taobao_Action/test/data')
        metrics=host.score(model,host.loader(root,fmap,params,split),list(fmap.labels),torch.device('cuda:0'),folder)
        if metrics['samples']!=read(root/'data/Taobao_Action/meta_data.json')['sample_size'][split]:
            raise ValueError('Incomplete sample coverage')
        if split=='valid':
            for task,values in metrics['per_task'].items():
                for name in ['auc','logloss']:
                    if abs(values[name]-recorded['validation']['per_task'][task][name])>1e-8:
                        raise ValueError('Validation replay mismatch: '+task+' '+name)
            write(output/'validation-replay-gate.json',{'status':'passed','tolerance':1e-8,'freeze_identity':freeze['identity']})
        evaluations[split]={'per_task':metrics['per_task'],**aggregate(metrics['per_task']),
            'samples':metrics['samples'],'inference_seconds':metrics['inference_seconds'],
            'batch_latency_ms_p50':metrics['batch_latency_ms_p50'],'batch_latency_ms_p95':metrics['batch_latency_ms_p95'],
            'historical_click_slices':metrics['slices']}
        write(folder/'result.json',evaluations[split])
    # Reuse the already evaluated DIN baseline rather than scoring it again.
    previous=read(root/'final-assessment/report.json');din=previous['test']['din']['metrics']
    baseline={'per_task':din['per_task'],**aggregate(din['per_task']),'samples':din['samples'],
        'source':'previous frozen-suite DIN test result; not re-executed'}
    test=evaluations['test']
    report={'status':'complete','completed_unix':time.time(),'evaluation_type':freeze['evaluation_type'],
        'selection_freeze':freeze,'validation_replay':evaluations['valid'],'test':test,'din_test_reference':baseline,
        'gain_vs_din':{'macro_auc':test['macro_auc']-baseline['macro_auc'],
            'macro_logloss_reduction':baseline['macro_logloss']-test['macro_logloss'],
            'per_task':{task:{name:test['per_task'][task][name]-baseline['per_task'][task][name] for name in ['auc','logloss']}
                for task in POLICY['tasks']}},
        'parameters':recorded['parameters'],'historical_train_seconds':recorded['train_seconds'],
        'evaluation_wall_seconds':time.monotonic()-started,'single_training_seed':True,'no_retraining':True,
        'budget_scope':'User-requested additional evaluation outside the completed48h search; all evaluation cost recorded separately.'}
    write(output/'report.json',report);write(output/'status.json',{'phase':'complete','at_unix':time.time()})
    print({'status':'complete','test_macro_auc':test['macro_auc'],'din_macro_auc':baseline['macro_auc'],
        'macro_auc_gain':report['gain_vs_din']['macro_auc'],'samples':test['samples'],
        'evaluation_wall_seconds':report['evaluation_wall_seconds']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args();root=args.root.resolve()
    output=args.output or root/'additional-assessments/multitask-audit-best-v1';output.mkdir(parents=True,exist_ok=True)
    if (output/'evaluation-start.json').exists() or (output/'selection-freeze.json').exists():
        raise SystemExit('Evaluation has already been started; inspect existing records before any retry')
    with (output/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(root,output)
        except BaseException as error:
            write(output/'failure.json',{'at_unix':time.time(),'error':repr(error)});raise
