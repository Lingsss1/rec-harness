"""One closed, frozen-suite assessment of existing checkpoints; no retraining."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import read,write,sha,identity
from selection import best_trial,trial_score,trial_logloss,validate_policy


def selection(root):
    protocol=read(root/'protocol.json')
    validate_policy(protocol.get('selection_policy'))
    rows=read(root/'references.json')['trials']
    states=[read(root/f'topics/gpu-{gpu}/state.json') for gpu in range(4)]
    if any(state['phase'] not in ['research_stopped','cancelled','failed_before_assessment'] for state in states):
        raise ValueError('Research must be closed before final selection')
    for state in states:rows+=state['trials']
    eligible=[row for row in rows if row['status']=='success' and row.get('metrics',{}).get('validation')]
    best=best_trial({'trials':eligible})
    din=next(row for row in eligible if row.get('baseline_kind')=='din')
    # The winning replacement control is compared with its intended mechanism
    # candidate from the same prospective study, chosen before any test reads.
    candidates=[row for row in eligible if row.get('study_ref')==best.get('study_ref') and row.get('role')=='candidate']
    matched=best_trial({'trials':candidates})
    picked=[('din',din),('selected',best)]
    if matched and matched['id']!=best['id']:picked.append(('matched_candidate',matched))
    protocol=read(root/'protocol.json');manifest=read(root/'dataset-source-manifest.json')
    for name,digest in protocol['source_sha256'].items():
        if sha(root/'source'/name)!=digest:raise ValueError('Frozen host source changed: '+name)
    suite=[]
    for key,row in picked:
        source=Path(row['model_path']);checkpoint=Path(row['output'])/'checkpoint.pt'
        if sha(source)!=row['source_sha256']:raise ValueError('Candidate source identity mismatch')
        if sha(checkpoint)!=row['metrics']['checkpoint_sha256']:raise ValueError('Checkpoint identity mismatch')
        if row['metrics']['data_identity']!=protocol['data_identity']:raise ValueError('Foreign data identity')
        suite.append({'key':key,'id':row['id'],'role':row['role'],'source_path':str(source),
            'source_sha256':sha(source),'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'config':row['config'],'effective_config':row['metrics']['effective_config'],'seed':row['seed'],
            'validation':row['metrics']['validation'],'parameters':row['metrics']['parameters'],
            'train_seconds':row['metrics']['train_seconds'],'training_job_wall_seconds':row['wall_seconds'],
            'study_ref':row.get('study_ref'),'selection_reason':
                'initial empirical baseline' if key=='din' else 'maximum validation four-task macro AUC under the frozen tie-breaking policy, including controls' if key=='selected' else
                'closest intended candidate from the same study as the selected replacement control'})
    freeze={'frozen_unix':time.time(),'selection_metric':protocol['primary'],'selection_policy':protocol['selection_policy'],
        'protocol_hash':protocol['identity'],'data_identity':protocol['data_identity'],
        'dataset_revision':manifest['revision'],'suite':suite,
        'heldout_policy':'One test evaluation per frozen checkpoint; no test-based model reselection or further search.',
        'budget_scope':'Post-search final evaluation requested by user; recorded separately outside original48h. No training or search is added.',
        'test_observations_read_at_freeze':False,'assessment_script_sha256':sha(Path(__file__)),
        'historical_status':read(root/'status.json'),
        'search_summary':{'rounds':sum(s['completed_research_rounds'] for s in states),
            'training_attempts':sum(len(s['trials']) for s in states),
            'successful_attempts':sum(t['status']=='success' for s in states for t in s['trials']),
            'job_wall_seconds':sum(t['wall_seconds'] for s in states for t in s['trials']),
            'observed_training_seeds':sorted({t['seed'] for t in eligible})}}
    freeze['identity']=identity(freeze);return freeze


def worker(root,key,split):
    assessment=root/'final-assessment';freeze=read(assessment/'selection-freeze.json')
    row=next(row for row in freeze['suite'] if row['key']==key)
    output=assessment/key/split;output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists() or (output/'worker-start.json').exists():
        raise ValueError('Assessment execution already started; inspect before any retry')
    write(output/'worker-start.json',{'started_unix':time.time(),'split':split,'pid':os.getpid(),
        'freeze_identity':freeze['identity'],'source_sha256':row['source_sha256'],'checkpoint_sha256':row['checkpoint_sha256']})
    sys.path.insert(0,str(root/'source'))
    import train as host
    import numpy as np
    import torch
    started=time.monotonic();seed=row['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    if sha(Path(row['source_path']))!=row['source_sha256'] or sha(Path(row['checkpoint_path']))!=row['checkpoint_sha256']:
        raise ValueError('Frozen suite artifact changed')
    protocol=read(root/'protocol.json')
    if protocol['identity']!=freeze['protocol_hash']:raise ValueError('Research protocol changed')
    model,fmap,params,effective=host.make_model(root,Path(row['source_path']),row['config'],output,seed)
    if effective!=row['effective_config']:raise ValueError('Restored effective configuration differs')
    state=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    if split=='test':
        gate=read(assessment/'validation-replay-gate.json')
        if gate['status']!='passed' or gate['freeze_identity']!=freeze['identity']:raise ValueError('Validation replay gate missing')
        params['test_data']=str(root/'data/Taobao_Action/test/data')
    iterator=host.loader(root,fmap,params,split)
    metrics=host.score(model,iterator,list(fmap.labels),torch.device('cuda:0'),output)
    expected=read(root/'data/Taobao_Action/meta_data.json')['sample_size'][split]
    if metrics['samples']!=expected:raise ValueError('Incomplete evaluated sample coverage')
    if split=='valid':
        for label,values in metrics['per_task'].items():
            recorded=row['validation']['per_task'][label]
            if abs(values['auc']-recorded['auc'])>1e-8 or abs(values['logloss']-recorded['logloss'])>1e-8:
                raise ValueError('Checkpoint validation replay mismatch: '+label)
    result={'status':'success','split':split,'id':row['id'],'key':key,'metrics':metrics,
        'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,
        'wall_seconds':time.monotonic()-started,'freeze_identity':freeze['identity'],
        'source_sha256':row['source_sha256'],'checkpoint_sha256':row['checkpoint_sha256'],
        'no_training':True,'post_48h_assessment':True,'history_policy':protocol['history_policy'],
        'hardware':torch.cuda.get_device_name(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    write(output/'result.json',result)


def run(root):
    assessment=root/'final-assessment';assessment.mkdir(exist_ok=True)
    if (assessment/'selection-freeze.json').exists():raise ValueError('Frozen assessment exists; do not automatically reopen or rerun')
    started=time.monotonic();freeze=selection(root);write(assessment/'selection-freeze.json',freeze)
    status=read(root/'status.json');status.update(assessment_launched=True,final_assessment='final-assessment',assessment_requested_unix=time.time())
    write(root/'status.json',status)
    # Validate every released file digest before parsing any held-out examples.
    for row in read(root/'dataset-source-manifest.json')['files']:
        if sha(root/'data/Taobao_Action'/row['path'])!=row['local_sha256']:raise ValueError('Released data digest mismatch: '+row['path'])

    def execute(item):
        index,row,split=item;output=assessment/row['key']/split;output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--worker',row['key'],'--split',split]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        with (output/'process.log').open('w') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            write(output/'process.json',{'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                'started_unix':time.time(),'argv':command,'split':split})
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        if code:raise ValueError('Assessment worker failed: '+str(output))
        return read(output/'result.json')

    for split in ['valid','test']:
        write(assessment/'status.json',{'phase':'validation_replay' if split=='valid' else 'heldout_test','at_unix':time.time(),
            'freeze_identity':freeze['identity'],'post_48h_assessment':True})
        with ThreadPoolExecutor(len(freeze['suite'])) as pool:
            results=list(pool.map(execute,[(i,row,split) for i,row in enumerate(freeze['suite'])]))
        if split=='valid':
            write(assessment/'validation-replay-gate.json',{'status':'passed','freeze_identity':freeze['identity'],
                'at_unix':time.time(),'models':[r['id'] for r in results],'tolerance':1e-8})
    tests={result['key']:result for result in results}
    primary=tests['selected']['metrics'];baseline=tests['din']['metrics']
    report={'status':'complete','completed_unix':time.time(),'selection_freeze':freeze,
        'test':tests,'selection_metric':freeze['selection_metric'],
        'primary_gain':{'macro_auc_absolute':primary['macro_auc']-baseline['macro_auc'],
            'macro_logloss_reduction':baseline['macro_logloss']-primary['macro_logloss'],
            'per_task':{task:{'auc_delta':primary['per_task'][task]['auc']-baseline['per_task'][task]['auc'],
                'logloss_delta':primary['per_task'][task]['logloss']-baseline['per_task'][task]['logloss']}
                for task in freeze['selection_policy']['tasks']}},
        'assessment_wall_seconds':time.monotonic()-started,
        'assessment_job_wall_seconds':sum(read(assessment/row['key']/split/'result.json')['wall_seconds'] for row in freeze['suite'] for split in ['valid','test']),
        'budget_scope':'Validation replay and final testing occurred after48h search; separately recorded, no extra search or training.',
        'limitations':['One training seed; no independent-seed uncertainty or significance claim.',
            'The best validation model is a replacement control, not evidence that every proposed mechanism is useful.',
            'Main selection metric is four-task macro AUC; every task must be examined for trade-offs.',
            'Official offline feedback semantics and other frozen benchmark deviations remain applicable.']}
    write(assessment/'report.json',report)
    write(assessment/'status.json',{'phase':'complete','at_unix':time.time(),'freeze_identity':freeze['identity']})
    print(json.dumps({'status':'complete','test_samples':primary['samples'],'selected_macro_auc':primary['macro_auc'],
        'baseline_macro_auc':baseline['macro_auc'],'macro_auc_gain':primary['macro_auc']-baseline['macro_auc'],'assessment_wall_seconds':report['assessment_wall_seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker');parser.add_argument('--split',choices=['valid','test'])
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.worker:worker(root,args.worker,args.split)
        else:
            (root/'final-assessment').mkdir(exist_ok=True)
            with (root/'final-assessment/assessment.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(root)
    except BaseException as error:
        folder=root/'final-assessment'
        if args.worker:folder=folder/args.worker/args.split
        write(folder/'failure.json',{'at_unix':time.time(),'error':repr(error)});raise
