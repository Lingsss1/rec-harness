"""Calibrate DIN, establish one shared reference, then run independent topics."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from common import read,write,sha,identity
import controller as host

HERE=Path(__file__).resolve().parent


def phase(root,name,**values):
    write(root/'status.json',{'phase':name,'updated_unix':time.time(),**values})


def audit(root):
    """Read train/valid only; test observations stay closed to research."""
    data=root/'data/Taobao_Action';meta=read(data/'meta_data.json');result={}
    labels=['is_click','cart','fav','buy']
    for split in ['train','valid']:
        files=sorted((data/split/'data').glob('*.parquet'))
        count=0;positives=np.zeros(4,dtype=np.int64);columns=None
        for path in files:
            parquet=pq.ParquetFile(path);count+=parquet.metadata.num_rows
            if columns is None:columns=parquet.schema.names
            elif parquet.schema.names!=columns:raise ValueError('Inconsistent data schema')
            table=parquet.read(columns=labels).to_pandas().to_numpy()
            if not np.isin(table,[0,1]).all():raise ValueError('Non-binary or missing labels')
            positives+=table.sum(0).astype(np.int64)
            stem=path.stem
            for side in ['user_info','item_info']:
                if not (data/split/side/(stem+'.parquet')).is_file():raise ValueError('Missing side block: '+stem)
        if count!=meta['sample_size'][split]:raise ValueError('Released split size mismatch: '+split)
        result[split]={'samples':count,'blocks':len(files),'columns':columns,
            'labels':{label:{'positive':int(value),'negative':count-int(value),'rate':float(value/count)}
                      for label,value in zip(labels,positives)}}
    result['test']='Closed; only public metadata count and file digests are used. No labels or predictions read.'
    result['assumptions']=[
        'Official user filtering and feature vocabulary counts use the released full benchmark window; this is inherited offline selection, not verified online construction.',
        'Publisher static user/ad snapshots have undocumented as-of dates.',
        'Auxiliary positives match user/category/brand over the next24h; they are not direct ad-ID conversion labels.',
        'history_policy determines whether historical delayed outcomes use official final feedback or conservative24h maturity.'
    ]
    write(root/'data-audit.json',result);return result


def base_protocol(root,policy):
    manifest=read(root/'dataset-source-manifest.json')
    identity_input={'repository':manifest['repo_id'],'revision':manifest['revision'],
        'files':[{'path':row['path'],'sha256':row['local_sha256']} for row in manifest['files']],
        'dataset':'Taobao_Action','history_policy':policy,'history_limit':100,
        'split':'released 2017-05-06..11 train, May12 valid, May13 test, Asia/Shanghai'}
    return {'campaign_root':str(root),'data_identity':identity(identity_input),'dataset_revision':manifest['revision'],
        'epochs':1,'batch_size':8192,'history_limit':100,'train_seed':20262027,'num_workers':0,
        'tasks':['is_click','cart','fav','buy'],'primary':'is_click ROC-AUC','secondary':'all four task AUC and unweighted Logloss; mean AUC reported separately',
        'dense_optimizer':'AdamW','dense_learning_rate':0.0001,'sparse_optimizer':'Adagrad','sparse_learning_rate':0.05,
        'gradient_clip':10.0,'gradient_clip_scope':'dense parameters only, following upstream split optimizer training',
        'precision':'bf16 autocast; TF32 disabled; no torch.compile','history_policy':policy,
        'history_visibility':('Exact released prefix/action inputs. Delayed feedback may not be mature at prediction time; results only establish this offline benchmark.' if policy=='official_offline' else
            'Exclude timestamp ties and future exposures. For age<=86400 seconds, keep only historical click/exposure; preserve full feedback only once window is strictly past. Conservative because event timestamps are absent from release.'),
        'drop_last':{'train':True,'valid':False},'full_validation_samples':2871767,
        'training':'scratch; official single-pass recipe adapted to DIN; sum four BCE unless declared training-loss hook',
        'baseline':'Initial empirical reference: DIN local activation over item-field history with action values and four separate heads; field/task adaptation, not exact original DIN reproduction',
        'research_goal':read(root/'research-scope.json')['objective'],
        'architecture_search':read(root/'research-scope.json')['architecture_search'],
        'research_budget':{key:read(root/'budget.json')[key] for key in ['seconds','start_event','start_unix','deadline_unix']},
        'test_policy':'closed during research; any final suite must be frozen before held-out scoring',
        'source_sha256':{name:sha(HERE/name) for name in ['baseline.py','train.py','visibility.py','data.py']},
        'upstream_commit':read(root/'setup.json')['upstream_commit'],
        'benchmark_deviations':['DIN is the initial empirical reference; unified research candidates can use entirely different architectures.',
            'One GPU per model; batch8192 per model (single-device rather than published distributed global batch).',
            'Click AUC selects models, four-task mean AUC is auxiliary.',
            'Validation includes every sample; upstream default drop_last may omit remainders.',
            'In-process DataLoader (num_workers=0): container shared memory is512MiB; all compared runs use the same loading policy.',
            'No torch.compile; BF16; TF32 disabled.']}


def run(root,policy):
    if (root/'STOP').exists() or (root/'topic-board.json').exists():raise ValueError('Use a new campaign for stopped/started research')
    write(root/'runner-process.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]})
    write(root/'visibility-choice.json',{'policy':policy,'at_unix':time.time(),'origin':'explicit campaign argument'})
    while read(root/'acquisition-status.json')['phase']!='downloaded_and_verified':
        if read(root/'acquisition-status.json')['phase']=='failed':raise ValueError('Acquisition failed; inspect acquisition-status.json')
        if (root/'STOP').exists():raise ValueError('Campaign stopped during acquisition')
        phase(root,'waiting_for_dataset',history_policy=policy);time.sleep(5)
    from budget import refresh
    budget=refresh(root);deadline=budget['deadline_unix']
    if time.time()>=deadline:raise ValueError('Research budget expired after dataset verification')
    phase(root,'data_audit',history_policy=policy);audit(root)
    from train import prepare
    prepare(root)
    protocol=base_protocol(root,policy)
    protocol.update(trial_cap_seconds=21600,estimated_unit_seconds=1800)
    protocol['identity']=identity(protocol);write(root/'protocol.json',protocol)
    phase(root,'din_calibration',history_policy=policy)
    result=host.train_job(data=root/'prepared',source=HERE/'baseline.py',config={},out=root/'calibration',
                          seed=protocol['train_seed'],gpu=0,protocol=protocol,deadline=deadline,calibration=20)
    if result['status']!='success':raise ValueError('DIN calibration failed; inspect calibration/process.log')
    metrics=result['metrics'];train_samples=read(root/'data-audit.json')['train']['samples']
    seconds_per_step=metrics['train_seconds']/metrics['steps']
    estimated_train=seconds_per_step*math.floor(train_samples/protocol['batch_size'])
    estimated_valid=max(60,estimated_train*0.2)
    estimated=estimated_train+estimated_valid+30
    protocol.pop('identity');protocol.update(estimated_unit_seconds=estimated,estimated_validation_seconds=estimated_valid,
        trial_cap_seconds=min(21600,max(900,2.5*estimated)),frozen_unix=time.time())
    protocol['identity']=identity(protocol);write(root/'protocol.json',protocol)
    budget.update(single_experiment_limit_seconds=protocol['trial_cap_seconds'],measured_calibration_seconds=metrics['wall_seconds'])
    write(root/'budget.json',budget)
    if deadline-time.time()<estimated:raise ValueError('Insufficient remaining budget for full DIN reference')
    phase(root,'din_baseline',history_policy=policy)
    row=host.train_job(data=root/'prepared',source=HERE/'baseline.py',config={},out=root/'baselines/din',
        seed=protocol['train_seed'],gpu=0,protocol=protocol,deadline=deadline)
    if row['status']!='success':raise ValueError('DIN baseline failed; inspect baselines/din/process.log')
    row.update(id='baseline-din',parent=None,role='baseline',baseline_kind='din',round=-1,
        variant_name='din_four_task',model_path=str(root/'baselines/din/model.py'),
        note='Sole shared reference; all four labels are trained, click AUC selects models.')
    write(root/'references.json',{'trials':[row],'protocol_hash':protocol['identity']})
    write(root/'baseline-summary.json',{'id':row['id'],'validation':row['metrics']['validation'],
        'parameters':row['metrics']['parameters'],'wall_seconds':row['wall_seconds'],
        'observed_vs_estimated_ratio':row['wall_seconds']/estimated})
    from topics import TopicPool
    questions=read(root/'topics.json')['topics']
    pool=TopicPool(root,budget['gpus'],questions,[row],protocol,deadline,decision_every=6)
    # TopicPool intentionally refuses existing runtime status. Preserve setup
    # receipt before handing this fresh research state to the original wrapper.
    shutil.copyfile(root/'status.json',root/'preparation-status.json');(root/'status.json').unlink()
    pool.run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--history-policy',required=True,choices=['official_offline','matured_24h'])
    args=parser.parse_args();root=args.root.resolve();root.mkdir(exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(root,args.history_policy)
        except BaseException as error:
            phase(root,'setup_failed',error=repr(error));raise
