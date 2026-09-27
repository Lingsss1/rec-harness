"""Observed development history lengths available before hypothesis generation."""
from collections import Counter
from pathlib import Path
import time

from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from .amazon_controls import load_config
from .amazon_report import audit_job
from . import history_diagnostic


def baseline_history(controller,folder,baseline):
    cfg=load_config(controller.study);folder=Path(folder)
    records=[e['payload'] for e in controller.store.events()
             if e['kind']=='baseline_history_completed' and e['payload']['round']==folder.name]
    if len(records)>1:raise ContractError('duplicate baseline history record')
    path=folder/'baseline-history.json'
    code={name:file_hash(name) for name in [__file__,history_diagnostic.__file__]}
    conditions=[(d,s) for d in cfg['datasets'] for s in cfg['development_seeds']]
    inputs={};rows=[];start=time.monotonic()
    # Audit every current receipt, including when reusing an aggregate report.
    for dataset,seed in conditions:
        result,_=audit_job(controller.study,cfg,baseline,dataset,seed)
        inputs[f'jobs/{baseline}-{dataset}-{seed}/completed.json']=digest(result)
        ranks=controller.study/'jobs'/f'{baseline}-{dataset}-{seed}'/'validation_ranks.json'
        inputs[str(ranks.relative_to(controller.study))]=file_hash(ranks)
    for dataset in cfg['datasets']:
        data=Path(cfg['data'])/(dataset+'_train.txt')
        if file_hash(data)!=cfg['data_hashes'][data.name]:raise ContractError('baseline history data changed')
        inputs[data.name]=file_hash(data)
    binding={'baseline':baseline,'protocol_hash':cfg['protocol_hash'],
             'conditions':[[d,s] for d,s in conditions],'input_hashes':inputs,'code_hashes':code}
    if records:
        ref=records[0]['report'];archived=controller.artifacts.verify(ref)
        if file_hash(path)!=ref['sha256']:raise ContractError('baseline history projection changed')
        report=read_json(archived)
        if report['binding']!=binding:raise ContractError('baseline history binding changed')
    else:
        for dataset in cfg['datasets']:
            data=Path(cfg['data'])/(dataset+'_train.txt');lengths=Counter()
            with data.open() as stream:
                for line in stream:
                    user,_=map(int,line.split());lengths[user]+=1
            if file_hash(data)!=inputs[data.name]:raise ContractError('baseline history data changed during read')
            for seed in cfg['development_seeds']:
                ranks_path=controller.study/'jobs'/f'{baseline}-{dataset}-{seed}'/'validation_ranks.json'
                ranks=read_json(ranks_path)
                if file_hash(ranks_path)!=inputs[str(ranks_path.relative_to(controller.study))]:
                    raise ContractError('baseline history ranks changed during read')
                slices=history_diagnostic.summarize(ranks,ranks,lengths)
                rows.append({'dataset':dataset,'seed':seed,
                    'slices':[{'history_length_min':p['history_length_min'],'history_length_max':p['history_length_max'],
                               'users':p['users'],'metrics':p['baseline']} for p in slices]})
        report={'schema':'amazon-baseline-history-v1','binding':binding,'rows':rows,
                'local_wall_seconds':time.monotonic()-start}
        report['hash']=digest(report);write_json(path,report)
        controller.store.event('baseline_history_completed',{'round':folder.name,
            'report':controller.artifacts.put(path),'report_hash':report['hash'],
            'scope':'Development baseline aggregate; no candidate, confirmation, new training or model inference.'})
    if report['hash']!=digest({k:v for k,v in report.items() if k!='hash'}):raise ContractError('baseline history hash changed')
    return {'report_hash':report['hash'],'baseline':baseline,'rows':report['rows'],
            'qualification':'Exploratory development distribution by total training history, not truncated model input length. No causal long-range deficit, new inference or confirmation evidence.'}
