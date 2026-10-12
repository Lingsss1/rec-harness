"""Development-only history slices bound to controller events before review."""
from pathlib import Path

from rec_harness.core import ContractError,digest,file_hash,read_json
from .amazon_controls import load_config
from . import history_diagnostic


def development_history(controller,folder,binding):
    directory=folder/'development-history'
    records=[e['payload'] for e in controller.store.events()
             if e['kind']=='development_history_completed' and e['payload']['round']==folder.name]
    if len(records)>1:raise ContractError('duplicate development history record')
    if records:
        ref=records[0]['report'];path=controller.artifacts.verify(ref)
        if file_hash(directory/'report.json')!=ref['sha256']:
            raise ContractError('development history projection changed')
        report=read_json(path)
    else:
        if directory.exists():
            suffix=len(list(folder.glob('development-history-interrupted-*')))+1
            directory.rename(folder/('development-history-interrupted-'+str(suffix)))
        report=history_diagnostic.diagnose(controller.study,directory,binding['base_model_id'],
            binding['candidate_model_version'],partitions=['development'])
    cfg=load_config(controller.study);plan=report['plan']
    if (report['hash']!=digest({k:v for k,v in report.items() if k!='hash'})
        or report['plan_hash']!=digest(plan)
        or plan['baseline']!=binding['base_model_id'] or plan['candidate']!=binding['candidate_model_version']
        or plan['protocol_hash']!=cfg['protocol_hash'] or plan['partitions']!=['development']
        or report['diagnostic_code_hash']!=file_hash(history_diagnostic.__file__)):
        raise ContractError('development history binding changed')
    expected={(dataset,seed) for dataset in cfg['datasets'] for seed in cfg['development_seeds']}
    if (len(report['rows'])!=len(expected)
        or {(r['dataset'],r['seed']) for r in report['rows']}!=expected
        or any(r['partition']!='development' for r in report['rows'])):
        raise ContractError('development history contains unexpected phases or conditions')
    for name,expected_hash in report['input_hashes'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ContractError('invalid history dependency path')
        path=controller.study/relative if relative.parts[0]=='jobs' else Path(cfg['data'])/relative
        if file_hash(path)!=expected_hash:raise ContractError('development history input changed')
    if not records:
        ref=controller.artifacts.put(directory/'report.json')
        controller.store.event('development_history_completed',{'round':folder.name,'report':ref,
            'report_hash':report['hash'],'scope':'development aggregate history slices; no new training'})
    # Explicit projection excludes input paths, user IDs, ranks and private phases.
    fields=['history_length_min','history_length_max','users','baseline','candidate','delta',
            'hit_wins','hit_losses','hit_unchanged']
    return {'report_hash':report['hash'],'baseline':plan['baseline'],'candidate':plan['candidate'],
        'rows':[{'dataset':row['dataset'],'seed':row['seed'],
                 'slices':[{key:part[key] for key in fields} for part in row['slices']]} for row in report['rows']],
        'qualification':'Exploratory development aggregates by total training-history length; not causal mechanism evidence or confirmation.'}
