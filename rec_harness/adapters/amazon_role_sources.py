"""Check every source promised by a generated-control role before reuse."""
from pathlib import Path
from rec_harness.core import ContractError,file_hash,safe_file,read_json


def freeze_source_history(config,study):
    if 'prior_development_sources' in config:
        raise ContractError('prior_development_sources is controller-owned')
    if 'source_history_studies' not in config:return
    paths=config.pop('source_history_studies')
    if not isinstance(paths,list) or any(not isinstance(p,str) or not p for p in paths):
        raise ContractError('source_history_studies must be study paths')
    from .amazon_controls import load_config
    from rec_harness.store import Store
    records=[]
    for path in dict.fromkeys(paths):
        root=Path(path).resolve()
        store=Store(root/'memory',read_only=True)
        try:
            previous=load_config(root,store)
            for key in ['protocol_hash','data_hashes','development_seeds']:
                if previous[key]!=study[key]:raise ContractError('source history changed '+key)
            records.append({'study':str(root),'event_head':store.verify()['head'],
                            'sources':development_sources(root,previous)})
        finally:store.close()
    config['prior_development_sources']=records


def development_sources(study,config,exclude=()):
    """Public development source identities only; never read confirmation results."""
    rows={}
    for version in sorted(config['model_versions']):
        if version in exclude:continue
        for dataset in config['datasets']:
            for seed in config['development_seeds']:
                path=Path(study)/'jobs'/f'{version}-{dataset}-{seed}'/'completed.json'
                if not path.is_file():continue
                sha=read_json(path).get('source_hash')
                if sha:
                    if sha!=config['model_versions'][version]['sha256']:
                        raise ContractError('development source receipt differs from registered model')
                    rows[(version,sha)]={'model_version':version,'source_hash':sha}
    return list(rows.values())


def verify_new_sources(directory,names,previous):
    known={row['source_hash']:row['model_version'] for row in previous}
    for name in names:
        sha=file_hash(safe_file(directory,name))
        if sha in known:
            raise ContractError('new source '+name+' duplicates previously evaluated model '+known[sha]+
                '; implement a distinct falsifiable candidate/control within the frozen proposal. '
                'Renaming a file or model does not make it new; cosmetic edits do not establish a distinct mechanism.')


def verify_generated_sources(directory,role,context,accepted):
    generated=context.get('generated_controls',[]) if role=='experiment' else []
    hashes=accepted.get('model_source_hashes')
    if not generated and hashes is None:return
    expected=['model.py',*[item['file'] for item in generated]]
    if not isinstance(hashes,dict) or len(set(expected))!=len(expected) or set(hashes)!=set(expected):
        raise ContractError('incomplete generated model source bindings')
    if hashes['model.py']!=accepted.get('source_hash'):
        raise ContractError('generated candidate source binding differs')
    for name,sha in hashes.items():
        if file_hash(safe_file(directory/'model',name))!=sha:
            raise ContractError('model source changed: '+name)
