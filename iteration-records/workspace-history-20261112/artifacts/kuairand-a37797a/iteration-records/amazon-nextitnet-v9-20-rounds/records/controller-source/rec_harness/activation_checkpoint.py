"""Resolve audited development weights locally, without exporting private rows."""
from pathlib import Path
from .core import ContractError
from .adapters.amazon_controls import load_config
from .adapters.amazon_report import audit_job


def resolve_checkpoint(config):
    source=config.get('checkpoint_source')
    if source is None:return None
    if not isinstance(source,dict) or set(source)!={'study','model_version','dataset','seed'}:
        raise ContractError('checkpoint source requires study/model_version/dataset/seed')
    root=Path(source['study']).resolve();study=load_config(root)
    seed=source['seed']
    if type(seed) is not int or seed not in study['development_seeds'] or seed in study['confirmation_seeds']:
        raise ContractError('activation probe accepts development checkpoints only')
    if source['dataset'] not in study['datasets']:raise ContractError('checkpoint dataset outside frozen study')
    if source['model_version'] not in study['model_versions']:raise ContractError('unregistered checkpoint model')
    if study['model_versions'][source['model_version']]['sha256']!=config['source']['sha256']:
        raise ContractError('checkpoint registered source differs from probe source')
    result,_=audit_job(root,study,source['model_version'],source['dataset'],seed)
    directory=root/'jobs'/f"{source['model_version']}-{source['dataset']}-{seed}"
    return {'path':str(directory/'checkpoint.pt'),'sha256':result['checkpoint_sha256']}
