"""Local-only deterministic sampling of frozen training histories for diagnostics."""
from pathlib import Path
import random
from .core import ContractError, file_hash
from .adapters.amazon_controls import load_config


def resolve_histories(config):
    sampling=config.get('history_sampling')
    if sampling is None:return None
    if not isinstance(sampling,dict) or set(sampling)!={'sample_size','seed'}:
        raise ContractError('history sampling requires sample_size and seed')
    if type(sampling['sample_size']) is not int or not 1<=sampling['sample_size']<=128 or type(sampling['seed']) is not int:
        raise ContractError('history sample must contain 1 to 128 histories and an integer seed')
    source=config.get('checkpoint_source')
    if not source:raise ContractError('real-history probe requires an audited development checkpoint')
    study=load_config(Path(source['study']).resolve())
    if source['seed'] not in study['development_seeds'] or source['seed'] in study['confirmation_seeds']:
        raise ContractError('real-history probe requires development scope')
    if source['dataset'] not in study['datasets']:raise ContractError('history dataset outside study')
    maxlen=config['settings']['MAXLEN']
    if type(maxlen) is not int or not 1<=maxlen<=512:raise ContractError('history length must be bounded by 512')
    name=source['dataset']+'_train.txt';path=Path(study['data'])/name
    expected=study['data_hashes'][name]
    if file_hash(path)!=expected:raise ContractError('frozen training histories changed')
    histories={}
    with path.open() as handle:
        for line in handle:
            user,item=map(int,line.split())
            if user<=0 or item<=0:raise ContractError('positive training identifiers required')
            histories.setdefault(user,[]).append(item)
    if file_hash(path)!=expected:raise ContractError('training histories changed while sampling')
    if len(histories)<sampling['sample_size']:raise ContractError('insufficient histories for frozen sample size')
    users=random.Random(sampling['seed']).sample(sorted(histories),sampling['sample_size'])
    sequences=[histories[user][-maxlen:] for user in users]
    lengths=[len(row) for row in sequences]
    counts={'all':len(lengths),'short':sum(n<=8 for n in lengths),'long':sum(n>8 for n in lengths)}
    # No user identifiers or validation targets enter the worker payload.
    return {'payload':{'sequences':[[0]*(maxlen-len(row))+row for row in sequences]},
        'train_hash':expected,'valid_positions':sum(lengths),
        'slice_counts':{key:value for key,value in counts.items() if value}}
