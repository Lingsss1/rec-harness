"""Unchanged accepted helper functions extracted from run_round_accumulation.py."""
import json,hashlib,os
from pathlib import Path
def require(condition, message):
    if not condition:
        raise ValueError(message)

def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def stat_identity(path):
    value = Path(path).stat()
    return dict(size=value.st_size,mtime_ns=value.st_mtime_ns,inode=value.st_ino,device=value.st_dev)

def normalize_layout(layout):
    """Only physical device index is normalized for cross-GPU comparison."""
    def visit(value):
        if isinstance(value, dict):
            return {key: ('cuda' if key == 'device' and isinstance(item, str) and item.startswith('cuda:') else visit(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value
    return visit(layout)

def metadata_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [metadata_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k):metadata_value(v) for k,v in value.items()}
    if callable(value):
        return {'callable': value.__module__+'.'+value.__qualname__}
    return str(value)

def optimizer_layout(model):
    named = list(model.named_parameters())
    names = {id(p):n for n,p in named}
    require(len(names) == len(named), 'Duplicate named parameter objects')
    def entry(p):
        return dict(name=names[id(p)], shape=list(p.shape), dtype=str(p.dtype),
                    device=str(p.device), requires_grad=p.requires_grad)
    groups, covered = {}, []
    for role in ('dense','sparse'):
        optimizer = getattr(model, role+'_optimizer')
        groups[role] = []
        for group in optimizer.param_groups:
            covered.extend(id(p) for p in group['params'])
            groups[role].append(dict(parameters=[entry(p) for p in group['params']],
                options={k:metadata_value(v) for k,v in group.items()
                         if k in optimizer.defaults and k != 'params' and not k.startswith('_')}))
    require(len(covered) == len(set(covered)) and set(covered) == {id(p) for _,p in named if p.requires_grad},
            'Optimizer coverage differs')
    return dict(named_parameters=[entry(p) for _,p in named], groups=groups,
        dense_parameter_order=[names[id(p)] for p in model._dense_params],
        sparse_parameter_order=[names[id(p)] for p in model._sparse_params],
        scope='Static registration/group/options/order only; not runtime buckets or matched optimization trajectories.')
