import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/inputs')
from activation_probe import activation_probe
from rec_harness.core import file_hash

spec=importlib.util.spec_from_file_location('reviewed_model','/inputs/model.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.manual_seed(13)
settings={key:getattr(module,key) for key in ['HIDDEN_UNITS','NUM_BLOCKS','KERNEL_SIZE','DROPOUT_RATE','MAXLEN']}
weights=None;initialization='random seed 13, no checkpoint';itemnum=127
checkpoint=Path('/inputs/checkpoint.pt')
if checkpoint.exists():
    checkpoint_hash=file_hash(checkpoint)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(weights,dict) or 'item_emb.weight' not in weights:raise ValueError('state_dict required')
    if any(not isinstance(value,torch.Tensor) or not torch.isfinite(value).all() for value in weights.values()):raise ValueError('invalid checkpoint tensor')
    embedding=weights['item_emb.weight']
    if embedding.ndim!=2 or embedding.shape[0]<17:raise ValueError('checkpoint vocabulary too small')
    if file_hash(checkpoint)!=checkpoint_hash:raise ValueError('checkpoint changed during load')
    itemnum=embedding.shape[0]-1;initialization='development checkpoint '+checkpoint_hash
model=module.NextItNetModel(itemnum,hidden_units=settings['HIDDEN_UNITS'],num_blocks=settings['NUM_BLOCKS'],
 kernel_size=settings['KERNEL_SIZE'],dropout_rate=settings['DROPOUT_RATE'],maxlen=settings['MAXLEN'])
if weights is not None:model.load_state_dict(weights,strict=True)
def state_hash():
 h=hashlib.sha256()
 for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
before=state_hash()
seq=torch.zeros((5,16),dtype=torch.long)
for row,length in enumerate([1,4,8,12,16]):seq[row,-length:]=torch.arange(1,length+1)
histories=Path('/inputs/histories.json')
if histories.exists():
    seq=torch.tensor(json.loads(histories.read_text())['sequences'],dtype=torch.long)
    if seq.ndim!=2 or seq.shape[1]!=settings['MAXLEN'] or not 1<=seq.shape[0]<=128:
        raise ValueError('invalid history shape')
    if seq.min()<0 or seq.max()>itemnum:raise ValueError('history items outside checkpoint vocabulary')
report=activation_probe(model,seq)
if histories.exists():report['history_hash']=file_hash(histories)
report.update(source_hash=file_hash('/inputs/model.py'),probe_hash=file_hash('/inputs/activation_probe.py'),
 initialization=initialization,settings=settings,unchanged_state=before==state_hash())
Path('/output/report.json').write_text(json.dumps(report,allow_nan=False,sort_keys=True)+'\n')
