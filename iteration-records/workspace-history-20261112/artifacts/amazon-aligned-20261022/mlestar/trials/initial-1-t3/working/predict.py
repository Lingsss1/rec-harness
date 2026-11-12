import os, math
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,heads=2,dropout=0.2,maxlen=128):
        super().__init__()
        self.itemnum,self.mask_id,self.maxlen = itemnum,itemnum+1,maxlen
        self.item = nn.Embedding(itemnum+2,hidden,padding_idx=0)
        self.position = nn.Embedding(maxlen,hidden)
        self.input_norm = nn.LayerNorm(hidden,eps=1e-8)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden,heads,dim_feedforward=4*hidden,dropout=dropout,activation='gelu',batch_first=True) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden,eps=1e-8))
        self.output_bias = nn.Parameter(torch.zeros(itemnum))
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                nn.init.normal_(module.weight,std=0.02)
                if isinstance(module,nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        padding = seq.eq(0)
        x = self.drop(self.input_norm(self.item(seq)+self.position(torch.arange(seq.shape[1],device=seq.device))))
        x = x.masked_fill(padding.unsqueeze(-1),0)
        for layer in self.layers:
            x = layer(x,src_key_padding_mask=padding)
            x = x.masked_fill(padding.unsqueeze(-1),0)
        return self.head(x)
    def predict(self,seq,candidates):
        z = self(seq)[:,-1]
        return (self.item(candidates)*z[:,None,:]).sum(-1)+self.output_bias[candidates-1]

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def predict(user_id,history,candidates,*,dataset_name=None):
    if not candidates: return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if ds not in _MODELS:
        ckpt = torch.load(os.path.join(_HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        model = Model(**ckpt['config'])
        model.load_state_dict(ckpt['state'])
        _MODELS[ds] = model.to(_DEVICE).eval()
    model = _MODELS[ds]
    h = [int(i) for i in history[-(model.maxlen-1):]]+[model.mask_id]
    seq = torch.tensor([[0]*(model.maxlen-len(h))+h],dtype=torch.long,device=_DEVICE)
    cand = torch.tensor([candidates],dtype=torch.long,device=_DEVICE)
    with torch.inference_mode(): return model.predict(seq,cand)[0].float().cpu().tolist()
