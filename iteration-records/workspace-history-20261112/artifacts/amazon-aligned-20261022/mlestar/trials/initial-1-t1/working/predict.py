import os, math
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,heads=1,dropout=0.3,maxlen=128):
        super().__init__()
        self.maxlen,self.hidden = maxlen,hidden
        self.item = nn.Embedding(itemnum+1,hidden,padding_idx=0)
        self.position = nn.Embedding(maxlen,hidden)
        self.drop = nn.Dropout(dropout)
        self.an = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True) for _ in range(blocks)])
        self.fn = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.Dropout(dropout),nn.ReLU(),nn.Linear(hidden,hidden),nn.Dropout(dropout)) for _ in range(blocks)])
        self.final = nn.LayerNorm(hidden,eps=1e-8)
        nn.init.normal_(self.item.weight,std=0.02)
        nn.init.normal_(self.position.weight,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        mask = (seq!=0).unsqueeze(-1)
        x = self.drop(self.item(seq)*math.sqrt(self.hidden)+self.position(torch.arange(seq.shape[1],device=seq.device)))*mask
        causal = torch.ones(seq.shape[1],seq.shape[1],device=seq.device,dtype=torch.bool).triu(1)
        for an,attn,fn,ff in zip(self.an,self.attn,self.fn,self.ff):
            q = an(x)
            x = q+attn(q,x,x,attn_mask=causal,need_weights=False)[0]
            x = fn(x)
            x = (x+ff(x))*mask
        return self.final(x)
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,-1,None,:]).sum(-1)

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
    h = [int(i) for i in history[-model.maxlen:]]
    seq = torch.tensor([[0]*(model.maxlen-len(h))+h],dtype=torch.long,device=_DEVICE)
    cand = torch.tensor([candidates],dtype=torch.long,device=_DEVICE)
    with torch.inference_mode(): return model.predict(seq,cand)[0].float().cpu().tolist()
