import os, math
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self,itemnum,hidden=128,blocks=2,dropout=0.3,maxlen=128):
        super().__init__()
        self.maxlen,self.hidden = maxlen,hidden
        self.item = nn.Embedding(itemnum+1,hidden,padding_idx=0)
        self.position = nn.Embedding(maxlen,hidden)
        self.input_norm = nn.LayerNorm(hidden,eps=1e-8)
        self.drop = nn.Dropout(dropout)
        self.filters = nn.ParameterList([nn.Parameter(torch.randn(maxlen//2+1,hidden,2)*0.02) for _ in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Linear(4*hidden,hidden),nn.Dropout(dropout)) for _ in range(blocks)])
        nn.init.normal_(self.item.weight,std=0.02)
        nn.init.normal_(self.position.weight,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        mask = (seq!=0).unsqueeze(-1)
        x = self.drop(self.input_norm(self.item(seq)+self.position(torch.arange(seq.shape[1],device=seq.device))))*mask
        for weight,n1,n2,ff in zip(self.filters,self.norm1,self.norm2,self.ff):
            z = torch.fft.rfft(x.float(),dim=1,norm='ortho')
            z = z*torch.view_as_complex(weight.contiguous())
            y = torch.fft.irfft(z,n=self.maxlen,dim=1,norm='ortho')
            x = n1(x+self.drop(y))*mask
            x = n2(x+ff(x))*mask
        return x[:,-1]
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,None,:]).sum(-1)

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
