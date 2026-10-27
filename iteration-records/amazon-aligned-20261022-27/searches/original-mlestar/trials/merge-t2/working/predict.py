import os
import torch
from torch import nn
class FilterBlock(nn.Module):
    def __init__(self,hidden,maxlen,dropout):
        super().__init__()
        self.maxlen=maxlen
        self.weight=nn.Parameter(torch.randn(maxlen//2+1,hidden,2)*0.02)
        self.drop=nn.Dropout(dropout); self.n1=nn.LayerNorm(hidden,eps=1e-8); self.n2=nn.LayerNorm(hidden,eps=1e-8)
        self.ff=nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Linear(4*hidden,hidden),nn.Dropout(dropout))
    def forward(self,x,keep):
        z=torch.fft.rfft(x.float(),dim=1,norm='ortho')
        y=torch.fft.irfft(z*torch.view_as_complex(self.weight.contiguous()),n=self.maxlen,dim=1,norm='ortho')
        x=self.n1(x+self.drop(y.to(x.dtype)))*keep
        return self.n2(x+self.ff(x))*keep

class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,dropout=0.3,maxlen=128):
        super().__init__()
        self.maxlen=maxlen
        self.item=nn.Embedding(itemnum+1,hidden,padding_idx=0); self.position=nn.Embedding(maxlen,hidden)
        self.norm=nn.LayerNorm(hidden,eps=1e-8); self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([FilterBlock(hidden,maxlen,dropout) for _ in range(blocks)])
        self.apply(self.initialize)
        with torch.no_grad(): self.item.weight[0].zero_()
    @staticmethod
    def initialize(m):
        if isinstance(m,(nn.Embedding,nn.Linear)):
            nn.init.normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,seq):
        keep=seq.ne(0).unsqueeze(-1)
        x=self.drop(self.norm(self.item(seq)+self.position(torch.arange(self.maxlen,device=seq.device))))*keep
        for block in self.blocks: x=block(x,keep)
        return x[:,-1]
    def predict(self,seq,candidates): return (self.item(candidates)*self(seq)[:,None,:]).sum(-1)

HERE=os.path.dirname(os.path.abspath(__file__))
DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
MODELS={}
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def predict(user_id,history,candidates,*,dataset_name=None):
    ds=DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if len(candidates)==0: return []
    if ds not in MODELS:
        ck=torch.load(os.path.join(HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        m=Model(**ck['config']); m.load_state_dict(ck['state']); MODELS[ds]=m.to(DEVICE).eval()
    m=MODELS[ds]; h=[int(i) for i in history[-m.maxlen:]]
    x=torch.tensor([[0]*(m.maxlen-len(h))+h],dtype=torch.long,device=DEVICE)
    c=torch.tensor([list(candidates)],dtype=torch.long,device=DEVICE)
    with torch.inference_mode(): return m.predict(x,c)[0].float().cpu().tolist()
