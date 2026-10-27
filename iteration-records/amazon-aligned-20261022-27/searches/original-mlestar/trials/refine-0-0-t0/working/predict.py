import os, math
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,heads=1,dropout=0.5,maxlen=128,init_std=1.0):
        super().__init__()
        self.maxlen,self.hidden=maxlen,hidden
        self.item=nn.Embedding(itemnum+1,hidden,padding_idx=0)
        self.position=nn.Embedding(maxlen,hidden)
        self.drop=nn.Dropout(dropout)
        self.an=nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.attn=nn.ModuleList([nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True) for _ in range(blocks)])
        self.fn=nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.Dropout(dropout),nn.ReLU(),nn.Linear(hidden,hidden),nn.Dropout(dropout)) for _ in range(blocks)])
        self.final=nn.LayerNorm(hidden,eps=1e-8)
        self.register_buffer('causal',torch.ones(maxlen,maxlen,dtype=torch.bool).triu(1),persistent=False)
        nn.init.normal_(self.item.weight,std=init_std)
        if init_std<0.1: nn.init.normal_(self.position.weight,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        keep=seq.ne(0).unsqueeze(-1)
        x=self.drop(self.item(seq)*math.sqrt(self.hidden)+self.position(torch.arange(seq.shape[1],device=seq.device)))*keep
        for an,attn,fn,ff in zip(self.an,self.attn,self.fn,self.ff):
            q=an(x)
            x=fn(q+attn(q,x,x,attn_mask=self.causal[:seq.shape[1],:seq.shape[1]],need_weights=False)[0])
            x=(x+ff(x))*keep
        return self.final(x)
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,-1,None,:]).sum(-1)

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
