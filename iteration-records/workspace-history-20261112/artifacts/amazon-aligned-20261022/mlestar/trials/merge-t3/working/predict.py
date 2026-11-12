import os
import torch
from torch import nn
class BertBlock(nn.Module):
    def __init__(self,h,heads,drop):
        super().__init__()
        self.attn=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True)
        self.n1=nn.LayerNorm(h,eps=1e-8); self.n2=nn.LayerNorm(h,eps=1e-8); self.drop=nn.Dropout(drop)
        self.ff=nn.Sequential(nn.Linear(h,4*h),nn.GELU(),nn.Linear(4*h,h),nn.Dropout(drop))
    def forward(self,x,padding):
        y=self.attn(x,x,x,key_padding_mask=padding,need_weights=False)[0]
        x=self.n1(x+self.drop(y))
        return self.n2(x+self.ff(x))

class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,heads=2,dropout=0.2,maxlen=128):
        super().__init__()
        self.itemnum,self.mask_id,self.maxlen=itemnum,itemnum+1,maxlen
        self.item=nn.Embedding(itemnum+2,hidden,padding_idx=0); self.position=nn.Embedding(maxlen,hidden)
        self.norm=nn.LayerNorm(hidden,eps=1e-8); self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([BertBlock(hidden,heads,dropout) for _ in range(blocks)])
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden,eps=1e-8))
        self.bias=nn.Parameter(torch.zeros(itemnum)); self.apply(self.initialize)
        with torch.no_grad(): self.item.weight[0].zero_()
    @staticmethod
    def initialize(m):
        if isinstance(m,(nn.Embedding,nn.Linear)):
            nn.init.normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,seq):
        padding=seq.eq(0); keep=(~padding).unsqueeze(-1)
        x=self.drop(self.norm(self.item(seq)+self.position(torch.arange(seq.shape[1],device=seq.device))))*keep
        for block in self.blocks: x=block(x,padding)*keep
        return self.head(x)
    def predict(self,seq,candidates):
        z=self(seq)[:,-1]
        return (self.item(candidates)*z[:,None,:]).sum(-1)+self.bias[candidates-1]

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
    m=MODELS[ds]; h=[int(i) for i in history[-(m.maxlen-1):]]+[m.mask_id]
    x=torch.tensor([[0]*(m.maxlen-len(h))+h],dtype=torch.long,device=DEVICE)
    c=torch.tensor([list(candidates)],dtype=torch.long,device=DEVICE)
    with torch.inference_mode(): return m.predict(x,c)[0].float().cpu().tolist()
