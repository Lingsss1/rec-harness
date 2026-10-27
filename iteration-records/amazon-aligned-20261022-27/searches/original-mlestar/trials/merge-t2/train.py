from __future__ import annotations
import os, json, random, inspect
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'working')

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

def read(path):
    d=defaultdict(list)
    with open(path) as f:
        for line in f:
            p=line.split()
            if len(p)==2: d[int(p[0])].append(int(p[1]))
    return dict(d)

def load(root,ds):
    train=read(os.path.join(root,ds+'_train.txt')); valid=read(os.path.join(root,ds+'_valid.txt'))
    n=max((i for split in (train,valid) for h in split.values() for i in h),default=0)
    path=os.path.join(root,'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f: stats=json.load(f)
        n=max(n,int(stats.get(ds,{}).get('itemnum',0)))
    return train,valid,n

def validation(train,valid,n,length):
    rng=random.Random(42); users=[u for u in valid if train.get(u) and valid[u]]
    if len(users)>200: users=rng.sample(users,200)
    xs=[]; cs=[]
    for u in users:
        h=train[u][-length:]; xs.append([0]*(length-len(h))+h)
        excluded=set(train[u])|set(valid[u]); neg=set()
        if n-len(excluded)<99: raise ValueError('Insufficient unique validation negatives')
        while len(neg)<99:
            j=rng.randint(1,n)
            if j not in excluded: neg.add(j)
        cs.append([valid[u][0]]+list(neg))
    if not xs: raise ValueError('No validation users')
    return torch.tensor(xs,dtype=torch.long),torch.tensor(cs,dtype=torch.long)

@torch.no_grad()
def evaluate(model,cases,device):
    model.eval(); ranks=[]
    for j in range(0,len(cases[0]),64):
        s=model.predict(cases[0][j:j+64].to(device),cases[1][j:j+64].to(device)); ranks.append((s[:,1:]>s[:,:1]).sum(1).cpu())
    r=torch.cat(ranks)
    return float((r<10).float().mean()),float(torch.where(r<10,1/torch.log2(r.float()+2),0).mean())

class Sampler:
    def __init__(self,train,n,length,k):
        self.hist=[np.asarray(h,dtype=np.int64) for h in train.values() if len(h)>=2 and len(set(h))<n]
        if not self.hist: raise ValueError('No trainable histories')
        self.seen=[np.unique(h) for h in self.hist]; self.n=n; self.length=length; self.k=k; self.rng=np.random.default_rng(42)
    def batch(self,b):
        x=np.zeros((b,self.length),dtype=np.int64); y=np.zeros(b,dtype=np.int64); neg=self.rng.integers(1,self.n+1,size=(b,self.k))
        for row,uid in enumerate(self.rng.integers(len(self.hist),size=b)):
            h=self.hist[uid]; end=int(self.rng.integers(1,len(h))); prefix=h[max(0,end-self.length):end]
            x[row,-len(prefix):]=prefix; y[row]=h[end]
            bad=np.isin(neg[row],self.seen[uid])
            while bad.any():
                neg[row,bad]=self.rng.integers(1,self.n+1,size=int(bad.sum())); bad=np.isin(neg[row],self.seen[uid])
        return x,y,neg

def writer():
    script='import os\nimport torch\nfrom torch import nn\n'+inspect.getsource(FilterBlock)+'\n'+inspect.getsource(Model)
    script+='''
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
'''
    with open(os.path.join(ROOT,'predict.py'),'w') as f: f.write(script)

def train_one(ds,root,device):
    train,valid,n=load(root,ds)
    cfg=dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),dropout=float(os.environ.get('SASREC_DROPOUT','0.3')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')))
    k=int(os.environ.get('FMLP_NEGATIVES','8'))
    if k<1: raise ValueError('Negative count must be positive')
    sampler=Sampler(train,n,cfg['maxlen'],k); cases=validation(train,valid,n,cfg['maxlen']); model=Model(**cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','200')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))*5
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            x,y,ng=[torch.as_tensor(a,device=device) for a in sampler.batch(batch)]
            opt.zero_grad(set_to_none=True); z=model(x)
            pl=(z*model.item(y)).sum(-1); nl=(z[:,None,:]*model.item(ng)).sum(-1)
            loss=F.softplus(-pl).mean()+F.softplus(nl).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best,best_epoch=metric,epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'merged_fmlp','best_epoch':epoch,'negative_count':k},os.path.join(ROOT,ds+'_model.pt'))
            if epoch-best_epoch>=patience: break
    del model,opt
    if device.type=='cuda': torch.cuda.empty_cache()
    return best[0]

def main():
    DATA_DIR=os.environ['GAGC_DATA_DIR']
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8,os.cpu_count() or 1)); os.makedirs(ROOT,exist_ok=True); writer()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores=[train_one(ds,DATA_DIR,device) for ds in DATASETS]
    print('Macro validation HR@10:',float(np.mean(scores)),flush=True)
if __name__=='__main__': main()
