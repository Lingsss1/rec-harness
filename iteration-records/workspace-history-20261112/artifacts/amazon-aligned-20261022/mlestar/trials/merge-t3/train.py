from __future__ import annotations
import os, json, random, inspect
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'working')

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
        h=train[u][-(length-1):]+[n+1]; xs.append([0]*(length-len(h))+h)
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
    def __init__(self,train,n,length,prob):
        self.hist=[np.asarray(h,dtype=np.int64) for h in train.values() if len(h)>=2]
        if not self.hist: raise ValueError('No trainable histories')
        self.n,self.length,self.prob=n,length,prob; self.rng=np.random.default_rng(42)
    def batch(self,b):
        x=np.zeros((b,self.length),dtype=np.int64); labels=np.zeros_like(x)
        for row,uid in enumerate(self.rng.integers(len(self.hist),size=b)):
            h=self.hist[uid]
            if self.rng.random()<0.5:
                end=len(h)-1 if self.rng.random()<0.5 else int(self.rng.integers(1,len(h)))
                prefix=h[max(0,end-(self.length-1)):end]
                seq=np.concatenate((prefix,np.asarray([self.n+1],dtype=np.int64)))
                x[row,-len(seq):]=seq; labels[row,-1]=h[end]
            else:
                end=len(h) if self.rng.random()<0.5 else int(self.rng.integers(2,len(h)+1))
                raw=h[max(0,end-self.length):end]; seq=raw.copy()
                chosen=self.rng.random(len(raw))<self.prob
                if not chosen.any(): chosen[int(self.rng.integers(len(raw)))]=True
                coins=self.rng.random(len(raw)); seq[chosen&(coins<0.8)]=self.n+1
                replace=chosen&(coins>=0.8)&(coins<0.9)
                seq[replace]=self.rng.integers(1,self.n+1,size=int(replace.sum()))
                x[row,-len(seq):]=seq; labels[row,-len(seq):]=np.where(chosen,raw,0)
        return x,labels

def writer():
    script='import os\nimport torch\nfrom torch import nn\n'+inspect.getsource(BertBlock)+'\n'+inspect.getsource(Model)
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
    m=MODELS[ds]; h=[int(i) for i in history[-(m.maxlen-1):]]+[m.mask_id]
    x=torch.tensor([[0]*(m.maxlen-len(h))+h],dtype=torch.long,device=DEVICE)
    c=torch.tensor([list(candidates)],dtype=torch.long,device=DEVICE)
    with torch.inference_mode(): return m.predict(x,c)[0].float().cpu().tolist()
'''
    with open(os.path.join(ROOT,'predict.py'),'w') as f: f.write(script)

def train_one(ds,root,device):
    train,valid,n=load(root,ds)
    cfg=dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','2')),dropout=float(os.environ.get('SASREC_DROPOUT','0.2')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')))
    if cfg['maxlen']<2: raise ValueError('BERT maxlen must be at least two')
    prob=float(os.environ.get('BERTREC_MASK_PROB',os.environ.get('BERT_MASK_PROB','0.2')))
    if not 0<prob<=1: raise ValueError('Invalid mask probability')
    sampler=Sampler(train,n,cfg['maxlen'],prob); cases=validation(train,valid,n,cfg['maxlen']); model=Model(**cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','200')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))*5
    chunk=max(1,min(128,4000000//max(n,1))); best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            x,lab=[torch.as_tensor(a,device=device) for a in sampler.batch(batch)]
            opt.zero_grad(set_to_none=True); features=model(x)[lab!=0]; targets=lab[lab!=0]-1
            leaf=features.detach().requires_grad_(True)
            for start in range(0,len(targets),chunk):
                end=min(start+chunk,len(targets))
                logits=F.linear(leaf[start:end],model.item.weight[1:n+1],model.bias)
                loss=F.cross_entropy(logits,targets[start:end],reduction='sum')/len(targets)
                loss.backward(); total+=float(loss.detach())
            features.backward(leaf.grad)
            nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best,best_epoch=metric,epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'merged_bert4rec','best_epoch':epoch,'mask_id':n+1,'mask_probability':prob},os.path.join(ROOT,ds+'_model.pt'))
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
