from __future__ import annotations
import os, json, random, math, inspect
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'working')

class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,heads=1,dropout=0.3,maxlen=128):
        super().__init__()
        self.maxlen,self.hidden=maxlen,hidden
        self.item=nn.Embedding(itemnum+1,hidden,padding_idx=0); self.position=nn.Embedding(maxlen,hidden)
        self.drop=nn.Dropout(dropout)
        self.an=nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.attn=nn.ModuleList([nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True) for _ in range(blocks)])
        self.fn=nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.Dropout(dropout),nn.ReLU(),nn.Linear(hidden,hidden),nn.Dropout(dropout)) for _ in range(blocks)])
        self.final=nn.LayerNorm(hidden,eps=1e-8)
        self.register_buffer('causal',torch.ones(maxlen,maxlen,dtype=torch.bool).triu(1),persistent=False)
        nn.init.normal_(self.item.weight,std=0.02); nn.init.normal_(self.position.weight,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        keep=seq.ne(0).unsqueeze(-1)
        x=self.drop(self.item(seq)*math.sqrt(self.hidden)+self.position(torch.arange(seq.shape[1],device=seq.device)))*keep
        for an,attn,fn,ff in zip(self.an,self.attn,self.fn,self.ff):
            q=an(x); x=fn(q+attn(q,x,x,attn_mask=self.causal[:seq.shape[1],:seq.shape[1]],need_weights=False)[0]); x=(x+ff(x))*keep
        return self.final(x)
    def predict(self,seq,candidates): return (self.item(candidates)*self(seq)[:,-1,None,:]).sum(-1)

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

def contrastive(a,b,targets,temperature):
    if len(a)<2: return (a.sum()+b.sum())*0
    a=F.normalize(a,dim=-1); b=F.normalize(b,dim=-1)
    logits=a@b.T/temperature
    same=targets[:,None].eq(targets[None,:]); same.fill_diagonal_(False)
    logits=logits.masked_fill(same,-1e4); labels=torch.arange(len(a),device=a.device)
    return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))*0.5

class Sampler:
    def __init__(self,train,n,length):
        self.hist=[np.asarray(h,dtype=np.int64) for h in train.values() if len(h)>=2 and len(set(h))<n]
        if not self.hist: raise ValueError('No trainable histories')
        self.seen=[np.unique(h) for h in self.hist]; self.n=n; self.length=length; self.rng=np.random.default_rng(42)
        self.index=defaultdict(list); counts=defaultdict(int); rng=random.Random(42)
        for row,h in enumerate(self.hist):
            positions=np.unique(np.linspace(1,len(h)-1,min(32,len(h)-1),dtype=np.int64))
            for end in positions:
                target=int(h[end]); counts[target]+=1; pool=self.index[target]
                if len(pool)<64: pool.append((row,int(end)))
                else:
                    j=rng.randrange(counts[target])
                    if j<64: pool[j]=(row,int(end))
    def batch(self,b):
        x=np.zeros((b,self.length),dtype=np.int64); y=np.zeros_like(x); neg=np.zeros_like(x); partner=np.zeros_like(x)
        targets=np.zeros(b,dtype=np.int64); paired=np.zeros(b,dtype=bool)
        for row,uid in enumerate(self.rng.integers(len(self.hist),size=b)):
            h=self.hist[uid]; end=len(h)-1 if self.rng.random()<0.5 else int(self.rng.integers(1,len(h)))
            start=max(0,end-self.length); a=h[start:end]; size=len(a)
            x[row,-size:]=a; y[row,-size:]=h[start+1:end+1]; targets[row]=h[end]
            z=self.rng.integers(1,self.n+1,size=size); bad=np.isin(z,self.seen[uid])
            while bad.any():
                z[bad]=self.rng.integers(1,self.n+1,size=int(bad.sum())); bad=np.isin(z,self.seen[uid])
            neg[row,-size:]=z
            pool=self.index.get(int(h[end]),[]); chosen=None
            if pool:
                offset=int(self.rng.integers(len(pool)))
                for j in range(len(pool)):
                    ref=pool[(offset+j)%len(pool)]
                    if ref!=(int(uid),end): chosen=ref; break
            if chosen is not None:
                r,e=chosen; ph=self.hist[r][max(0,e-self.length):e]; partner[row,-len(ph):]=ph; paired[row]=True
            else: partner[row]=x[row]
        return x,y,neg,partner,targets,paired

def writer():
    script='import os, math\nimport torch\nfrom torch import nn\n'+inspect.getsource(Model)
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
    cfg=dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','1')),dropout=float(os.environ.get('SASREC_DROPOUT','0.3')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')))
    sampler=Sampler(train,n,cfg['maxlen']); cases=validation(train,valid,n,cfg['maxlen']); model=Model(**cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','100')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))*5
    lu=float(os.environ.get('DUOREC_LAMBDA_U','0.05')); ls=float(os.environ.get('DUOREC_LAMBDA_S','0.05')); temperature=float(os.environ.get('DUOREC_TEMPERATURE','0.2'))
    if temperature<=0: raise ValueError('Temperature must be positive')
    last_weight=float(os.environ.get('SASREC_LAST_WEIGHT','0.25'))
    if not 0<=last_weight<=1: raise ValueError('Last-position weight must be between zero and one')
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            x,y,ng,p,targets,paired=[torch.as_tensor(a,device=device) for a in sampler.batch(batch)]
            opt.zero_grad(set_to_none=True); features=model(x); mask=y.ne(0)
            pl=(features[mask]*model.item(y[mask])).sum(-1); nl=(features[mask]*model.item(ng[mask])).sum(-1)
            point_loss=F.softplus(-pl)+F.softplus(nl)
            per_user_counts=mask.sum(1).clamp_min(1).to(point_loss.dtype)
            position_weights=(1-last_weight)/per_user_counts[:,None]
            position_weights=position_weights.expand_as(y).clone()
            position_weights[:,-1]+=last_weight
            loss=(point_loss*position_weights[mask]).sum()/x.shape[0]
            a=features[:,-1]
            if lu: loss=loss+lu*contrastive(a,model(x)[:,-1],targets,temperature)
            if ls and int(paired.sum())>1:
                loss=loss+ls*contrastive(a[paired],model(p[paired])[:,-1],targets[paired],temperature)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best,best_epoch=metric,epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'merged_duorec','best_epoch':epoch,'lambda_u':lu,'lambda_s':ls},os.path.join(ROOT,ds+'_model.pt'))
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
