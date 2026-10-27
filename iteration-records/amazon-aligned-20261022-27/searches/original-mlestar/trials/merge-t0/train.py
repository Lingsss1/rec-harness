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

def read(path):
    result=defaultdict(list)
    with open(path) as f:
        for line in f:
            p=line.split()
            if len(p)==2: result[int(p[0])].append(int(p[1]))
    return dict(result)

def load(root,ds):
    train=read(os.path.join(root,ds+'_train.txt')); valid=read(os.path.join(root,ds+'_valid.txt'))
    n=max((i for split in (train,valid) for h in split.values() for i in h),default=0)
    path=os.path.join(root,'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f: stats=json.load(f)
        n=max(n,int(stats.get(ds,{}).get('itemnum',0)))
    return train,valid,n

def validation(train,valid,n,length):
    rng=random.Random(42)
    users=[u for u in valid if train.get(u) and valid[u]]
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
        s=model.predict(cases[0][j:j+64].to(device),cases[1][j:j+64].to(device))
        ranks.append((s[:,1:]>s[:,:1]).sum(1).cpu())
    r=torch.cat(ranks)
    return float((r<10).float().mean()),float(torch.where(r<10,1/torch.log2(r.float()+2),0).mean())

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
    small=ds=='Electronics'
    cfg=dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','1')),dropout=float(os.environ.get('SASREC_DROPOUT','0.3' if small else '0.5')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')),init_std=0.02 if small else 1.0)
    length=cfg['maxlen']; histories=[np.asarray(h,dtype=np.int64) for h in train.values() if len(h)>=2 and len(set(h))<n]
    if not histories: raise ValueError('No trainable histories')
    seen=[np.unique(h) for h in histories]
    seqs=np.zeros((len(histories),length),dtype=np.int64); pos=np.zeros_like(seqs)
    for row,h in enumerate(histories):
        size=min(len(h)-1,length); seqs[row,-size:]=h[-size-1:-1]; pos[row,-size:]=h[-size:]
    cases=validation(train,valid,n,length)
    model=Model(**cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    rng=np.random.default_rng(42)
    k=int(os.environ.get('GSASREC_NEGATIVES','256' if small else '64')); t=float(os.environ.get('GSASREC_T','0.75' if small else '0.5'))
    if k<1 or not 0<=t<=1: raise ValueError('Invalid negative count or calibration')
    beta=1-t+t*k/max(n-1,1)
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','100' if small else '200')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))*5
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            rows=rng.integers(len(histories),size=batch); s=seqs[rows]; p=pos[rows]; ix=p!=0
            negatives=[]
            for row,mask in zip(rows,ix):
                z=rng.integers(1,n+1,size=(int(mask.sum()),k)); bad=np.isin(z,seen[row])
                while bad.any():
                    z[bad]=rng.integers(1,n+1,size=int(bad.sum())); bad=np.isin(z,seen[row])
                negatives.append(z)
            negatives=np.concatenate(negatives)
            s=torch.as_tensor(s,device=device); p=torch.as_tensor(p,device=device)
            opt.zero_grad(set_to_none=True)
            features=model(s)[p!=0]; targets=p[p!=0]; leaf=features.detach().requires_grad_(True)
            chunk=max(16,min(256,1048576//(k*cfg['hidden'])))
            for start in range(0,len(targets),chunk):
                end=min(start+chunk,len(targets)); z=leaf[start:end]
                pl=(z*model.item(targets[start:end])).sum(-1)
                ni=torch.as_tensor(negatives[start:end],device=device)
                nl=(model.item(ni)*z[:,None,:]).sum(-1)
                loss=(-beta*F.logsigmoid(pl)+F.softplus(nl).sum(-1)).sum()/((k+1)*len(targets))
                loss.backward(); total+=float(loss.detach())
            features.backward(leaf.grad)
            nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best,best_epoch=metric,epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'merged_gsasrec','best_epoch':epoch,'negative_count':k,'calibration_t':t},os.path.join(ROOT,ds+'_model.pt'))
            if epoch-best_epoch>=patience: break
    del model,opt
    if device.type=='cuda': torch.cuda.empty_cache()
    return best[0]

def main():
    DATA_DIR=os.environ['GAGC_DATA_DIR']
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    os.makedirs(ROOT,exist_ok=True); writer()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores=[train_one(ds,DATA_DIR,device) for ds in DATASETS]
    print('Macro validation HR@10:',float(np.mean(scores)),flush=True)
if __name__=='__main__': main()
