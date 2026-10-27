from __future__ import annotations
import os, json, random, math, shutil
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']

class FeedForward(nn.Module):
    def __init__(self, h, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h,h), nn.Dropout(drop), nn.ReLU(), nn.Linear(h,h), nn.Dropout(drop))
    def forward(self,x):
        return x + self.net(x)

class Model(nn.Module):
    def __init__(self, itemnum, hidden=64, blocks=2, heads=1, dropout=0.5, maxlen=128):
        super().__init__()
        self.itemnum, self.hidden, self.maxlen = itemnum, hidden, maxlen
        self.item = nn.Embedding(itemnum+1,hidden,padding_idx=0)
        nn.init.normal_(self.item.weight,0,1)
        with torch.no_grad(): self.item.weight[0].zero_()
        self.position = nn.Embedding(maxlen,hidden)
        self.drop = nn.Dropout(dropout)
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True) for _ in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff = nn.ModuleList([FeedForward(hidden,dropout) for _ in range(blocks)])
        self.final = nn.LayerNorm(hidden,eps=1e-8)
        self.register_buffer('causal',torch.ones(maxlen,maxlen,dtype=torch.bool).triu(1),persistent=False)
    def forward(self,seq):
        keep = seq.ne(0).unsqueeze(-1)
        x = self.drop(self.item(seq)*math.sqrt(self.hidden)+self.position(torch.arange(seq.shape[1],device=seq.device))) * keep
        for a,n1,n2,f in zip(self.attn,self.norm1,self.norm2,self.ff):
            q=n1(x)
            z=a(q,x,x,attn_mask=self.causal[:seq.shape[1],:seq.shape[1]],need_weights=False)[0]
            x=f(n2(q+z))*keep
        return self.final(x)
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,-1,None,:]).sum(-1)


def read_split(path):
    out=defaultdict(list)
    with open(path) as f:
        for line in f:
            p=line.split()
            if len(p)==2: out[int(p[0])].append(int(p[1]))
    return dict(out)


def load_data(root,ds):
    train=read_split(os.path.join(root,ds+'_train.txt'))
    valid=read_split(os.path.join(root,ds+'_valid.txt'))
    itemnum=max([max(v,default=0) for v in train.values()]+[max(v,default=0) for v in valid.values()])
    path=os.path.join(root,'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f: stats=json.load(f)
        itemnum=max(itemnum,int(stats.get(ds,{}).get('itemnum',0)))
    return train,valid,itemnum


def validation_cases(train,valid,itemnum,maxlen):
    rng=random.Random(42)
    users=[u for u in valid if u in train and valid[u]]
    if len(users)>200: users=rng.sample(users,200)
    seqs=[]; candidates=[]
    for u in users:
        hist=train[u][-maxlen:]
        seqs.append([0]*(maxlen-len(hist))+hist)
        excluded=set(train[u])|set(valid[u])
        if itemnum-len(excluded)<99: raise ValueError('Insufficient unique validation negatives')
        negatives=set()
        while len(negatives)<99:
            n=rng.randint(1,itemnum)
            if n not in excluded: negatives.add(n)
        candidates.append([valid[u][0]]+list(negatives))
    if not users: raise ValueError('No validation users')
    return np.asarray(seqs,dtype=np.int64),np.asarray(candidates,dtype=np.int64)


@torch.no_grad()
def evaluate(model,cases,device):
    model.eval(); hits=[]; ndcg=[]
    for start in range(0,len(cases[0]),64):
        x=torch.as_tensor(cases[0][start:start+64],device=device)
        c=torch.as_tensor(cases[1][start:start+64],device=device)
        s=model.predict(x,c)
        rank=(s[:,1:]>s[:,:1]).sum(1)
        hits.extend((rank<10).float().cpu().tolist())
        ndcg.extend(torch.where(rank<10,1/torch.log2(rank.float()+2),0).cpu().tolist())
    return float(np.mean(hits)),float(np.mean(ndcg))


class Sampler:
    def __init__(self,train,itemnum,maxlen,k):
        self.hist=[np.asarray(v,dtype=np.int64) for v in train.values() if len(v)>=2]
        self.seen=[np.unique(v) for v in self.hist]
        self.itemnum,self.maxlen,self.k=itemnum,maxlen,k
        self.rng=np.random.default_rng(42)
        if not self.hist: raise ValueError('No training sequences')
        if any(len(s)>=itemnum for s in self.seen): raise ValueError('No negative items for a training user')
    def batch(self,b):
        x=np.zeros((b,self.maxlen),dtype=np.int64); y=np.zeros_like(x)
        neg=np.zeros((b,self.maxlen,self.k),dtype=np.int64)
        for row,uid in enumerate(self.rng.integers(len(self.hist),size=b)):
            hist=self.hist[uid]; n=min(len(hist)-1,self.maxlen)
            x[row,-n:]=hist[-n-1:-1]; y[row,-n:]=hist[-n:]
            z=self.rng.integers(1,self.itemnum+1,size=(n,self.k))
            bad=np.isin(z,self.seen[uid])
            while bad.any():
                z[bad]=self.rng.integers(1,self.itemnum+1,size=int(bad.sum()))
                bad=np.isin(z,self.seen[uid])
            neg[row,-n:]=z
        return x,y,neg


def negative_sum(feat,ids,weight):
    return F.softplus((feat[:,None,:]*F.embedding(ids,weight)).sum(-1)).sum()


def train_one(ds,root,working,device):
    train,valid,itemnum=load_data(root,ds)
    cfg=dict(itemnum=itemnum,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','1')),dropout=float(os.environ.get('SASREC_DROPOUT','0.5')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')))
    k=int(os.environ.get('GSASREC_NEGATIVES','64')); t=float(os.environ.get('GSASREC_T','0.5'))
    if k<1 or not 0<=t<=1: raise ValueError('Invalid gBCE parameters')
    beta=1-t+t*k/max(itemnum-1,1)
    sampler=Sampler(train,itemnum,cfg['maxlen'],k)
    cases=validation_cases(train,valid,itemnum,cfg['maxlen'])
    model=Model(**cfg).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','200')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            x,y,n=[torch.as_tensor(a,device=device) for a in sampler.batch(batch)]
            optimizer.zero_grad(set_to_none=True)
            features=model(x); mask=y.ne(0); feat=features[mask]; targets=y[mask]; negatives=n[mask]
            positive=(feat*model.item(targets)).sum(-1)
            numerator=-beta*F.logsigmoid(positive).sum()
            for start in range(0,len(feat),256):
                numerator=numerator+checkpoint(negative_sum,feat[start:start+256],negatives[start:start+256],model.item.weight,use_reentrant=False)
            loss=numerator/((k+1)*len(feat))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            total+=float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best=metric; best_epoch=epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'sasrec_gbce','negatives':k,'calibration_t':t,'best_epoch':epoch},os.path.join(working,ds+'_model.pt'))
            if epoch-best_epoch>=patience*5: break
    del model,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return best[0]


def write_predict_script(working):
    shutil.copyfile(os.path.abspath(__file__),os.path.join(working,'model_def.py'))
    script='''import os, importlib.util, torch
HERE=os.path.dirname(os.path.abspath(__file__))
DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
_models={}
_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_spec=importlib.util.spec_from_file_location('gsasrec_model_def',os.path.join(HERE,'model_def.py'))
_mod=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
def predict(user_id,history,candidates,*,dataset_name=None):
    ds=DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if not candidates: return []
    if ds not in _models:
        ck=torch.load(os.path.join(HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        model=_mod.Model(**ck['config'])
        model.load_state_dict(ck['state']); model.to(_device).eval()
        _models[ds]=model
    model=_models[ds]
    hist=list(history[-model.maxlen:])
    seq=[0]*(model.maxlen-len(hist))+hist
    with torch.inference_mode():
        return model.predict(torch.tensor([seq],dtype=torch.long,device=_device),torch.tensor([candidates],dtype=torch.long,device=_device))[0].cpu().tolist()
'''
    with open(os.path.join(working,'predict.py'),'w') as f: f.write(script)


def main():
    DATA_DIR=os.environ['GAGC_DATA_DIR']
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working=os.path.join(os.path.dirname(os.path.abspath(__file__)),'working'); os.makedirs(working,exist_ok=True)
    write_predict_script(working)
    scores=[train_one(ds,DATA_DIR,working,device) for ds in DATASETS]
    print('Macro validation HR@10:',float(np.mean(scores)),flush=True)

if __name__=='__main__': main()
