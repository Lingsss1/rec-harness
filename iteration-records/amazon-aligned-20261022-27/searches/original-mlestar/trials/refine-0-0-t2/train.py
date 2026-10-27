from __future__ import annotations
import os, json, random, shutil
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']

class FilterBlock(nn.Module):
    def __init__(self,hidden,maxlen,dropout):
        super().__init__()
        self.maxlen=maxlen
        self.complex_weight=nn.Parameter(torch.randn(maxlen//2+1,hidden,2)*0.02)
        self.drop=nn.Dropout(dropout)
        self.norm1=nn.LayerNorm(hidden,eps=1e-8)
        self.norm2=nn.LayerNorm(hidden,eps=1e-8)
        self.ff=nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Linear(4*hidden,hidden),nn.Dropout(dropout))
    def forward(self,x):
        z=torch.fft.rfft(x.float(),dim=1,norm='ortho')
        weight=torch.view_as_complex(self.complex_weight.contiguous())
        y=torch.fft.irfft(z*weight,n=self.maxlen,dim=1,norm='ortho')
        x=self.norm1(x+self.drop(y.to(x.dtype)))
        return self.norm2(x+self.ff(x))

class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,dropout=0.3,maxlen=128):
        super().__init__()
        self.itemnum,self.maxlen=itemnum,maxlen
        self.item=nn.Embedding(itemnum+1,hidden,padding_idx=0)
        self.position=nn.Embedding(maxlen,hidden)
        self.emb_norm=nn.LayerNorm(hidden,eps=1e-8)
        self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([FilterBlock(hidden,maxlen,dropout) for _ in range(blocks)])
        self.apply(self.initialize)
        with torch.no_grad(): self.item.weight[0].zero_()
    @staticmethod
    def initialize(module):
        if isinstance(module,(nn.Embedding,nn.Linear)):
            nn.init.normal_(module.weight,0,0.02)
            if isinstance(module,nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self,seq):
        keep=seq.ne(0).unsqueeze(-1)
        x=self.drop(self.emb_norm(self.item(seq)+self.position(torch.arange(self.maxlen,device=seq.device))))*keep
        for block in self.blocks: x=block(x)*keep
        return x[:,-1,:]
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,None,:]).sum(-1)


def read_split(path):
    out=defaultdict(list)
    with open(path) as f:
        for line in f:
            p=line.split()
            if len(p)==2: out[int(p[0])].append(int(p[1]))
    return dict(out)


def load_data(root,ds):
    train=read_split(os.path.join(root,ds+'_train.txt')); valid=read_split(os.path.join(root,ds+'_valid.txt'))
    itemnum=max([max(v,default=0) for v in train.values()]+[max(v,default=0) for v in valid.values()])
    path=os.path.join(root,'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f: stats=json.load(f)
        itemnum=max(itemnum,int(stats.get(ds,{}).get('itemnum',0)))
    return train,valid,itemnum


def validation_cases(train,valid,itemnum,maxlen):
    rng=random.Random(42); users=[u for u in valid if u in train and valid[u]]
    if len(users)>200: users=rng.sample(users,200)
    seqs=[]; candidates=[]
    for u in users:
        h=train[u][-maxlen:]; seqs.append([0]*(maxlen-len(h))+h)
        excluded=set(train[u])|set(valid[u]); negatives=set()
        if itemnum-len(excluded)<99: raise ValueError('Insufficient unique validation negatives')
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
        x=torch.as_tensor(cases[0][start:start+64],device=device); c=torch.as_tensor(cases[1][start:start+64],device=device)
        s=model.predict(x,c); rank=(s[:,1:]>s[:,:1]).sum(1)
        hits.extend((rank<10).float().cpu().tolist())
        ndcg.extend(torch.where(rank<10,1/torch.log2(rank.float()+2),0).cpu().tolist())
    return float(np.mean(hits)),float(np.mean(ndcg))


class Sampler:
    def __init__(self,train,itemnum,maxlen):
        self.hist=[np.asarray(v,dtype=np.int64) for v in train.values() if len(v)>=2]
        self.seen=[set(v.tolist()) for v in self.hist]
        self.itemnum,self.maxlen=itemnum,maxlen; self.rng=np.random.default_rng(42)
        if not self.hist: raise ValueError('No training sequences')
        if any(len(s)>=itemnum for s in self.seen): raise ValueError('No training negatives')
    def batch(self,b):
        x=np.zeros((b,self.maxlen),dtype=np.int64); y=np.zeros(b,dtype=np.int64); n=np.zeros(b,dtype=np.int64)
        for row,uid in enumerate(self.rng.integers(len(self.hist),size=b)):
            hist=self.hist[uid]; j=int(self.rng.integers(1,len(hist)))
            h=hist[max(0,j-self.maxlen):j]
            x[row,-len(h):]=h; y[row]=hist[j]
            z=int(self.rng.integers(1,self.itemnum+1))
            while z in self.seen[uid]: z=int(self.rng.integers(1,self.itemnum+1))
            n[row]=z
        return x,y,n


def train_one(ds,root,working,device):
    train,valid,itemnum=load_data(root,ds)
    cfg=dict(itemnum=itemnum,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),dropout=float(os.environ.get('SASREC_DROPOUT','0.3')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')))
    sampler=Sampler(train,itemnum,cfg['maxlen']); cases=validation_cases(train,valid,itemnum,cfg['maxlen'])
    model=Model(**cfg).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','200')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            x,y,n=[torch.as_tensor(a,device=device) for a in sampler.batch(batch)]
            optimizer.zero_grad(set_to_none=True)
            features=model(x)
            pos=(features*model.item(y)).sum(-1)
            negative_embeddings=model.item(n)
            neg=(features*negative_embeddings).sum(-1)
            shared_scores=features@negative_embeddings.T
            invalid=x.unsqueeze(-1).eq(n.view(1,1,-1)).any(dim=1)
            invalid=invalid|y.unsqueeze(1).eq(n.unsqueeze(0))
            shared_scores=shared_scores.masked_fill(invalid,float('-inf'))
            logits=torch.cat((pos.unsqueeze(1),shared_scores),dim=1)
            labels=torch.zeros(x.size(0),dtype=torch.long,device=device)
            ranking_loss=F.cross_entropy(logits,labels)
            auxiliary_loss=F.softplus(-pos).mean()+F.softplus(neg).mean()
            loss=ranking_loss+0.1*auxiliary_loss
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best=metric; best_epoch=epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'fmlp','filter_bins':cfg['maxlen']//2+1,'best_epoch':epoch},os.path.join(working,ds+'_model.pt'))
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
_spec=importlib.util.spec_from_file_location('fmlp_model_def',os.path.join(HERE,'model_def.py'))
_mod=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
def predict(user_id,history,candidates,*,dataset_name=None):
    ds=DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if not candidates: return []
    if ds not in _models:
        ck=torch.load(os.path.join(HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        if ck['architecture']!='fmlp': raise ValueError('Wrong checkpoint architecture')
        model=_mod.Model(**ck['config']); model.load_state_dict(ck['state']); model.to(_device).eval()
        _models[ds]=model
    model=_models[ds]; hist=list(history[-model.maxlen:]); seq=[0]*(model.maxlen-len(hist))+hist
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
