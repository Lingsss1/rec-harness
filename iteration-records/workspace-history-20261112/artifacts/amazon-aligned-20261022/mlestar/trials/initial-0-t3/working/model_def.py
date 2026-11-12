from __future__ import annotations
import os, json, random, shutil
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']

class BertBlock(nn.Module):
    def __init__(self,h,heads,drop):
        super().__init__()
        self.attn=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True)
        self.norm1=nn.LayerNorm(h,eps=1e-8); self.norm2=nn.LayerNorm(h,eps=1e-8)
        self.drop=nn.Dropout(drop)
        self.ff=nn.Sequential(nn.Linear(h,4*h),nn.GELU(),nn.Linear(4*h,h),nn.Dropout(drop))
    def forward(self,x,padding):
        z=self.attn(x,x,x,key_padding_mask=padding,need_weights=False)[0]
        x=self.norm1(x+self.drop(z))
        return self.norm2(x+self.ff(x))

class Model(nn.Module):
    def __init__(self,itemnum,hidden=64,blocks=2,heads=2,dropout=0.2,maxlen=128):
        super().__init__()
        self.itemnum,self.maxlen,self.mask_id=itemnum,maxlen,itemnum+1
        self.item=nn.Embedding(itemnum+2,hidden,padding_idx=0)
        self.position=nn.Embedding(maxlen,hidden)
        self.norm=nn.LayerNorm(hidden,eps=1e-8); self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([BertBlock(hidden,heads,dropout) for _ in range(blocks)])
        self.output=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden,eps=1e-8))
        self.bias=nn.Parameter(torch.zeros(itemnum))
        self.apply(self.initialize)
        with torch.no_grad(): self.item.weight[0].zero_()
    @staticmethod
    def initialize(module):
        if isinstance(module,(nn.Embedding,nn.Linear)):
            nn.init.normal_(module.weight,0,0.02)
            if isinstance(module,nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self,seq):
        padding=seq.eq(0); keep=(~padding).unsqueeze(-1)
        x=self.drop(self.norm(self.item(seq)+self.position(torch.arange(seq.shape[1],device=seq.device))))*keep
        for block in self.blocks: x=block(x,padding)*keep
        return self.output(x)
    def predict(self,seq,candidates):
        feat=self(seq)[:,-1,:]
        return (self.item(candidates)*feat[:,None,:]).sum(-1)+self.bias[candidates-1]


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
        h=train[u][-(maxlen-1):] if maxlen>1 else []
        h=h+[itemnum+1]; seqs.append([0]*(maxlen-len(h))+h)
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
    def __init__(self,train,itemnum,maxlen,probability):
        self.hist=[np.asarray(v,dtype=np.int64) for v in train.values() if len(v)>=2]
        self.itemnum,self.maxlen,self.probability=itemnum,maxlen,probability
        self.rng=np.random.default_rng(42)
        if not self.hist: raise ValueError('No training sequences')
    def batch(self,b):
        x=np.zeros((b,self.maxlen),dtype=np.int64); labels=np.zeros_like(x)
        for row,uid in enumerate(self.rng.integers(len(self.hist),size=b)):
            hist=self.hist[uid]
            if self.rng.random()<0.5:
                # Exact inference format, with a target drawn exclusively from train.
                j=int(self.rng.integers(1,len(hist)))
                h=hist[max(0,j-(self.maxlen-1)):j] if self.maxlen>1 else hist[:0]
                seq=np.concatenate([h,np.asarray([self.itemnum+1],dtype=np.int64)])
                x[row,-len(seq):]=seq; labels[row,-1]=hist[j]
            else:
                # Bidirectional cloze task on a randomly ending training window.
                end=int(self.rng.integers(2,len(hist)+1)); h=hist[max(0,end-self.maxlen):end].copy(); size=len(h)
                chosen=self.rng.random(size)<self.probability
                if not chosen.any(): chosen[int(self.rng.integers(size))]=True
                targets=np.where(chosen,h,0)
                coins=self.rng.random(size)
                replace_mask=chosen&(coins<0.8); replace_random=chosen&(coins>=0.8)&(coins<0.9)
                h[replace_mask]=self.itemnum+1
                h[replace_random]=self.rng.integers(1,self.itemnum+1,size=int(replace_random.sum()))
                x[row,-size:]=h; labels[row,-size:]=targets
        return x,labels


def catalog_loss(features,targets,weight,bias):
    # Only real items are output classes; neither PAD nor MASK is scored.
    return F.cross_entropy(F.linear(features,weight,bias),targets,reduction='sum')


def train_one(ds,root,working,device):
    train,valid,itemnum=load_data(root,ds)
    cfg=dict(itemnum=itemnum,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','2')),dropout=float(os.environ.get('SASREC_DROPOUT','0.2')),maxlen=int(os.environ.get('SASREC_MAXLEN','128')))
    probability=float(os.environ.get('BERTREC_MASK_PROB','0.2'))
    sampler=Sampler(train,itemnum,cfg['maxlen'],probability); cases=validation_cases(train,valid,itemnum,cfg['maxlen'])
    model=Model(**cfg).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    epochs=int(os.environ.get('SASREC_EPOCHS','200')); steps=int(os.environ.get('SASREC_BATCHES','200')); batch=int(os.environ.get('SASREC_BATCH','128')); patience=int(os.environ.get('SASREC_PATIENCE','20'))
    # Bound each transient full-catalog logit matrix to roughly 8M entries.
    chunk=max(1,min(64,8000000//max(itemnum,1)))
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for _ in range(steps):
            x,labels=[torch.as_tensor(a,device=device) for a in sampler.batch(batch)]
            optimizer.zero_grad(set_to_none=True)
            mask=labels.ne(0); features=model(x)[mask]; targets=labels[mask]-1
            weight=model.item.weight[1:itemnum+1]
            loss=features.sum()*0
            for start in range(0,len(features),chunk):
                # Recompute logits during backward rather than retaining every chunk.
                loss=loss+checkpoint(catalog_loss,features[start:start+chunk],targets[start:start+chunk],weight,model.bias,use_reentrant=False)
            loss=loss/len(features)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            metric=evaluate(model,cases,device)
            print(f'{ds} epoch={epoch} loss={total/steps:.6f} val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}',flush=True)
            if metric>best:
                best=metric; best_epoch=epoch
                torch.save({'config':cfg,'state':model.state_dict(),'architecture':'bert4rec','mask_id':itemnum+1,'output_head':'gelu_layernorm_tied_real_items_with_bias','mask_probability':probability,'best_epoch':epoch},os.path.join(working,ds+'_model.pt'))
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
_spec=importlib.util.spec_from_file_location('bert4rec_model_def',os.path.join(HERE,'model_def.py'))
_mod=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
def predict(user_id,history,candidates,*,dataset_name=None):
    ds=DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if not candidates: return []
    if ds not in _models:
        ck=torch.load(os.path.join(HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        if ck['architecture']!='bert4rec': raise ValueError('Wrong checkpoint architecture')
        model=_mod.Model(**ck['config']); model.load_state_dict(ck['state']); model.to(_device).eval()
        if model.mask_id!=ck['mask_id']: raise ValueError('Wrong mask ID')
        _models[ds]=model
    model=_models[ds]
    hist=list(history[-(model.maxlen-1):]) if model.maxlen>1 else []
    hist.append(model.mask_id); seq=[0]*(model.maxlen-len(hist))+hist
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
