from __future__ import annotations
import os, json, random, math, inspect
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'working')

class Model(nn.Module):
    def __init__(self,itemnum,hidden=128,blocks=2,dropout=0.3,maxlen=128):
        super().__init__()
        self.maxlen,self.hidden = maxlen,hidden
        self.item = nn.Embedding(itemnum+1,hidden,padding_idx=0)
        self.position = nn.Embedding(maxlen,hidden)
        self.input_norm = nn.LayerNorm(hidden,eps=1e-8)
        self.drop = nn.Dropout(dropout)
        self.filters = nn.ParameterList([nn.Parameter(torch.randn(maxlen//2+1,hidden,2)*0.02) for _ in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Linear(4*hidden,hidden),nn.Dropout(dropout)) for _ in range(blocks)])
        nn.init.normal_(self.item.weight,std=0.02)
        nn.init.normal_(self.position.weight,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        mask = (seq!=0).unsqueeze(-1)
        x = self.drop(self.input_norm(self.item(seq)+self.position(torch.arange(seq.shape[1],device=seq.device))))*mask
        for weight,n1,n2,ff in zip(self.filters,self.norm1,self.norm2,self.ff):
            z = torch.fft.rfft(x.float(),dim=1,norm='ortho')
            z = z*torch.view_as_complex(weight.contiguous())
            y = torch.fft.irfft(z,n=self.maxlen,dim=1,norm='ortho')
            x = n1(x+self.drop(y))*mask
            x = n2(x+ff(x))*mask
        return x[:,-1]
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,None,:]).sum(-1)

def read_split(ds,split):
    result = defaultdict(list)
    with open(os.path.join(DATA_DIR,ds+'_'+split+'.txt')) as f:
        for line in f:
            p = line.split()
            if len(p)==2: result[int(p[0])].append(int(p[1]))
    return dict(result)

def load(ds):
    train,valid = read_split(ds,'train'),read_split(ds,'valid')
    path = os.path.join(DATA_DIR,'dataset_stats.json')
    stats = {}
    if os.path.isfile(path):
        with open(path) as f: stats = json.load(f)
    n = int(stats[ds]['itemnum']) if ds in stats else max(i for split in (train,valid) for h in split.values() for i in h)
    return train,valid,n

def validation_data(train,valid,n,length):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users)>200: users = rng.sample(users,200)
    seqs,candidates = [],[]
    for u in users:
        h = train[u][-length:]
        seqs.append([0]*(length-len(h))+h)
        excluded = set(train[u])|set(valid[u])
        if n-len(excluded)<99: raise ValueError('Insufficient validation negatives')
        neg = set()
        while len(neg)<99:
            i = rng.randint(1,n)
            if i not in excluded: neg.add(i)
        candidates.append([valid[u][0]]+list(neg))
    if not seqs: raise ValueError('No validation users')
    return torch.tensor(seqs,dtype=torch.long),torch.tensor(candidates,dtype=torch.long)

@torch.no_grad()
def evaluate(model,validation,device):
    model.eval()
    seq,cand = validation
    ranks = []
    for start in range(0,len(seq),64):
        scores = model.predict(seq[start:start+64].to(device),cand[start:start+64].to(device))
        ranks.append((scores[:,1:]>scores[:,:1]).sum(1).cpu())
    rank = torch.cat(ranks)
    return float((rank<10).float().mean()),float(torch.where(rank<10,1/torch.log2(rank.float()+2),0).mean())

def writer():
    script = 'import os, math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'+inspect.getsource(Model)
    script += '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def predict(user_id,history,candidates,*,dataset_name=None):
    if not candidates: return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if ds not in _MODELS:
        ckpt = torch.load(os.path.join(_HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        model = Model(**ckpt['config'])
        model.load_state_dict(ckpt['state'])
        _MODELS[ds] = model.to(_DEVICE).eval()
    model = _MODELS[ds]
    h = [int(i) for i in history[-model.maxlen:]]
    seq = torch.tensor([[0]*(model.maxlen-len(h))+h],dtype=torch.long,device=_DEVICE)
    cand = torch.tensor([candidates],dtype=torch.long,device=_DEVICE)
    with torch.inference_mode(): return model.predict(seq,cand)[0].float().cpu().tolist()
'''
    with open(os.path.join(ROOT,'predict.py'),'w') as f: f.write(script)

def train_one(ds,device):
    train,valid,n = load(ds)
    length = int(os.environ.get('SASREC_MAXLEN','128'))
    cfg = dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','128')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),dropout=float(os.environ.get('SASREC_DROPOUT','0.3')),maxlen=length)
    model = Model(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    histories = [h for h in train.values() if len(h)>=2 and len(set(h))<n]
    if not histories: raise ValueError('No trainable histories')
    excluded = [np.unique(h) for h in histories]
    validation = validation_data(train,valid,n,length)
    rng = np.random.default_rng(42)
    batch = int(os.environ.get('SASREC_BATCH','256'))
    steps = int(os.environ.get('SASREC_BATCHES','200'))
    epochs = int(os.environ.get('SASREC_EPOCHS','200'))
    k = int(os.environ.get('FMLP_NEGATIVES','8'))
    best,best_epoch = (-1.0,-1.0),0
    for epoch in range(1,epochs+1):
        model.train()
        total = 0.0
        for _ in range(steps):
            rows = rng.integers(len(histories),size=batch)
            seq = np.zeros((batch,length),dtype=np.int64)
            pos = np.zeros(batch,dtype=np.int64)
            neg = rng.integers(1,n+1,size=(batch,k))
            for b,row in enumerate(rows):
                h = histories[row]
                end = len(h)-1 if rng.random()<0.5 else int(rng.integers(1,len(h)))
                prefix = h[max(0,end-length):end]
                seq[b,-len(prefix):],pos[b] = prefix,h[end]
                bad = np.isin(neg[b],excluded[row])
                while bad.any():
                    neg[b,bad] = rng.integers(1,n+1,size=int(bad.sum()))
                    bad = np.isin(neg[b],excluded[row])
            s,p,ng = [torch.as_tensor(x,device=device) for x in (seq,pos,neg)]
            opt.zero_grad(set_to_none=True)
            z = model(s)
            pl = (z*model.item(p)).sum(-1)
            nl = (z[:,None,:]*model.item(ng)).sum(-1)
            loss = F.softplus(-pl).mean()+F.softplus(nl).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step()
            total += float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            score = evaluate(model,validation,device)
            print(f'[{ds}] epoch={epoch} loss={total/steps:.5f} val_HR@10={score[0]:.4f} NDCG@10={score[1]:.4f}',flush=True)
            if score>best:
                best,best_epoch = score,epoch
                torch.save({'architecture':'fmlp','config':cfg,'state':model.state_dict(),'filter_bins':length//2+1},os.path.join(ROOT,ds+'_model.pt'))
            if epoch-best_epoch>=int(os.environ.get('SASREC_PATIENCE','20'))*5: break
    return best[0]

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    os.makedirs(ROOT,exist_ok=True)
    writer()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = [train_one(ds,device) for ds in DATASETS]
    print('Macro validation HR@10:',sum(scores)/len(scores),flush=True)
if __name__=='__main__': main()
