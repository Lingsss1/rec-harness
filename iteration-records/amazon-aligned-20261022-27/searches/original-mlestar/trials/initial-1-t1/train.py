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
    def __init__(self,itemnum,hidden=64,blocks=2,heads=1,dropout=0.3,maxlen=128):
        super().__init__()
        self.maxlen,self.hidden = maxlen,hidden
        self.item = nn.Embedding(itemnum+1,hidden,padding_idx=0)
        self.position = nn.Embedding(maxlen,hidden)
        self.drop = nn.Dropout(dropout)
        self.an = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True) for _ in range(blocks)])
        self.fn = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-8) for _ in range(blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.Dropout(dropout),nn.ReLU(),nn.Linear(hidden,hidden),nn.Dropout(dropout)) for _ in range(blocks)])
        self.final = nn.LayerNorm(hidden,eps=1e-8)
        nn.init.normal_(self.item.weight,std=0.02)
        nn.init.normal_(self.position.weight,std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        mask = (seq!=0).unsqueeze(-1)
        x = self.drop(self.item(seq)*math.sqrt(self.hidden)+self.position(torch.arange(seq.shape[1],device=seq.device)))*mask
        causal = torch.ones(seq.shape[1],seq.shape[1],device=seq.device,dtype=torch.bool).triu(1)
        for an,attn,fn,ff in zip(self.an,self.attn,self.fn,self.ff):
            q = an(x)
            x = q+attn(q,x,x,attn_mask=causal,need_weights=False)[0]
            x = fn(x)
            x = (x+ff(x))*mask
        return self.final(x)
    def predict(self,seq,candidates):
        return (self.item(candidates)*self(seq)[:,-1,None,:]).sum(-1)

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

def contrastive(a,b,targets,temperature=0.2,eligible=None):
    a,b = F.normalize(a,dim=-1),F.normalize(b,dim=-1)
    logits = a@b.T/temperature
    eye = torch.eye(len(a),dtype=torch.bool,device=a.device)
    false_negative = targets[:,None].eq(targets[None,:]) & ~eye
    logits = logits.masked_fill(false_negative,-1e4)
    labels = torch.arange(len(a),device=a.device)
    loss = (F.cross_entropy(logits,labels,reduction='none')+F.cross_entropy(logits.T,labels,reduction='none'))*0.5
    if eligible is None: return loss.mean()
    if not eligible.any(): return (a.sum()+b.sum())*0
    return loss[eligible].mean()

def train_one(ds,device):
    train,valid,n = load(ds)
    length = int(os.environ.get('SASREC_MAXLEN','128'))
    cfg = dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','1')),dropout=float(os.environ.get('SASREC_DROPOUT','0.3')),maxlen=length)
    model = Model(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    histories = [h for h in train.values() if len(h)>=2 and len(set(h))<n]
    if not histories: raise ValueError('No training histories')
    excluded = [np.unique(h) for h in histories]
    # Bounded references, never padded prefix copies; training interactions only.
    partners = defaultdict(list)
    index_rng = random.Random(42)
    seen = defaultdict(int)
    for row,h in enumerate(histories):
        for end in range(max(1,len(h)-32),len(h)):
            target = h[end]
            seen[target] += 1
            bucket = partners[target]
            if len(bucket)<64: bucket.append((row,end))
            else:
                j = index_rng.randrange(seen[target])
                if j<64: bucket[j] = (row,end)
    validation = validation_data(train,valid,n,length)
    rng = np.random.default_rng(42)
    epochs = int(os.environ.get('SASREC_EPOCHS','200'))
    steps = int(os.environ.get('SASREC_BATCHES','100'))
    batch = int(os.environ.get('SASREC_BATCH','128'))
    lu = float(os.environ.get('DUOREC_LAMBDA_U','0.05'))
    ls = float(os.environ.get('DUOREC_LAMBDA_S','0.05'))
    best,best_epoch = (-1.0,-1.0),0
    for epoch in range(1,epochs+1):
        model.train()
        total = 0.0
        for _ in range(steps):
            seq = np.zeros((batch,length),dtype=np.int64)
            pos = np.zeros_like(seq); neg = np.zeros_like(seq); partner_seq = np.zeros_like(seq)
            targets = np.zeros(batch,dtype=np.int64); eligible = np.zeros(batch,dtype=bool)
            rows = rng.integers(len(histories),size=batch)
            for b,row in enumerate(rows):
                h = histories[row]
                end = len(h)-1 if rng.random()<0.5 else int(rng.integers(1,len(h)))
                start = max(0,end-length)
                a,p = h[start:end],h[start+1:end+1]
                seq[b,-len(a):],pos[b,-len(p):] = a,p
                v = rng.integers(1,n+1,size=len(a))
                bad = np.isin(v,excluded[row])
                while bad.any():
                    v[bad] = rng.integers(1,n+1,size=int(bad.sum()))
                    bad = np.isin(v,excluded[row])
                neg[b,-len(a):] = v
                targets[b] = h[end]
                pool = partners.get(h[end],[])
                chosen = None
                if pool:
                    offset = int(rng.integers(len(pool)))
                    for j in range(len(pool)):
                        ref = pool[(offset+j)%len(pool)]
                        if ref!=(int(row),end): chosen = ref; break
                if chosen is None:
                    partner_seq[b] = seq[b]
                else:
                    r,e = chosen
                    ph = histories[r][max(0,e-length):e]
                    partner_seq[b,-len(ph):] = ph
                    eligible[b] = True
            s,p,ng,sp = [torch.as_tensor(x,device=device) for x in (seq,pos,neg,partner_seq)]
            tar = torch.as_tensor(targets,device=device)
            opt.zero_grad(set_to_none=True)
            feats = model(s)
            mask = p!=0
            pl = (feats[mask]*model.item(p[mask])).sum(-1)
            nl = (feats[mask]*model.item(ng[mask])).sum(-1)
            bce = F.softplus(-pl).mean()+F.softplus(nl).mean()
            second = model(s)[:,-1]
            partner = model(sp)[:,-1]
            loss = bce+lu*contrastive(feats[:,-1],second,tar)+ls*contrastive(feats[:,-1],partner,tar,eligible=torch.as_tensor(eligible,device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step()
            total += float(loss.detach())
        if epoch%5==0 or epoch==epochs:
            score = evaluate(model,validation,device)
            print(f'[{ds}] epoch={epoch} loss={total/steps:.5f} val_HR@10={score[0]:.4f} NDCG@10={score[1]:.4f}',flush=True)
            if score>best:
                best,best_epoch = score,epoch
                torch.save({'architecture':'duorec','config':cfg,'state':model.state_dict(),'lambda_u':lu,'lambda_s':ls},os.path.join(ROOT,ds+'_model.pt'))
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
