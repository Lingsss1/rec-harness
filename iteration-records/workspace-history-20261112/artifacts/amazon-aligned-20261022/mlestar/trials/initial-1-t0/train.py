from __future__ import annotations
import os, json, random, math, inspect
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')

class Model(nn.Module):
    def __init__(self, itemnum, hidden=64, blocks=2, heads=1, dropout=0.3, maxlen=128):
        super().__init__()
        self.maxlen, self.hidden = maxlen, hidden
        self.item = nn.Embedding(itemnum+1, hidden, padding_idx=0)
        self.position = nn.Embedding(maxlen, hidden)
        self.drop = nn.Dropout(dropout)
        self.an = nn.ModuleList([nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True) for _ in range(blocks)])
        self.fn = nn.ModuleList([nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(hidden, hidden), nn.Dropout(dropout)) for _ in range(blocks)])
        self.final = nn.LayerNorm(hidden, eps=1e-8)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self, seq):
        mask = (seq != 0).unsqueeze(-1)
        x = self.drop(self.item(seq)*math.sqrt(self.hidden) + self.position(torch.arange(seq.shape[1], device=seq.device))) * mask
        causal = torch.ones(seq.shape[1], seq.shape[1], device=seq.device, dtype=torch.bool).triu(1)
        for an, attn, fn, ff in zip(self.an, self.attn, self.fn, self.ff):
            q = an(x)
            y = attn(q, x, x, attn_mask=causal, need_weights=False)[0]
            x = q + y
            x = fn(x)
            x = (x + ff(x))*mask
        return self.final(x)
    def predict(self, seq, candidates):
        return (self.item(candidates)*self(seq)[:, -1, None, :]).sum(-1)

def read_split(ds, split):
    result = defaultdict(list)
    with open(os.path.join(DATA_DIR, ds+'_'+split+'.txt')) as f:
        for line in f:
            p = line.split()
            if len(p)==2: result[int(p[0])].append(int(p[1]))
    return dict(result)

def load(ds):
    train, valid = read_split(ds, 'train'), read_split(ds, 'valid')
    path = os.path.join(DATA_DIR, 'dataset_stats.json')
    stats = {}
    if os.path.isfile(path):
        with open(path) as f: stats = json.load(f)
    n = int(stats[ds]['itemnum']) if ds in stats else max(i for split in (train, valid) for h in split.values() for i in h)
    return train, valid, n

def validation_data(train, valid, n, length):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users)>200: users = rng.sample(users, 200)
    seqs, candidates = [], []
    for u in users:
        hist = train[u][-length:]
        seqs.append([0]*(length-len(hist))+hist)
        excluded = set(train[u]) | set(valid[u])
        if n-len(excluded)<99: raise ValueError('Insufficient validation negatives')
        neg = set()
        while len(neg)<99:
            i = rng.randint(1,n)
            if i not in excluded: neg.add(i)
        candidates.append([valid[u][0]]+list(neg))
    if not seqs: raise ValueError('No validation users')
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(candidates, dtype=torch.long)

@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    ranks = []
    seq, cand = validation
    for start in range(0,len(seq),64):
        scores = model.predict(seq[start:start+64].to(device),cand[start:start+64].to(device))
        ranks.append((scores[:,1:]>scores[:,:1]).sum(1).cpu())
    rank = torch.cat(ranks)
    return float((rank<10).float().mean()), float(torch.where(rank<10, 1/torch.log2(rank.float()+2), 0).mean())

def writer():
    script = 'import os, math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n' + inspect.getsource(Model)
    script += '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates: return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if ds not in _MODELS:
        ckpt = torch.load(os.path.join(_HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        model = Model(**ckpt['config'])
        model.load_state_dict(ckpt['state'])
        _MODELS[ds] = model.to(_DEVICE).eval()
    model = _MODELS[ds]
    hist = [int(i) for i in history[-model.maxlen:]]
    seq = torch.tensor([[0]*(model.maxlen-len(hist))+hist],dtype=torch.long,device=_DEVICE)
    cand = torch.tensor([candidates],dtype=torch.long,device=_DEVICE)
    with torch.inference_mode(): return model.predict(seq,cand)[0].float().cpu().tolist()
'''
    with open(os.path.join(ROOT,'predict.py'),'w') as f: f.write(script)

def train_one(ds, device):
    train, valid, n = load(ds)
    length = int(os.environ.get('SASREC_MAXLEN','128'))
    cfg = dict(itemnum=n, hidden=int(os.environ.get('SASREC_HIDDEN','64')), blocks=int(os.environ.get('SASREC_BLOCKS','2')), heads=int(os.environ.get('SASREC_HEADS','1')), dropout=float(os.environ.get('SASREC_DROPOUT','0.3')), maxlen=length)
    model = Model(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.001')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    histories = [h for h in train.values() if len(h)>=2 and len(set(h))<n]
    if not histories: raise ValueError('No trainable histories')
    seqs = np.zeros((len(histories),length),dtype=np.int64)
    pos = np.zeros_like(seqs)
    forbidden = []
    for row,h in enumerate(histories):
        a,b = h[:-1][-length:],h[1:][-length:]
        seqs[row,-len(a):],pos[row,-len(b):] = a,b
        forbidden.append(np.unique(h))
    validation = validation_data(train, valid, n, length)
    rng = np.random.default_rng(42)
    epochs = int(os.environ.get('SASREC_EPOCHS','200'))
    steps = int(os.environ.get('SASREC_BATCHES','100'))
    batch = int(os.environ.get('SASREC_BATCH','128'))
    k = int(os.environ.get('GSASREC_NEGATIVES','256'))
    t = float(os.environ.get('GSASREC_T','0.75'))
    beta = 1-t+t*k/max(n-1,1)
    best, best_epoch = (-1.0,-1.0), 0
    patience = int(os.environ.get('SASREC_PATIENCE','20'))
    for epoch in range(1,epochs+1):
        model.train()
        total = 0.0
        for _ in range(steps):
            rows = rng.integers(len(histories),size=batch)
            s,p = seqs[rows],pos[rows]
            ix = p!=0
            negative_rows = []
            for row,mask in zip(rows,ix):
                neg = rng.integers(1,n+1,size=(int(mask.sum()),k))
                bad = np.isin(neg, forbidden[row])
                while bad.any():
                    neg[bad] = rng.integers(1,n+1,size=int(bad.sum()))
                    bad = np.isin(neg, forbidden[row])
                negative_rows.append(neg)
            negatives = np.concatenate(negative_rows,axis=0)
            s = torch.as_tensor(s,device=device)
            p = torch.as_tensor(p,device=device)
            optimizer.zero_grad(set_to_none=True)
            features = model(s)[p!=0]
            # Backpropagate output chunks separately, then the shared encoder once.
            leaf = features.detach().requires_grad_(True)
            targets = p[p!=0]
            for start in range(0,len(targets),256):
                end = min(start+256,len(targets))
                z = leaf[start:end]
                pl = (z*model.item(targets[start:end])).sum(-1)
                ni = torch.as_tensor(negatives[start:end],device=device)
                nl = (model.item(ni)*z[:,None,:]).sum(-1)
                loss = (-beta*F.logsigmoid(pl)+F.softplus(nl).sum(-1)).sum()/(len(targets)*(k+1))
                loss.backward()
                total += float(loss.detach())
            features.backward(leaf.grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
        if epoch%5==0 or epoch==epochs:
            score = evaluate(model,validation,device)
            print(f'[{ds}] epoch={epoch} loss={total/steps:.6f} val_HR@10={score[0]:.4f} NDCG@10={score[1]:.4f}',flush=True)
            if score>best:
                best,best_epoch = score,epoch
                torch.save({'architecture':'gsasrec','config':cfg,'state':model.state_dict(),'negative_count':k,'calibration_t':t},os.path.join(ROOT,ds+'_model.pt'))
            if epoch-best_epoch>=patience*5: break
    return best[0]

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ROOT,exist_ok=True)
    writer()
    scores = [train_one(ds,device) for ds in DATASETS]
    print('Macro validation HR@10:',sum(scores)/len(scores),flush=True)
if __name__=='__main__': main()
