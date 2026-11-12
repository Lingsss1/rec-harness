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
    def __init__(self,itemnum,hidden=64,blocks=2,heads=2,dropout=0.2,maxlen=128):
        super().__init__()
        self.itemnum,self.mask_id,self.maxlen = itemnum,itemnum+1,maxlen
        self.item = nn.Embedding(itemnum+2,hidden,padding_idx=0)
        self.position = nn.Embedding(maxlen,hidden)
        self.input_norm = nn.LayerNorm(hidden,eps=1e-8)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden,heads,dim_feedforward=4*hidden,dropout=dropout,activation='gelu',batch_first=True) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden,eps=1e-8))
        self.output_bias = nn.Parameter(torch.zeros(itemnum))
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                nn.init.normal_(module.weight,std=0.02)
                if isinstance(module,nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
        with torch.no_grad(): self.item.weight[0].zero_()
    def forward(self,seq):
        padding = seq.eq(0)
        x = self.drop(self.input_norm(self.item(seq)+self.position(torch.arange(seq.shape[1],device=seq.device))))
        x = x.masked_fill(padding.unsqueeze(-1),0)
        for layer in self.layers:
            x = layer(x,src_key_padding_mask=padding)
            x = x.masked_fill(padding.unsqueeze(-1),0)
        return self.head(x)
    def predict(self,seq,candidates):
        z = self(seq)[:,-1]
        return (self.item(candidates)*z[:,None,:]).sum(-1)+self.output_bias[candidates-1]

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
        h = train[u][-(length-1):]+[n+1]
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
    h = [int(i) for i in history[-(model.maxlen-1):]]+[model.mask_id]
    seq = torch.tensor([[0]*(model.maxlen-len(h))+h],dtype=torch.long,device=_DEVICE)
    cand = torch.tensor([candidates],dtype=torch.long,device=_DEVICE)
    with torch.inference_mode(): return model.predict(seq,cand)[0].float().cpu().tolist()
'''
    with open(os.path.join(ROOT,'predict.py'),'w') as f: f.write(script)

def train_one(ds,device):
    train,valid,n = load(ds)
    length = int(os.environ.get('SASREC_MAXLEN','128'))
    if length<2: raise ValueError('BERT maxlen must be at least two')
    cfg = dict(itemnum=n,hidden=int(os.environ.get('SASREC_HIDDEN','64')),blocks=int(os.environ.get('SASREC_BLOCKS','2')),heads=int(os.environ.get('SASREC_HEADS','2')),dropout=float(os.environ.get('SASREC_DROPOUT','0.2')),maxlen=length)
    model = Model(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=float(os.environ.get('SASREC_LR','0.0005')),betas=(0.9,0.98),weight_decay=float(os.environ.get('SASREC_WD','0')))
    histories = [h for h in train.values() if len(h)>=2]
    if not histories: raise ValueError('No training histories')
    validation = validation_data(train,valid,n,length)
    rng = np.random.default_rng(42)
    batch = int(os.environ.get('SASREC_BATCH','128'))
    steps = int(os.environ.get('SASREC_BATCHES','100'))
    epochs = int(os.environ.get('SASREC_EPOCHS','200'))
    mask_prob = float(os.environ.get('BERT_MASK_PROB','0.2'))
    best,best_epoch = (-1.0,-1.0),0
    for epoch in range(1,epochs+1):
        model.train()
        total = 0.0
        for _ in range(steps):
            seq = np.zeros((batch,length),dtype=np.int64)
            labels = np.zeros_like(seq)
            for b,row in enumerate(rng.integers(len(histories),size=batch)):
                h = histories[row]
                if rng.random()<0.5:
                    end = len(h)-1 if rng.random()<0.5 else int(rng.integers(1,len(h)))
                    prefix = h[max(0,end-(length-1)):end]+[n+1]
                    seq[b,-len(prefix):] = prefix
                    labels[b,-1] = h[end]
                else:
                    end = len(h) if rng.random()<0.5 else int(rng.integers(2,len(h)+1))
                    raw = np.asarray(h[max(0,end-length):end],dtype=np.int64)
                    corrupted = raw.copy()
                    selected = rng.random(len(raw))<mask_prob
                    if not selected.any(): selected[int(rng.integers(len(raw)))] = True
                    action = rng.random(len(raw))
                    corrupted[selected & (action<0.8)] = n+1
                    replace = selected & (action>=0.8) & (action<0.9)
                    corrupted[replace] = rng.integers(1,n+1,size=int(replace.sum()))
                    seq[b,-len(raw):] = corrupted
                    lab = np.zeros(len(raw),dtype=np.int64)
                    lab[selected] = raw[selected]
                    labels[b,-len(raw):] = lab
            s = torch.as_tensor(seq,device=device)
            lab = torch.as_tensor(labels,device=device)
            opt.zero_grad(set_to_none=True)
            features = model(s)[lab!=0]
            targets = lab[lab!=0]-1
            leaf = features.detach().requires_grad_(True)
            # Exact catalog CE, with logits allocated only for a bounded row chunk.
            chunk = max(1,min(128,4000000//max(n,1)))
            for start in range(0,len(targets),chunk):
                end = min(start+chunk,len(targets))
                logits = F.linear(leaf[start:end],model.item.weight[1:n+1],model.output_bias)
                loss = F.cross_entropy(logits,targets[start:end],reduction='sum')/len(targets)
                loss.backward()
                total += float(loss.detach())
            features.backward(leaf.grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step()
        if epoch%5==0 or epoch==epochs:
            score = evaluate(model,validation,device)
            print(f'[{ds}] epoch={epoch} loss={total/steps:.5f} val_HR@10={score[0]:.4f} NDCG@10={score[1]:.4f}',flush=True)
            if score>best:
                best,best_epoch = score,epoch
                torch.save({'architecture':'bert4rec','config':cfg,'state':model.state_dict(),'mask_id':n+1,'output_head':'transform_tied_bias'},os.path.join(ROOT,ds+'_model.pt'))
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
