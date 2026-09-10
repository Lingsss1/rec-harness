"""Frozen CPU/GPU harness. Imports upstream Model + SeqDataset, never upstream evolver."""
import argparse, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',required=True); ap.add_argument('--model',required=True); ap.add_argument('--output',required=True); ap.add_argument('--final',action='store_true'); ap.add_argument('--checkpoint'); a=ap.parse_args()
    p=load(a.protocol); root=Path(p['upstream']); data=root/'data_cache/CDs_and_Vinyl'; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for name,h in p['dataHashes'].items():
        if digest(data/name)!=h: raise ValueError('Frozen dataset changed: '+name)
    torch.set_num_threads(2); random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed']); torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('model_candidate',a.model); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location('upstream_utils',root/'simulator/util_functions.py'); util=importlib.util.module_from_spec(spec); spec.loader.exec_module(util)
    train=load(data/'train.json'); valid=load(data/'valid.json'); candidates=load(data/'validation_candidate.json')
    itemnum=max(load(data/'item2id.json').values()); usernum=max(load(data/'user2id.json').values())
    train_ids=sorted(train['History'],key=int); random.Random(p['seed']).shuffle(train_ids); train_ids=train_ids[:p['trainUsers']]
    subset={'History':{u:train['History'][u] for u in train_ids}}
    args=argparse.Namespace(dataset=str(data),device=p['device'],hidden_units=p['hidden'],maxlen=p['maxlen'],dropout_rate=0.2)
    model=module.Model(usernum,itemnum,args,subset).to(p['device'])
    for name,param in model.named_parameters():
        if param.ndim>1: torch.nn.init.xavier_normal_(param.data)
    model.pos_emb.weight.data[0,:]=0; model.item_emb.weight.data[0,:]=0
    started=time.monotonic(); losses=[]
    if a.final:
        model.load_state_dict(torch.load(a.checkpoint,map_location=p['device'],weights_only=True)); valid=load(data/'test.json'); candidates=load(data/'test_candidate.json')
        validation_history=load(data/'valid.json')['History']
        history={u:train['History'][u]+validation_history.get(u,[]) for u in train['History']}
    else:
        history=train['History']; loader=DataLoader(util.SeqDataset(subset,itemnum,p['maxlen']),batch_size=p['batchSize'],num_workers=0)
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.98))
        for epoch in range(p['epochs']):
            model.train(); total=0
            for u,seq,pos,neg in loader:
                optimizer.zero_grad(); loss=model(np.array(list(map(int,u))),seq.numpy(),pos.numpy(),neg.numpy())
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                loss.backward(); optimizer.step(); total+=float(loss.detach())
            losses.append(total/len(loader)); print(json.dumps({'epoch':epoch+1,'loss':losses[-1]}),flush=True)
        torch.save(model.state_dict(),out/'checkpoint.pt')
    users=sorted(set(history)&set(valid['History'])&set(candidates),key=int); random.Random(p['seed']+1).shuffle(users); users=users[:p['evalUsers']]
    model.eval(); records=[]; embeddings=[]
    with torch.no_grad():
        for u in users:
            seq=np.zeros((1,p['maxlen']),dtype=np.int64); hist=history[u][-p['maxlen']:]; seq[0,-len(hist):]=hist
            # Fixed upstream candidates; stable tie handling and no candidate resampling.
            items=candidates[u]; scores=model(np.array([int(u)]),seq,item_indices=np.array([items])).cpu().numpy()[0]
            if not np.isfinite(scores).all(): raise ValueError('Non-finite prediction')
            rank=int(np.where(np.argsort(-scores,kind='stable')==0)[0][0]); hr=float(rank<5); ndcg=1/np.log2(rank+2) if rank<5 else 0.0
            records.append({'user':u,'rank':rank,'hr':hr,'ndcg':float(ndcg)})
            embeddings.append(model.log2feats(seq)[0,-1].cpu().numpy())
    if not records: raise ValueError('Empty evaluation')
    np.save(out/'embeddings.npy',np.array(embeddings))
    receipt={'protocolId':p['id'],'codeIdentity':digest(a.model),'split':'test' if a.final else 'validation','metrics':{'ndcg':float(np.mean([r['ndcg'] for r in records])),'hr':float(np.mean([r['hr'] for r in records]))},'records':records,'losses':losses,'seconds':time.monotonic()-started,'device':p['device'],'trainUsers':len(train_ids),'evalUsers':len(users),'epochs':p['epochs']}
    (out/'evaluation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)); print(json.dumps({k:v for k,v in receipt.items() if k!='records'}),flush=True)
if __name__=='__main__': main()
