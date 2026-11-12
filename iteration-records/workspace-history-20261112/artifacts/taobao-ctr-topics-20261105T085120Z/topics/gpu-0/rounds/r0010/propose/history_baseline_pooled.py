"""Task-side references: field MLP, masked mean history, and DIN-style attention.

The target-aware model inherits DIN's local activation idea (arxiv:1706.06978).
It is a field-adapted reference, not an exact reproduction of the author code.
"""
import torch
from torch import nn

CONFIG={'kind':'din','embedding_dim':16,'hidden':[128,64],'lr':0.001}


class Model(nn.Module):
    def __init__(self,meta,config=None):
        super().__init__()
        config={**CONFIG,**(config or {})}
        self.config=config
        self.kind=config['kind']
        self.category=meta['category_field_index'];self.brand=meta['brand_field_index']
        dim=config['embedding_dim']
        self.embeddings=nn.ModuleList([nn.Embedding(size,dim,padding_idx=0) for size in meta['buckets']])
        self.action=nn.Embedding(5,dim,padding_idx=0)
        self.recency=nn.Embedding(33,dim,padding_idx=0)
        self.attention=nn.Sequential(nn.Linear(dim*4,64),nn.PReLU(),nn.Linear(64,1))
        layers=[]
        width=len(meta['fields'])*dim+(dim if self.kind!='static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width,hidden),nn.ReLU()]);width=hidden
        layers.append(nn.Linear(width,1))
        self.output=nn.Sequential(*layers)

    def forward(self,batch):
        fields=[embedding(batch['static'][:,i]) for i,embedding in enumerate(self.embeddings)]
        features=torch.cat(fields,dim=-1)
        if self.kind!='static':
            ages=(torch.log2(batch['history_age_seconds']+1).long()+1).clamp(0,32)
            history=(self.embeddings[self.category](batch['history_category'])+
                     self.embeddings[self.brand](batch['history_brand'])+
                     self.action(batch['history_type'])+self.recency(ages))
            mask=batch['history_mask']
            if self.kind=='pooled':
                weights=mask.float()/mask.sum(-1,keepdim=True).clamp_min(1)
            else:
                query=(fields[self.category]+fields[self.brand])[:,None,:].expand_as(history)
                scores=self.attention(torch.cat([query,history,query-history,query*history],dim=-1)).squeeze(-1)
                scores=scores.masked_fill(~mask,-1e4)
                weights=scores.softmax(-1)*mask.float()
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
            features=torch.cat([features,(weights.unsqueeze(-1)*history).sum(1)],dim=-1)
        return self.output(features).squeeze(-1)
