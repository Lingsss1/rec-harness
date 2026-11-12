"""DIN-style reference plus a residual branch of marked event counts.

Only supplied causal history is used. Equality is on encoded category/brand IDs,
not verified raw identities. Counts are restricted to the supplied history cap.
The four proposal variants collapse the target/action axes of the SAME count
cube. Parameter counts and operations match; effective input rank does not.
No claim of scientific priority: explicit interest intensity predates this study.
"""
import torch
from torch import nn

CONFIG={'kind':'din','embedding_dim':16,'hidden':[128,64],'lr':0.001,
        'target_resolved':True,'action_resolved':False,'count_hidden':32}


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
        if self.kind != 'din':
            raise ValueError('This study fixes the backbone to the DIN-style reference')
        self.target_resolved=bool(config['target_resolved'])
        self.action_resolved=bool(config['action_resolved'])
        self.count_head=nn.Sequential(nn.Linear(48,config['count_hidden']),
                                      nn.ReLU(),nn.Linear(config['count_hidden'],1))
        # Added AFTER the complete backbone: preserve its seeded initialization.
        # Initial logits are exactly the reference logits for matching seed/meta.
        nn.init.zeros_(self.count_head[-1].weight)
        nn.init.zeros_(self.count_head[-1].bias)

    def _backbone_forward(self,batch):
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


    def _count_cube(self,batch):
        # [B,L,4] disjoint routes: neither, category only, brand only, both.
        # Zero is padding/missing; never count two zero IDs as a match.
        mask=batch['history_mask']
        target_category=batch['static'][:,self.category,None]
        target_brand=batch['static'][:,self.brand,None]
        category=(batch['history_category']==target_category) & (target_category>0)
        brand=(batch['history_brand']==target_brand) & (target_brand>0)
        routes=torch.stack((~category & ~brand,category & ~brand,
                            ~category & brand,category & brand),dim=-1)
        routes=routes.to(dtype=torch.float32)*mask.unsqueeze(-1)
        actions=torch.stack(tuple(batch['history_type']==i for i in (1,2,3,4)),dim=-1)
        actions=actions.to(dtype=torch.float32)
        age=batch['history_age_seconds'].clamp_min(0)
        # Total count and exponential recency mass with 1-day / 7-day time constants.
        # Fixed constants, no label fitting or wider history acquisition.
        kernels=torch.stack((torch.ones_like(age),torch.exp(-age/86400.0),
                             torch.exp(-age/604800.0)),dim=-1)
        # [B,4 target routes,4 actions,3 kernels]; no [B,L,4,4,3] tensor.
        marked=routes.unsqueeze(-1)*actions.unsqueeze(-2)
        cube=torch.bmm(marked.flatten(2).transpose(1,2),kernels).reshape(-1,4,4,3)
        if not self.target_resolved:
            cube=cube.sum(dim=1,keepdim=True).expand(-1,4,-1,-1)/4.0
        if not self.action_resolved:
            cube=cube.sum(dim=2,keepdim=True).expand(-1,-1,4,-1)/4.0
        return cube

    def _count_features(self,batch):
        # Fixed scale; all-zero empty history stays zero in all four variants.
        return torch.log1p(self._count_cube(batch)).flatten(1)/4.859812404361672

    def forward(self,batch):
        logits=self._backbone_forward(batch)
        correction=self.count_head(self._count_features(batch)).squeeze(-1)
        return logits+correction

    def diagnostics(self,batch):
        # Host calls on a fixed small validation prefix; descriptive only.
        features=self._count_features(batch)
        correction=self.count_head(features).squeeze(-1)
        mask=batch['history_mask']
        target_category=batch['static'][:,self.category,None]
        target_brand=batch['static'][:,self.brand,None]
        category=mask & (target_category>0) & (batch['history_category']==target_category)
        brand=mask & (target_brand>0) & (batch['history_brand']==target_brand)
        empty=~mask.any(dim=1)
        cube=self._count_cube(batch)
        result={
            'residual_abs_mean':correction.abs().mean(),
            'residual_mean':correction.mean(),
            'feature_rms':features.square().mean().sqrt(),
            'empty_fraction':empty.float().mean(),
            'category_match_fraction':category.any(dim=1).float().mean(),
            'brand_match_fraction':brand.any(dim=1).float().mean(),
            'joint_match_fraction':(category & brand).any(dim=1).float().mean(),
            'empty_residual_abs':(correction.abs()*empty).sum()/empty.sum().clamp_min(1),
            'represented_count_mean':cube[:,:,:,0].sum(dim=(1,2)).mean(),
        }
        for i,name in enumerate(('pv','cart','fav','buy')):
            present=(mask & (batch['history_type']==i+1)).any(dim=1)
            result['action_'+name+'_fraction']=present.float().mean()
        return result
