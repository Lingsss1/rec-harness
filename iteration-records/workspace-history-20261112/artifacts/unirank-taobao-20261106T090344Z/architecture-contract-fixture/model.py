import torch
from torch import nn
CONFIG={'embedding_dim':4,'token_dim':16}
class Model(nn.Module):
    def __init__(self,feature_map,model_id=None,gpu=-1,embedding_dim=4,token_dim=16,**kwargs):
        super().__init__()
        self.labels=list(feature_map.labels)
        self.user_fields=[n for n,s in feature_map.features.items() if s['type']!='meta' and s.get('source') not in ['item','action']]
        self.item_fields=[n for n,s in feature_map.features.items() if s['type']!='meta' and s.get('source') in ['item','action']]
        self.emb=nn.ModuleDict({n:nn.Embedding(s['vocab_size'],embedding_dim) for n,s in feature_map.features.items() if s['type']!='meta'})
        self.field_proj=nn.Linear(embedding_dim,token_dim)
        self.event_proj=nn.Linear(len(self.item_fields)*embedding_dim,token_dim)
        self.task_tokens=nn.Parameter(torch.randn(len(self.labels),token_dim)*.02)
        self.interact=nn.MultiheadAttention(token_dim,2,batch_first=True)
        self.shared_head=nn.Linear(token_dim,1)
    def forward(self,inputs):
        batch,items,mask,_=inputs
        context=self.field_proj(torch.stack([self.emb[n](batch[n]) for n in self.user_fields],dim=1))
        events=self.event_proj(torch.cat([self.emb[n](items[n]) for n in self.item_fields],dim=-1))
        tasks=self.task_tokens.unsqueeze(0).expand(mask.size(0),-1,-1)
        tokens=torch.cat([context,events,tasks],dim=1)
        prefix=torch.zeros((mask.size(0),len(self.user_fields)),dtype=torch.bool,device=mask.device)
        suffix=torch.zeros((mask.size(0),1+len(self.labels)),dtype=torch.bool,device=mask.device)
        padding=torch.cat([prefix,~mask.bool(),suffix],dim=1)
        mixed,_=self.interact(tokens,tokens,tokens,key_padding_mask=padding,need_weights=False)
        values=torch.sigmoid(self.shared_head(mixed[:,-len(self.labels):]))
        return {label+'_pred':values[:,i] for i,label in enumerate(self.labels)}
