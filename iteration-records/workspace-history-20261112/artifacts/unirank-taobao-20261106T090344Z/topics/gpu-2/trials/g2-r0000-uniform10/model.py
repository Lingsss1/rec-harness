"""Count-preserving block compression before DIN local activation.

All learned modules and widths are inherited from baseline-din. Routing uses
only visible historical and target category/brand IDs; no current action.
Novelty and predictive/cost benefit are unverified prospective hypotheses.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG={'embedding_dim':16,'attention_hidden_units':[64,32],
        'tower_hidden_units':[128,64],'attention_normalization':'none','compression_mode':'uniform10'}


class Model(MultiTaskModel):
    def __init__(self,feature_map,model_id='DIN_Taobao_Action',gpu=-1,
                 embedding_dim=16,attention_hidden_units=(64,32),
                 tower_hidden_units=(128,64),attention_normalization='none',num_tasks=None,
                 task='binary_classification',compression_mode='uniform10',**kwargs):
        if num_tasks is not None and num_tasks!=len(feature_map.labels):raise ValueError('Task count mismatch')
        super().__init__(feature_map,model_id=model_id,gpu=gpu,
                         num_tasks=len(feature_map.labels),task=task,**kwargs)
        if attention_normalization!='none':raise ValueError('DIN baseline uses unnormalized local activation weights')
        if compression_mode not in ('match20','rotated20','uniform20','uniform10'):
            raise ValueError('Unknown compression mode')
        self.compression_mode=compression_mode
        self.embedding_dim=embedding_dim
        self.embedding_layer=FeatureEmbeddingDict(feature_map,embedding_dim)
        self.item_fields=[name for name,spec in feature_map.features.items()
                          if spec.get('source')=='item' and spec['type']!='meta']
        self.user_fields=[name for name,spec in feature_map.features.items()
                          if spec.get('source') not in ['item','action'] and spec['type']!='meta']
        self.key_width=len(self.item_fields)*embedding_dim
        self.action_field=next((name for name,spec in feature_map.features.items() if spec.get('source')=='action'),None)
        self.value_width=self.key_width+(embedding_dim if self.action_field else 0)
        layers=[];width=4*self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width,hidden),Dice(hidden)]);width=hidden
        layers.append(nn.Linear(width,1));self.local_activation=nn.Sequential(*layers)
        input_width=len(self.user_fields)*embedding_dim+self.key_width+self.value_width
        self.towers=nn.ModuleList()
        for _ in feature_map.labels:
            tower=[];width=input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width,hidden),Dice(hidden)]);width=hidden
            tower.append(nn.Linear(width,1));self.towers.append(nn.Sequential(*tower))
        self.reset_parameters();self.model_to_device()

    def _represent(self,inputs):
        user,item,mask=self.get_inputs(inputs)
        user_emb=self.embedding_layer(user)
        item_emb=self.embedding_layer(item)
        target=torch.cat([item_emb[name][:,-1] for name in self.item_fields],dim=-1)
        keys=torch.cat([item_emb[name][:,:-1] for name in self.item_fields],dim=-1)
        values=keys
        if self.action_field:values=torch.cat([values,item_emb[self.action_field][:,:-1]],dim=-1)
        context=torch.cat([user_emb[name] for name in self.user_fields],dim=-1)
        return context,target,keys,values,mask,item

    def _compress(self, values, mask, item):
        batch,length,width=values.shape
        # Fixed released history input, with pre-padding; no history truncation.
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        block_size=5 if self.compression_mode=='uniform20' else 10
        groups=length//block_size
        valid=mask.bool().reshape(batch,groups,block_size)
        grouped=values.reshape(batch,groups,block_size,width)
        if self.compression_mode.startswith('uniform'):
            assignment=valid.unsqueeze(-2).to(values.dtype)
        else:
            match=torch.zeros_like(mask,dtype=torch.bool)
            for field in ('cate_id','brand'):
                ids=item[field]
                historical=ids[:,:-1]
                target=ids[:,-1:]
                # Zero is missing/padding, not positive evidence of a match.
                match=match | ((historical==target) & (historical!=0) & (target!=0))
            match=match.reshape(batch,groups,block_size) & valid
            if self.compression_mode=='rotated20':
                # Rotate membership among VALID events only, keeping each slot's
                # occupancy exactly equal to match20, including partial blocks.
                n=valid.long().sum(-1,keepdim=True)
                rank=valid.long().cumsum(-1)-1
                start=block_size-n
                source=start+(rank+n//2).remainder(n.clamp_min(1))
                source=source.clamp(0,block_size-1)
                match=match.gather(-1,source) & valid
            assignment=torch.stack((valid & ~match,match),dim=-2).to(values.dtype)
        slots=assignment.shape[-2]
        assignment=assignment.reshape(batch*groups,slots,block_size)
        counts=assignment.sum(-1)
        sums=torch.bmm(assignment,grouped.reshape(batch*groups,block_size,width))
        means=sums/counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch,groups*slots,width),counts.reshape(batch,groups*slots)

    def forward(self,inputs):
        context,target,keys,values,mask,item=self._represent(inputs)
        # Values contain every key field plus historical action. Compress them
        # together so item/action alignment is preserved within each summary.
        values,counts=self._compress(values,mask,item)
        keys=values[:,:,:self.key_width]
        batch,length,_=keys.shape
        query=target[:,None,:].expand(-1,length,-1)
        interactions=torch.cat([query,keys,query-keys,query*keys],dim=-1)
        weights=self.local_activation(interactions.reshape(batch*length,-1)).reshape(batch,length)
        # Retain event multiplicity under unnormalized DIN weights. Constant
        # activation recovers the original masked sum exactly in real arithmetic.
        weights=weights*counts.to(weights.dtype)
        pooled=torch.bmm(weights[:,None,:],values).squeeze(1)
        features=torch.cat([context,target,pooled],dim=-1)
        return {label+'_pred':torch.sigmoid(tower(features)) for label,tower in zip(self.feature_map.labels,self.towers)}
