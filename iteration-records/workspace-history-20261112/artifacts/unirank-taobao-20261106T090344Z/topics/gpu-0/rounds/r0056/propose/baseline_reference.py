"""DIN local-activation pooling adapted to UniRank's four-task Taobao schema.

Mechanism references: Zhou et al. DIN (KDD 2018), arXiv:1706.06978;
DeepCTR-Torch AttentionSequencePoolingLayer/LocalActivationUnit.
This is a field and task adaptation, not a claim of exact paper reproduction.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG={'embedding_dim':16,'attention_hidden_units':[64,32],
        'tower_hidden_units':[128,64],'attention_normalization':'none'}


class Model(MultiTaskModel):
    def __init__(self,feature_map,model_id='DIN_Taobao_Action',gpu=-1,
                 embedding_dim=16,attention_hidden_units=(64,32),
                 tower_hidden_units=(128,64),attention_normalization='none',num_tasks=None,
                 task='binary_classification',**kwargs):
        if num_tasks is not None and num_tasks!=len(feature_map.labels):raise ValueError('Task count mismatch')
        super().__init__(feature_map,model_id=model_id,gpu=gpu,
                         num_tasks=len(feature_map.labels),task=task,**kwargs)
        if attention_normalization!='none':raise ValueError('DIN baseline uses unnormalized local activation weights')
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
        return context,target,keys,values,mask

    def forward(self,inputs):
        context,target,keys,values,mask=self._represent(inputs)
        batch,length,_=keys.shape
        if length:
            query=target[:,None,:].expand(-1,length,-1)
            interactions=torch.cat([query,keys,query-keys,query*keys],dim=-1)
            weights=self.local_activation(interactions.reshape(batch*length,-1)).reshape(batch,length)
            weights=weights*mask.to(weights.dtype)
            pooled=torch.bmm(weights[:,None,:],values).squeeze(1)
        else:pooled=target.new_zeros((batch,self.value_width))
        features=torch.cat([context,target,pooled],dim=-1)
        return {label+'_pred':torch.sigmoid(tower(features)) for label,tower in zip(self.feature_map.labels,self.towers)}
