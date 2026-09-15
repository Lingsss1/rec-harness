"""Minimal two-stream routing around native work attention, original release inputs.

This is not full LoopCTR: no added FFN/MoE/depth supervision, prediction never writes work.
"""
from functools import partial
import torch
from torch import nn
from .QFormerCross8 import QFormerCross8
from .QFormerCross8NoFeedback import NoFeedbackStage

DYNAMIC={'fixed':(), 'static':(), 'dynamic_read':(0,), 'dynamic_write':(1,2),
         'dynamic_bypass':(3,4), 'dynamic_all':(0,1,2,3,4), 'collapsed_all':(0,1,2,3,4)}

class StreamRouter(nn.Module):
    def __init__(self,width,mode):
        super().__init__()
        if mode not in DYNAMIC:raise ValueError(mode)
        self.width=width;self.mode=mode;self.rows=DYNAMIC[mode]
        # Zero construction consumes no RNG; only active dynamic projections are registered.
        self.logits=nn.Parameter(torch.zeros(5+len(self.rows)*width))
    def forward(self,a,b):
        if a.shape!=b.shape or a.shape[-1]!=self.width:raise ValueError('Aligned stream states required')
        base=self.logits[:5]
        if self.mode=='fixed':base=base*0
        output=base.expand(*a.shape[:-1],5)
        if self.rows:
            x=(a+b).to(self.logits.dtype)/2
            x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
            offsets=x@self.logits[5:].reshape(len(self.rows),self.width).T
            # Python row mapping is static, so no data-dependent hard routing.
            parts=[]
            for row in range(5):
                parts.append(output[...,row]+(offsets[...,self.rows.index(row)] if row in self.rows else 0))
            output=torch.stack(parts,-1)
        coeff=output.tanh().to(a.dtype)
        return tuple(coeff[...,i:i+1]*(.25 if i>=3 else .5) for i in range(5))

class MultiStreamStage(NoFeedbackStage):
    def __init__(self,*args,stream_mode='static',depth_plan=(1,1),**kwargs):
        super().__init__(*args,**kwargs)
        self.router=StreamRouter(kwargs.get('token_dim',args[0] if args else None),stream_mode)
        self.stream_mode=stream_mode
        self.depth_plan=tuple(depth_plan)
        if self.depth_plan not in ((1,1),(1,2),(2,1)):raise ValueError(depth_plan)
        if self.work_steps!=2 or self.entry_weight!=.5:raise ValueError('Exactly two native work calls required')
    def work(self,u,history,mask,round_index):
        for _ in range(self.depth_plan[round_index]):u=self.layers[1](u,history,mask)
        return u
    def depth_reference(self,queries,history,mask=None):
        e=self.layers[0](queries,history,mask)
        w=self.work(e,history,mask,0);y=self.layers[2](e,w,None)
        w=self.work((w+e)/2,history,mask,1);y=self.layers[2](y,w,None)
        out=self.post_self_norm(y+self.dropout(self.post_self_attn(y,y,y)))
        return out,self.dropout(self.post_ffn(out))
    def forward_components(self,queries,history,mask=None):
        e=self.layers[0](queries,history,mask);a=b=e
        r,wa,wb,pa,pb=self.router(a,b)
        mean=(a+b)/2;shift=r*(a-b);u=mean+shift
        raw=self.work(u,history,mask,0);d=raw-u
        da=pa*(b-a);db=pb*(a-b)
        a1=raw+(a-u)+da+wa*d
        b1=b+db+wb*d
        y1=self.layers[2](e,a1,None)
        if self.stream_mode=='collapsed_all':a1=b1=(a1+b1)/2
        r,wa,wb,pa,pb=self.router(a1,b1)
        shift=r*(a1-b1);u=(a1+b1)/2+shift
        raw=self.work(u,history,mask,1);d=raw-u
        da=pa*(b1-a1);db=pb*(a1-b1)
        # Mean readout of the two updated streams, rearranged for exact native zero-gate parity.
        z2=raw-shift+(da+db)/2+(wa+wb)/2*d
        y2=self.layers[2](y1,z2,None)
        out=self.post_self_norm(y2+self.dropout(self.post_self_attn(y2,y2,y2)))
        return out,self.dropout(self.post_ffn(out))

class QFormerCross8MultiStream(QFormerCross8):
    def __init__(self,feature_map,model_id='QFormerCross8MultiStream',stream_mode='dynamic_write',depth_plan=(1,1),**kwargs):
        super().__init__(feature_map,model_id=model_id,
            _unified_qformer_cls=partial(MultiStreamStage,stream_mode=stream_mode,depth_plan=depth_plan,work_steps=2,entry_weight=.5),**kwargs)
