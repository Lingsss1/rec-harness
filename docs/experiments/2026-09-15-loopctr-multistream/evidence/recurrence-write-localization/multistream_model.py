"""Round-localized dynamic writing with the exact previous parameter layout."""
from functools import partial
import torch
from .original_streams import MultiStreamStage as OriginalStage
from .QFormerCross8 import QFormerCross8
CONTROLS=('full','first_only','second_only','collapsed_write','tied_write','no_static_routes','first_only_no_routes','second_only_no_routes','collapsed_no_routes')
class MultiStreamStage(OriginalStage):
 def __init__(self,*args,stream_control='full',**kwargs):
  super().__init__(*args,**kwargs)
  if stream_control not in CONTROLS:raise ValueError(stream_control)
  self.stream_control=stream_control
 def routed(self,a,b,step):
  c=self.stream_control;r,wa,wb,pa,pb=self.router(a,b)
  disabled=(step==1 and c.startswith('first_only')) or (step==0 and c.startswith('second_only'))
  if disabled:
   # Keep shared learned static biases; remove only input-dependent write offsets.
   wa=(self.router.logits[1].tanh()*.5).to(a.dtype).expand_as(wa)
   wb=(self.router.logits[2].tanh()*.5).to(a.dtype).expand_as(wb)
  if c=='tied_write':wa=wb=(wa+wb)/2
  if c=='no_static_routes' or c.endswith('_no_routes'):r=r*0;pa=pa*0;pb=pb*0
  return r,wa,wb,pa,pb
 def forward_components(self,queries,history,mask=None):
  e=self.layers[0](queries,history,mask);a=b=e
  r,wa,wb,pa,pb=self.routed(a,b,0)
  mean=(a+b)/2;shift=r*(a-b);u=mean+shift
  raw=self.layers[1](u,history,mask);d=raw-u
  da=pa*(b-a);db=pb*(a-b)
  a1=raw+(a-u)+da+wa*d;b1=b+db+wb*d
  y1=self.layers[2](e,a1,None)
  if self.stream_control in ('collapsed_write','collapsed_no_routes'):a1=b1=(a1+b1)/2
  r,wa,wb,pa,pb=self.routed(a1,b1,1)
  shift=r*(a1-b1);u=(a1+b1)/2+shift
  raw=self.layers[1](u,history,mask);d=raw-u
  da=pa*(b1-a1);db=pb*(a1-b1)
  z2=raw-shift+(da+db)/2+(wa+wb)/2*d
  y2=self.layers[2](y1,z2,None)
  out=self.post_self_norm(y2+self.dropout(self.post_self_attn(y2,y2,y2)))
  return out,self.dropout(self.post_ffn(out))
class QFormerCross8MultiStream(QFormerCross8):
 def __init__(self,feature_map,model_id='QFormerCross8MultiStream',stream_mode='dynamic_write',stream_control='full',**kwargs):
  if stream_mode!='dynamic_write':raise ValueError('Only dynamic writing tested in this round')
  super().__init__(feature_map,model_id=model_id,_unified_qformer_cls=partial(MultiStreamStage,stream_mode=stream_mode,stream_control=stream_control,work_steps=2,entry_weight=.5),**kwargs)
