import os,sys,types,importlib,copy,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;T=R.parent
sys.path[:0]=[str(T/'recurrence-single-history-read'),str(T.parent/'runtime/deps'),str(T.parent/'runtime/UniRank')]
import test_single_history_model as fixture
import torch
torch.set_num_threads(1)
pkg=types.ModuleType('local_test');pkg.__path__=[str(R),str(R/'source-snapshot')];sys.modules[pkg.__name__]=pkg
M=importlib.import_module(pkg.__name__+'.multistream_model');P=importlib.import_module(pkg.__name__+'.QFormerCross8NoFeedback')
KW=dict(token_dim=16,num_heads=4,num_layers=3,ffn_dim=32,num_queries=4,dropout=0,qk_norm=True,work_steps=2,entry_weight=.5)
class Tests(unittest.TestCase):
 def test_explicit_state_and_gradient_oracle(self):
  q=fixture.wave((4,4,16));h=fixture.wave((4,5,16),.5);mask=fixture.mask_rows().bool();parent=P.NoFeedbackStage(**KW).double()
  for control in M.CONTROLS:
   m=M.MultiStreamStage(**KW,stream_mode='dynamic_write',stream_control=control).double();m.load_state_dict(parent.state_dict(),strict=False)
   torch.testing.assert_close(m(q,h,mask),parent(q,h,mask),rtol=0,atol=0)
   with torch.no_grad():m.router.logits.copy_(torch.linspace(-.04,.05,m.router.logits.numel()))
   manual=copy.deepcopy(m);a=b=prediction=manual.layers[0](q,h,mask)
   for step in range(2):
    flat=manual.router.logits;x=(a+b)/2;x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6);bias=flat[:5];off=x@flat[5:].reshape(2,16).T
    enabled=not ((step==1 and control.startswith('first_only')) or (step==0 and control.startswith('second_only')))
    w1=.5*(bias[1]+(off[...,0:1] if enabled else 0)).tanh();w2=.5*(bias[2]+(off[...,1:2] if enabled else 0)).tanh()
    if control=='tied_write':w1=w2=(w1+w2)/2
    nr=control=='no_static_routes' or control.endswith('_no_routes');r=.5*bias[0].tanh()*(0 if nr else 1);pa=.25*bias[3].tanh()*(0 if nr else 1);pb=.25*bias[4].tanh()*(0 if nr else 1)
    u=(a+b)/2+r*(a-b);d=manual.layers[1](u,h,mask)-u;aa=a+pa*(b-a)+(1+w1)*d;bb=b+pb*(a-b)+(step+w2)*d
    prediction=manual.layers[2](prediction,aa if step==0 else (aa+bb)/2,None);a,b=aa,bb
    if step==0 and control in ('collapsed_write','collapsed_no_routes'):a=b=(a+b)/2
   out=manual.post_self_norm(prediction+manual.post_self_attn(prediction,prediction,prediction));expected=manual.combine_post_ffn(out,manual.post_ffn(out));actual=m(q,h,mask)
   torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9);actual.square().sum().backward();expected.square().sum().backward()
   for (_,x),(_,y) in zip(m.named_parameters(),manual.named_parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=1e-7,atol=1e-8)
 def test_full_matches_trained_parent_and_no_feedback(self):
  q=fixture.wave((4,4,16));h=fixture.wave((4,5,16));mask=fixture.mask_rows().bool();m=M.MultiStreamStage(**KW,stream_mode='dynamic_write').double();p=M.OriginalStage(**KW,stream_mode='dynamic_write').double()
  with torch.no_grad():m.router.logits.fill_(.013)
  p.load_state_dict(m.state_dict());torch.testing.assert_close(m(q,h,mask),p(q,h,mask),rtol=0,atol=0)
  for c in M.CONTROLS:
   m.stream_control=c;before=[];hk=m.layers[1].register_forward_pre_hook(lambda m,a:before.append(a[0].clone()));m(q,h,mask);hk.remove();after=[];h1=m.layers[1].register_forward_pre_hook(lambda m,a:after.append(a[0].clone()));h2=m.layers[2].register_forward_hook(lambda m,a,o:o+3);m(q,h,mask);h1.remove();h2.remove()
   for x,y in zip(before,after):torch.testing.assert_close(x,y,rtol=0,atol=0)
if __name__=='__main__':unittest.main()
