import os,sys,types,importlib,copy,tempfile,unittest,json,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;T=R.parent
sys.path[:0]=[str(T/'recurrence-single-history-read'),str(T.parent/'runtime/deps'),str(T.parent/'runtime/UniRank')]
import test_single_history_model as fixture
import torch
torch.set_num_threads(1)
pkg=types.ModuleType('streams_test');pkg.__path__=[str(R),str(R/'source-snapshot')];sys.modules[pkg.__name__]=pkg
M=importlib.import_module(pkg.__name__+'.multistream_model');P=importlib.import_module(pkg.__name__+'.QFormerCross8NoFeedback')
KW=dict(token_dim=16,num_heads=4,num_layers=3,ffn_dim=32,num_queries=4,dropout=0,qk_norm=True,work_steps=2,entry_weight=.5)
class Tests(unittest.TestCase):
 def test_zero_and_nonzero_explicit_stream_oracle(self):
  q=fixture.wave((4,4,16));h=fixture.wave((4,5,16),.5);mask=fixture.mask_rows().bool();parent=P.NoFeedbackStage(**KW).double()
  for mode,depth in [('dynamic_write',(1,1)),('dynamic_write',(1,2)),('dynamic_write',(2,1))]:
   m=M.MultiStreamStage(**KW,stream_mode=mode,depth_plan=depth).double();m.load_state_dict(parent.state_dict(),strict=False)
   reference=m.depth_reference(q,h,mask);reference=m.combine_post_ffn(*reference)
   torch.testing.assert_close(m(q,h,mask),reference,rtol=0,atol=0)
   if depth==(1,1):torch.testing.assert_close(m(q,h,mask),parent(q,h,mask),rtol=0,atol=0)
   if mode=='fixed':continue
   with torch.no_grad():m.router.logits.copy_(torch.linspace(-.04,.05,m.router.logits.numel()))
   manual=copy.deepcopy(m);e=manual.layers[0](q,h,mask);a=b=e;prediction=e
   for step in range(2):
    r,wa,wb,pa,pb=manual.router(a,b);u=(a+b)/2+r*(a-b);raw=u
    for _ in range(depth[step]):raw=manual.layers[1](raw,h,mask)
    d=raw-u
    aa=a+pa*(b-a)+(1+wa)*d
    bb=b+pb*(a-b)+((0 if step==0 else 1)+wb)*d
    z=aa if step==0 else (aa+bb)/2
    prediction=manual.layers[2](prediction,z,None);a,b=aa,bb
    if mode=='collapsed_all':a=b=(a+b)/2
   out=manual.post_self_norm(prediction+manual.post_self_attn(prediction,prediction,prediction));expected=manual.combine_post_ffn(out,manual.post_ffn(out));actual=m(q,h,mask)
   torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9)
   actual.square().sum().backward();expected.square().sum().backward()
   for (_,a),(_,b) in zip(m.named_parameters(),manual.named_parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-7,atol=1e-8)
 def test_prediction_does_not_write_work(self):
  m=M.MultiStreamStage(**KW,stream_mode='dynamic_all').double();q=fixture.wave((4,4,16));h=fixture.wave((4,5,16));mask=fixture.mask_rows().bool()
  with torch.no_grad():m.router.logits.fill_(.01)
  before=[];hk=m.layers[1].register_forward_pre_hook(lambda m,a:before.append(a[0].clone()));m(q,h,mask);hk.remove()
  after=[];h1=m.layers[1].register_forward_pre_hook(lambda m,a:after.append(a[0].clone()));h2=m.layers[2].register_forward_hook(lambda m,a,o:o+3);m(q,h,mask);h1.remove();h2.remove()
  for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test_full_constructor_rng_native_params_and_optimizer_registration(self):
  with tempfile.TemporaryDirectory() as tmp:
   fm,batch=fixture.synthetic_map_batch(tmp)
   kw=dict(task='binary_classification',num_queries=8,num_tasks=4,token_dim=16,embedding_dim=16,num_heads=4,num_ns_layers=2,num_unified_layers=3,ffn_ratio=2.,qk_norm=True,tower_hidden_units=[16,8],max_len=5,gpu=-1,loss=['binary_crossentropy']*4,dense_optimizer='AdamW',dense_learning_rate=1e-4,sparse_optimizer='Adagrad',sparse_learning_rate=.05,dense_weight_decay=0,enable_bf16=False,enable_torch_compile=False,gradient_checkpointing=False,net_dropout=0,metrics=['AUC'],verbose=0,model_root=tmp,query_fusion_fixed_scales=[1]*8,query_fusion_training_scales=[1]*8)
   torch.manual_seed(20262027);parent=P.QFormerCross8NoFeedback(fm,**kw);state=copy.deepcopy(parent.state_dict());rng=torch.get_rng_state().clone()
   for mode in M.DYNAMIC:
    torch.manual_seed(20262027);obj=M.QFormerCross8MultiStream(fm,stream_mode=mode,**kw)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for n,v in state.items():torch.testing.assert_close(obj.state_dict()[n],v,rtol=0,atol=0)
    gate=obj.unified_qformer.router.logits;self.assertTrue(any(p is gate for p in obj._dense_params))
    for k,v in parent(batch).items():torch.testing.assert_close(obj(batch)[k],v,rtol=0,atol=0)
if __name__=='__main__':
 start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 print(json.dumps(dict(state='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,wall_seconds=time.monotonic()-start,production_data_reads=0,training_updates=0)));sys.exit(not result.wasSuccessful())
