"""Independent numerical checks of the exact reviewed cosine aggregation."""
import ast
import copy
from pathlib import Path

from rec_harness.core import ContractError, digest, file_hash, write_json
from .amazon_prefix_candidate import COSINE_VERSION, expected_sources


def aggregation(source, *, zero_logits=False):
    """Extract the reviewed operation, without importing the full model module."""
    import torch
    tree=ast.parse(Path(source).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NextItNetModel')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode_sequence')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='preceding_count' for t in n.targets))
    body=copy.deepcopy(fn.body[start:])
    if zero_logits:
        first=next(n for n in body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='logits' for t in n.targets))
        first.value=ast.BinOp(left=first.value,op=ast.Mult(),right=ast.Constant(value=0.0))
    body[-1].value=ast.Tuple(elts=[ast.Name(id='weights',ctx=ast.Load()),
                                  ast.Name(id='context',ctx=ast.Load()),body[-1].value],ctx=ast.Load())
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='x'),ast.arg(arg='valid')],
                       vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
    module=ast.Module(body=[ast.FunctionDef(name='operation',args=args,body=body,decorator_list=[])],type_ignores=[])
    namespace={'torch':torch,'nn':torch.nn}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    return namespace['operation']


def inspect_cosine(config, source, output):
    import torch
    source=Path(source)
    if source.read_bytes()!=expected_sources(config,'cosine_context')[COSINE_VERSION].encode():
        raise ContractError('cosine source exceeds reviewed transformation')
    torch.set_num_threads(1);torch.manual_seed(20260924)
    operation=aggregation(source);uniform_operation=aggregation(source,zero_logits=True)
    valid=torch.tensor([[0,0,0,0],[0,0,0,1],[0,1,0,1],[1,1,1,1]],dtype=torch.float64).unsqueeze(-1)
    x=torch.randn(4,4,3,dtype=torch.float64,requires_grad=True)
    rows=[];means=[];counts=[]
    for b in range(4):
        row=[];mean=[];count=[]
        for t in range(4):
            predecessors=[j for j in range(t) if valid[b,j,0]==1]
            count.append(len(predecessors))
            if predecessors and valid[b,t,0]==1:
                q=x[b,t]/x[b,t].norm().clamp_min(1e-6)
                states=torch.stack([x[b,j] for j in predecessors])
                logits=torch.stack([(q*(x[b,j]/x[b,j].norm().clamp_min(1e-6))).sum() for j in predecessors])
                a=logits.exp()/logits.exp().sum()
                row.append((a[:,None]*states).sum(0));mean.append(states.mean(0))
            else:
                row.append(x.new_zeros(3));mean.append(x.new_zeros(3))
        rows.append(torch.stack(row));means.append(torch.stack(mean));counts.append(count)
    reference=torch.stack(rows);uniform=torch.stack(means);counts=torch.tensor(counts).unsqueeze(-1)
    weights,context,encoded=operation(x,valid)
    torch.testing.assert_close(context,reference,atol=1e-12,rtol=1e-12)
    expected=torch.where(counts>0,.85*x+.15*reference,x)*valid
    torch.testing.assert_close(encoded,expected,atol=1e-12,rtol=1e-12)
    grad=torch.autograd.grad(encoded.sum(),x,retain_graph=True)[0]
    reference_grad=torch.autograd.grad(expected.sum(),x,retain_graph=True)[0]
    torch.testing.assert_close(grad,reference_grad,atol=1e-11,rtol=1e-11)
    _,zero_context,zero_encoded=uniform_operation(x,valid)
    torch.testing.assert_close(zero_context,uniform,atol=1e-12,rtol=1e-12)
    incumbent=torch.where(counts>0,.85*x+.15*uniform,x)*valid
    torch.testing.assert_close(zero_encoded,incumbent,atol=1e-12,rtol=1e-12)
    mask=(counts<=1).expand_as(x)
    torch.testing.assert_close(encoded[mask],incumbent[mask],atol=1e-12,rtol=1e-12)
    assert torch.count_nonzero(encoded[valid.expand_as(x)==0])==0
    for t in range(4):
        changed=x.detach().clone();changed[:,t+1:]+=19
        torch.testing.assert_close(operation(changed,valid)[2][:,:t+1],encoded[:,:t+1],atol=1e-12,rtol=1e-12)
    special=torch.tensor([[[1.,0.],[0.,1.],[1.,0.]]],dtype=torch.float64)
    v=torch.ones(1,3,1,dtype=torch.float64)
    w,c,_=operation(special,v);swapped=special[:,[1,0,2]];ws,cs,_=operation(swapped,v)
    assert w[0,2,0]>w[0,2,1]
    torch.testing.assert_close(w[0,2,:2],ws[0,2,:2].flip(0))
    torch.testing.assert_close(c[:,2],cs[:,2])
    assert not torch.allclose(c[:,2],special[:,:2].mean(1))
    dtype_checks={}
    for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        for length in (1,4):
            for scale in (0.,1e-9,1.):
                z=(torch.randn(3,length,3)*scale).to(dtype).requires_grad_()
                validity=torch.ones(3,length,1,dtype=dtype);validity[0]=0
                result=operation(z,validity)
                assert all(torch.isfinite(item).all() for item in result)
                result[2].float().square().sum().backward()
                assert torch.isfinite(z.grad).all()
                assert result[2].dtype==dtype
        dtype_checks[str(dtype)]=True
    assert torch.autograd.gradcheck(lambda states:operation(states,valid)[2],(x,))
    report={'source_hash':file_hash(source),'inspector_hash':file_hash(__file__),
            'checks':{'explicit_loop_outputs_and_gradients':True,'zero_logits_matches_uniform':True,
                      'content_swap_discrimination':True,'zero_or_one_predecessor_matches_incumbent':True,
                      'future_causality':True,'padding_zero':True,'double_precision_gradcheck':True,
                      'finite_degenerate_inputs_by_dtype':dtype_checks},
            'max_context_error':float((context-reference).abs().max().detach()),
            'scope':'synthetic aggregation correctness; no recommendation performance conclusion'}
    report['hash']=digest(report);write_json(output,report);return report
