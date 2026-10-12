"""Check the exact reviewed aggregation against an independent loop and derivatives."""
import ast
import copy
from pathlib import Path

from rec_harness.core import ContractError,digest,file_hash,write_json
from .amazon_prefix_candidate import expected_sources,RECENCY_VERSION


def inspect_recency(config, source, output):
    import torch
    source=Path(source)
    if source.read_bytes()!=expected_sources(config,'linear_recency')[RECENCY_VERSION].encode():
        raise ContractError('recency source exceeds reviewed transformation')
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NextItNetModel')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode_sequence')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='weights' for t in n.targets))
    functions={}
    for name in ('encoded','context_only'):
        body=copy.deepcopy(fn.body[start:])
        if name=='context_only':body[-1]=ast.Return(value=ast.Name(id='context',ctx=ast.Load()))
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='x'),ast.arg(arg='valid')],vararg=None,
                           kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
        module=ast.Module(body=[ast.FunctionDef(name=name,args=args,body=body,decorator_list=[])],type_ignores=[])
        namespace={'torch':torch,'nn':torch.nn}
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
        functions[name]=namespace[name]
    torch.set_num_threads(1);torch.manual_seed(20260924)
    valid=torch.tensor([[0,0,0,0,0],[0,0,0,0,1],[0,1,0,1,1],[1,1,1,1,1]],dtype=torch.float64).unsqueeze(-1)
    x=torch.randn((4,5,2),dtype=torch.float64,requires_grad=True)
    expected=[];uniform=[];counts=[]
    for b in range(4):
        row=[];plain=[];number=[]
        for t in range(5):
            predecessors=[j for j in range(t) if valid[b,j,0].item()==1]
            n=len(predecessors);number.append(n)
            row.append(sum((i+1)*x[b,j] for i,j in enumerate(predecessors))/(n*(n+1)/2) if n else x.new_zeros(2))
            plain.append(sum(x[b,j] for j in predecessors)/n if n else x.new_zeros(2))
        expected.append(torch.stack(row));uniform.append(torch.stack(plain));counts.append(number)
    expected=torch.stack(expected);uniform=torch.stack(uniform)
    actual=functions['context_only'](x,valid)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    count=torch.tensor(counts).unsqueeze(-1)
    output_tensor=functions['encoded'](x,valid)
    reference=torch.where(count>0,0.85*x+0.15*expected,x)*valid
    torch.testing.assert_close(output_tensor,reference,atol=1e-12,rtol=1e-12)
    old=torch.where(count>0,0.85*x+0.15*uniform,x)*valid
    mask=(count<=1).expand_as(x)
    torch.testing.assert_close(output_tensor[mask],old[mask],atol=1e-12,rtol=1e-12)
    assert torch.equal(output_tensor[valid.expand_as(x)==0],torch.zeros_like(output_tensor[valid.expand_as(x)==0]))
    scalar=torch.tensor([2.,4.,8.,16.],dtype=torch.float64).reshape(1,4,1)
    torch.testing.assert_close(functions['context_only'](scalar,torch.ones_like(scalar)),
                              torch.tensor([0.,2.,10/3,34/6],dtype=torch.float64).reshape(1,4,1))
    for t in range(5):
        perturbed=x.detach().clone();perturbed[:,t:]+=17
        torch.testing.assert_close(functions['context_only'](perturbed,valid)[:,:t+1],actual[:,:t+1],atol=1e-12,rtol=1e-12)
    grad=torch.autograd.grad(actual[3,4].sum(),x,retain_graph=True)[0]
    reference_grad=torch.zeros_like(x)
    for j in range(4):reference_grad[3,j]=(j+1)/10
    torch.testing.assert_close(grad,reference_grad,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(functions['context_only'],(x,valid))
    report={'source_hash':file_hash(source),'inspector_hash':file_hash(__file__),
            'checks':{'explicit_loop':True,'hand_computed_contexts':True,'padding_zero':True,
                      'zero_or_one_predecessor_matches_incumbent':True,'causal_context':True,
                      'analytic_gradient':True,'double_precision_gradcheck':True},
            'max_context_error':float((actual-expected).abs().max().detach()),
            'scope':'synthetic aggregation semantics; no performance conclusion'}
    report['hash']=digest(report);write_json(output,report)
    return report
