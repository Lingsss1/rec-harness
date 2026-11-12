"""Local masked activation and residual-bypass probes; only aggregates leave RAM.

Runs a supplied model in evaluation mode. Call inside a controller-owned isolated
worker for untrusted models. This function itself is not a sandbox.
"""
import torch
from rec_harness.core import ContractError


def activation_probe(model,sequence):
    if sequence.ndim!=2 or sequence.dtype!=torch.long or sequence.shape[1]==0:
        raise ContractError('nonempty [batch,length] integer sequence required')
    mask=sequence.ne(0)
    if not mask.any() or not mask[:,-1].all():raise ContractError('nonempty right-aligned histories required')
    if ((~mask[:,1:])&mask[:,:-1]).any():raise ContractError('padding must precede history')
    blocks=list(model.blocks)
    if not blocks:raise ContractError('residual blocks required')
    modes={module:module.training for module in model.modules()};handles=[];summaries={}
    def tensor(value):
        if not isinstance(value,torch.Tensor) or value.ndim!=3 or value.shape[:2]!=sequence.shape or not torch.isfinite(value).all():
            raise ContractError('finite [batch,length,hidden] activation required')
        return value
    def capture(index):
        def hook(module,args,output):
            before=tensor(args[0]);after=tensor(output)
            if before.shape!=after.shape:raise ContractError('residual input/output shape mismatch')
            selected=after[mask].double();delta=(after-before)[mask].double()
            if index in summaries:raise ContractError('shared or repeatedly executed block requires an explicit call-site adapter')
            summaries[index]={'activation_rms':float(selected.square().mean().sqrt()),
                'residual_rms':float(delta.square().mean().sqrt()),
                'zero_fraction':float(selected.eq(0).double().mean()),'valid_positions':int(mask.sum())}
        return hook
    try:
        model.eval()
        for index,block in enumerate(blocks):handles.append(block.register_forward_hook(capture(index)))
        with torch.no_grad():baseline=tensor(model.encode_sequence(sequence)).detach().clone()
        for handle in handles:handle.remove()
        handles=[]
        if len(summaries)!=len(blocks):raise ContractError('declared block not executed')
        lengths=mask.sum(1);slices={'all':torch.ones_like(lengths,dtype=torch.bool),'short':lengths<=8,'long':lengths>8}
        rows=[]
        for index,block in enumerate(blocks):
            # Bypass the whole residual block, preserving its input and external masking.
            handles=[block.register_forward_hook(lambda module,args,output:args[0])]
            with torch.no_grad():changed=tensor(model.encode_sequence(sequence)).detach()
            handles[0].remove();handles=[]
            delta=(changed[:,-1]-baseline[:,-1]).double()
            effect={}
            for name,select in slices.items():
                if not select.any():continue
                reference=baseline[select,-1].double().square().mean().sqrt()
                rms=delta[select].square().mean().sqrt()
                effect[name]={'histories':int(select.sum()),'tail_change_rms':float(rms),
                    'tail_reference_rms':float(reference),
                    'relative_tail_change':float(rms/reference) if reference>0 else None}
            rows.append({'block_index':index,**summaries[index],'bypass':effect})
        return {'schema':'masked-residual-probe-v1','blocks':rows,
            'scope':'local activation and representation sensitivity on supplied histories; not recommendation accuracy or causal mechanism proof',
            'limitations':['Block bypass may produce out-of-distribution representations.','Relative change is undefined when reference RMS is zero.',
                'No interaction IDs, per-history arrays, raw activations, weights or ranks are returned.']}
    finally:
        for handle in handles:handle.remove()
        for module,mode in modes.items():module.training=mode
