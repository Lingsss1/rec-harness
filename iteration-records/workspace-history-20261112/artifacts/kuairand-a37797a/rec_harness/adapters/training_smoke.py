"""Synthetic gradient-step contract checks before spending real training budget."""
import argparse
import importlib.util
from pathlib import Path
import sys

from rec_harness.core import ContractError,digest,file_hash,read_json,write_json


def stress_contract(model, *, itemnum=127, length=16, checker=None):
    import torch
    if checker is None:
        from .model_contract import check_contract
        checker=check_contract
    before=checker(model,itemnum=itemnum,length=length)
    losses=[]
    def aligned(seq,pos,neg):
        with torch.no_grad():
            encoded=model.encode_sequence(seq);positive,negative=model(seq,pos,neg)
            if not torch.allclose(positive,(encoded*model.item_emb(pos)).sum(-1),atol=1e-5,rtol=1e-5) or not torch.allclose(negative,(encoded*model.item_emb(neg)).sum(-1),atol=1e-5,rtol=1e-5):
                raise ContractError('forward scoring differs from encoded representations')
    for step in range(3):
        model.eval();model.zero_grad(set_to_none=True)
        seq=(torch.arange(2*length).reshape(2,length)%(itemnum-2))+1
        seq[0,:length//2]=0
        pos=torch.where(seq!=0,seq+1,0);neg=torch.where(seq!=0,itemnum,0)
        aligned(seq,pos,neg)
        positive,negative=model(seq,pos,neg)
        loss=torch.nn.functional.softplus(negative-positive)[pos!=0].mean()
        if not torch.isfinite(loss):raise ContractError('synthetic training loss is nonfinite')
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if not parameter.requires_grad:continue
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ContractError('synthetic training gradient unavailable or nonfinite')
                norm=parameter.grad.norm()
                if not torch.isfinite(norm):raise ContractError('synthetic gradient norm is nonfinite')
                if norm>0:parameter.add_(parameter.grad,alpha=-0.05/float(norm))
        after=checker(model,itemnum=itemnum,length=length)
        aligned(seq,pos,neg)
        losses.append(float(loss.detach()))
    return {'steps':3,'losses':losses,'before':before,'after':after,'forward_alignment':True,
            'scope':'normalized synthetic gradient stress; does not alter the frozen performance-training optimizer or checkpoints'}


def run_training_smoke(python,source,directory,recipe,artifacts,uid_registry):
    from rec_harness.cpu_job import run_cpu_job
    directory=Path(directory)
    paths={'model.py':Path(source['path']),'worker.py':Path(__file__),
           'model_contract.py':Path(__file__).with_name('model_contract.py')}
    files={name:{'path':str(path.resolve()),'sha256':file_hash(path)} for name,path in paths.items()}
    if files['model.py']['sha256']!=source['sha256']:raise ContractError('training-smoke source changed')
    receipt=run_cpu_job(directory,recipe,files,'worker.py',['--source-hash',source['sha256']],
        python=python,uid_registry=uid_registry,artifacts=artifacts,timeout=90)
    if receipt['process']['status']!='success':raise ContractError('synthetic training contract failed; see '+str(directory))
    report=read_json(directory/'results/report.json')
    if report['source_hash']!=source['sha256'] or report['worker_hash']!=files['worker.py']['sha256'] or report['checker_hash']!=files['model_contract.py']['sha256'] or report['hash']!=digest({k:v for k,v in report.items() if k!='hash'}):
        raise ContractError('synthetic training report identity changed')
    if report['checks']['steps']!=3 or report['checks']['forward_alignment'] is not True or not all(report['checks'][stage][key] is True for stage in ['before','after'] for key in ['causal_input_gradient','tail_readout','empty_history_finite','padding_gradient_zero']):
        raise ContractError('synthetic training checks incomplete')
    return {'status':'success','report_ref':artifacts.put(directory/'results/report.json'),'receipt':receipt}


def main():
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--source-hash',required=True);args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).parent))
    from model_contract import check_contract
    source=Path('/inputs/model.py')
    if file_hash(source)!=args.source_hash:raise ContractError('staged source mismatch')
    spec=importlib.util.spec_from_file_location('candidate',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(20260923)
    model=module.NextItNetModel(127,hidden_units=module.HIDDEN_UNITS,num_blocks=module.NUM_BLOCKS,
        kernel_size=module.KERNEL_SIZE,dropout_rate=module.DROPOUT_RATE,maxlen=module.MAXLEN)
    report={'source_hash':args.source_hash,'worker_hash':file_hash(__file__),
            'checker_hash':file_hash(Path(__file__).with_name('model_contract.py')),
            'checks':stress_contract(model,length=min(16,module.MAXLEN),checker=check_contract)}
    report['hash']=digest(report);write_json('/output/report.json',report)


if __name__=='__main__':main()
