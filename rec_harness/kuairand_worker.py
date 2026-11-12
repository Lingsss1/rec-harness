"""Host-owned CUDA checks, objective, training and exhaustive research scoring."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from .core import ContractError, canonical, digest, file_hash, read_json, write_json


def load_candidate(path, cfg):
    from .kuairand_source_contract import review_source
    review_source(Path(path).read_text())
    spec = importlib.util.spec_from_file_location('research_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.build(dict(cfg))
    if not callable(getattr(encoder, 'encode_all', None)) or not callable(getattr(encoder, 'score_queries', None)):
        raise ContractError('candidate needs encode_all(ids,times) and score_queries(context,ids,rows,positions,candidates)')
    return encoder


def shared_initialization(encoder, cfg):
    """Share all shape/name-compatible untrained baseline parameters, never weights from training."""
    from longseq_bench.models import ModelConfig, HSTUEncoder, DAREEncoder
    from longseq_bench.ccformer import CCFormerEncoder
    from longseq_bench.train import seed_everything
    seed_everything(42)
    config = ModelConfig(**cfg)
    cls = DAREEncoder if config.name == 'dare' else CCFormerEncoder if config.name == 'ccformer' else HSTUEncoder
    baseline = cls(config).state_dict(); state = encoder.state_dict()
    shared = {k: v for k, v in baseline.items() if k in state and state[k].shape == v.shape}
    encoder.load_state_dict({**state, **shared})
    seed_everything(42)
    return {'shared_parameter_names': sorted(shared), 'shared_parameter_names_hash': digest(sorted(shared)),
        'confounds': 'New/differently shaped parameters and architecture-specific dropout draws may differ; sampler order is shared.'}


def validate_treatment(treatment):
    if set(treatment) != {'optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'loss', 'temperature'}:
        raise ContractError('unknown or missing training treatment field')
    if treatment['optimizer'] not in {'adamw', 'adam', 'sgd'} or treatment['scheduler'] not in {'constant', 'cosine'}:
        raise ContractError('unregistered optimizer or scheduler')
    if treatment['loss'] not in {'sampled_softmax', 'bpr'}:
        raise ContractError('unregistered objective')
    for key in ('learning_rate', 'temperature'):
        if not isinstance(treatment[key], (float, int)) or not math.isfinite(treatment[key]) or treatment[key] <= 0:
            raise ContractError('invalid ' + key)
    if not math.isfinite(treatment['weight_decay']) or treatment['weight_decay'] < 0:
        raise ContractError('invalid weight decay')


def full_size(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from longseq_bench.models import BenchmarkModel, ModelConfig
    from longseq_bench.train import seed_everything
    rank = int(os.environ.get('RANK', '0')); local = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group('nccl', timeout=timedelta(hours=24))
    seed_everything(42)
    cfg = read_json(Path(args.benchmark)/'configs/benchmark.json')
    model = BenchmarkModel(ModelConfig(name=args.model, **cfg['model']),
        cfg['training']['negatives'], cfg['training']['temperature'], cfg['training']['loss_chunk']).cuda()
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    b, h = cfg['training']['local_batch_size'], cfg['model']['max_history']
    ids = torch.randint(1, cfg['model']['num_items'] + 1, (b, h + 1), device='cuda')
    batch = {'ids': ids[:, :-1], 'labels': ids[:, 1:],
        'times': torch.arange(h + 1, device='cuda').expand(b, -1) * 1000}
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    loss = wrapped(batch)
    if not torch.isfinite(loss): raise ContractError('nonfinite full-size loss')
    loss.backward()
    if not any(p.grad is not None for p in model.parameters()): raise ContractError('missing full-size gradients')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ContractError('nonfinite full-size gradients')
    torch.cuda.synchronize()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    write_json(out/('rank-%d.json' % rank), {'kind': 'synthetic_full_size', 'model': args.model,
        'rank': rank, 'world_size': world, 'local_batch': b, 'history': h, 'negatives': 128,
        'loss': float(loss), 'seconds': time.perf_counter()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'passed': True})
    if world > 1: dist.destroy_process_group()


def wrapper(encoder, cfg, treatment):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    class HostModel(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = encoder
        def logits(self, context, ids, rows, positions, candidates):
            result = self.encoder.score_queries(context, ids, rows, positions, candidates).float()
            if result.shape != candidates.shape or not torch.isfinite(result).all():
                raise ContractError('invalid candidate scores')
            return result
        def forward(self, batch):
            ids, times, labels = batch['ids'], batch['times'], batch['labels']
            context = self.encoder.encode_all(ids, times)
            rows, positions = labels.ne(0).nonzero(as_tuple=True)
            if not len(rows): raise ContractError('no supervised positions')
            loss = torch.zeros((), device=ids.device)
            for left in range(0, len(rows), cfg['training']['loss_chunk']):
                r, p = rows[left:left+cfg['training']['loss_chunk']], positions[left:left+cfg['training']['loss_chunk']]
                positive = labels[r, p, None]
                negatives = torch.randint(1, cfg['model']['num_items']+1, (len(r), 128), device=ids.device)
                candidates = torch.cat([positive, negatives], -1)
                def part(r, p, candidates, context):
                    logits = self.logits(context, ids, r, p, candidates) / treatment['temperature']
                    collision = candidates[:, 1:].eq(candidates[:, :1])
                    if treatment['loss'] == 'sampled_softmax':
                        logits = torch.cat([logits[:, :1], logits[:, 1:].masked_fill(collision, -5e4)], -1)
                        return -torch.nn.functional.log_softmax(logits, -1)[:, 0].sum()
                    values = torch.nn.functional.softplus(logits[:, 1:] - logits[:, :1])
                    return values.masked_fill(collision, 0).sum() / 128
                loss = loss + checkpoint(part, r, p, candidates, context, use_reentrant=False)
            return loss / len(rows)
    return HostModel()


def checks(args):
    import torch
    from longseq_bench.train import seed_everything
    seed_everything(42)
    cfg = read_json(args.config); treatment = read_json(args.treatment)
    validate_treatment(treatment)
    # Keep real embedding width/catalog; small sequence checks isolate causality.
    small = {**cfg['model'], 'name': args.model, 'max_history': 16,
             'truncate_keep': 8, 'sla_local': 4, 'sla_global': 2}
    device = args.device
    encoder = load_candidate(args.candidate, small).to(device)
    host = wrapper(encoder, {**cfg, 'model': small}, treatment).to(device)
    ids = torch.randint(1, small['num_items']+1, (2, 16), device=device)
    times = torch.arange(17, device=device).expand(2, -1) * 1000
    labels = ids.roll(-1, dims=1)
    host.train(); loss = host({'ids': ids, 'times': times, 'labels': labels}); loss.backward()
    if not torch.isfinite(loss) or not any(p.grad is not None and p.grad.abs().sum() > 0 for p in host.parameters()):
        raise ContractError('no finite nonzero training gradients')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in host.parameters()):
        raise ContractError('nonfinite gradients')
    host.eval()
    with torch.no_grad():
        rows = torch.tensor([0, 1], device=device); pos = torch.tensor([5, 5], device=device)
        candidates = torch.randint(1, small['num_items']+1, (2, 4), device=device)
        context = encoder.encode_all(ids, times)
        reference = host.logits(context, ids, rows, pos, candidates)
        alternate = ids.clone(); alternate[:, 7:] = ids[:, 7:].roll(1, 0)
        altered_times = times.clone(); altered_times[:, 7:] += 1000000
        causal = host.logits(encoder.encode_all(alternate, altered_times), alternate, rows, pos, candidates)
        if not torch.allclose(reference, causal, atol=2e-5, rtol=2e-4):
            raise ContractError('future-history causality check failed')
        prefix_ids, prefix_times = ids[:, :6], times[:, :7]
        physical = host.logits(encoder.encode_all(prefix_ids, prefix_times), prefix_ids, rows, pos, candidates)
        if not torch.allclose(reference, physical, atol=2e-5, rtol=2e-4):
            raise ContractError('physical-prefix equivalence failed')
        single = host.logits(encoder.encode_all(ids[:1], times[:1]), ids[:1], rows[:1], pos[:1], candidates[:1])
        if not torch.allclose(reference[:1], single, atol=2e-5, rtol=2e-4):
            raise ContractError('cross-user dependence')
        separate = torch.cat([host.logits(context, ids, rows, pos, candidates[:, i:i+1]) for i in range(4)], -1)
        if not torch.allclose(reference, separate, atol=2e-5, rtol=2e-4):
            raise ContractError('cross-candidate dependence')
        changed = candidates.clone(); changed[:, 1:] = changed[:, 1:].roll(1, 0)
        if not torch.allclose(reference[:, 0], host.logits(context, ids, rows, pos, changed)[:, 0], atol=2e-5, rtol=2e-4):
            raise ContractError('candidate independence failed')
    write_json(args.output, {'passed': True, 'kind': 'synthetic_admission', 'candidate_hash': file_hash(args.candidate),
        'checks': ['finite_loss', 'finite_nonzero_gradients', 'causal_future_items_and_times', 'physical_prefix_equivalence',
            'user_independence', 'candidate_independence', 'candidate_chunk_invariance']})


def score_model(model, loader, device, num_items, chunk, *, metrics=True):
    import torch
    model.eval(); totals = {k: 0. for k in ['ndcg@10', 'ndcg@50', 'hr@10', 'hr@50', 'mrr', 'mrr_exact']}
    count = 0
    with torch.inference_mode():
        for cpu in loader:
            batch = {k: v.to(device) for k, v in cpu.items()}
            ids, times = batch['ids'], batch['times']
            rows = torch.arange(len(ids), device=device); pos = batch['length'] - 1
            context = model.encoder.encode_all(ids, times)
            if metrics:
                targets = batch['target']
                positive = model.logits(context, ids, rows, pos, targets[:, None]).squeeze(-1)
                seen = torch.zeros(len(ids), num_items+1, dtype=torch.bool, device=device)
                seen.scatter_(1, ids, True); repeated = seen[rows, targets]
                better = torch.zeros(len(ids), dtype=torch.long, device=device)
            for left in range(1, num_items+1, chunk):
                candidates = torch.arange(left, min(left+chunk, num_items+1), device=device)[None].expand(len(ids), -1)
                logits = model.logits(context, ids, rows, pos, candidates)
                if metrics:
                    eligible = ~seen.gather(1, candidates)
                    ahead = (logits > positive[:, None]) | ((logits == positive[:, None]) & (candidates < targets[:, None]))
                    better += (ahead & eligible & candidates.ne(targets[:, None])).sum(1)
            if metrics:
                exact = better + 1; ranks = torch.where(repeated | (exact > 2500), 2501, exact)
                for k in (10, 50):
                    hit = ranks <= k
                    totals['hr@%d' % k] += float(hit.float().sum())
                    totals['ndcg@%d' % k] += float((hit / torch.log2(ranks.float()+1)).sum())
                totals['mrr'] += float((1./ranks.float()).sum())
                totals['mrr_exact'] += float(torch.where(repeated, 0., 1./exact.float()).sum())
            count += len(ids)
    if not count: raise ContractError('empty scoring users')
    return {**{k: v/count for k, v in totals.items()}, 'users': count} if metrics else {'users': count}


def train(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    from longseq_bench.data import verify_cache
    from longseq_bench.train import seed_everything, rng_state, restore_rng, atomic_save
    from .kuairand_data import ResearchDataset
    cfg, treatment = read_json(args.config), read_json(args.treatment)
    if args.stop_after_epoch is not None and not args.synthetic:
        raise ContractError('early-stop injection is reserved for synthetic recovery acceptance')
    validate_treatment(treatment)
    if (cfg['training']['epochs'], cfg['training']['seed'], cfg['training']['global_batch_size'],
        cfg['training']['negatives'], cfg['training']['precision']) != (101, 42, 128, 128, 'fp32'):
        raise ContractError('frozen training contract changed')
    local = int(os.environ.get('LOCAL_RANK', '0')); rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 2: raise ContractError('formal research requires exactly two GPUs')
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(hours=24))
    meta = verify_cache(args.cache)
    if meta['max_history'] != cfg['model']['max_history']: raise ContractError('cache/model history mismatch')
    seed_everything(42)
    encoder = load_candidate(args.candidate, {**cfg['model'], 'name': args.model})
    initialization = shared_initialization(encoder, {**cfg['model'], 'name': args.model})
    model = wrapper(encoder, cfg, treatment).to(device)
    opt_type = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[treatment['optimizer']]
    kwargs = {'lr': treatment['learning_rate'], 'weight_decay': treatment['weight_decay']}
    if treatment['optimizer'] != 'sgd': kwargs['betas'] = (.9, .98)
    optimizer = opt_type(model.parameters(), **kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 101) if treatment['scheduler'] == 'cosine' else None
    fingerprint = digest({'candidate': file_hash(args.candidate), 'config': cfg, 'treatment': treatment,
        'cache': meta, 'worker': file_hash(__file__), 'torch': torch.__version__, 'numpy': np.__version__, 'world': world})
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'last.pt'; start = 0; records = []
    train_data, val_data = ResearchDataset(args.cache, 'train'), ResearchDataset(args.cache, 'validation')
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=0, drop_last=True)
    loader = DataLoader(train_data, batch_size=64, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=cfg['evaluation']['batch_size'], shuffle=False)
    panel_ids = list(range(min(128, len(val_data))))
    panel = DataLoader(Subset(val_data, panel_ids), batch_size=cfg['evaluation']['batch_size'], shuffle=False)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['fingerprint'] != fingerprint: raise ContractError('resume fingerprint mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if scheduler: scheduler.load_state_dict(saved['scheduler'])
        start, records = saved['next_epoch'], saved['records']; restore_rng(saved['rng_states'][rank])
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False)
    for epoch in range(start, 101):
        sampler.set_epoch(epoch); wrapped.train(); started = time.perf_counter(); total = 0.
        torch.cuda.reset_peak_memory_stats()
        for cpu in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in cpu.items()}
            loss = wrapped(batch)
            if not torch.isfinite(loss): raise ContractError('nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ContractError('nonfinite training gradients')
            optimizer.step(); total += float(loss.detach())
        if scheduler: scheduler.step()
        torch.cuda.synchronize(); costs = [None] * world
        dist.all_gather_object(costs, {'rank': rank, 'seconds': time.perf_counter()-started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'loss': total/max(1, len(loader))})
        if rank == 0:
            # Full validation at the fixed final epoch; no final holdout is read.
            records.append({'epoch': epoch, 'rank_costs': costs})
        states = [None] * world; dist.all_gather_object(states, rng_state())
        if rank == 0:
            atomic_save({'fingerprint': fingerprint, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None, 'next_epoch': epoch+1,
                'records': records, 'rng_states': states}, checkpoint)
            write_json(output/'progress.json', {'epoch': epoch, 'epochs': 101, 'cost': records[-1]})
        dist.barrier()
        if args.synthetic and args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            dist.destroy_process_group()
            return
    if rank == 0 and not (output/'result.json').exists():
        started = time.perf_counter()
        metrics = score_model(model, val_loader, device, cfg['model']['num_items'], cfg['evaluation']['candidate_chunk'])
        torch.cuda.synchronize(); evaluation_seconds = time.perf_counter()-started
        # One complete panel warmup followed by three identical catalog passes.
        timings = []
        for repeat in range(4):
            torch.cuda.synchronize(); started = time.perf_counter()
            score_model(model, panel, device, cfg['model']['num_items'], cfg['evaluation']['candidate_chunk'], metrics=False)
            torch.cuda.synchronize()
            if repeat: timings.append(time.perf_counter()-started)
        write_json(output/'result.json', {'kind': 'synthetic_research_validation' if args.synthetic else 'kuairand_research_validation',
            'synthetic': args.synthetic, 'partition': 'validation',
            'selection': 'fixed_final_epoch', 'seed': 42, 'epochs': 101, 'fingerprint': fingerprint,
            'candidate_hash': file_hash(args.candidate), 'metrics': metrics, 'evaluation_seconds': evaluation_seconds,
            'treatment_hash': file_hash(args.treatment), 'config_hash': digest(cfg), 'initialization': initialization,
            'training_gpu_seconds': sum(c['seconds'] for r in records for c in r['rank_costs']),
            'scoring_seconds': statistics.median(timings), 'scoring_repeats': timings,
            'scoring_protocol': {**cfg['evaluation'], 'warmup_passes': 1, 'panel_indices_hash': digest(panel_ids), 'panel_users': len(panel_ids)},
            'parameters': sum(p.numel() for p in model.parameters()),
            'peak_memory_bytes': max(c['peak_allocated_bytes'] for r in records for c in r['rank_costs']),
            'mechanism_status': 'single_seed_provisional'})
    dist.barrier(); dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['full-size', 'check', 'train'])
    p.add_argument('--benchmark', required=True); p.add_argument('--model', required=True)
    p.add_argument('--candidate'); p.add_argument('--config'); p.add_argument('--treatment')
    p.add_argument('--cache'); p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='CPU is only for synthetic admission checks')
    p.add_argument('--synthetic', action='store_true', help='quarantine recovery acceptance from formal research metrics')
    p.add_argument('--stop-after-epoch', type=int)
    a = p.parse_args(); sys.path.insert(0, str(Path(a.benchmark).resolve()))
    {'full-size': full_size, 'check': checks, 'train': train}[a.action](a)


if __name__ == '__main__': main()
