"""Host-only performance probes; never edit or resume a formal run differently."""
import argparse
import copy
from dataclasses import replace
import os
from pathlib import Path
import statistics
import sys
import time

from .core import ContractError, canonical, digest, read_json, write_json


def run(args):
    sys.path.insert(0, str(Path(args.benchmark).resolve()))
    import torch
    from torch.utils.data import DataLoader, Subset
    from longseq_bench.data import AlignedDataset
    from longseq_bench.evaluation import evaluate
    from longseq_bench.models import BenchmarkModel, ModelConfig
    from longseq_bench.train import seed_everything
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '4':
        raise ContractError('isolated probe must use GPU 4 only')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    document = read_json(Path(args.benchmark)/'configs/benchmark.json')
    report = {'kind': 'host_only_performance_probe', 'formal_result': False,
        'formal_training_modified': False, 'research_context_eligible': False, 'results': {}}
    torch.cuda.set_device(0)
    for name in ('hstu', 'ultra_hstu'):
        cfg = ModelConfig(name=name, **document['model'])
        tc = document['training']
        seed_everything(42)
        model = BenchmarkModel(cfg, tc['negatives'], tc['temperature'], tc['loss_chunk']).cuda()
        reference = Path(args.baseline)/name
        manifest = read_json(reference/'run.json')
        saved = torch.load(reference/'last.pt', map_location='cpu', weights_only=False)
        if saved['fingerprint'] != manifest['fingerprint']: raise ContractError('checkpoint identity changed')
        model.load_state_dict(saved['model']); del saved
        panel = Subset(AlignedDataset(args.cache, 'eval'), list(range(64)))
        loader = DataLoader(panel, batch_size=document['evaluation']['batch_size'], shuffle=False)
        evaluations = []; reference_metrics = None
        for chunk in (256, 2048, 8192):
            # Warmup covers the entire scoring panel, then use identical repeats.
            evaluate(model, loader, 'cuda', cfg.num_items, chunk)
            measurements = []; metrics = None
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                torch.cuda.synchronize(); start = time.perf_counter()
                metrics = evaluate(model, loader, 'cuda', cfg.num_items, chunk)
                torch.cuda.synchronize(); measurements.append(time.perf_counter()-start)
            if reference_metrics is None: reference_metrics = metrics
            evaluations.append({'candidate_chunk': chunk, 'seconds_median': statistics.median(measurements),
                'seconds': measurements, 'same_panel_aggregate_metrics': metrics == reference_metrics,
                'metrics_hash': digest(metrics), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        # Compare computation-only training cost, with the same local batch and random stream.
        del model; torch.cuda.empty_cache()
        seed_everything(42)
        ids = torch.randint(1, cfg.num_items+1, (64, 1025), device='cuda')
        batch = {'ids': ids[:, :-1], 'labels': ids[:, 1:],
            'times': torch.arange(1025, device='cuda').expand(64, -1)*1000}
        training = []
        for checkpoint in (True, False):
            seed_everything(42)
            candidate = BenchmarkModel(replace(cfg, checkpoint_blocks=checkpoint),
                tc['negatives'], tc['temperature'], tc['loss_chunk']).cuda().train()
            timings = []; losses = []; gradient = None
            for repeat in range(4):
                seed_everything(2026)
                candidate.zero_grad(set_to_none=True); torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
                loss = candidate(batch); loss.backward(); torch.cuda.synchronize()
                if repeat:
                    timings.append(time.perf_counter()-start); losses.append(float(loss.detach()))
                if repeat == 3:
                    gradient = digest([float(p.grad.float().norm()) for p in candidate.parameters() if p.grad is not None])
            training.append({'checkpoint_blocks': checkpoint, 'forward_backward_seconds_median': statistics.median(timings),
                'seconds': timings, 'losses': losses, 'gradient_norms_hash': gradient,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'includes_ddp_optimizer_or_data_loading': False})
            del candidate; torch.cuda.empty_cache()
        del batch, ids; torch.cuda.empty_cache()
        report['results'][name] = {'evaluation_panel_users': 64, 'checkpoint_fingerprint': manifest['fingerprint'],
            'evaluation': evaluations, 'synthetic_full_size_forward_backward': training}
        write_json(output/'report.json', report)
        print(canonical({'model': name, **report['results'][name]}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', default='artifacts/kuairand-a37797a/benchmarks/kuairand-longseq')
    p.add_argument('--baseline', default='artifacts/kuairand-research/baseline')
    p.add_argument('--cache', default='artifacts/kuairand-research/data/cache-l1024')
    p.add_argument('--output', required=True)
    a = p.parse_args(); run(a)


if __name__ == '__main__': main()
