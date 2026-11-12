"""Compare isolated and co-located DDP work on spare GPUs; no formal result edits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .core import ContractError, canonical, read_json, write_json
from .kuairand_baseline import command, environment


def worker(args):
    sys.path.insert(0, str(Path(args.benchmark).resolve()))
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from longseq_bench.models import ModelConfig, BenchmarkModel
    from longseq_bench.train import seed_everything
    local = int(os.environ['LOCAL_RANK']); rank = int(os.environ['RANK'])
    torch.cuda.set_device(local); dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    seed_everything(42); cfg = read_json(Path(args.benchmark)/'configs/benchmark.json')
    mc = ModelConfig(name=args.model, **cfg['model']); tc = cfg['training']
    model = BenchmarkModel(mc, tc['negatives'], tc['temperature'], tc['loss_chunk']).cuda()
    wrapped = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    ids = torch.randint(1, mc.num_items+1, (64, 1025), device='cuda')
    batch = {'ids': ids[:, :-1], 'labels': ids[:, 1:],
        'times': torch.arange(1025, device='cuda').expand(64, -1)*1000}
    wrapped(batch).backward(); torch.cuda.synchronize(); model.zero_grad(set_to_none=True)
    ready = Path(args.output); ready.mkdir(parents=True, exist_ok=True)
    write_json(ready/('%s-ready-%d.json' % (args.model, rank)), {'pid': os.getpid(), 'rank': rank})
    gate = ready/'go.json'; deadline = time.monotonic()+180
    while not gate.exists():
        if time.monotonic() > deadline: raise ContractError('collocation start barrier timed out')
        time.sleep(.02)
    start_at = read_json(gate)['start_at']
    while time.time() < start_at: time.sleep(.002)
    durations = []; losses = []; torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True); seed_everything(2026)
        torch.cuda.synchronize(); start = time.perf_counter()
        loss = wrapped(batch); loss.backward(); torch.cuda.synchronize()
        durations.append(time.perf_counter()-start); losses.append(float(loss.detach()))
    ended = time.time()
    write_json(ready/('%s-rank-%d.json' % (args.model, rank)), {'model': args.model, 'rank': rank,
        'step_seconds': durations, 'median_seconds': statistics.median(durations), 'losses': losses,
        'started_at': started, 'ended_at': ended, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    dist.destroy_process_group()


def run(args):
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    workspace = Path(__file__).resolve().parents[1]; bench = Path(args.benchmark).resolve()
    python = str(Path(args.python).absolute()); env = environment(bench, workspace, [6, 7])
    results = {}
    for mode, names in [('hstu-alone', ['hstu']), ('dare-alone', ['dare']), ('shared-pair', ['hstu', 'dare'])]:
        target = root/mode; target.mkdir()
        def one(name):
            return command([python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '-m',
                'rec_harness.kuairand_collocation_probe', 'worker', '--benchmark', str(bench), '--model', name,
                '--output', str(target), '--steps', str(args.steps)], target/name/'job', cwd=workspace, env=env, gpus=[6, 7])
        with ThreadPoolExecutor(len(names)) as pool:
            futures = [pool.submit(one, name) for name in names]
            deadline = time.monotonic()+240
            while len(list(target.glob('*-ready-*.json'))) != 2*len(names):
                for f in futures:
                    if f.done(): f.result()
                if time.monotonic() > deadline: raise ContractError('probe initialization timed out')
                time.sleep(.2)
            write_json(target/'go.json', {'start_at': time.time()+1})
            for f in futures: f.result()
        rows = [read_json(p) for p in target.glob('*-rank-*.json')]
        results[mode] = {'rank_timings': rows,
            'work_makespan_seconds': max(r['ended_at'] for r in rows)-min(r['started_at'] for r in rows)}
        write_json(root/'progress.json', results)
    serial = results['hstu-alone']['work_makespan_seconds'] + results['dare-alone']['work_makespan_seconds']
    shared = results['shared-pair']['work_makespan_seconds']
    report = {'kind': 'synthetic_ddp_collocation_probe', 'formal_result': False, 'gpus': [6, 7],
        'same_global_batch_128': True, 'results': results, 'serial_work_seconds': serial,
        'shared_work_seconds': shared, 'aggregate_throughput_speedup': serial/shared,
        'qualification': 'Full-size synthetic forward/backward only. Epoch evaluation overlap and steady-state real-data costs require separate observation.'}
    write_json(root/'report.json', report); print(canonical(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'worker']); p.add_argument('--output', required=True)
    p.add_argument('--benchmark', default='artifacts/kuairand-a37797a/benchmarks/kuairand-longseq')
    p.add_argument('--python', default='artifacts/RecHarness-6lyc/.venv/bin/python')
    p.add_argument('--model'); p.add_argument('--steps', type=int, default=3)
    a = p.parse_args(); (run if a.action == 'run' else worker)(a)


if __name__ == '__main__': main()
