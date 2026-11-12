"""Randomized crossover confirmation of frozen KV-read wall latency.

All variants are resident for this timing-only confirmation. Use the primary
performance run, which isolates variants, for memory measurements.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from rec_harness.attention_width import paired_interval
from rec_harness.core import ContractError, canonical, file_hash, read_json, write_json
from rec_harness.kuairand_attention_performance import native_memory, compress_memory, read_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance-run', default='artifacts/kuairand-research/attention-width/hstu-performance-20261103-01')
    parser.add_argument('--output', required=True)
    parser.add_argument('--users', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    performance = Path(args.performance_run).resolve()
    if read_json(performance / 'status.json')['status'] != 'completed':
        raise ContractError('finish primary performance run before crossover')
    pc = read_json(performance / 'contract.json')
    implementation = ROOT / 'rec_harness/kuairand_attention_performance.py'
    if file_hash(implementation) != pc['implementation_sha256']:
        raise ContractError('primary benchmark implementation differs')
    source = Path(pc['source_run'])
    c = read_json(source / 'contract.json')
    old_args = c['args']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str((ROOT / old_args['benchmark']).resolve()))
    from longseq_bench.models import BenchmarkModel, ModelConfig
    from longseq_bench.data import AlignedDataset
    baseline = ROOT / old_args['baseline']
    manifest = read_json(baseline / 'run.json')
    checkpoint = baseline / 'last.pt'
    if file_hash(checkpoint) != c['checkpoint_sha256']:
        raise ContractError('source checkpoint differs')
    cfg, tc = ModelConfig(**manifest['model']), manifest['training']
    model = BenchmarkModel(cfg, tc['negatives'], tc['temperature'], tc['loss_chunk']).eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model']); del saved
    model = model.cuda().eval()
    bank = torch.load(source / 'query-bank.pt', weights_only=True)['calibration'].cuda()
    dataset = AlignedDataset(ROOT / old_args['cache'], 'train')
    panel = c['panel_users'][:args.users]
    budgets = [1024, 128, 256, 512]
    write_json(output / 'contract.json', dict(args=vars(args), users=panel, budgets=budgets,
        gpu=torch.cuda.get_device_name(0), visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        primary_implementation_sha256=file_hash(implementation), source_sha256=file_hash(__file__),
        checkpoint_sha256=c['checkpoint_sha256'], randomized_order=True,
        memory_measurement=False, fixed_query_anchor=True, precision='fp32'))
    rows = []
    started = time.time()
    with torch.inference_mode():
        for ordinal, user in enumerate(panel):
            item = dataset[user]
            raw = native_memory(model, item['ids'][None].cuda(), item['times'][None].cuda())
            memories = {1024: raw}
            for budget in budgets[1:]:
                memories[budget] = compress_memory(raw, bank, budget, old_args['seed'] + user * 100)
            for budget in budgets:
                for _ in range(5): read_memory(model, memories[budget])
            torch.cuda.synchronize()
            generator = random.Random(42 + user)
            for repeat in range(args.rounds):
                order = budgets.copy(); generator.shuffle(order)
                for budget in order:
                    start = time.perf_counter()
                    result = read_memory(model, memories[budget])
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    del result
                    rows.append(dict(user_index=user, round=repeat, budget=budget, milliseconds=elapsed))
            del raw, memories
            write_json(output / 'status.json', dict(status='running', completed_users=ordinal + 1,
                                                    planned_users=len(panel), wall_seconds=time.time()-started))
            print(canonical(dict(completed_users=ordinal + 1, planned_users=len(panel))), flush=True)
    summary = []
    for budget in budgets:
        values = torch.tensor([r['milliseconds'] for r in rows if r['budget'] == budget], dtype=torch.float64)
        per_user_difference = []
        for user in panel:
            base = [r['milliseconds'] for r in rows if r['user_index'] == user and r['budget'] == 1024]
            changed = [r['milliseconds'] for r in rows if r['user_index'] == user and r['budget'] == budget]
            per_user_difference.append(sum(changed)/len(changed) - sum(base)/len(base))
        summary.append(dict(budget=budget, samples=len(values), p50_ms=float(values.quantile(.5)),
            p95_ms=float(values.quantile(.95)), mean_ms=float(values.mean()),
            mean_time_change_ci95=paired_interval(per_user_difference)))
    with (output / 'timings.jsonl').open('w') as handle:
        for row in rows: handle.write(canonical(row)+'\n')
    with (output / 'summary.csv').open('w', newline='') as handle:
        flat = [dict(budget=r['budget'], p50_ms=r['p50_ms'], p95_ms=r['p95_ms'], mean_ms=r['mean_ms'],
                     mean_change_ms=r['mean_time_change_ci95']['mean'],
                     change_ci_low=r['mean_time_change_ci95']['low'],
                     change_ci_high=r['mean_time_change_ci95']['high']) for r in summary]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write_json(output / 'report.json', dict(kind='randomized_cached_read_crossover', summary=summary,
        users=len(panel), rounds_per_user=args.rounds,
        qualification='Fixed context/query anchor, portable eager PyTorch, timing-only; variants coexist in this confirmation.'))
    write_json(output / 'status.json', dict(status='completed', users=len(panel), wall_seconds=time.time()-started))
    print(canonical(summary), flush=True)


if __name__ == '__main__': main()
