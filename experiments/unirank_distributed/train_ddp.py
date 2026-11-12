"""DDP training at unchanged global batch/order, with globally synchronized Dice BN.

Each rank reads the same deterministic official global batch and forwards only
its contiguous slice. This deliberately avoids changing the released loader's
rank-dependent block order. Validation/test keep the original single-GPU scorer.
"""
import argparse
import copy
from datetime import timedelta
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def sliced(inputs, rank, world):
    n = len(inputs[2])
    lo, hi = n * rank // world, n * (rank + 1) // world
    batch, items, mask, multi = inputs
    return ({k: v[lo:hi] for k, v in batch.items()}, {k: v[lo:hi] for k, v in items.items()},
            mask[lo:hi], [v[lo:hi] for v in multi]), lo, hi


def synthetic(fmap, n=97):
    generator = torch.Generator().manual_seed(8881)
    batch, items = {}, {}
    for name, spec in fmap.features.items():
        if spec['type'] == 'meta':
            continue
        target = items if spec.get('source') in ['item', 'action'] else batch
        shape = (n, 101) if target is items else (n, spec.get('max_len', 4)) if spec['type'] == 'sequence' else (n,)
        if 'vocab_size' in spec:
            target[name] = torch.randint(0, min(spec['vocab_size'], 31), shape, generator=generator)
        else:
            target[name] = torch.rand(shape, generator=generator)
        if spec.get('source') == 'action':
            target[name][:, -1] = 0
    lengths = torch.arange(n).remainder(101)
    valid = (torch.arange(100)[None, :] >= 100 - lengths[:, None]).float()
    multi = [valid * ((torch.arange(100)[None, :] + i) % 3 == 0) for i in range(len(fmap.labels))]
    for index, label in enumerate(fmap.labels):
        batch[label] = (torch.arange(n) + index).remainder(3).eq(0).float()
    return batch, items, valid, multi


def parity(host, raw, reference, ddp, fmap, device, rank, world, output):
    from common import write
    inputs = synthetic(fmap)
    local, lo, hi = sliced(inputs, rank, world)
    n = len(inputs[2])
    checks = []
    for amp in [False, True]:
        raw.zero_grad(set_to_none=True)
        raw.train()
        if rank == 0:
            reference.zero_grad(set_to_none=True)
            reference.train()
        features = host.features_only(local, fmap.labels, device)
        y = torch.stack([local[0][t] for t in fmap.labels], -1).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            pred = host.predict(ddp, features, fmap.labels)
        loss = torch.nn.functional.binary_cross_entropy(pred, y, reduction='none').mean(0).sum()
        loss = loss * (world * (hi - lo) / n)
        loss.backward()
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        global_loss /= world
        padded = pred.detach().new_zeros(((n + world - 1) // world, len(fmap.labels)))
        padded[:hi - lo] = pred.detach()
        gathered = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded)
        valid_check = torch.ones(1, dtype=torch.int64, device=device)
        if rank == 0:
            all_pred = torch.cat([p[:n * (r + 1) // world - n * r // world] for r, p in enumerate(gathered)])
            full_features = host.features_only(inputs, fmap.labels, device)
            target = torch.stack([inputs[0][t] for t in fmap.labels], -1).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                before = host.predict(reference, full_features, fmap.labels)
            before_loss = torch.nn.functional.binary_cross_entropy(before, target, reduction='none').mean(0).sum()
            before_loss.backward()
            differences, norms = [], []
            for (name, p), (_, q) in zip(raw.named_parameters(), reference.named_parameters()):
                if p.grad is None or q.grad is None:
                    assert p.grad is None and q.grad is None, name
                    continue
                differences.append(torch.sum((p.grad.float() - q.grad.float()).square()))
                norms.append(torch.sum(q.grad.float().square()))
            relative = float(torch.sqrt(torch.stack(differences).sum() / torch.stack(norms).sum().clamp_min(1e-20)))
            delta = float((all_pred - before).abs().max())
            loss_delta = float((global_loss - before_loss).abs())
            passed = relative < (.08 if amp else .001) and delta < (.03 if amp else .0005) and loss_delta < (.02 if amp else .0005)
            checks.append({'precision': 'bf16' if amp else 'fp32', 'passed': passed,
                'max_prediction_difference': delta, 'loss_difference': loss_delta,
                'gradient_relative_l2': relative, 'global_batch': n,
                'rank_sizes': [n * (r + 1) // world - n * r // world for r in range(world)]})
            if not passed:
                valid_check.zero_()
        dist.broadcast(valid_check, src=0)
        if not valid_check.item():
            if rank == 0:
                write(output / 'parity.json', {'status': 'failed', 'checks': checks})
            raise ValueError('Global-batch DDP/SyncBN parity check failed')
    if rank == 0:
        write(output / 'parity.json', {'status': 'passed', 'world_size': world, 'checks': checks,
            'semantics': 'unequal local chunks, globally weighted six-task loss, synchronized BN',
            'bitwise_equivalence_claimed': False})


def run(args):
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'source'))
    import train as host
    from common import read, write, sha
    folder = root / 'datasets' / args.dataset
    protocol = read(folder / 'protocol.json')
    rank, world, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    assert protocol['epochs'] == 1 and protocol['batch_size'] == 8192
    assert not read(folder / 'parameters.json').get('distributed', False)
    for name, expected in protocol['source_sha256'].items():
        assert sha(root / 'source' / name) == expected, name
    assert sha(folder / (args.kind + '-config.json')) == protocol['variant_config_sha256'][args.kind]
    assert read(folder / 'acquisition-status.json')['phase'] == 'downloaded_and_verified'
    torch.set_num_threads(4)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    started = time.monotonic()
    random.seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed'])
    torch.cuda.manual_seed_all(protocol['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    if rank == 0:
        write(output / 'progress.json', {'phase': 'initializing_ddp', 'at_unix': time.time(), 'world_size': world})
    raw, fmap, params, effective, source = host.make_model(root, folder, args.kind, output, device=local_rank)
    reference = copy.deepcopy(raw) if args.check_parity and rank == 0 else None
    raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw)
    ddp = DistributedDataParallel(raw, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=True, find_unused_parameters=True, gradient_as_bucket_view=True, bucket_cap_mb=64)
    if args.check_parity:
        parity(host, raw, reference, ddp, fmap, device, rank, world, output)
        dist.barrier()
        dist.destroy_process_group()
        return
    labels = list(fmap.labels)
    assert labels == protocol['tasks']
    embedding_ids = {id(p) for m in raw.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters(recurse=False)}
    sparse = [p for p in raw.parameters() if p.requires_grad and id(p) in embedding_ids]
    dense = [p for p in raw.parameters() if p.requires_grad and id(p) not in embedding_ids]
    optimizers = [torch.optim.AdamW(dense, lr=protocol['dense_learning_rate'], weight_decay=0, foreach=False),
                  torch.optim.Adagrad(sparse, lr=protocol['embedding_learning_rate'], foreach=False)]
    if rank == 0:
        write(output / 'progress.json', {'phase': 'loading_train', 'at_unix': time.time(), 'world_size': world})
    train = host.loader(folder, fmap, params, 'train')
    assert train.global_num_samples == protocol['sample_size']['train']
    train.set_epoch(0)
    raw.train()
    dist.barrier()
    epoch_started = previous = time.monotonic()
    durations, examples, steps, loss_sum = [], 0, 0, 0.0
    for inputs in train:
        global_n = len(inputs[2])
        local, lo, hi = sliced(inputs, rank, world)
        targets = torch.stack([local[0][label] for label in labels], -1).to(device).float()
        features = host.features_only(local, labels, device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values = host.predict(ddp, features, labels)
        loss = torch.nn.functional.binary_cross_entropy(values, targets, reduction='none').mean(0).sum()
        loss = loss * (world * (hi - lo) / global_n)
        assert torch.isfinite(loss)
        assert not hasattr(raw, 'training_loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense, protocol['gradient_clip'])
        for optimizer in optimizers:
            optimizer.step()
        torch.cuda.synchronize()
        steps += 1
        examples += global_n
        loss_sum += float(loss.detach()) * global_n
        now = time.monotonic()
        durations.append(now - previous)
        previous = now
        if steps % 10 == 0:
            stats = torch.tensor([float(np.mean(durations[3:] or durations)), float(loss.detach())], device=device)
            dist.all_reduce(stats[:1], op=dist.ReduceOp.MAX)
            dist.all_reduce(stats[1:])
            stats[1] /= world
            if rank == 0:
                avg, logged_loss = stats.tolist()
                write(output / 'progress.json', {'phase': 'calibration' if args.calibration_steps else 'train',
                    'epoch': 1, 'steps': steps, 'total_steps': len(train), 'examples': examples,
                    'loss': logged_loss, 'mean_step_seconds': avg, 'eta_training_seconds': (len(train) - steps) * avg,
                    'wall_seconds': time.monotonic() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                    'world_size': world, 'global_batch_size': 8192, 'at_unix': time.time()})
        if args.calibration_steps and steps >= args.calibration_steps:
            break
    dist.barrier()
    training_seconds = time.monotonic() - epoch_started
    peak = torch.tensor([torch.cuda.max_memory_allocated()], dtype=torch.int64, device=device)
    peaks = [torch.empty_like(peak) for _ in range(world)]
    dist.all_gather(peaks, peak)
    summary = torch.tensor([training_seconds, float(np.mean(durations[3:] or durations)), loss_sum], device=device, dtype=torch.float64)
    dist.all_reduce(summary[:2], op=dist.ReduceOp.MAX)
    dist.all_reduce(summary[2:])
    summary[2] /= world
    training_seconds, avg, global_loss_sum = summary.tolist()
    counts = {'parameters': sum(p.numel() for p in raw.parameters()),
              'embedding_parameters': sum(p.numel() for p in sparse), 'dense_parameters': sum(p.numel() for p in dense)}
    gpu_peaks = [int(p.item()) for p in peaks]
    if rank == 0 and args.calibration_steps:
        write(output / 'result.json', {'status': 'success', 'calibration_only': True, 'steps': steps,
            'training_examples': examples, 'train_seconds': training_seconds, 'wall_seconds': time.monotonic() - started,
            'world_size': world, 'global_batch_size': 8192, 'steady_step_seconds': avg,
            'projected_train_seconds': len(train) * avg, 'peak_gpu_bytes': max(gpu_peaks),
            'peak_gpu_bytes_per_rank': gpu_peaks, 'training_loss_hook_calls': 0, **counts,
            'training_gpu_seconds': training_seconds * world, 'executor_sha256': sha(Path(__file__))})
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    del optimizers, ddp, train
    dist.barrier()
    dist.destroy_process_group()
    if args.calibration_steps or rank != 0:
        return
    # The extra ranks release their GPUs. Reuse the original full-batch scorer
    # on rank0; no changed metric implementation or test-data partition.
    torch.cuda.empty_cache()
    valid = host.loader(folder, fmap, params, 'valid')
    validation = host.score(raw, valid, labels, device, output, 'valid', protocol['sample_size']['valid'])
    torch.save(raw.state_dict(), output / 'checkpoint.pt')
    checkpoint_sha = sha(output / 'checkpoint.pt')
    write(output / 'selection.json', {'selected_unix': time.time(), 'criterion': protocol['selection'],
        'validation': validation, 'checkpoint_sha256': checkpoint_sha, 'protocol_hash': protocol['identity'],
        'source_sha256': sha(source), 'test_used_for_selection': False, 'world_size': world,
        'global_batch_size': 8192, 'synchronized_batch_norm': True})
    del valid
    torch.cuda.empty_cache()
    raw.load_state_dict(torch.load(output / 'checkpoint.pt', map_location=device, weights_only=True))
    test = host.loader(folder, fmap, params, 'test')
    heldout = host.score(raw, test, labels, device, output, 'test', protocol['sample_size']['test'])
    result = {'status': 'success', 'dataset': args.dataset, 'kind': args.kind, 'seed': protocol['seed'],
        'source_sha256': sha(source), 'executor_sha256': sha(Path(__file__)), 'effective_config': effective,
        'protocol_hash': protocol['identity'], 'dataset_revision': protocol['revision'],
        'validation': validation, 'test': heldout, 'optimizer_steps': steps, 'training_examples': examples,
        'train_seconds': training_seconds, 'wall_seconds': time.monotonic() - started,
        'epochs': [{'epoch': 1, 'train_seconds': training_seconds, 'training_loss': global_loss_sum / examples,
                    'validation': validation}], 'training_loss_hook_calls': 0, **counts,
        'peak_gpu_bytes': max(gpu_peaks), 'peak_gpu_bytes_per_rank': gpu_peaks,
        'world_size': world, 'global_batch_size': 8192, 'training_gpu_seconds': training_seconds * world,
        'synchronized_batch_norm': True, 'evaluation_world_size': 1,
        'hardware': torch.cuda.get_device_name(), 'initialization': 'from_scratch',
        'checkpoint_sha256': checkpoint_sha, 'selection_policy': protocol['selection'],
        'test_used_for_selection': False, 'one_epoch_fixed_recipe': True, 'single_seed': True}
    write(output / 'result.json', result)
    write(output / 'progress.json', {'phase': 'complete', 'at_unix': time.time(), 'test_macro_auc': heldout['auc'],
          'wall_seconds': result['wall_seconds'], 'world_size': world, 'evaluation_world_size': 1})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', default='KuaiRand')
    parser.add_argument('--kind', choices=['joint1', 'joint3'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-steps', type=int, default=0)
    parser.add_argument('--check-parity', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / ('failure-rank' + os.environ.get('RANK', '0') + '.txt')).write_text(repr(error) + '\n')
        raise
