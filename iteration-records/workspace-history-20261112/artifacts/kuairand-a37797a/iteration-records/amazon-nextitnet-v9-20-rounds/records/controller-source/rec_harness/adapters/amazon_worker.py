"""Frozen NextItNet training/evaluation worker; consumes train+validation only.

Run as a standalone script in the torch environment. Model source defines only
architecture/hyperparameters; this worker owns sampling, training and evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    p = Path(path)
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    temporary.replace(p)


def read_split(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        user, item = map(int, line.split())
        result.setdefault(user, []).append(item)
    return result


def evaluation_rows(train, valid, itemnum, count, seed):
    rng = random.Random(seed)
    users = sorted(u for u in valid if train.get(u) and valid[u])
    rng.shuffle(users)
    rows = []
    for user in users[:count]:
        positive = valid[user][0]
        seen = set(train[user]) | {positive, 0}
        if itemnum - len(seen - {0}) < 99:
            raise ValueError('insufficient eligible negatives')
        candidates = [positive]
        while len(candidates) < 100:
            item = rng.randint(1, itemnum)
            if item not in seen:
                seen.add(item)
                candidates.append(item)
        rows.append({'user_id': user, 'history': train[user], 'candidates': candidates})
    if not rows:
        raise ValueError('no evaluation users')
    return rows


def evaluate(model, rows, maxlen, device, batch_size=256):
    model.eval()
    ranks = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            seq = np.zeros((len(batch), maxlen), dtype=np.int64)
            for i, row in enumerate(batch):
                history = row['history'][-maxlen:]
                seq[i, -len(history):] = history
            candidates = torch.tensor([r['candidates'] for r in batch], device=device)
            encoded = model.encode_sequence(torch.as_tensor(seq, device=device))[:, -1]
            scores = (model.item_emb(candidates) * encoded[:, None, :]).sum(-1)
            if not torch.isfinite(scores).all():
                raise ValueError('nonfinite public scores')
            # Positive appears at index 0; stable descending ordering matches project ties.
            ranks.extend((scores[:, 1:] > scores[:, :1]).sum(-1).cpu().tolist())
    metrics = {}
    for k in (5, 10, 20):
        metrics[f'HR@{k}'] = sum(r < k for r in ranks) / len(ranks)
        metrics[f'NDCG@{k}'] = sum(1 / math.log2(r + 2) if r < k else 0 for r in ranks) / len(ranks)
    return metrics, ranks


def sample_batch(train, users, itemnum, batch_size, maxlen, rng):
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for i in range(batch_size):
        history = train[rng.choice(users)]
        seen = set(history)
        if len(seen) >= itemnum:
            raise ValueError('user has no training negative')
        relevant = history[-(maxlen + 1):]
        length = len(relevant) - 1
        seq[i, -length:] = relevant[:-1]
        pos[i, -length:] = relevant[1:]
        for j in range(maxlen - 1, maxlen - length - 1, -1):
            item = rng.randint(1, itemnum)
            while item in seen:
                item = rng.randint(1, itemnum)
            neg[i, j] = item
    return seq, pos, neg


def correctness(model, maxlen, itemnum, device):
    model.eval()
    seq = torch.zeros((2, maxlen), dtype=torch.long, device=device)
    seq[:, -3:] = torch.tensor([[1, 2, 3], [2, 3, 4]], device=device)
    with torch.no_grad():
        encoded = model.encode_sequence(seq)
        changed = seq.clone(); changed[:, -1] = min(itemnum, 5)
        alternate = model.encode_sequence(changed)
        if not torch.allclose(encoded[:, :-1], alternate[:, :-1], atol=1e-5, rtol=1e-5):
            raise ValueError('future information leaked into causal representations')
        candidates = torch.tensor([1, 2, 3], device=device)
        predicted = model.score_candidates(seq[:1], candidates)
        reference = model.item_emb(candidates) @ encoded[0, -1]
        if not torch.allclose(predicted, reference, atol=1e-5, rtol=1e-5):
            raise ValueError('right-aligned readout contract violated')
    return {'causal_prefix': True, 'tail_readout': True, 'parameters': sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batches', type=int, default=200)
    parser.add_argument('--eval-users', type=int, default=1000)
    parser.add_argument('--eval-seed', type=int, default=20260923)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise RuntimeError('immutable result already exists')
    torch.set_num_threads(args.threads)
    os.environ['GAGC_SEED'] = str(args.seed)
    spec = importlib.util.spec_from_file_location('candidate_model', args.source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # Match project baseline; record statistical, not bitwise reproducibility.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False  # Preserve FP32 convolution ranking for independent CPU rescore.
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for Amazon comparison')
    device = torch.device('cuda:0')
    train_path = Path(args.data) / f'{args.dataset}_train.txt'
    valid_path = Path(args.data) / f'{args.dataset}_valid.txt'
    train, valid = read_split(train_path), read_split(valid_path)
    itemnum = max(max(items) for data in (train, valid) for items in data.values())
    users = sorted(u for u in train if len(train[u]) >= 2)
    rows = evaluation_rows(train, valid, itemnum, args.eval_users, args.eval_seed)
    dump(output / 'validation_candidates.json', [{k: v for k, v in r.items() if k != 'history'} for r in rows])
    hyperparameters = {k: getattr(module, k) for k in ('MAXLEN', 'HIDDEN_UNITS', 'NUM_BLOCKS', 'KERNEL_SIZE', 'DROPOUT_RATE', 'LR', 'WEIGHT_DECAY', 'BATCH_SIZE')}
    metadata = {'dataset': args.dataset, 'seed': args.seed, 'source_hash': sha(args.source), 'worker_hash': sha(__file__),
                'data_hashes': {'train': sha(train_path), 'validation': sha(valid_path)}, 'candidates_hash': sha(output / 'validation_candidates.json'),
                'protocol': vars(args), 'hyperparameters': hyperparameters, 'torch': torch.__version__,
                'device': torch.cuda.get_device_name(), 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_deterministic': False, 'itemnum': itemnum, 'n_users': len(train), 'initialization': 'fresh'}
    dump(output / 'manifest.json', metadata)
    model = module.NextItNetModel(itemnum, hidden_units=module.HIDDEN_UNITS, num_blocks=module.NUM_BLOCKS,
                                  kernel_size=module.KERNEL_SIZE, dropout_rate=module.DROPOUT_RATE, maxlen=module.MAXLEN).to(device)
    checks = correctness(model, module.MAXLEN, itemnum, device)
    dump(output / 'correctness.json', checks)
    optimizer = torch.optim.Adam(model.parameters(), lr=module.LR, weight_decay=module.WEIGHT_DECAY)
    rng = random.Random(args.seed)
    best, best_epoch, best_metrics, best_ranks = -1, 0, None, None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for _ in range(args.batches):
            arrays = sample_batch(train, users, itemnum, module.BATCH_SIZE, module.MAXLEN, rng)
            seq, pos, neg = [torch.as_tensor(a, device=device) for a in arrays]
            positive, negative = model(seq, pos, neg)
            loss = F.softplus(negative - positive)[pos != 0].mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item())
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, ranks = evaluate(model, rows, module.MAXLEN, device)
            record = {'epoch': epoch, 'loss': sum(losses) / len(losses), 'metrics': metrics, 'elapsed_seconds': time.monotonic() - started}
            with (output / 'curve.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            if metrics['HR@10'] > best:
                best, best_epoch, best_metrics, best_ranks = metrics['HR@10'], epoch, metrics, ranks
                torch.save(model.state_dict(), output / 'checkpoint.pt')
            if epoch - best_epoch >= args.patience: break
    torch.cuda.synchronize()
    dump(output / 'validation_ranks.json', {'user_ids': [r['user_id'] for r in rows], 'ranks': best_ranks})
    result = {'status': 'success', 'metrics': best_metrics, 'best_epoch': best_epoch, 'completed_epoch': epoch,
              'elapsed_seconds': time.monotonic() - started, 'allocated_gpu_seconds': time.monotonic() - started,
              'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'checkpoint_sha256': sha(output / 'checkpoint.pt'),
              'source_hash': metadata['source_hash'], 'worker_hash': metadata['worker_hash'], 'candidates_hash': metadata['candidates_hash']}
    dump(output / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
