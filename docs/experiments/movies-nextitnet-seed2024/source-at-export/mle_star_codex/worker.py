"""Host-owned training/scoring; generated code supplies only a NextItNet model.

Sampling, BPR, optimizer, training length, candidate lists and ranking stay fixed.
Candidate generation never receives test observations. Evaluation rows reproduce
6lyc/RecHarness's raw-data fallback (seed 42, 99 unique negatives).
"""
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

CHECK_VERSION = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def read(path):
    result = {}
    with Path(path).open() as f:
        for line in f:
            u, i = map(int, line.split())
            result.setdefault(u, []).append(i)
    return result


def rows_for(train, labels, itemnum, count, valid=None):
    rng = random.Random(42)
    users = [u for u in labels if u in train and labels[u]]
    if len(users) > count:
        users = rng.sample(users, count)
    all_items = set(range(1, itemnum + 1))
    rows = []
    for u in users:
        history = train[u] + (valid.get(u, []) if valid is not None else [])
        target = labels[u][0]
        seen = set(history) | set(labels[u]) | {target, 0}
        pool = list(all_items - seen - {target})
        if len(pool) < 99:
            raise ValueError('fewer than 99 eligible negatives')
        rows.append({'user': u, 'history': history, 'candidates': [target] + rng.sample(pool, 99)})
    if not rows:
        raise ValueError('no eligible evaluation users')
    return rows


def metrics(scores):
    if scores.ndim != 2 or scores.shape[1] != 100 or not np.isfinite(scores).all():
        raise ValueError('invalid prediction matrix')
    ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
    return {f'{name}@{k}': float(np.mean(ranks < k) if name == 'HR' else
            np.mean(np.where(ranks < k, 1 / np.log2(ranks + 2), 0)))
            for k in (5, 10, 20) for name in ('HR', 'NDCG')}


def predict(model, rows, maxlen, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            seq = np.zeros((len(batch), maxlen), dtype=np.int64)
            for i, row in enumerate(batch):
                h = row['history'][-maxlen:]
                if h:
                    seq[i, -len(h):] = h
            seq_t = torch.as_tensor(seq, device=device)
            # Preserve public scoring, including the original baseline readout.
            outputs.extend(model.score_candidates(seq_t[i:i + 1],
                torch.tensor(row['candidates'], device=device)).cpu().numpy()[None]
                for i, row in enumerate(batch))
    scores = np.concatenate(outputs)
    metrics(scores)
    return scores


def batch(train, users, itemnum, size, maxlen, rng):
    seq = np.zeros((size, maxlen), dtype=np.int64)
    pos, neg = np.zeros_like(seq), np.zeros_like(seq)
    for i in range(size):
        h = train[rng.choice(users)]
        seen = set(h)
        if len(seen) >= itemnum:
            raise ValueError('no training negatives')
        h = h[-(maxlen + 1):]
        n = len(h) - 1
        seq[i, -n:], pos[i, -n:] = h[:-1], h[1:]
        for j in range(maxlen - n, maxlen):
            x = rng.randint(1, itemnum)
            while x in seen:
                x = rng.randint(1, itemnum)
            neg[i, j] = x
    return seq, pos, neg


def load(source, itemnum, device):
    spec = importlib.util.spec_from_file_location('nextitnet_candidate', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    model = m.NextItNetModel(itemnum, hidden_units=m.HIDDEN_UNITS, num_blocks=m.NUM_BLOCKS,
        kernel_size=m.KERNEL_SIZE, dropout_rate=m.DROPOUT_RATE, maxlen=m.MAXLEN).to(device)
    return m, model


def check(model, maxlen, device):
    """Check public contracts without prescribing embedding/scoring internals.

    Shared-table dot products and last-position readout are diagnostic references,
    not validity requirements. BPR accepts any finite, role-consistent scores.
    """
    model.eval()
    seq = torch.zeros((2, maxlen), dtype=torch.long, device=device)
    seq[:, -3:] = torch.tensor([[1, 2, 3], [2, 3, 4]], device=device)
    with torch.no_grad():
        enc = model.encode_sequence(seq)
        altered = seq.clone()
        altered[:, -1] = 5
        other = model.encode_sequence(altered)
        assert enc.shape[:2] == seq.shape and torch.isfinite(enc).all()
        assert torch.allclose(enc[:, :-1], other[:, :-1], atol=1e-5, rtol=1e-5), 'causality'
        c = torch.tensor([1, 2, 3], device=device)
        public_scores = model.score_candidates(seq[:1], c)
        assert public_scores.shape == c.shape and torch.isfinite(public_scores).all(), 'public scorer'
        perm = torch.tensor([2, 0, 1], device=device)
        assert torch.allclose(model.score_candidates(seq[:1], c[perm]),
                              public_scores[perm], atol=1e-5, rtol=1e-5), 'candidate order dependence'
        assert torch.allclose(model.score_candidates(seq[:1], c[:2]),
                              public_scores[:2], atol=1e-5, rtol=1e-5), 'candidate set dependence'
        pos, neg = seq.roll(1, dims=1), seq.roll(2, dims=1)
        p, n = model(seq, pos, neg)
        assert p.shape == pos.shape and torch.isfinite(p).all(), 'positive training score shape/finiteness'
        assert n.shape == neg.shape and torch.isfinite(n).all(), 'negative training score shape/finiteness'
        altered_p, altered_n = model(altered, pos, neg)
        assert torch.allclose(altered_p[:, :-1], p[:, :-1], atol=1e-5, rtol=1e-5), 'training causality'
        assert torch.allclose(altered_n[:, :-1], n[:, :-1], atol=1e-5, rtol=1e-5), 'training causality'
        swapped_p, swapped_n = model(seq, neg, pos)
        assert torch.allclose(swapped_p, n, atol=1e-5, rtol=1e-5), 'training target role dependence'
        assert torch.allclose(swapped_n, p, atol=1e-5, rtol=1e-5), 'training target role dependence'
        replaced_pos = pos.clone()
        replaced_pos[:, -1] = 5
        replaced_p, replaced_n = model(seq, replaced_pos, neg)
        assert torch.allclose(replaced_p[:, :-1], p[:, :-1], atol=1e-5, rtol=1e-5), 'training target-set dependence'
        assert torch.allclose(replaced_n, n, atol=1e-5, rtol=1e-5), 'training target-set dependence'
        single_p, single_n = model(seq[:1], pos[:1], neg[:1])
        assert torch.allclose(single_p, p[:1], atol=1e-5, rtol=1e-5), 'training batch dependence'
        assert torch.allclose(single_n, n[:1], atol=1e-5, rtol=1e-5), 'training batch dependence'
        assert torch.allclose(model.encode_sequence(seq[:1]), enc[:1], atol=1e-5, rtol=1e-5), 'batch independence'
        # Optional architecture audits preserve the earlier readout investigation.
        # Models may legitimately use a different table, projection or pooling.
        shared, tail = None, None
        try:
            reference_p = (enc * model.item_emb(pos)).sum(-1)
            reference_n = (enc * model.item_emb(neg)).sum(-1)
            shared = bool(torch.allclose(p, reference_p, atol=1e-5, rtol=1e-5)
                          and torch.allclose(n, reference_n, atol=1e-5, rtol=1e-5))
            tail = bool(torch.allclose(public_scores,
                model.item_emb(c) @ enc[0, -1], atol=1e-5, rtol=1e-5))
        except (AttributeError, RuntimeError, TypeError):
            pass
        return {'check_version': CHECK_VERSION, 'causal': True,
                'training_scores_valid': True, 'training_causal': True,
                'training_target_role_independent': True, 'training_target_set_independent': True,
                'shared_embedding_training_scores_match': shared, 'tail_readout_matches': tail}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batches', type=int, default=200)
    p.add_argument('--eval-users', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--mode', choices=['train', 'test'], default='train')
    p.add_argument('--checkpoint', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'result.json').exists():
        raise RuntimeError('refusing to overwrite result')
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')
    train_path = a.data / 'trainval/Movies_and_TV_train.txt'
    valid_path = a.data / 'trainval/Movies_and_TV_valid.txt'
    train, valid = read(train_path), read(valid_path)
    stats = json.loads((a.data / 'trainval/dataset_stats.json').read_text())
    itemnum = stats['Movies_and_TV']['itemnum']
    if a.mode == 'test':
        labels = read(a.data / 'test/Movies_and_TV_test.txt')
        rows = rows_for(train, labels, itemnum, 10000, valid)
    else:
        rows = rows_for(train, valid, itemnum, a.eval_users)
    row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    module, model = load(a.source, itemnum, device)
    save(a.output / 'manifest.json', {'source_sha256': sha(a.source), 'worker_sha256': sha(__file__),
        'check_version': CHECK_VERSION,
        'train_sha256': sha(train_path), 'valid_sha256': sha(valid_path), 'rows_sha256': row_hash,
        'seed': a.seed, 'eval_seed': 42, 'itemnum': itemnum, 'n_users': len(rows),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'protocol': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}})
    if a.mode == 'test':
        model.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True))
        save(a.output / 'correctness.json', check(model, module.MAXLEN, device))
        scores = predict(model, rows, module.MAXLEN, device)
        np.save(a.output / 'scores.npy', scores)
        save(a.output / 'result.json', {'status': 'success', 'metrics': metrics(scores), 'rows_sha256': row_hash})
        return
    save(a.output / 'correctness.json', check(model, module.MAXLEN, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=module.LR, weight_decay=module.WEIGHT_DECAY)
    rng = random.Random(a.seed)
    users = [u for u in train if len(train[u]) >= 2]
    best, best_epoch = -1, 0
    start = time.monotonic()
    for epoch in range(1, a.epochs + 1):
        model.train()
        loss_sum = 0
        for _ in range(a.batches):
            arrays = batch(train, users, itemnum, module.BATCH_SIZE, module.MAXLEN, rng)
            seq, pos, neg = [torch.as_tensor(x, device=device) for x in arrays]
            positive, negative = model(seq, pos, neg)
            # Match upstream BPR, including its epsilon.
            loss = (-torch.log(torch.sigmoid(positive - negative) + 1e-8) * (pos != 0)).sum() / (pos != 0).sum().clamp(min=1)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += loss.item()
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            scores = predict(model, rows, module.MAXLEN, device)
            m = metrics(scores)
            record = {'epoch': epoch, 'loss': loss_sum / a.batches, 'metrics': m, 'seconds': time.monotonic() - start}
            with (a.output / 'curve.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            if m['HR@10'] > best:
                best, best_epoch, best_metrics = m['HR@10'], epoch, m
                torch.save(model.state_dict(), a.output / 'checkpoint.pt')
                np.save(a.output / 'scores.npy', scores)
            if epoch - best_epoch >= a.patience:
                break
    model.load_state_dict(torch.load(a.output / 'checkpoint.pt', map_location=device, weights_only=True))
    save(a.output / 'correctness-after-training.json', check(model, module.MAXLEN, device))
    save(a.output / 'result.json', {'status': 'success', 'metrics': best_metrics,
        'best_epoch': best_epoch, 'completed_epochs': epoch, 'rows_sha256': row_hash,
        'checkpoint_sha256': sha(a.output / 'checkpoint.pt')})


if __name__ == '__main__':
    main()
