"""Per-dataset SASRec — trains one model per Amazon Reviews dataset.

Architecture (aligned with LLM-SRec SeqRec/sasrec, adapted for the RecHarness raw .txt data format):
  - One SASRec model per dataset: Movies_and_TV, Industrial_and_Scientific,
    Electronics, CDs_and_Vinyl
  - Checkpoints saved to ./working/{dataset}_model.pt
  - Official SASRec positional encoding: positions 0..maxlen-1 with padding masked after dropout
  - Official PointWiseFeedForward: Conv1d → Dropout → ReLU → Conv1d → Dropout + residual
  - Official BCEWithLogits loss on positive and negative sequence logits
  - Early stopping with patience=20 (checked every 5 epochs)
  - Writes predict.py routing to per-dataset models via dataset_name kwarg

Key hyper-parameters (all tunable by RecHarness):
  HIDDEN_UNITS : embedding / attention dimension
  NUM_BLOCKS   : number of self-attention blocks
  NUM_HEADS    : number of attention heads
  DROPOUT_RATE : dropout on attention + FFN
  LR           : Adam learning rate
  WEIGHT_DECAY : Adam weight decay
  BATCH_SIZE   : training batch size
  NUM_BATCHES  : gradient steps per epoch
  NUM_EPOCHS   : max training epochs
  MAXLEN       : max history length
  PATIENCE     : early stopping patience (epochs without val improvement)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# ── Hyper-parameters ──────────────────────────────────────────────────
HIDDEN_UNITS = int(os.environ.get("SASREC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("SASREC_HEADS", "1"))
DROPOUT_RATE = float(os.environ.get("SASREC_DROPOUT", "0.5"))
LR = float(os.environ.get("SASREC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("SASREC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("SASREC_BATCH", "128"))
NUM_BATCHES = int(os.environ.get("SASREC_BATCHES", "200"))
### >>> Self_EvolveRec-BLOCK-START: Predeclared four-variant search and bounded training
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = 5
EVAL_EVERY = 10
VARIANTS = ("baseline", "gBCE_only", "gate_only", "combined")
NUM_NEGATIVES = 128
RECENT_WINDOW = 5
POSITION_CHUNK = 512
SEARCH_SECONDS = 43200.0
TRAINING_FRACTION = 0.9

DATA_DIR = os.environ["GAGC_DATA_DIR"]

import shutil
import time
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def reset_training_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four dataset routes during training and serving
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Data loading ──────────────────────────────────────────────────────


def _read_split(path: str) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                d[int(parts[0])].append(int(parts[1]))
    return dict(d)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return stats[dataset_name]["itemnum"]
    # fallback: scan train file
    itemnum = 0
    path = os.path.join(data_dir, f"{dataset_name}_train.txt")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    user_valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    itemnum = _load_itemnum(data_dir, dataset_name)
    return user_train, user_valid, itemnum


# ── Batch sampler ─────────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Reuse eligible users and avoid unused single-negative sampling
def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    users=None,
    sample_single_negative=True,
):
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No users with at least two training interactions")
    ### <<< Self_EvolveRec-BLOCK-END
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ### >>> Self_EvolveRec-BLOCK-START: Construct exclusions only for the baseline sampler
        ts = set(history) if sample_single_negative else set()
        if sample_single_negative and len(ts) >= itemnum:
            raise ValueError("Baseline sampler requires an unseen training negative")
        ### <<< Self_EvolveRec-BLOCK-END
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            ### >>> Self_EvolveRec-BLOCK-START: Preserve baseline negatives only for BCE variants
            if nxt != 0 and sample_single_negative:
                neg_item = random.randint(1, itemnum)
                while neg_item in ts:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            ### <<< Self_EvolveRec-BLOCK-END
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


# ── Model ─────────────────────────────────────────────────────────────


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure the shared encoder and independent recent pathway
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        use_gate: bool = False,
        recent_window: int = 5,
    ):
        super().__init__()
        if itemnum <= 1:
            raise ValueError("At least two real items are required")
        if recent_window > maxlen or recent_window < 1:
            raise ValueError("recent_window must be between one and maxlen")
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.recent_window = recent_window
        if use_gate:
            # Gate construction must not shift shared encoder initialization
            # relative to the ungated ablation with the same training seed.
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Linear(2 * hidden_units, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, math.log(0.1 / 0.9))
        ### <<< Self_EvolveRec-BLOCK-END

        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.item_emb.weight.data.normal_(0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    ### >>> Self_EvolveRec-BLOCK-START: Mask-safe shared encoding of independent raw five-item windows
    def encode_full(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        if M < 1 or M > self.maxlen:
            raise ValueError("Invalid sequence length")
        timeline_mask = seq.eq(0)
        pos = torch.arange(M, device=seq.device).unsqueeze(0)
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        seq_emb = seq_emb.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        causal = torch.ones(M, M, dtype=torch.bool, device=seq.device).triu(1)
        blocked = causal.unsqueeze(0) | timeline_mask.unsqueeze(1)
        # Padded queries have no meaningful output. Give them key zero to
        # prevent all-masked softmax rows, then clear their states below.
        blocked = blocked.masked_fill(timeline_mask.unsqueeze(-1), True)
        blocked[:, :, 0] = blocked[:, :, 0] & ~timeline_mask
        attention_mask = blocked.repeat_interleave(self.num_heads, dim=0)

        for i in range(len(self.attention_layers)):
            states = seq_emb.transpose(0, 1)
            queries = self.attention_layernorms[i](states)
            attn_out, _ = self.attention_layers[i](
                queries,
                states,
                states,
                attn_mask=attention_mask,
                need_weights=False,
            )
            seq_emb = (queries + attn_out).transpose(0, 1)
            seq_emb = seq_emb.masked_fill(timeline_mask.unsqueeze(-1), 0.0)
            seq_emb = self.forward_layers[i](self.forward_layernorms[i](seq_emb))
            seq_emb = seq_emb.masked_fill(timeline_mask.unsqueeze(-1), 0.0)
        return self.last_layernorm(seq_emb).masked_fill(
            timeline_mask.unsqueeze(-1), 0.0
        )

    def raw_recent_windows(self, seq, positions):
        # Each row is an independently encoded raw suffix ending at a
        # supervised position. Positions within every window are 0..W-1.
        padded = F.pad(seq, (self.recent_window - 1, 0), value=0)
        windows = padded.unfold(1, self.recent_window, 1)
        return windows[positions[:, 0], positions[:, 1]].contiguous()

    def encode_recent(self, windows):
        return self.encode_full(windows)[:, -1, :]

    def mix_features(self, full, recent):
        gate = torch.sigmoid(self.gate(torch.cat((full, recent), dim=-1)))
        return (1.0 - gate) * full + gate * recent

    def supervised_features(self, seq, valid):
        full = self.encode_full(seq)[valid]
        # Empty diagnostic histories have no recent windows. Avoid passing
        # a zero-sized batch through MultiheadAttention.
        if not self.use_gate or full.shape[0] == 0:
            return full
        positions = valid.nonzero(as_tuple=False)
        windows = self.raw_recent_windows(seq, positions)
        recent_chunks = []
        for chunk in windows.split(POSITION_CHUNK):
            if self.training and torch.is_grad_enabled():
                recent_chunks.append(
                    checkpoint(
                        self.encode_recent,
                        chunk,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                )
            else:
                recent_chunks.append(self.encode_recent(chunk))
        if not recent_chunks:
            return full
        return self.mix_features(full, torch.cat(recent_chunks, dim=0))

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.use_gate:
            return self.encode_full(seq)
        valid = seq.ne(0)
        features = self.supervised_features(seq, valid)
        result = features.new_zeros((*seq.shape, self.hidden_units))
        return result.masked_scatter(valid.unsqueeze(-1), features.reshape(-1))

    def final_features(self, seq):
        full = self.encode_full(seq)[:, -1, :]
        if not self.use_gate:
            return full
        recent_ids = seq[:, -self.recent_window :]
        if recent_ids.shape[1] < self.recent_window:
            recent_ids = F.pad(
                recent_ids, (self.recent_window - recent_ids.shape[1], 0)
            )
        features = self.mix_features(full, self.encode_recent(recent_ids))
        return features.masked_fill(seq[:, -1:].eq(0), 0.0)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    ### <<< Self_EvolveRec-BLOCK-END

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        log_feats = self.log2feats(seq)
        if pos is None or neg is None:
            return log_feats
        pos_embs = self.item_emb(pos)
        neg_embs = self.item_emb(neg)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    ### >>> Self_EvolveRec-BLOCK-START: Serve the same final-window mixture used during training
    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.predict_score(self.final_features(seq), candidates)

    ### <<< Self_EvolveRec-BLOCK-END


# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: SASRec,
    user_train: dict,
    user_valid: dict,
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Fix validation sampling without resetting training randomness
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END

    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    model.eval()
    with torch.no_grad():
        for u in val_users:
            pos_items = user_valid[u]
            if not pos_items:
                continue
            target = pos_items[0]

            hist = user_train[u][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int32)
            for i, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - i] = item

            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            neg_pool: set[int] = set()
            interacted = set(user_train[u]) | set(pos_items)
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Reuse the fixed validation-only candidate RNG
                ni = rng.randint(1, itemnum)
                ### <<< Self_EvolveRec-BLOCK-END
                if ni not in interacted:
                    neg_pool.add(ni)
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            scores = model.predict(seq_t, cand_t)[0].cpu().numpy().tolist()
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    model.train()
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Exact requested gBCE reduction, component ablations and checkpoint ledger
def gbce_chunk_sum(model, features, positive, negative, beta):
    positive_logits = model.predict_score(features, positive).float()
    negative_logits = model.predict_score(features, negative).float()
    return (
        beta * F.softplus(-positive_logits) + F.softplus(negative_logits).sum(dim=-1)
    ).sum() / (NUM_NEGATIVES + 1)


def train_one_dataset(
    dataset_name: str,
    device: torch.device,
    data_dir: str,
    variant: str,
    working_dir: str,
    seconds: float,
):
    started = synchronized_time(device)
    deadline = started + seconds
    reset_training_seed()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    users = [u for u, items in user_train.items() if len(items) >= 2]
    use_gate = variant in ("gate_only", "combined")
    use_gbce = variant in ("gBCE_only", "combined")
    beta = 0.25 + 0.75 * NUM_NEGATIVES / (itemnum - 1) if itemnum > 1 else None
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        use_gate=use_gate,
        recent_window=RECENT_WINDOW,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    # A fresh initial checkpoint prevents accidentally reusing an older trial.
    torch.save(model.state_dict(), ckpt_path)
    best_hr10 = None
    best_epoch = 0
    stale_evaluations = 0
    completed_epoch = 0
    stopped_for_budget = False
    evaluations = []
    step_seconds = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        completed_steps = 0
        for _ in range(NUM_BATCHES):
            if synchronized_time(device) >= deadline:
                stopped_for_budget = True
                break
            step_started = time.perf_counter()
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                users=users,
                sample_single_negative=not use_gbce,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            valid = pos_t.ne(0)
            optimizer.zero_grad(set_to_none=True)
            features = model.supervised_features(seq_t, valid)
            positives = pos_t[valid]
            if positives.numel() == 0:
                raise ValueError("Empty supervised batch")
            if use_gbce:
                loss = features.new_zeros((), dtype=torch.float32)
                for start in range(0, positives.numel(), POSITION_CHUNK):
                    stop = min(start + POSITION_CHUNK, positives.numel())
                    # Uniform independent draws, with replacement. Padding is
                    # excluded; target and history collisions are intentional.
                    negatives = torch.randint(
                        1,
                        itemnum + 1,
                        (stop - start, NUM_NEGATIVES),
                        device=device,
                    )
                    feature_chunk = features[start:stop]
                    positive_chunk = positives[start:stop]

                    # Recompute candidate embeddings during backward instead
                    # of retaining all positions x 128 x hidden activations.
                    def chunk_loss(h, p, n):
                        return gbce_chunk_sum(model, h, p, n, beta)

                    loss = loss + checkpoint(
                        chunk_loss,
                        feature_chunk,
                        positive_chunk,
                        negatives,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                loss = loss / positives.numel()
            else:
                negatives = torch.as_tensor(neg, dtype=torch.long, device=device)[valid]
                positive_logits = model.predict_score(features, positives).float()
                negative_logits = model.predict_score(features, negatives).float()
                loss = (
                    F.softplus(-positive_logits).mean()
                    + F.softplus(negative_logits).mean()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {variant}/{dataset_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
            completed_steps += 1
            step_seconds.append(synchronized_time(device) - step_started)
        if completed_steps != NUM_BATCHES:
            break
        completed_epoch = epoch
        if epoch % EVAL_EVERY == 0:
            if synchronized_time(device) >= deadline:
                stopped_for_budget = True
                break
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = float(val_m["HR@10"])
            evaluations.append({"epoch": epoch, **val_m})
            print(
                f"[{variant}/{dataset_name}] epoch={epoch} "
                f"loss={total_loss / NUM_BATCHES:.6f} HR@10={hr10:.4f}",
                flush=True,
            )
            # Strict improvement retains the earliest checkpoint on ties.
            if best_hr10 is None or hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                stale_evaluations = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                stale_evaluations += 1
            if stale_evaluations >= PATIENCE:
                break

    metadata = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "use_gate": use_gate,
        "recent_window": RECENT_WINDOW,
        "variant": variant,
        "dataset_name": dataset_name,
        "dataset_routes": DATASETS,
        "positional_policy": "padded_slots_0_to_branch_length_minus_1",
        "loss": "requested_gBCE" if use_gbce else "baseline_two_term_BCE",
        "num_negatives": NUM_NEGATIVES if use_gbce else 1,
        "beta": beta if use_gbce else None,
        "calibration_t": 0.75 if use_gbce else None,
        "sampler": (
            "uniform_with_replacement_including_collisions"
            if use_gbce
            else "baseline_uniform_excluding_training_history"
        ),
        "training_seed": 42,
        "preprocessing_seed": 0,
        "best_epoch": best_epoch,
        "best_hr10": best_hr10,
    }
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    record = {
        "variant": variant,
        "dataset": dataset_name,
        "best_hr10": best_hr10,
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "evaluations": evaluations,
        "selection_opportunities": len(evaluations),
        "budget_exhausted": stopped_for_budget,
        "allocated_seconds": seconds,
        "elapsed_seconds": synchronized_time(device) - started,
        "mean_step_seconds": float(np.mean(step_seconds)) if step_seconds else None,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    with open(os.path.join(working_dir, f"{dataset_name}_search.json"), "w") as f:
        json.dump(record, f, indent=2)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


### <<< Self_EvolveRec-BLOCK-END


# ── Write unified predict.py ──────────────────────────────────────────


def write_predict_script(working_dir: str) -> None:
    script = '''"""Per-dataset SASRec predict — routes by dataset_name kwarg.

Supports both legacy signature:
    predict(user_id, history, candidates) -> list[float]
and per-dataset signature (used by the RecHarness harness when available):
    predict(user_id, history, candidates, *, dataset_name) -> list[float]
"""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Serve every required category regardless of environment filtering
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

_models: dict = {}
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SASRec


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail clearly when a required category artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model or metadata for required dataset {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the checkpoint's selected pathway architecture
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            use_gate=meta.get("use_gate", False),
            recent_window=meta.get("recent_window", 5),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit routes instead of silently using another category
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    with torch.no_grad():
        cand_t   = torch.tensor([candidates], dtype=torch.long, device=_device)
        scores   = model.predict(seq_t, cand_t)[0].cpu().tolist()

    return scores
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written → {predict_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Macro-only variant selection and measured serving consistency checks
def check_selected_predictor(working_dir, device, deadline):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "selected_predictor_check", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    # Validate one category at a time: loading all four models plus a
    # duplicate checkpoint can otherwise exhaust GPU memory.
    results = {}
    for ds in DATASETS:
        if synchronized_time(device) >= deadline:
            results[ds] = {"status": "not_run_budget_exhausted"}
            continue
        with open(os.path.join(working_dir, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        predictor._models.clear()
        predictor._metas.clear()
        predictor.DATASETS = [ds]
        predictor._ensure_loaded()
        kwargs = {
            k: meta[k]
            for k in (
                "itemnum",
                "hidden_units",
                "num_blocks",
                "num_heads",
                "maxlen",
                "use_gate",
                "recent_window",
            )
        }
        direct = SASRec(**kwargs, dropout_rate=0.0).to(device).eval()
        direct.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{ds}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        user_train, _, _ = load_dataset(ds, DATA_DIR)
        histories = [[], [1], [1, min(2, meta["itemnum"])]]
        histories += [history[-MAXLEN:] for history in list(user_train.values())[:5]]
        candidates = list(range(1, min(meta["itemnum"], 32) + 1))
        max_error = 0.0
        feature_error = 0.0
        with torch.no_grad():
            for history in histories:
                history = history[-MAXLEN:]
                seq = torch.tensor(
                    [[0] * (MAXLEN - len(history)) + history],
                    dtype=torch.long,
                    device=device,
                )
                items = torch.tensor([candidates], device=device)
                expected = direct.predict(seq, items)
                served = torch.tensor(
                    predictor.predict(0, history, candidates, dataset_name=ds),
                    device=device,
                ).unsqueeze(0)
                if (
                    not torch.isfinite(expected).all()
                    or not torch.isfinite(served).all()
                ):
                    raise AssertionError(f"Nonfinite predictor scores for {ds}")
                max_error = max(max_error, (expected - served).abs().max().item())
                feature_error = max(
                    feature_error,
                    (direct.log2feats(seq)[:, -1] - direct.final_features(seq))
                    .abs()
                    .max()
                    .item(),
                )
            recent_error = recent_score_error = None
            gate_by_length = {}
            disagreement = []
            if direct.use_gate:
                a = torch.ones((1, MAXLEN), dtype=torch.long, device=device)
                b = a.clone()
                b[:, :-RECENT_WINDOW] = min(2, meta["itemnum"])
                # Exercise the actual training window gatherer, rather than
                # merely encoding two manually identical sliced tensors.
                positions = torch.tensor([[0, MAXLEN - 1]], device=device)
                ra = direct.encode_recent(direct.raw_recent_windows(a, positions))
                rb = direct.encode_recent(direct.raw_recent_windows(b, positions))
                recent_error = (ra - rb).abs().max().item()
                recent_score_error = (
                    (direct.predict_score(ra, items) - direct.predict_score(rb, items))
                    .abs()
                    .max()
                    .item()
                )
                for history in list(user_train.values())[:128]:
                    if synchronized_time(device) >= deadline:
                        break
                    history = history[-MAXLEN:]
                    if not history:
                        continue
                    seq = torch.tensor(
                        [[0] * (MAXLEN - len(history)) + history],
                        dtype=torch.long,
                        device=device,
                    )
                    full = direct.encode_full(seq)[:, -1]
                    recent = direct.encode_recent(seq[:, -RECENT_WINDOW:])
                    gate = torch.sigmoid(
                        direct.gate(torch.cat((full, recent), -1))
                    ).item()
                    stratum = (
                        "1_to_5"
                        if len(history) <= 5
                        else "6_to_20" if len(history) <= 20 else "over_20"
                    )
                    gate_by_length.setdefault(stratum, []).append(gate)
                    disagreement.append(
                        (1.0 - F.cosine_similarity(full, recent)).item()
                    )
            if max_error > 1e-4 or feature_error > 1e-4:
                raise AssertionError(f"Training/predictor mismatch for {ds}")
            if recent_error is not None and (
                recent_error > 1e-6 or recent_score_error > 1e-6
            ):
                raise AssertionError(
                    f"Recent pathway depends on older history for {ds}"
                )
            results[ds] = {
                "status": "passed",
                "predictor_max_abs_error": max_error,
                "final_feature_max_abs_error": feature_error,
                "recent_prefix_invariance_max_abs_error": recent_error,
                "recent_score_prefix_invariance_max_abs_error": recent_score_error,
                "gate_by_observed_history_length": {
                    key: {
                        "count": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": min(values),
                        "max": max(values),
                    }
                    for key, values in gate_by_length.items()
                },
                "mean_full_recent_cosine_distance": (
                    float(np.mean(disagreement)) if disagreement else None
                ),
            }
        del direct
        predictor._models.clear()
        predictor._metas.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "metrics": results,
        "metric_definitions": {
            "predictor_max_abs_error": "Maximum absolute score difference between a directly loaded checkpoint and generated predict.py on empty, short, and public training histories.",
            "final_feature_max_abs_error": "Maximum absolute difference between log2feats(sequence) at the last position and the optimized serving feature.",
            "recent_prefix_invariance_max_abs_error": "Maximum recent-feature change after changing only interactions before the final five.",
            "recent_score_prefix_invariance_max_abs_error": "Maximum recent-branch candidate-score change after changing only interactions before the final five.",
            "gate_by_observed_history_length": "Measured scalar gate summary by truncated public history length; exploratory, not a selection metric.",
            "mean_full_recent_cosine_distance": "Mean one minus cosine similarity of full and independently encoded recent features on public training histories; exploratory.",
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = synchronized_time(device)
    final_deadline = started + SEARCH_SECONDS
    training_deadline = started + SEARCH_SECONDS * TRAINING_FRACTION
    per_run_seconds = (
        SEARCH_SECONDS * TRAINING_FRACTION / (len(VARIANTS) * len(DATASETS))
    )
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    search_dir = os.path.join(working_dir, "variant_search")
    os.makedirs(search_dir, exist_ok=True)
    ledger = {
        "budget_seconds": SEARCH_SECONDS,
        "training_fraction": TRAINING_FRACTION,
        "per_category_variant_seconds": per_run_seconds,
        "selection_metric": "macro_validation_HR@10",
        "variant_tie_order": list(VARIANTS),
        "checkpoint_tie_rule": "earliest strict-improvement checkpoint",
        "runs": [],
        "macro_scores": {},
    }
    ledger_path = os.path.join(search_dir, "ledger.json")
    for variant in VARIANTS:
        variant_dir = os.path.join(search_dir, variant)
        records = []
        for ds in DATASETS:
            remaining = max(0.0, training_deadline - synchronized_time(device))
            record = train_one_dataset(
                ds,
                device,
                DATA_DIR,
                variant,
                variant_dir,
                min(per_run_seconds, remaining),
            )
            records.append(record)
            ledger["runs"].append(record)
            ledger["elapsed_seconds"] = synchronized_time(device) - started
            with open(ledger_path, "w") as f:
                json.dump(ledger, f, indent=2)
        scores = [record["best_hr10"] for record in records]
        ledger["macro_scores"][variant] = (
            sum(scores) / len(DATASETS)
            if all(score is not None for score in scores)
            else None
        )
    eligible = [v for v in VARIANTS if ledger["macro_scores"][v] is not None]
    # No partial-category macro scores or fabricated validation results.
    selected = (
        max(eligible, key=lambda v: ledger["macro_scores"][v])
        if eligible
        else "baseline"
    )
    ledger["selected_variant"] = selected
    ledger["selection_status"] = (
        "validation_selected"
        if eligible
        else "no_complete_validation_schedule_baseline_fallback"
    )
    selected_dir = os.path.join(search_dir, selected)
    for ds in DATASETS:
        for suffix in ("model.pt", "meta.pkl"):
            filename = f"{ds}_{suffix}"
            shutil.copy2(
                os.path.join(selected_dir, filename),
                os.path.join(working_dir, filename),
            )
    # Selection is frozen before diagnostics. These checks never change it.
    write_predict_script(working_dir)
    diagnostics = check_selected_predictor(working_dir, device, final_deadline)
    with open(os.path.join(working_dir, "architecture_checks.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    ledger["elapsed_seconds"] = synchronized_time(device) - started
    ledger["diagnostic_definitions_file"] = "architecture_checks.json"
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=2)
    print(
        f"Selected variant={selected}; macro validation HR@10={ledger['macro_scores'][selected]}",
        flush=True,
    )


### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


