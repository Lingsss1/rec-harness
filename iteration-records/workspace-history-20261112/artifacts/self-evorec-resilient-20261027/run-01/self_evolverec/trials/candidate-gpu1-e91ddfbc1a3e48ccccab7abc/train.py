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
### >>> Self_EvolveRec-BLOCK-START: Match the host training default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Declare residual and loss ablations without changing data inputs
DATA_DIR = os.environ["GAGC_DATA_DIR"]
# baseline, gbce, recent_bce, combined, and full_gbce are separate trials.
VARIANT = os.environ.get("SASREC_VARIANT", "combined")
if VARIANT not in {"baseline", "gbce", "recent_bce", "combined", "full_gbce"}:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
RESIDUAL_MODE = (
    "recent"
    if VARIANT in {"recent_bce", "combined"}
    else "full" if VARIANT == "full_gbce" else "none"
)
USE_GBCE = VARIANT in {"gbce", "combined", "full_gbce"}
NEGATIVE_COUNT = 32
GBCE_T = 0.75
POSITION_CHUNK = 512
RECENT_WINDOW = 5
if MAXLEN < RECENT_WINDOW:
    raise ValueError("SASREC_MAXLEN must be at least five")
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four datasets
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


### >>> Self_EvolveRec-BLOCK-START: Skip unused single-negative sampling for gBCE
def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    sample_single_negative: bool = True,
):
    ### <<< Self_EvolveRec-BLOCK-END
    users = [u for u, items in user_train.items() if len(items) >= 2]
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            ### >>> Self_EvolveRec-BLOCK-START: Preserve baseline sampling and avoid saturated-catalog loops
            if nxt != 0 and sample_single_negative:
                if len(ts) < itemnum:
                    neg_item = random.randint(1, itemnum)
                    while neg_item in ts:
                        neg_item = random.randint(1, itemnum)
                else:
                    neg_item = random.randrange(1, itemnum)
                    neg_item += int(neg_item >= nxt)
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
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        ### >>> Self_EvolveRec-BLOCK-START: Persist residual architecture configuration
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        residual_mode: str = "none",
        recent_window: int = 5,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum

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

        ### >>> Self_EvolveRec-BLOCK-START: Zero-initialized gated shared-encoder residual
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if residual_mode not in {"none", "recent", "full"}:
            raise ValueError(residual_mode)
        if not 1 <= recent_window <= maxlen:
            raise ValueError("Invalid recent window")
        self.residual_mode = residual_mode
        self.recent_window = recent_window
        if residual_mode != "none":
            self.residual_scale = nn.Parameter(torch.zeros(()))
            self.residual_gate = nn.Linear(2 * hidden_units + 1, 1)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.constant_(self.residual_gate.bias, -2.0)

    def _encode(self, seq: torch.Tensor, recent: bool = False) -> torch.Tensor:
        ### <<< Self_EvolveRec-BLOCK-END
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_emb(pos)
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = seq == 0
        seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)
        ### >>> Self_EvolveRec-BLOCK-START: Safe padding masks for independently encoded recent windows
        attention_mask = ~torch.tril(
            torch.ones((M, M), dtype=torch.bool, device=seq.device)
        )
        if recent:
            # MultiheadAttention boolean True means blocked. Valid queries cannot
            # attend padding. A padded query can attend itself to avoid all-masked rows.
            blocked = attention_mask.unsqueeze(0) | timeline_mask.unsqueeze(1)
            diagonal = torch.eye(M, dtype=torch.bool, device=seq.device).unsqueeze(0)
            blocked = torch.where(timeline_mask.unsqueeze(-1), ~diagonal, blocked)
            attention_mask = (
                blocked.repeat_interleave(self.attention_layers[0].num_heads, dim=0)
                if self.attention_layers
                else blocked
            )
        ### <<< Self_EvolveRec-BLOCK-END

        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            queries = self.attention_layernorms[i](seq_emb)
            ### >>> Self_EvolveRec-BLOCK-START: Avoid materializing unused attention weights
            attn_out, _ = self.attention_layers[i](
                queries,
                seq_emb,
                seq_emb,
                attn_mask=attention_mask,
                need_weights=False,
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        ### >>> Self_EvolveRec-BLOCK-START: Fuse causal five-item prefixes using shared encoder parameters
        features = self.last_layernorm(seq_emb)
        if recent:
            features = features.masked_fill(timeline_mask.unsqueeze(-1), 0.0)
        return features

    def _windows(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(seq, (self.recent_window - 1, 0)).unfold(
            1, self.recent_window, 1
        )

    def _fuse(
        self, full: torch.Tensor, windows: torch.Tensor, prefix_lengths: torch.Tensor
    ) -> torch.Tensor:
        if self.residual_mode == "none":
            return full
        recent = (
            self._encode(windows, recent=True)[:, -1]
            if self.residual_mode == "recent"
            else full
        )
        lengths = prefix_lengths.to(full.dtype).log1p().unsqueeze(-1)
        gate = torch.sigmoid(self.residual_gate(torch.cat((full, recent, lengths), -1)))
        return full + self.residual_scale * gate * recent

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        full = self._encode(seq)
        if self.residual_mode == "none":
            return full
        if not self.training and self.residual_scale.detach().item() == 0.0:
            return full
        indices = (seq.reshape(-1) != 0).nonzero(as_tuple=True)[0]
        windows = self._windows(seq).reshape(-1, self.recent_window)
        lengths = (seq != 0).cumsum(-1).reshape(-1)
        original = full.reshape(-1, self.hidden_units)
        if indices.numel() == 0:
            return full
        updates = [
            self._fuse(original[ids], windows[ids], lengths[ids])
            for ids in indices.split(POSITION_CHUNK)
        ]
        # Copy the full feature tensor once instead of once per chunk.
        fused = original.index_copy(0, indices, torch.cat(updates, dim=0))
        return fused.reshape_as(full)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        item_features = self.item_emb(items)
        if item_features.shape == features.shape:
            # Aligned next-item scores: B x L x H against B x L x H.
            return (features * item_features).sum(-1)
        if items.ndim == 1:
            # A shared candidate list or full catalog.
            return features.matmul(item_features.transpose(-1, -2))
        # Per-example candidates: B x K, or B x L x K for sequence features.
        return (features.unsqueeze(-2) * item_features).sum(-1)
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

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        ### >>> Self_EvolveRec-BLOCK-START: Encode only the final recent prefix during recommendation
        final_feat = self._encode(seq)[:, -1, :]
        if self.residual_mode != "none":
            windows = self._windows(seq)[:, -1, :]
            lengths = (seq != 0).sum(-1)
            final_feat = self._fuse(final_feat, windows, lengths)
        item_embs = self.item_emb(candidates)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Accumulate chunk gradients through a leaf proxy and verify scoring equivalence


def backward_objective(
    model, seq, positive, negative, use_gbce, chunk_size=POSITION_CHUNK
):
    full = model._encode(seq)
    residual = model.residual_mode != "none"
    # All variants backpropagate through the full encoder exactly once.
    proxy = full.detach().requires_grad_(True)
    indices = (positive.reshape(-1) != 0).nonzero(as_tuple=True)[0]
    count = indices.numel()
    if count == 0:
        return 0.0
    targets = positive.reshape(-1)[indices]
    if use_gbce:
        if model.itemnum < 2:
            raise ValueError("Negative sampling requires at least two items")
        # Uniform with replacement over all nonpadding items except this target.
        negatives = torch.randint(
            1, model.itemnum, (count, NEGATIVE_COUNT), device=seq.device
        )
        negatives += (negatives >= targets[:, None]).long()
    else:
        negatives = negative.reshape(-1)[indices, None]
    windows = model._windows(seq).reshape(-1, model.recent_window)
    lengths = (seq != 0).cumsum(-1).reshape(-1)
    flat = proxy.reshape(-1, model.hidden_units)
    total_loss = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        ids = indices[start:stop]
        h = flat[ids]
        if residual:
            h = model._fuse(h, windows[ids], lengths[ids])
        pos_logits = (h * model.item_emb(targets[start:stop])).sum(-1).float()
        neg_logits = (
            (h[:, None] * model.item_emb(negatives[start:stop])).sum(-1).float()
        )
        positive_loss = torch.nn.functional.softplus(-pos_logits)
        negative_loss = torch.nn.functional.softplus(neg_logits).sum(-1)
        if use_gbce:
            beta = 1.0 - GBCE_T + GBCE_T * NEGATIVE_COUNT / (model.itemnum - 1)
            per_position = (beta * positive_loss + negative_loss) / (NEGATIVE_COUNT + 1)
        else:
            per_position = positive_loss + negative_loss
        loss = per_position.sum() / count
        # Free every chunk graph immediately, including architecture ablations.
        loss.backward()
        detached_loss = loss.detach()
        total_loss = total_loss + detached_loss
    if proxy.grad is None:
        raise RuntimeError("Chunk losses did not produce full-feature gradients")
    full.backward(proxy.grad)
    # Synchronize once per batch rather than once per position chunk.
    return total_loss.item()


def check_chunk_gradients(model, seq, positive, negative, use_gbce):
    # Tiny dropout-free check includes a nonzero residual, testing recent-encoder
    # and gate gradients as well as the full encoder and shared item embeddings.
    previous_training = model.training
    model.eval()
    residual = model.residual_mode != "none"
    saved_scale = model.residual_scale.detach().clone() if residual else None
    devices = [seq.device.index] if seq.is_cuda else []
    try:
        if residual:
            with torch.no_grad():
                model.residual_scale.zero_()
                reference = model._encode(seq)
                actual = model.log2feats(seq)
                if not torch.equal(reference, actual):
                    raise RuntimeError("Zero-residual baseline equivalence failed")
                candidates = (
                    torch.arange(1, min(model.itemnum, 16) + 1, device=seq.device)
                    .unsqueeze(0)
                    .expand(seq.shape[0], -1)
                )
                expected_scores = (
                    model.item_emb(candidates)
                    .matmul(reference[:, -1].unsqueeze(-1))
                    .squeeze(-1)
                )
                torch.testing.assert_close(
                    model.predict(seq, candidates),
                    expected_scores,
                    rtol=0.0,
                    atol=0.0,
                )
                model.residual_scale.fill_(0.1)
        gradients = []
        losses = []
        for chunk in (1000000, 3):
            model.zero_grad(set_to_none=True)
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(4242)
                losses.append(
                    backward_objective(
                        model, seq, positive, negative, use_gbce, chunk_size=chunk
                    )
                )
            gradients.append(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            )
        if not math.isclose(losses[0], losses[1], rel_tol=2e-5, abs_tol=2e-6):
            raise RuntimeError("Chunked loss verification failed")
        if gradients[0].keys() != gradients[1].keys():
            raise RuntimeError("Chunked gradient support differs")
        for name in gradients[0]:
            torch.testing.assert_close(
                gradients[0][name], gradients[1][name], rtol=2e-3, atol=2e-5, msg=name
            )
    finally:
        model.zero_grad(set_to_none=True)
        if residual:
            with torch.no_grad():
                model.residual_scale.copy_(saved_scale)
        model.train(previous_training)


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
    ### >>> Self_EvolveRec-BLOCK-START: Fixed validation candidates without modifying training RNG
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
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
                ### >>> Self_EvolveRec-BLOCK-START: Use the local validation RNG
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


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    ### >>> Self_EvolveRec-BLOCK-START: Initialize each independent category with training seed 42
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum < 2 or not any(len(items) >= 2 for items in user_train.values()):
        raise ValueError(f"Insufficient training data for {dataset_name}")
    print(
        f"  [{dataset_name}] users={len(user_train)}  items={itemnum}"
        f"  variant={VARIANT}  negatives={NEGATIVE_COUNT if USE_GBCE else 1}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the declared residual architecture
        maxlen=MAXLEN,
        residual_mode=RESIDUAL_MODE,
        recent_window=RECENT_WINDOW,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### >>> Self_EvolveRec-BLOCK-START: Verify chunk backward once before optimizer updates
    gradient_checked = False
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even when HR is zero
    best_hr10 = -float("inf")
    ### <<< Self_EvolveRec-BLOCK-END
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Train the shared recent encoder with bounded chunk activations
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                sample_single_negative=not USE_GBCE,
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            if not gradient_checked:
                # Include a real valid training row even when sampled histories
                # are short, and preserve the five-slot recent-window convention.
                check_width = min(MAXLEN, 8)
                check_chunk_gradients(
                    model,
                    seq_t[:2, -check_width:],
                    pos_t[:2, -check_width:],
                    neg_t[:2, -check_width:],
                    USE_GBCE,
                )
                gradient_checked = True
            optimizer.zero_grad(set_to_none=True)
            batch_loss = backward_objective(model, seq_t, pos_t, neg_t, USE_GBCE)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += batch_loss
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the final epoch even for short host runs
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ### <<< Self_EvolveRec-BLOCK-END
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Serialize exact scoring architecture and objective configuration
                "maxlen": MAXLEN,
                "residual_mode": RESIDUAL_MODE,
                "recent_window": RECENT_WINDOW,
                "variant": VARIANT,
                "negative_count": NEGATIVE_COUNT if USE_GBCE else 1,
                "gbce_t": GBCE_T if USE_GBCE else None,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


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
### >>> Self_EvolveRec-BLOCK-START: Retain all four category routes in the generated predictor
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
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            ### >>> Self_EvolveRec-BLOCK-START: Load the trained gated residual architecture
            maxlen=meta["maxlen"],
            residual_mode=meta.get("residual_mode", "none"),
            recent_window=meta.get("recent_window", 5),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit category routes
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"No trained checkpoint for dataset {dataset_name}")
    # Preserve the legacy fallback only when no dataset was supplied.
    if dataset_name and dataset_name in _models:
    ### <<< Self_EvolveRec-BLOCK-END
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)

    all_hr10: list[float] = []
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    write_predict_script(working_dir)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


