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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host data boundary and declare the six controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]

CONFIGURATIONS = (
    {"name": "baseline", "pooling": "none", "gbce": False, "components": 0},
    {"name": "gbce_only", "pooling": "none", "gbce": True, "components": 1},
    {"name": "uniform_only", "pooling": "uniform", "gbce": False, "components": 1},
    {"name": "recency_only", "pooling": "recency", "gbce": False, "components": 1},
    {"name": "support_only", "pooling": "support", "gbce": False, "components": 1},
    {"name": "combined", "pooling": "combined", "gbce": True, "components": 2},
)
NEGATIVE_COUNT = 128
GBCE_T = 0.75
CATEGORY_SECONDS = 1800.0


def reset_training_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

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


### >>> Self_EvolveRec-BLOCK-START: Avoid redundant user scans and bypass rejection sampling for gBCE
def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    users=None,
    sample_negatives: bool = True,
):
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No users with at least two training interactions")
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Expose training-only histories for collision accounting
    batch_seq, batch_pos, batch_neg = [], [], []
    sample_batch.last_histories = []
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        sample_batch.last_histories.append(history)
        ### <<< Self_EvolveRec-BLOCK-END
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            ### >>> Self_EvolveRec-BLOCK-START: Preserve baseline negatives and handle saturated histories
            if nxt != 0 and sample_negatives:
                if idx == maxlen - 1 and len(ts - {0}) >= itemnum:
                    raise ValueError(
                        "Baseline rejection sampler has no eligible negative"
                    )
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
    ### >>> Self_EvolveRec-BLOCK-START: Persist the pooling architecture in the model constructor
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        pooling_mode: str = "none",
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

        ### >>> Self_EvolveRec-BLOCK-START: Causal support-weighted residual with matched pooling controls
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if pooling_mode not in {"none", "uniform", "recency", "support", "combined"}:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")
        self.pooling_mode = pooling_mode
        if pooling_mode != "none":
            # Keep shared SASRec initialization and subsequent dropout randomness
            # identical across pooling controls. The projection is deterministic.
            with torch.random.fork_rng(devices=[]):
                self.pool_projection = nn.Linear(hidden_units, hidden_units, bias=False)
            with torch.no_grad():
                self.pool_projection.weight.copy_(torch.eye(hidden_units))
            self.pool_gate = nn.Parameter(torch.tensor(-3.0))

    def history_pool(self, seq: torch.Tensor) -> torch.Tensor:
        # Static embeddings: no positional embeddings or dropout enter this pool.
        e = self.item_emb(seq)
        valid = seq.ne(0)
        length = seq.shape[1]
        position = torch.arange(length, device=seq.device)
        age = position[:, None] - position[None, :]
        mask = valid[:, :, None] & valid[:, None, :] & age.ge(0)[None]
        with torch.autocast(device_type=seq.device.type, enabled=False):
            weights = torch.ones(
                seq.shape[0], length, length, device=seq.device, dtype=torch.float32
            )
            if self.pooling_mode in {"support", "combined"}:
                z = torch.nn.functional.normalize(e.detach().float(), dim=-1, eps=1e-8)
                similarity = torch.bmm(z, z.transpose(1, 2)).clamp_min(0)
                pair_mask = valid[:, :, None] & valid[:, None, :]
                pair_mask = (
                    pair_mask
                    & ~torch.eye(length, device=seq.device, dtype=torch.bool)[None]
                )
                similarity = similarity * pair_mask
                # C[j,t] includes every k <= t, including k > j. Causal masking
                # before cumsum would incorrectly remove that evidence.
                cumulative = similarity.cumsum(-1)
                count = valid.cumsum(-1).float()
                support = (
                    cumulative.transpose(1, 2) / (count - 1).clamp_min(1)[:, :, None]
                )
                weights = weights * (1.0 + support)
            if self.pooling_mode in {"recency", "combined"}:
                weights = weights * torch.exp(-0.05 * age.clamp_min(0).float())[None]
            weights = weights * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.bmm(weights.to(e.dtype), e)
        ### <<< Self_EvolveRec-BLOCK-END

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_emb(pos)
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = seq == 0
        seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)
        attention_mask = ~torch.tril(
            torch.ones((M, M), dtype=torch.bool, device=seq.device)
        )

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

        ### >>> Self_EvolveRec-BLOCK-START: Use the identical augmented representation in training and inference
        features = self.last_layernorm(seq_emb)
        if self.pooling_mode != "none":
            features = features + self.pool_gate.sigmoid() * self.pool_projection(
                self.history_pool(seq)
            )
        return features * (~timeline_mask).unsqueeze(-1)
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
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(candidates)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)


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
    ### >>> Self_EvolveRec-BLOCK-START: Fix validation candidates without perturbing the training RNG
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    previous_training = model.training
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
            ### >>> Self_EvolveRec-BLOCK-START: Validate candidate availability and reuse fixed evaluation randomness
            interacted = set(user_train[u]) | set(pos_items)
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError("Validation requires 99 distinct eligible negatives")
            while len(neg_pool) < 99:
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore evaluation caller state
    model.train(previous_training)
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Exact memory-bounded generalized BCE and architecture checks


def generalized_bce(model, features, positives, negatives):
    from torch.utils.checkpoint import checkpoint

    k = negatives.shape[-1]
    if model.itemnum <= 1:
        raise ValueError("gBCE requires at least two real catalog items")
    if positives.numel() == 0 or k == 0:
        raise ValueError("gBCE requires valid positives and at least one negative draw")
    if features.ndim != 2 or positives.ndim != 1 or negatives.ndim != 2:
        raise ValueError("gBCE expects features P x D, positives P, negatives P x K")
    if (
        features.shape[0] != positives.numel()
        or negatives.shape[0] != positives.numel()
    ):
        raise ValueError("gBCE feature, positive and negative counts must match")
    beta = 1.0 - GBCE_T + GBCE_T * k / (model.itemnum - 1)
    positive_logits = (features * model.item_emb(positives)).sum(-1).float()
    total = beta * torch.nn.functional.softplus(-positive_logits).sum()

    def negative_sum(feats, ids):
        logits = (feats[:, None, :] * model.item_emb(ids)).sum(-1).float()
        return torch.nn.functional.softplus(logits).sum()

    # Checkpoint the expensive embedding gathers, retaining the exact sum over
    # independent draws while avoiding B*L*128*d retained embedding activations.
    for ids in negatives.split(16, dim=-1):
        if torch.is_grad_enabled():
            total = total + checkpoint(
                negative_sum,
                features,
                ids,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            total = total + negative_sum(features, ids)
    return total / ((k + 1) * positives.numel())


def check_architecture(model, device):
    """Small deterministic checks; preserve every training random stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    previous_training = model.training
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            n = min(4, model.maxlen)
            sequence = torch.tensor(
                [[0] + [1 + i % model.itemnum for i in range(n - 1)]],
                dtype=torch.long,
                device=device,
            )
            features = model.log2feats(sequence)
            assert torch.isfinite(features).all()
            torch.testing.assert_close(
                features[sequence == 0], torch.zeros_like(features[sequence == 0])
            )
            if n > 2:
                changed = sequence.clone()
                changed[:, -1] = 1 + int(sequence[0, -1]) % model.itemnum
                torch.testing.assert_close(
                    features[:, :-1],
                    model.log2feats(changed)[:, :-1],
                    atol=2e-5,
                    rtol=2e-5,
                )
            if model.pooling_mode != "none":
                pooled = model.history_pool(sequence)
                embedding = model.item_emb(sequence).float()
                reference = torch.zeros_like(pooled)
                for t in range(n):
                    keys = [j for j in range(t + 1) if sequence[0, j] != 0]
                    if not keys:
                        continue
                    weights = []
                    for j in keys:
                        support = 0.0
                        if model.pooling_mode in {"support", "combined"}:
                            zj = torch.nn.functional.normalize(
                                embedding[0, j], dim=0, eps=1e-8
                            )
                            for q in keys:
                                if q != j:
                                    zq = torch.nn.functional.normalize(
                                        embedding[0, q], dim=0, eps=1e-8
                                    )
                                    support += float(torch.dot(zj, zq).clamp_min(0))
                            support /= max(len(keys) - 1, 1)
                        w = 1.0 + support
                        if model.pooling_mode in {"recency", "combined"}:
                            w *= math.exp(-0.05 * (t - j))
                        weights.append(w)
                    weights = torch.tensor(weights, device=device, dtype=torch.float32)
                    weights /= weights.sum()
                    reference[0, t] = (weights[:, None] * embedding[0, keys]).sum(0)
                torch.testing.assert_close(pooled, reference, atol=2e-5, rtol=2e-5)
                if n > 1:
                    torch.testing.assert_close(pooled[0, 1], embedding[0, 1])
                saved_padding = model.item_emb.weight[0].clone()
                try:
                    model.item_emb.weight[0].fill_(123.0)
                    torch.testing.assert_close(
                        pooled, model.history_pool(sequence), atol=2e-5, rtol=2e-5
                    )
                    torch.testing.assert_close(
                        features, model.log2feats(sequence), atol=2e-5, rtol=2e-5
                    )
                finally:
                    model.item_emb.weight[0].copy_(saved_padding)
            f = model.item_emb.weight[1:2].detach().clone()
            p = torch.ones(1, dtype=torch.long, device=device)
            negatives = torch.ones((1, NEGATIVE_COUNT), dtype=torch.long, device=device)
            beta = 1 - GBCE_T + GBCE_T * NEGATIVE_COUNT / (model.itemnum - 1)
            positive_score = (f * model.item_emb(p)).sum(-1).float()
            negative_score = (f[:, None] * model.item_emb(negatives)).sum(-1).float()
            reference_loss = (
                -beta * torch.nn.functional.logsigmoid(positive_score)
                - torch.nn.functional.logsigmoid(-negative_score).sum(-1)
            ).mean() / (NEGATIVE_COUNT + 1)
            torch.testing.assert_close(
                generalized_bce(model, f, p, negatives), reference_loss
            )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train(previous_training)


# ── Per-dataset training ──────────────────────────────────────────────
### <<< Self_EvolveRec-BLOCK-END

### >>> Self_EvolveRec-BLOCK-START: Six controlled configurations with completed-epoch checkpoint selection


def train_one_dataset(
    dataset_name: str,
    device: torch.device,
    data_dir: str,
    configuration=None,
    working_dir=None,
):
    import time

    configuration = CONFIGURATIONS[0] if configuration is None else configuration
    if working_dir is None:
        working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    started = time.monotonic()
    reset_training_seed()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 1 or not user_valid:
        raise ValueError(f"Invalid training/validation data for {dataset_name}")
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError(f"No usable training users for {dataset_name}")
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        pooling_mode=configuration["pooling"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    # check_architecture preserves the post-initialization RNG states itself.
    # Do not reset them: doing so changes the baseline training random stream.
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    latest_path = os.path.join(working_dir, f"{dataset_name}_latest.pt")
    manifest_path = os.path.join(working_dir, f"{dataset_name}_manifest.json")
    manifest = {
        "configuration": configuration,
        "dataset": dataset_name,
        "training_seed": 42,
        "preprocessing_seed": 0,
        "optimizer": {
            "name": "Adam",
            "lr": LR,
            "betas": [0.9, 0.98],
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": 5.0,
        },
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "max_epochs": NUM_EPOCHS,
        "maxlen": MAXLEN,
        "category_wall_seconds_cap": CATEGORY_SECONDS,
        "checkpoint_schedule": "epoch 1, multiples of 10, final completed epoch",
        "checkpoint_selection": "category validation HR@10, then earliest epoch",
        "negative_sampler": (
            "uniform replacement, IDs 1..N, no exclusions"
            if configuration["gbce"]
            else "baseline history rejection"
        ),
        "negative_count": NEGATIVE_COUNT if configuration["gbce"] else 1,
        "alpha": NEGATIVE_COUNT / (itemnum - 1) if configuration["gbce"] else None,
        "beta": (
            1 - GBCE_T + GBCE_T * NEGATIVE_COUNT / (itemnum - 1)
            if configuration["gbce"]
            else None
        ),
        "evaluations": [],
        "epochs": [],
        "completed_epochs": 0,
        "current_positive_collisions": 0,
        "known_history_collisions": 0,
        "negative_draws": 0,
        "unavailable_probes": [
            "review-dependent probes",
            "metadata categories",
            "simulator diversity",
            "simulator conformity",
        ],
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def write_manifest():
        manifest["charged_wall_seconds"] = time.monotonic() - started
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    best_hr10, best_epoch = -1.0, None
    completed, last_evaluated = 0, None
    evaluation_seconds = 0.0
    step_seconds = 0.0
    # Conservatively reserve time for final evaluation, reloads and manifests.
    # The host remains the authority for charged GPU time and hard termination.
    # DEBUG: Reserve time for the previously disconnected diagnostic workflow,
    # final validation, serialization, and checkpoint reloads within the category cap.
    reserve_seconds = 180.0

    def evaluate_checkpoint(epoch):
        nonlocal best_hr10, best_epoch, last_evaluated, evaluation_seconds
        record = {"epoch": epoch, "status": "started"}
        manifest["evaluations"].append(record)
        write_manifest()
        begin = time.monotonic()
        try:
            metrics = evaluate_val(
                model, user_train, user_valid, itemnum, device, MAXLEN
            )
            synchronize()
            record.update(status="completed", metrics=metrics)
            hr10 = float(metrics["HR@10"])
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            last_evaluated = epoch
        except Exception as exc:
            record.update(status="failed", error=repr(exc))
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - begin
            evaluation_seconds = max(evaluation_seconds, record["wall_seconds"])
            write_manifest()

    try:
        manifest["architecture_checks"] = {"status": "started"}
        write_manifest()
        try:
            check_architecture(model, device)
            synchronize()
            manifest["architecture_checks"] = {"status": "completed"}
        except Exception as exc:
            manifest["architecture_checks"] = {"status": "failed", "error": repr(exc)}
            raise
        write_manifest()
        for epoch in range(1, NUM_EPOCHS + 1):
            if time.monotonic() - started >= CATEGORY_SECONDS - max(
                reserve_seconds, 2 * evaluation_seconds
            ):
                break
            model.train()
            total_loss, updates = 0.0, 0
            complete = True
            epoch_started = time.monotonic()
            for _ in range(NUM_BATCHES):
                if time.monotonic() - started + max(
                    1.0, 2 * step_seconds
                ) >= CATEGORY_SECONDS - max(reserve_seconds, 2 * evaluation_seconds):
                    complete = False
                    break
                step_started = time.monotonic()
                seq, pos, neg = sample_batch(
                    user_train,
                    itemnum,
                    BATCH_SIZE,
                    MAXLEN,
                    users=users,
                    sample_negatives=not configuration["gbce"],
                )
                seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
                pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
                valid = pos_t.ne(0)
                if not valid.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                if configuration["gbce"]:
                    features = model.log2feats(seq_t)[valid]
                    positive_ids = pos_t[valid]
                    negative_ids = torch.randint(
                        1,
                        itemnum + 1,
                        (positive_ids.numel(), NEGATIVE_COUNT),
                        device=device,
                    )
                    loss = generalized_bce(model, features, positive_ids, negative_ids)
                    # One device-to-host transfer replaces per-row GPU kernels
                    # and synchronizations. Every replacement draw is counted.
                    drawn_ids = negative_ids.detach().cpu().numpy()
                    positive_array = pos[pos != 0]
                    manifest["current_positive_collisions"] += int(
                        np.count_nonzero(drawn_ids == positive_array[:, None])
                    )
                    manifest["negative_draws"] += negative_ids.numel()
                    offset = 0
                    for row, history_ids in enumerate(sample_batch.last_histories):
                        row_count = int(np.count_nonzero(pos[row]))
                        if row_count:
                            known = np.fromiter(set(history_ids), dtype=np.int64)
                            manifest["known_history_collisions"] += int(
                                np.count_nonzero(
                                    np.isin(
                                        drawn_ids[offset : offset + row_count],
                                        known,
                                        assume_unique=False,
                                    )
                                )
                            )
                            offset += row_count
                    if offset != drawn_ids.shape[0]:
                        raise RuntimeError("Collision accounting lost batch alignment")
                else:
                    neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
                    positive_logits, negative_logits = model(seq_t, pos_t, neg_t)
                    loss = (
                        torch.nn.functional.softplus(-positive_logits[valid]).mean()
                        + torch.nn.functional.softplus(negative_logits[valid]).mean()
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss: {configuration['name']}/{dataset_name}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach())
                updates += 1
                synchronize()
                step_seconds = max(step_seconds, time.monotonic() - step_started)
            if not complete:
                # Partial epochs are charged but cannot become selected checkpoints.
                if completed:
                    model.load_state_dict(
                        torch.load(latest_path, map_location=device, weights_only=True)
                    )
                break
            if not updates:
                raise RuntimeError("Epoch contained no valid positive positions")
            completed = epoch
            torch.save(model.state_dict(), latest_path)
            manifest["completed_epochs"] = completed
            manifest["epochs"].append(
                {
                    "epoch": epoch,
                    "loss": total_loss / updates,
                    "updates": updates,
                    "wall_seconds": time.monotonic() - epoch_started,
                }
            )
            write_manifest()
            if epoch == 1 or epoch % 10 == 0:
                evaluate_checkpoint(epoch)
        if not completed:
            raise RuntimeError("Category cap reached before completing an epoch")
        if last_evaluated != completed:
            evaluate_checkpoint(completed)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.eval()
        # DEBUG: Execute the diagnostic adapter on the actual selected model.
        # Keep diagnostics inside this category's existing charged time window.
        from types import SimpleNamespace
        from diagnosis_tools import DiagnosisProbe

        def diagnostic_scores(features, items):
            embeddings = model.item_emb(items)
            return (features.unsqueeze(-2) * embeddings).sum(-1)

        def diagnostic_batches():
            # Preserve the original single-negative diagnostic protocol even
            # when optimization used multiple replacement negatives.
            diagnostic_users = [
                u
                for u in users
                if len({i for i in user_train[u] if 1 <= i <= itemnum}) < itemnum
            ]
            if not diagnostic_users:
                return
            for _ in range(10):
                if time.monotonic() >= diagnostic_deadline:
                    return
                sequence, positive, negative = sample_batch(
                    user_train,
                    itemnum,
                    BATCH_SIZE,
                    MAXLEN,
                    users=diagnostic_users,
                    sample_negatives=True,
                )
                yield (
                    None,
                    torch.as_tensor(sequence, dtype=torch.long, device=device),
                    torch.as_tensor(positive, dtype=torch.long, device=device),
                    torch.as_tensor(negative, dtype=torch.long, device=device),
                )

        diagnostic_deadline = min(
            started + CATEGORY_SECONDS - 15.0,
            time.monotonic() + 120.0,
        )
        view = SimpleNamespace(
            model=model,
            item_emb=model.item_emb,
            log2feats=model.log2feats,
            predict_score=diagnostic_scores,
            user_train=user_train,
            # No permitted text was loaded by this training script. Report that
            # evidence as unavailable rather than substituting invented metadata.
            meta_data={},
            review_data={},
            dataset_name=dataset_name,
            maxlen=MAXLEN,
            diagnostic_deadline=diagnostic_deadline,
        )
        manifest["diagnosis"] = {"status": "started"}
        write_manifest()
        diagnosis = DiagnosisProbe(view, device).run_full_diagnosis(
            diagnostic_batches()
        )
        synchronize()
        diagnosis_path = os.path.join(working_dir, f"{dataset_name}_diagnosis.json")
        with open(diagnosis_path, "w") as f:
            json.dump(diagnosis, f, indent=2, allow_nan=False)
        manifest["diagnosis"] = {
            "status": "completed",
            "path": diagnosis_path,
            "budget_exhausted": diagnosis["metrics"]["diagnostic_budget_exhausted"],
        }
        write_manifest()
        meta = {
            "itemnum": itemnum,
            "hidden_units": HIDDEN_UNITS,
            "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS,
            "dropout_rate": DROPOUT_RATE,
            "maxlen": MAXLEN,
            "pooling_mode": configuration["pooling"],
            "configuration": configuration["name"],
            "selected_epoch": best_epoch,
        }
        with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        manifest.update(
            best_hr10=best_hr10, selected_epoch=best_epoch, status="completed"
        )
        print(
            f"[{configuration['name']}/{dataset_name}] HR@10={best_hr10:.4f} epoch={best_epoch}",
            flush=True,
        )
        return best_hr10
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        synchronize()
        write_manifest()


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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

### >>> Self_EvolveRec-BLOCK-START: Keep all four inference routes active
DATASETS = list(DEFAULT_DATASETS)
_models: dict = {}
_metas: dict = {}
_device = None
### <<< Self_EvolveRec-BLOCK-END


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
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained residual architecture
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            pooling_mode=meta.get("pooling_mode", "none"),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject a missing explicit dataset route
    if dataset_name is not None:
        if dataset_name not in _models:
            raise KeyError(f"Missing trained dataset route: {dataset_name}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
    ### <<< Self_EvolveRec-BLOCK-END
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

### >>> Self_EvolveRec-BLOCK-START: Compare six routed configurations and freeze a reload-verified winner


def main():
    import importlib.util
    import shutil
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    DATASETS[:] = DEFAULT_DATASETS
    results = []
    for order, configuration in enumerate(CONFIGURATIONS):
        configuration_dir = os.path.join(
            working_dir, "configurations", configuration["name"]
        )
        os.makedirs(configuration_dir, exist_ok=True)
        per_category, failures = {}, {}
        begin = time.monotonic()
        for ds in DEFAULT_DATASETS:
            try:
                per_category[ds] = train_one_dataset(
                    ds, device, DATA_DIR, configuration, configuration_dir
                )
            except Exception as exc:
                failures[ds] = repr(exc)
                print(f"[{configuration['name']}/{ds}] failed: {exc!r}", flush=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        complete = len(per_category) == len(DEFAULT_DATASETS) and not failures
        result = {
            "configuration": configuration,
            "order": order,
            "category_hr10": per_category,
            "failures": failures,
            "macro_hr10": sum(per_category.values()) / 4 if complete else None,
            "charged_wall_seconds": time.monotonic() - begin,
            "directory": configuration_dir,
        }
        results.append(result)
        with open(os.path.join(working_dir, "search_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    eligible = [r for r in results if r["macro_hr10"] is not None]
    if not eligible:
        raise RuntimeError(
            "No configuration completed all four categories; see search_results.json"
        )
    winner = min(
        eligible,
        key=lambda r: (
            -r["macro_hr10"],
            r["configuration"]["components"],
            r["charged_wall_seconds"],
            r["order"],
        ),
    )
    for ds in DEFAULT_DATASETS:
        for suffix in ("model.pt", "meta.pkl"):
            filename = f"{ds}_{suffix}"
            shutil.copyfile(
                os.path.join(winner["directory"], filename),
                os.path.join(working_dir, filename),
            )
    write_predict_script(working_dir)

    # The generated predictor imports train.py, reconstructs the serialized
    # architecture and must agree with independently reloaded selected models.
    spec = importlib.util.spec_from_file_location(
        "verified_trial_predictor", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    for ds in DEFAULT_DATASETS:
        with open(os.path.join(working_dir, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        reference = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            pooling_mode=meta["pooling_mode"],
        ).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{ds}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.eval()
        histories, _, _ = load_dataset(ds, DATA_DIR)
        user = next(u for u, history in histories.items() if history)
        history = histories[user][-meta["maxlen"] :]
        candidates = list(range(1, min(meta["itemnum"], 16) + 1))
        sequence = torch.tensor(
            [[0] * (meta["maxlen"] - len(history)) + history],
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            expected = reference.predict(
                sequence, torch.tensor([candidates], device=device)
            )[0].cpu()
        actual = torch.tensor(
            predictor.predict(user, history, candidates, dataset_name=ds)
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        del reference
    frozen = {
        "configuration": winner["configuration"],
        "category_hr10": winner["category_hr10"],
        "macro_hr10": winner["macro_hr10"],
        "predictor_reload_agreement": True,
        "selection": "category checkpoints first; equal-weight four-category macro second",
        "tie_rules": [
            "fewer added components",
            "lower charged wall time",
            "declared order",
        ],
        "training_robustness": "single seed; not measured",
        "held_out_metrics": "unavailable",
    }
    with open(os.path.join(working_dir, "frozen_configuration.json"), "w") as f:
        json.dump(frozen, f, indent=2)
    print(
        f"Selected {winner['configuration']['name']}: macro validation HR@10={winner['macro_hr10']:.4f}",
        flush=True,
    )


### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


