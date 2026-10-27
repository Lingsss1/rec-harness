# === main_code.py ===
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
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "300"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-provided data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
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


### >>> Self_EvolveRec-BLOCK-START: Reuse eligible training users without changing sampling probabilities
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, users=None
):
    if users is None:
        users = [
            u
            for u, items in user_train.items()
            if len(items) >= 2 and len(set(items)) < itemnum
        ]
    if not users:
        raise ValueError(
            "No training users have both a next-item pair and an unseen negative"
        )
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
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in ts:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


### >>> Self_EvolveRec-BLOCK-START: Sparse training-only directed transition residual
# Fixed fusion does not fit parameters using graph-derived training examples.
# Complementary user-fold graphs are required before adding a learned gate;
# this experiment deliberately keeps SASRec training and its loss unchanged.
FUSION_ALPHAS = (0.0, 0.1, 0.3, 1.0)


def build_transition_graph(user_train, itemnum):
    frequency = np.zeros(itemnum + 1, dtype=np.int64)
    edge_parts = []
    stride = itemnum + 1
    for history in user_train.values():
        items = np.asarray(history, dtype=np.int64)
        if items.size:
            np.add.at(frequency, items, 1)
        if items.size >= 2:
            edge_parts.append(items[:-1] * stride + items[1:])
    if edge_parts:
        keys, counts = np.unique(np.concatenate(edge_parts), return_counts=True)
    else:
        keys = np.empty(0, dtype=np.int64)
        counts = np.empty(0, dtype=np.int64)
    frequency[0] = 0
    return {
        "keys": torch.from_numpy(keys),
        "counts": torch.from_numpy(counts.astype(np.float32)),
        "frequency": torch.from_numpy(frequency),
        "window": 3,
        "rho": 0.7,
        "shrink": 10.0,
    }


class TransitionResidual(nn.Module):
    def __init__(
        self,
        itemnum,
        keys,
        counts,
        frequency,
        window=3,
        rho=0.7,
        shrink=10.0,
        alpha=0.0,
    ):
        super().__init__()
        self.itemnum = int(itemnum)
        self.window = int(window)
        self.rho = float(rho)
        self.alpha = float(alpha)
        keys = keys.long()
        counts = counts.float()
        frequency = frequency.float().clone()
        frequency[0] = 0.0
        if frequency.sum().item() == 0:
            frequency[1:] = 1.0
        p0 = frequency / frequency.sum().clamp_min(1.0)
        outgoing = torch.zeros(itemnum + 1, dtype=torch.float32)
        if keys.numel():
            outgoing.scatter_add_(0, keys // (itemnum + 1), counts)
        denominator = outgoing + float(shrink)
        self.register_buffer("keys", keys)
        self.register_buffer("correction", counts / denominator[keys // (itemnum + 1)])
        self.register_buffer("prior_coefficient", float(shrink) / denominator)
        self.register_buffer("p0", p0)

    def forward(self, sequence, candidates):
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(sequence.shape[0], -1)
        length = sequence.shape[1]
        width = min(self.window, length)
        if width == 0:
            q = self.p0[candidates]
        else:
            # Gather the last W nonpadding items, including prefixes with gaps.
            positions = torch.arange(length, device=sequence.device).expand_as(sequence)
            positions = positions.masked_fill(sequence == 0, -1)
            recent = positions.topk(width, dim=1, largest=True, sorted=True).values
            valid = recent >= 0
            anchors = sequence.gather(1, recent.clamp_min(0))
            weights = self.rho ** torch.arange(
                width, device=sequence.device, dtype=torch.float32
            )
            weights = weights.unsqueeze(0) * valid
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            prior_mass = (weights * self.prior_coefficient[anchors]).sum(1)
            prior_mass = torch.where(
                valid.any(1), prior_mass, torch.ones_like(prior_mass)
            )
            q = prior_mass.unsqueeze(1) * self.p0[candidates]
            if self.keys.numel():
                for k in range(width):
                    query = anchors[:, k : k + 1] * (self.itemnum + 1) + candidates
                    index = torch.searchsorted(self.keys, query.contiguous())
                    safe_index = index.clamp_max(self.keys.numel() - 1)
                    matched = (index < self.keys.numel()) & (
                        self.keys[safe_index] == query
                    )
                    q = (
                        q
                        + weights[:, k : k + 1] * matched * self.correction[safe_index]
                    )
        # An item's score never depends on other supplied candidates.
        residual = torch.log1p(q * float(self.itemnum))
        return residual.masked_fill(candidates == 0, 0.0)


### <<< Self_EvolveRec-BLOCK-END

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
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
    ):
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

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

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

        return self.last_layernorm(seq_emb)

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

    ### >>> Self_EvolveRec-BLOCK-START: Shared deployment scoring for fixed transition fusion
    def set_transition(self, graph_state, alpha=0.0):
        self.transition = TransitionResidual(
            self.itemnum, alpha=alpha, **graph_state
        ).to(self.item_emb.weight.device)

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(candidates)
        scores = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        transition = getattr(self, "transition", None)
        if transition is not None and transition.alpha != 0.0:
            scores = scores + transition.alpha * transition(seq, candidates)
        return scores

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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible validation without perturbing training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives for every fusion coefficient
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError("Validation requires 99 unique unseen negative items")
            while len(neg_pool) < 99:
                ni = rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            ### <<< Self_EvolveRec-BLOCK-END
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
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)

    ### >>> Self_EvolveRec-BLOCK-START: Prepare eligible users once and prevent saturated-history sampling loops
    sampling_users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not sampling_users:
        raise ValueError(f"{dataset_name}: no eligible next-item training users")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Reject empty training schedules and initialize a fresh trial checkpoint
    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError("SASREC_EPOCHS and SASREC_BATCHES must both be positive")
    best_hr10 = -1.0
    best_epoch = 0
    torch.save(model.state_dict(), ckpt_path)
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke cached-user sampling in every training step
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, users=sampling_users
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
            indices = pos_t != 0
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the final epoch for short training schedules
        should_validate = epoch % 5 == 0 or epoch == NUM_EPOCHS
        ### <<< Self_EvolveRec-BLOCK-END
        if should_validate:
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

    ### >>> Self_EvolveRec-BLOCK-START: Evaluate fixed residuals using one frozen baseline checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    graph_state = build_transition_graph(user_train, itemnum)
    graph_path = os.path.join(working_dir, f"{dataset_name}_transition.pt")
    torch.save(graph_state, graph_path)
    model.set_transition(graph_state)
    fusion_metrics = {}
    for alpha in FUSION_ALPHAS:
        model.transition.alpha = alpha
        metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
        fusion_metrics[alpha] = metrics
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "experiment": "fixed_directed_transition",
                    "alpha": alpha,
                    "window": graph_state["window"],
                    "rho": graph_state["rho"],
                    "shrink": graph_state["shrink"],
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
    model.transition.alpha = 0.0
    with open(
        os.path.join(working_dir, f"{dataset_name}_fusion_validation.json"), "w"
    ) as f:
        json.dump(fusion_metrics, f, indent=2)
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
            },
            f,
        )

    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    ### >>> Self_EvolveRec-BLOCK-START: Defer coefficient selection until all four categories are evaluated
    return fusion_metrics
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

_models: dict = {}
_metas: dict = {}
_device = None


### >>> Self_EvolveRec-BLOCK-START: Preserve evaluator randomness when importing the seeded training module
def _load_model_class():
    import random
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        spec = importlib.util.spec_from_file_location(
            f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the trained SASRec architecture from train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SASRec
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Require complete artifacts for every dataset route
        for required_path in (meta_path, ckpt_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(required_path)
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Restore the selected transition residual alongside the neural checkpoint
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        graph_path = os.path.join(_HERE, f"{ds}_transition.pt")
        alpha = float(meta.get("transition_alpha", 0.0))
        if os.path.isfile(graph_path):
            graph_state = torch.load(graph_path, map_location="cpu", weights_only=True)
            m.set_transition(graph_state, alpha=alpha)
        elif alpha != 0.0:
            raise FileNotFoundError(graph_path)
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes while retaining the legacy default
    if not _models:
        raise RuntimeError("No trained dataset models were loaded")
    ds = dataset_name if dataset_name is not None else next(iter(_models))
    if ds not in _models:
        raise ValueError(f"Unknown dataset route: {ds!r}")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Exact itemwise fusion across complete candidate lists in bounded chunks
    if not candidates:
        return []
    with torch.no_grad():
        # Encode once; every chunk uses the same prefix and itemwise residual.
        final_feat = model.log2feats(seq_t)[:, -1, :]
        transition = getattr(model, "transition", None)
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor([candidates[start:start + 4096]], dtype=torch.long, device=_device)
            chunk_scores = model.item_emb(cand_t).matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            if transition is not None and transition.alpha != 0.0:
                chunk_scores = chunk_scores + transition.alpha * transition(seq_t, cand_t)
            scores.extend(chunk_scores[0].cpu().tolist())
    return scores
    ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Select one shared coefficient by four-category macro validation HR@10
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError("This matched task requires all four Amazon datasets")
    results = {}
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        results[ds] = train_one_dataset(ds, device, DATA_DIR)

    # Integer hit totals avoid floating-point differences in mathematical ties.
    macro = {
        alpha: sum(results[ds][alpha]["HR@10"] for ds in DATASETS) / len(DATASETS)
        for alpha in FUSION_ALPHAS
    }
    selected_alpha = FUSION_ALPHAS[0]
    for alpha in FUSION_ALPHAS[1:]:
        if macro[alpha] > macro[selected_alpha] + 1e-12:
            selected_alpha = alpha
    for ds in DATASETS:
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta["transition_alpha"] = selected_alpha
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
    with open(os.path.join(working_dir, "fusion_selection.json"), "w") as f:
        json.dump(
            {
                "experiment": "fixed_directed_transition",
                "alphas": list(FUSION_ALPHAS),
                "macro_validation_hr10": macro,
                "selected_alpha": selected_alpha,
                "per_dataset": results,
            },
            f,
            indent=2,
        )
    write_predict_script(working_dir)
    print(
        f"\nAll datasets done. Selected alpha={selected_alpha:g}; "
        f"average val HR@10={macro[selected_alpha]:.4f}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Retain the compatibility view and evaluate the actual deployed module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual deployed nn.Module")
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._residual_pairs = []

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute directional concentration without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)

            # Sum of all pairwise dot products equals the squared norm
            # of the vector sum. Subtract self-pairs, including zero vectors.
            total = x.sum(dim=0, dtype=torch.float64)
            diagonal = x.double().square().sum()
            mean_cos = (
                ((total.square().sum() - diagonal) / (k * (k - 1))).clamp(-1, 1).item()
            )
            return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic consumption and require explicit residual-off semantics
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        self._residual_pairs = []
        margins_all = []
        base_scorer = getattr(self.model, "predict_score_base", None)
        if base_scorer is not None and not callable(base_scorer):
            raise TypeError("view.predict_score_base must be callable when provided")
        feature_layout = getattr(self.model, "feature_layout", None)
        if feature_layout is None:
            feature_layout = getattr(self.module, "feature_layout", None)
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Reject malformed batches without consuming an extra diagnostic batch
        for step, batch in enumerate(islice(dataloader, num_batches)):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    f"Diagnostic batch {step} must contain (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned diagnostic tensors before model scoring
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative tensors must share shape B x L"
                )
            log_feats = self.model.log2feats(seq_t)
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError(
                    "view.log2feats must return aligned B x L x D features"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "view.predict_score must return aligned B x L x 2 scores"
                )
            margin = all_logits[..., 0].float() - all_logits[..., 1].float()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Compare residual-off scores only through an explicit scoring contract
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            if valid.any():
                if (
                    not torch.isfinite(all_logits[valid]).all()
                    or not torch.isfinite(margin[valid]).all()
                ):
                    raise ValueError("Nonfinite deployed diagnostic scores or margins")
                margins_all.append(margin[valid].detach().float().cpu())
                base_logits = None
                if base_scorer is not None:
                    base_logits = base_scorer(log_feats, all_items)
                elif feature_layout == "sasrec_recent_concat":
                    dim = self.model.item_emb.weight.shape[-1]
                    if log_feats.shape[-1] != 2 * dim:
                        raise ValueError(
                            "sasrec_recent_concat requires exactly 2D features"
                        )
                    base_logits = (
                        log_feats[..., :dim].unsqueeze(-2)
                        * self.model.item_emb(all_items)
                    ).sum(-1)
                if base_logits is not None:
                    if base_logits.shape != all_logits.shape:
                        raise ValueError(
                            "Residual-off scores must share the deployed B x L x 2 shape"
                        )
                    if not torch.isfinite(base_logits[valid]).all():
                        raise ValueError("Nonfinite residual-off diagnostic scores")
                    self._residual_pairs.append(
                        torch.stack([base_logits[valid], all_logits[valid]], dim=1)
                        .detach()
                        .float()
                        .cpu()
                    )
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical valid positions for margins and breakdown counts
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Missing diagnostic samples are unavailable rather than healthy zeros
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Restore deployed mode and report evidence-qualified paired residual diagnostics
    def run_full_diagnosis(self, train_loader):
        # Preserve mixed submodule modes as well as the root module's mode.
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            with torch.no_grad():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=eps, topk_categories=20
                    )
                )
        finally:
            for module, training in prior_modes:
                module.training = training

        paired = {
            "status": "unavailable",
            "reason": "No valid paired observations with view.predict_score_base or explicit feature_layout='sasrec_recent_concat'",
        }
        if self._residual_pairs:
            scores = torch.cat(self._residual_pairs, dim=0)
            # Axis 1: residual off / deployed; axis 2: positive / negative.
            outranks = scores[..., 1] > scores[..., 0]
            ties = scores[..., 1] == scores[..., 0]
            paired = {
                "status": "available",
                "candidate_protocol": "Identical supplied positive/negative pairs; positive-first stable ties",
                "positions": int(scores.shape[0]),
                "residual_off_mean_positive_rank": float(
                    (1 + outranks[:, 0].long()).float().mean()
                ),
                "deployed_mean_positive_rank": float(
                    (1 + outranks[:, 1].long()).float().mean()
                ),
                "residual_off_negative_outranking_fraction": float(
                    outranks[:, 0].float().mean()
                ),
                "deployed_negative_outranking_fraction": float(
                    outranks[:, 1].float().mean()
                ),
                "residual_off_tie_fraction": float(ties[:, 0].float().mean()),
                "deployed_tie_fraction": float(ties[:, 1].float().mean()),
                "corrected_pairs": int((outranks[:, 0] & ~outranks[:, 1]).sum()),
                "regressed_pairs": int((~outranks[:, 0] & outranks[:, 1]).sum()),
                "mean_absolute_score_change": float(
                    (scores[:, 1] - scores[:, 0]).abs().mean()
                ),
            }
        self._residual_pairs = []

        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "residual_fixed_candidate_comparison": paired,
                "unavailable_observations": {
                    "item_categories": None,
                    "reviews_and_ratings": None,
                    "item_prices_and_stores": None,
                    "product_specifications": None,
                    "simulator_diversity_and_conformity_traits": None,
                },
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Mean sampled pairwise embedding cosine mapped to [0,1]; approximately 0.5 "
                    "means mean cosine zero. Measures directional concentration only, not recommendation "
                    "diversity, satisfaction, or simulator traits. Zero vectors use F.normalize semantics; "
                    "null means fewer than two catalog embeddings."
                ),
                "pos_neg_margin_mean": (
                    "Mean deployed positive-minus-negative score on valid training positions. "
                    "Depends on score scale; null means no valid observations."
                ),
                "pos_neg_margin_p10": "Empirical lower tenth percentile of deployed training-pair margins; scale dependent.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid supplied training pairs where the negative strictly outscores the positive.",
                "pos_neg_small_margin_rate": "Fraction of deployed training-pair margins below eps=0.5; scale dependent and not calibrated across models.",
                "pos_neg_margin_category_breakdown": (
                    "Counts valid low-margin and wrong-rank positions. Category lists remain empty "
                    "because category observations are unavailable."
                ),
                "residual_fixed_candidate_comparison": (
                    "Paired comparison on identical supplied positive/negative training candidates. "
                    "Residual-off scores use view.predict_score_base when provided, or the first D features "
                    "dotted with raw candidate embeddings when feature_layout='sasrec_recent_concat' is explicitly declared. "
                    "Feature width alone does not establish scoring semantics. Deployed scores use view.predict_score. "
                    "Positive rank is 1 or 2 with positive-first ties. These training-pair "
                    "diagnostics are not pinned validation HR@10 or full-catalog ranks and do not select coefficients."
                ),
                "unavailable_observations": (
                    "Null denotes unavailable evidence. Histories and embedding similarity do not establish "
                    "ownership, dislike, satisfaction, compatibility, complementarity, redundancy, current need, "
                    "or simulator traits."
                ),
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
