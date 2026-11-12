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
### >>> Self_EvolveRec-BLOCK-START: Match the host training horizon
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Required data boundary and reproducible fusion controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]
FUSION = os.environ.get("SASREC_FUSION", "context")
OBJECTIVE = os.environ.get("SASREC_OBJECTIVE", "ce")
RECENT_WINDOW = 5
NEGATIVE_COUNT = 32
VALIDATION_INTERVAL = 10
if FUSION not in {"none", "fixed", "global", "context", "state"}:
    raise ValueError("Unsupported SASREC_FUSION")
if OBJECTIVE not in {"bce", "ce"}:
    raise ValueError("Unsupported SASREC_OBJECTIVE")
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


### >>> Self_EvolveRec-BLOCK-START: Exact uniform subset sampling with compact history exclusion
class TrainingSampler:
    def __init__(self, user_train, itemnum, seed=1042):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, items in user_train.items() if len(items) >= 2]
        if not self.users:
            raise ValueError("No users with two training interactions")
        self.excluded = {
            u: np.unique(np.asarray(user_train[u], dtype=np.int64)) for u in self.users
        }
        # Map uniform ranks in the eligible catalog back to actual item IDs.
        # Keep only valid catalog IDs in the exclusion mapping.
        self.excluded = {
            u: ids[(ids >= 1) & (ids <= itemnum)] for u, ids in self.excluded.items()
        }
        self.rank_thresholds = {
            u: ids - np.arange(len(ids), dtype=np.int64) - 1
            for u, ids in self.excluded.items()
        }
        self.user_rng = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.targets = self.reduced = self.skipped = 0

    def sample(self, batch_size, maxlen, num_negatives=1):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
        for b in range(batch_size):
            u = self.user_rng.choice(self.users)
            history = self.user_train[u]
            length = min(maxlen, len(history) - 1)
            seq[b, -length:] = history[-length - 1 : -1]
            pos[b, -length:] = history[-length:]
            excluded = self.excluded[u]
            # Targets already belong to this full observed training history.
            available = self.itemnum - int(
                ((excluded >= 1) & (excluded <= self.itemnum)).sum()
            )
            k = min(num_negatives, available)
            self.targets += length
            self.reduced += length * (k < num_negatives)
            if k == 0:
                self.skipped += length
                continue
            # Each row is an independent uniform subset of eligible catalog ranks.
            # The bijection below removes rejection loops and their collision bias.
            ranks = np.stack(
                [
                    self.rng.choice(available, size=k, replace=False)
                    for _ in range(length)
                ]
            )
            draws = (
                ranks
                + 1
                + np.searchsorted(self.rank_thresholds[u], ranks, side="right")
            )
            neg[b, -length:, :k] = draws
        return seq, pos, neg[..., 0] if num_negatives == 1 else neg


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    # Preserve the B x L negative interface used by compatibility loaders.
    return TrainingSampler(user_train, itemnum, seed=random.getrandbits(32)).sample(
        batch_size, maxlen
    )


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
        ### >>> Self_EvolveRec-BLOCK-START: Serialize residual architecture controls
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        fusion: str = "context",
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

        ### >>> Self_EvolveRec-BLOCK-START: Causal recent-history residual and matched gate initialization
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if fusion not in {"none", "fixed", "global", "context", "state"}:
            raise ValueError(f"Unknown fusion mode: {fusion}")
        self.fusion = fusion
        self.recent_window = int(recent_window)
        if self.recent_window < 1:
            raise ValueError("recent_window must be positive")
        if fusion != "none":
            decay = math.log(2.0) / 2.0
            self.recent_decay = nn.Parameter(torch.tensor(math.log(math.expm1(decay))))
        if fusion == "global":
            self.global_gate = nn.Parameter(torch.tensor(-2.0))
        if fusion in {"context", "state"}:
            self.recent_gate = nn.Sequential(
                nn.Linear(3 * hidden_units, hidden_units),
                nn.GELU(),
                nn.Linear(hidden_units, 1),
            )
            nn.init.zeros_(self.recent_gate[-1].weight)
            nn.init.constant_(self.recent_gate[-1].bias, -2.0)

    def _recent_features(self, seq, h):
        embeddings = self.item_emb(seq)
        numerator = torch.zeros_like(embeddings)
        denominator = embeddings.new_zeros((*seq.shape, 1))
        decay = torch.nn.functional.softplus(self.recent_decay)
        for lag in range(min(self.recent_window, seq.shape[1])):
            if lag == 0:
                shifted = embeddings
                valid = seq.ne(0).unsqueeze(-1)
            else:
                shifted = torch.nn.functional.pad(embeddings[:, :-lag], (0, 0, lag, 0))
                valid = torch.nn.functional.pad(
                    seq[:, :-lag].ne(0), (lag, 0)
                ).unsqueeze(-1)
            weight = torch.exp(-decay * lag) * valid.to(embeddings.dtype)
            numerator = numerator + shifted * weight
            denominator = denominator + weight
        raw = numerator / denominator.clamp_min(1e-8)
        h_norm = h.norm(dim=-1, keepdim=True).detach()
        raw_norm = raw.norm(dim=-1, keepdim=True)
        recent = raw * (h_norm / raw_norm.clamp_min(1e-8))
        if self.fusion == "fixed":
            gate = h.new_full((*seq.shape, 1), 1.0 / (1.0 + math.exp(2.0)))
        elif self.fusion == "global":
            gate = self.global_gate.sigmoid().expand(*seq.shape, 1)
        else:
            # State-only control has the same parameter count as the contextual gate.
            inputs = (h, recent, h * recent) if self.fusion == "context" else (h, h, h)
            gate = self.recent_gate(torch.cat(inputs, dim=-1)).sigmoid()
        return raw, recent, gate

    @torch.no_grad()
    def recent_diagnostics(self, seq):
        was_training = self.training
        self.eval()
        try:
            if self.fusion == "none":
                return {"metrics": {}, "metric_definitions": {}}
            h = self._sasrec_features(seq)
            raw, recent, gate = self._recent_features(seq, h)
            valid = seq.ne(0)
            if not valid.any():
                return {"metrics": {}, "metric_definitions": {}}
            hn = h.norm(dim=-1)[valid]
            rn = raw.norm(dim=-1)[valid]
            g = gate.squeeze(-1)[valid]
            metrics = {
                "state_norm_mean": hn.mean().item(),
                "recent_raw_norm_mean": rn.mean().item(),
                "recent_raw_to_state_norm_mean": (rn / hn.clamp_min(1e-8))
                .mean()
                .item(),
                "gate_mean": g.mean().item(),
                "gate_below_005_fraction": (g < 0.05).float().mean().item(),
                "gate_above_095_fraction": (g > 0.95).float().mean().item(),
            }
            definitions = {
                "state_norm_mean": "Mean pre-fusion SASRec L2 norm at nonpadding inputs.",
                "recent_raw_norm_mean": "Mean recent-pool L2 norm before norm matching at nonpadding inputs.",
                "recent_raw_to_state_norm_mean": "Mean pre-matching recent/state norm ratio; state norm denominator clamped at 1e-8.",
                "gate_mean": "Mean scalar residual gate at nonpadding inputs.",
                "gate_below_005_fraction": "Fraction of gates below 0.05; a diagnostic convention.",
                "gate_above_095_fraction": "Fraction of gates above 0.95; a diagnostic convention.",
            }
            return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            self.train(was_training)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        h = self._sasrec_features(seq)
        if self.fusion == "none":
            return h
        _, recent, gate = self._recent_features(seq, h)
        return (h + gate * recent) * seq.ne(0).unsqueeze(-1)

    def _sasrec_features(self, seq: torch.Tensor) -> torch.Tensor:
        ### <<< Self_EvolveRec-BLOCK-END
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
                queries, seq_emb, seq_emb, attn_mask=attention_mask, need_weights=False
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
    ### >>> Self_EvolveRec-BLOCK-START: Isolate deterministic validation randomness from training
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
                ### >>> Self_EvolveRec-BLOCK-START: Fixed local validation candidate stream
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
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    ### >>> Self_EvolveRec-BLOCK-START: Identical per-category initialization and explicit architecture settings
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        fusion=FUSION,
        recent_window=RECENT_WINDOW,
    ).to(device)
    sampler = TrainingSampler(user_train, itemnum)
    print(
        f"  [{dataset_name}] fusion={FUSION} objective={OBJECTIVE} "
        f"parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the earliest checkpoint even when validation hits are zero
    best_hr10 = -1.0
    best_epoch = 0
    validation_history = []
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Stable candidate-set cross-entropy including reduced negative counts
            count = NEGATIVE_COUNT if OBJECTIVE == "ce" else 1
            seq, pos, neg = sampler.sample(BATCH_SIZE, MAXLEN, count)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            eligible_target = (
                (neg_t != 0).any(dim=-1) if neg_t.ndim == 3 else neg_t.ne(0)
            )
            indices = pos_t.ne(0) & eligible_target
            if not indices.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            if OBJECTIVE == "bce":
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                )
                loss = loss + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )
            else:
                features = model.log2feats(seq_t)[indices]
                targets = pos_t[indices]
                negatives = neg_t[indices]
                positive = (features * model.item_emb(targets)).sum(-1).float()
                # Keep one joint denominator. Reducing an entirely masked chunk
                # separately can produce NaN gradients when k is smaller than 32.
                logit_chunks = [positive.unsqueeze(1)]
                for start in range(0, negatives.shape[1], 8):
                    candidates = negatives[:, start : start + 8]
                    logits = torch.einsum(
                        "nd,nkd->nk", features, model.item_emb(candidates)
                    ).float()
                    logit_chunks.append(
                        logits.masked_fill(candidates.eq(0), float("-inf"))
                    )
                candidate_logits = torch.cat(logit_chunks, dim=1)
                loss = torch.nn.functional.cross_entropy(
                    candidate_logits,
                    torch.zeros(len(targets), dtype=torch.long, device=device),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Fixed ten-epoch validation cadence
        if epoch % VALIDATION_INTERVAL == 0 or epoch == NUM_EPOCHS:
            ### <<< Self_EvolveRec-BLOCK-END
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Archive measured validation and sampling diagnostics
            hr10 = val_m["HR@10"]
            validation_history.append({"epoch": epoch, **val_m})
            with open(
                os.path.join(working_dir, f"{dataset_name}_validation.json"), "w"
            ) as f:
                json.dump(validation_history, f, indent=2)
            print(
                f"  [{dataset_name}] sampling_targets={sampler.targets} "
                f"reduced_k_fraction={sampler.reduced / max(sampler.targets, 1):.6f} "
                f"skipped_fraction={sampler.skipped / max(sampler.targets, 1):.6f}",
                flush=True,
            )
            if FUSION != "none":
                diagnostic = model.recent_diagnostics(seq_t)
                print(
                    f"  [{dataset_name}] recent_diagnostics={json.dumps(diagnostic)}",
                    flush=True,
                )
            ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Preserve the baseline patience measured in validation checks
            if epoch - best_epoch >= PATIENCE * VALIDATION_INTERVAL:
                ### <<< Self_EvolveRec-BLOCK-END
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    ### >>> Self_EvolveRec-BLOCK-START: Verify saved-model scoring parity and causal feature computation
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        probe_seq = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
        probe_history = next(items for items in user_train.values() if len(items) >= 2)
        visible = probe_history[-min(MAXLEN, len(probe_history)) :]
        probe_seq[0, -len(visible) :] = torch.as_tensor(visible, device=device)
        probe_candidates = torch.arange(
            1, min(itemnum, 16) + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        features = model.log2feats(probe_seq)
        direct_scores = torch.einsum(
            "bd,bkd->bk", features[:, -1], model.item_emb(probe_candidates)
        )
        torch.testing.assert_close(
            model.predict(probe_seq, probe_candidates),
            direct_scores,
            rtol=1e-4,
            atol=1e-5,
        )
        if MAXLEN > 1:
            cut = MAXLEN - 1
            changed = probe_seq.clone()
            changed[:, cut:] = changed[:, cut:] % itemnum + 1
            torch.testing.assert_close(
                model.log2feats(changed)[:, :cut],
                features[:, :cut],
                rtol=1e-4,
                atol=1e-5,
            )
    print(f"  [{dataset_name}] scoring parity and causality checks passed", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Persist architecture, objective, and resolved training settings
                "maxlen": MAXLEN,
                "fusion": FUSION,
                "recent_window": RECENT_WINDOW,
                "objective": OBJECTIVE,
                "negative_count": NEGATIVE_COUNT if OBJECTIVE == "ce" else 1,
                "negative_exclusion": "entire observed training history, including later training positions",
                "negative_sampling": "uniform unique without replacement; padding excluded",
                "importance_correction": False,
                "training_seed": 42,
                "negative_seed": 1042,
                "hidden_units_resolved": HIDDEN_UNITS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "Adam",
                "betas": (0.9, 0.98),
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "num_epochs": NUM_EPOCHS,
                "patience_checks": PATIENCE,
                "validation_interval": VALIDATION_INTERVAL,
                "best_epoch": best_epoch,
                "best_local_validation_hr10": best_hr10,
                "parameter_count": sum(p.numel() for p in model.parameters()),
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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

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
        ### >>> Self_EvolveRec-BLOCK-START: Require complete category artifacts in generated inference
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing metadata or checkpoint for {ds}")
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
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained recent-history architecture in generated inference
            maxlen=meta["maxlen"],
            fusion=meta["fusion"],
            recent_window=meta["recent_window"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on missing category models during generated inference
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        raise RuntimeError("No category checkpoints were loaded")
    ### <<< Self_EvolveRec-BLOCK-END

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


