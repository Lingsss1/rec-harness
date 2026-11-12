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
### >>> Self_EvolveRec-BLOCK-START: Honor the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Explicit data boundary and reproducible intervention controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]
VARIANT = os.environ.get("SASREC_VARIANT", "combined")
if VARIANT not in {"baseline", "objective_only", "residual_only", "combined"}:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
USE_RESIDUAL = VARIANT in {"residual_only", "combined"}
USE_CANDIDATE_CE = VARIANT in {"objective_only", "combined"}
RECENT_WINDOW = 3
NUM_NEGATIVES = 32
CE_TEMPERATURE = 1.0
CE_CHUNK_SIZE = 512
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four dataset routes
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


### >>> Self_EvolveRec-BLOCK-START: Exact user-aware uniform sampling and memory-safe candidate CE
class TrainingSampler:
    def __init__(self, user_train, itemnum, candidate_ce=False):
        self.user_train = user_train
        self.itemnum = itemnum
        self.candidate_ce = candidate_ce
        self.users = [u for u, history in user_train.items() if len(history) >= 2]
        if not self.users:
            raise ValueError("Training requires at least one history of length two")
        self.seen = {
            u: {int(i) for i in user_train[u] if 1 <= int(i) <= itemnum}
            for u in self.users
        }
        self.dense_pools = {}
        self.skipped_positions = 0
        self.user_rng = random.Random(42)
        self.negative_rng = np.random.default_rng(42)
        required = NUM_NEGATIVES if candidate_ce else 1
        for user in self.users:
            available = self.itemnum - len(self.seen[user])
            if available < required:
                raise ValueError(
                    f"Training user {user} has {available} eligible negatives; "
                    f"the configured objective requires {required}"
                )

    def state_dict(self):
        return {
            "user_rng": self.user_rng.getstate(),
            "negative_rng": self.negative_rng.bit_generator.state,
        }

    def _draw(self, user, count):
        excluded = self.seen[user]
        available = self.itemnum - len(excluded)
        if available < count:
            raise ValueError(f"User {user} has fewer than {count} eligible negatives")
        # Dense histories use an explicit eligible set. Otherwise preserve the
        # order of accepted uniform draws; sorting before truncation is biased.
        if available <= 4 * count or len(excluded) * 2 > self.itemnum:
            if user not in self.dense_pools:
                self.dense_pools[user] = np.asarray(
                    [i for i in range(1, self.itemnum + 1) if i not in excluded],
                    dtype=np.int32,
                )
            pool = self.dense_pools[user]
            return pool[
                self.negative_rng.choice(len(pool), count, replace=False)
            ].tolist()
        selected, selected_set = [], set()
        for _ in range(8):
            if len(selected) == count:
                break
            draws = self.negative_rng.integers(
                1, self.itemnum + 1, size=max(16, 2 * (count - len(selected)))
            )
            for value in draws:
                value = int(value)
                if value not in excluded and value not in selected_set:
                    selected.append(value)
                    selected_set.add(value)
                    if len(selected) == count:
                        break
        if len(selected) < count:
            remaining = np.asarray(
                [
                    i
                    for i in range(1, self.itemnum + 1)
                    if i not in excluded and i not in selected_set
                ],
                dtype=np.int32,
            )
            indices = self.negative_rng.choice(
                len(remaining), count - len(selected), replace=False
            )
            selected.extend(remaining[indices].tolist())
        return selected

    def sample(self, batch_size, maxlen):
        batch_seq = np.zeros((batch_size, maxlen), dtype=np.int32)
        batch_pos = np.zeros_like(batch_seq)
        neg_shape = (
            (batch_size, maxlen, NUM_NEGATIVES)
            if self.candidate_ce
            else batch_seq.shape
        )
        batch_neg = np.zeros(neg_shape, dtype=np.int32)
        # Keep identities aligned internally; diagnostic tuples remain unchanged.
        self.batch_users = []
        for row in range(batch_size):
            user = self.user_rng.choice(self.users)
            self.batch_users.append(user)
            history = self.user_train[user]
            nxt = history[-1]
            idx = maxlen - 1
            excluded = self.seen[user]
            for item in reversed(history[:-1]):
                batch_seq[row, idx] = item
                batch_pos[row, idx] = nxt
                if self.candidate_ce:
                    negatives = self._draw(user, NUM_NEGATIVES)
                    batch_neg[row, idx, : len(negatives)] = negatives
                    if not negatives:
                        self.skipped_positions += 1
                elif len(excluded) < self.itemnum:
                    # Both objectives use the same complete-training-history
                    # exclusion and uniform eligible-item distribution.
                    batch_neg[row, idx] = self._draw(user, 1)[0]
                else:
                    self.skipped_positions += 1
                nxt = item
                idx -= 1
                if idx < 0:
                    break
        return batch_seq, batch_pos, batch_neg


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    # Keep the original B x L diagnostic-loader sampling contract.
    return TrainingSampler(user_train, itemnum).sample(batch_size, maxlen)


def backward_candidate_ce(model, seq, positive, negative):
    valid = (positive != 0) & negative.ne(0).any(dim=-1)
    count = int(valid.sum().item())
    if count == 0:
        return None
    features = model.log2feats(seq)[valid]
    # Accumulate gradients at compact leaves, then backpropagate through both
    # tied-embedding paths together. Avoid a catalog-sized embedding gradient
    # allocation and accumulation for every scoring chunk.
    feature_leaf = features.detach().requires_grad_(True)
    positive = positive[valid]
    negative = negative[valid]
    all_candidates = torch.cat((positive[:, None], negative), dim=1)
    unique_ids, inverse = torch.unique(
        all_candidates.reshape(-1), sorted=True, return_inverse=True
    )
    candidate_indices = inverse.reshape_as(all_candidates)
    unique_embeddings = model.item_emb(unique_ids)
    embedding_leaf = unique_embeddings.detach().requires_grad_(True)
    total_loss = features.new_zeros(())
    for start in range(0, count, CE_CHUNK_SIZE):
        stop = min(start + CE_CHUNK_SIZE, count)
        candidates = torch.cat(
            (positive[start:stop, None], negative[start:stop]), dim=1
        )
        negative_ids = candidates[:, 1:]
        invalid = (negative_ids == 0) | (negative_ids == candidates[:, :1])
        # Defensive duplicate masking; the sampler normally guarantees uniqueness.
        ordered, order = negative_ids.sort(dim=1)
        duplicate_sorted = torch.zeros_like(ordered, dtype=torch.bool)
        duplicate_sorted[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
        duplicate = torch.zeros_like(duplicate_sorted).scatter(
            1, order, duplicate_sorted
        )
        invalid = invalid | duplicate
        logits = (
            torch.einsum(
                "bd,bkd->bk",
                feature_leaf[start:stop],
                embedding_leaf[candidate_indices[start:stop]],
            ).float()
            / CE_TEMPERATURE
        )
        logits = torch.cat(
            (logits[:, :1], logits[:, 1:].masked_fill(invalid, -torch.inf)), dim=1
        )
        loss = (
            nn.functional.cross_entropy(
                logits,
                torch.zeros(stop - start, dtype=torch.long, device=seq.device),
                reduction="sum",
            )
            / count
        )
        loss.backward()
        total_loss.add_(loss.detach())
    torch.autograd.backward(
        (features, unique_embeddings),
        (feature_leaf.grad, embedding_leaf.grad),
    )
    return float(total_loss)


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
    ### >>> Self_EvolveRec-BLOCK-START: Persisted directional recent-history architecture
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        use_recent_residual: bool = False,
        recent_window: int = 3,
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

        ### >>> Self_EvolveRec-BLOCK-START: Context gate and independent directional low-rank projections
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.use_recent_residual = bool(use_recent_residual)
        self.recent_window = int(recent_window)
        self.residual_rank = max(1, hidden_units // 4)
        self.last_recent_gate = None
        if self.recent_window < 1:
            raise ValueError("recent_window must be positive")
        if self.use_recent_residual:
            self.recent_source = nn.Linear(hidden_units, self.residual_rank, bias=False)
            self.recent_target = nn.Linear(hidden_units, self.residual_rank, bias=False)
            self.context_gate = nn.Linear(2 * hidden_units + 1, 1)
            nn.init.xavier_uniform_(self.recent_source.weight)
            nn.init.xavier_uniform_(self.recent_target.weight)
            nn.init.zeros_(self.context_gate.weight)
            nn.init.constant_(self.context_gate.bias, -2.0)
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid computing unused attention weights
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

        ### >>> Self_EvolveRec-BLOCK-START: Causal context-gated directional residual with compatible feature dimensions
        features = self.last_layernorm(seq_emb)
        mask = (~timeline_mask).unsqueeze(-1).to(features.dtype)
        if self.use_recent_residual:
            embeddings = self.item_emb(seq) * mask
            positions = torch.arange(M, device=seq.device)[None, :].expand(B, -1)
            latest = torch.where(~timeline_mask, positions, -1).cummax(dim=1).values
            cursor = latest
            recent_sum = torch.zeros_like(embeddings)
            recent_count = torch.zeros_like(mask)
            # Follow previous observed events rather than previous tensor slots.
            # This remains causal with left padding and with internal zero gaps.
            for _ in range(self.recent_window):
                present = (cursor >= 0).unsqueeze(-1)
                gathered = embeddings.gather(
                    1,
                    cursor.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.hidden_units),
                )
                recent_sum = recent_sum + gathered * present
                recent_count = recent_count + present.to(mask.dtype)
                previous = latest.gather(1, (cursor - 1).clamp_min(0))
                cursor = torch.where(cursor > 0, previous, -1)
            recent = recent_sum / recent_count.clamp_min(1.0)
            observed = mask.cumsum(dim=1).log1p()
            gate = self.context_gate(
                torch.cat((features, recent, observed), dim=-1)
            ).sigmoid()
            self.last_recent_gate = gate.detach()[~timeline_mask]
            source = gate * self.recent_source(recent)
            # Exact reassociation:
            # dot(source, target(e)) == dot(source @ target.weight, e).
            # Returning d-dimensional effective features preserves existing
            # diagnostic view.log2feats / predict_score dot-product contracts.
            # Both projections receive gradients through this feature path.
            features = features + source.matmul(self.recent_target.weight)
        return features * mask
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

    ### >>> Self_EvolveRec-BLOCK-START: Explicit paired and candidate-set scoring for prediction and diagnostics
    def predict_score(
        self,
        features: torch.Tensor,
        items: torch.Tensor,
        *,
        candidate_set: bool = False,
    ) -> torch.Tensor:
        # Default: one aligned item per feature when the leading shapes match.
        # candidate_set=True disambiguates a shared catalog whose length happens
        # to equal the batch size. Candidate sets may be shared or per feature.
        embeddings = self.item_emb(items)
        if not candidate_set and items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            return features.matmul(embeddings.transpose(0, 1))
        if items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError(
            f"Incompatible scoring shapes: features={tuple(features.shape)}, "
            f"items={tuple(items.shape)}; use shared candidate IDs or "
            "one candidate set per feature"
        )

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        final_feat = self.log2feats(seq)[:, -1, :]
        return self.predict_score(final_feat, candidates, candidate_set=True)

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
    ### >>> Self_EvolveRec-BLOCK-START: Fixed validation randomness without perturbing training
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    was_training = model.training
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
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic validation candidates with explicit feasibility check
            available = itemnum - len({i for i in interacted if 1 <= i <= itemnum})
            if available < 99:
                raise ValueError(
                    f"User {u} has fewer than 99 eligible validation negatives"
                )
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's model mode after validation
    model.train(was_training)
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }
    ### <<< Self_EvolveRec-BLOCK-END


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    ### >>> Self_EvolveRec-BLOCK-START: Reset training seeds independently for each category
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    sampler = TrainingSampler(user_train, itemnum, candidate_ce=USE_CANDIDATE_CE)
    print(
        f"  [{dataset_name}] users={len(user_train)}  items={itemnum}  variant={VARIANT}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the residual in its designated experimental arms
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        use_recent_residual=USE_RESIDUAL,
        recent_window=RECENT_WINDOW,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Reject empty training and prevent stale checkpoint reuse
    if NUM_EPOCHS < 1 or NUM_BATCHES < 1 or BATCH_SIZE < 1:
        raise ValueError("Epochs, batches per epoch, and batch size must be positive")
    best_hr10 = -float("inf")
    best_epoch = 0
    # Overwrite artifacts from any earlier invocation of this trial directory.
    torch.save(model.state_dict(), ckpt_path)
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Chunked uncorrected candidate-set CE with one optimizer step
            seq, pos, neg = sampler.sample(BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if USE_CANDIDATE_CE:
                loss_value = backward_candidate_ce(model, seq_t, pos_t, neg_t)
                if loss_value is None:
                    continue
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                indices = (pos_t != 0) & (neg_t != 0)
                if not indices.any():
                    continue
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                )
                loss = loss + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )
                loss.backward()
                loss_value = float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss_value
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Fixed twenty-epoch validation schedule including the final epoch
        validate_now = epoch % 20 == 0 or epoch == NUM_EPOCHS
        ### <<< Self_EvolveRec-BLOCK-END
        if validate_now:
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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed training horizon without validation-patience stopping
            # All four objective/residual arms use the same epoch horizon.
            # Local validation is a checkpoint proxy; the host reports the
            # independently evaluated pinned benchmark selection metric.
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    ### >>> Self_EvolveRec-BLOCK-START: Save final resumable training state separately from selected inference weights
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_local_hr10": best_hr10,
            "variant": VARIANT,
            "dataset": dataset_name,
        },
        os.path.join(working_dir, f"{dataset_name}_training_state.pt"),
    )
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                ### >>> Self_EvolveRec-BLOCK-START: Record architecture and resolved training intervention
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
                "use_recent_residual": USE_RESIDUAL,
                "recent_window": RECENT_WINDOW,
                "residual_architecture": "context_gated_directional_low_rank_v1",
                "residual_rank": max(1, HIDDEN_UNITS // 4),
                "gate_bias_initialization": -2.0,
                "feature_contract": "d-dimensional exact effective dot-product features",
                "fitted_recent_gate": None,
                "variant": VARIANT,
                "training_seed": 42,
                "preprocessing_seed": 0,
                "loss": (
                    "uncorrected_candidate_set_ce"
                    if USE_CANDIDATE_CE
                    else "baseline_bce"
                ),
                "num_negatives": NUM_NEGATIVES if USE_CANDIDATE_CE else 1,
                "temperature": CE_TEMPERATURE,
                "ce_chunk_size": CE_CHUNK_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "adam_betas": (0.9, 0.98),
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "epoch_cap": NUM_EPOCHS,
                "best_epoch": best_epoch,
                "skipped_training_positions": sampler.skipped_positions,
                "item_id_mapping": "unchanged public split integer IDs; padding=0",
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Verify selected checkpoint reconstruction and score parity
    selected_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state)
    model.eval()
    reloaded = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        maxlen=MAXLEN,
        use_recent_residual=USE_RESIDUAL,
        recent_window=RECENT_WINDOW,
    ).to(device)
    reloaded.load_state_dict(selected_state)
    reloaded.eval()
    probe_user = sampler.users[0]
    probe_history = user_train[probe_user][-MAXLEN:]
    probe_sequence = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
    probe_sequence[0, -len(probe_history) :] = torch.tensor(
        probe_history, device=device
    )
    probe_candidates = torch.arange(1, min(itemnum, 100) + 1, device=device)[None, :]
    with torch.no_grad():
        torch.testing.assert_close(
            model.predict(probe_sequence, probe_candidates),
            reloaded.predict(probe_sequence, probe_candidates),
            rtol=1e-5,
            atol=1e-6,
        )
    del reloaded
    gate_values = model.last_recent_gate
    gate_mean = (
        float(gate_values.mean())
        if gate_values is not None and gate_values.numel()
        else None
    )
    print(
        f"  [{dataset_name}] Best local val HR@10={best_hr10:.4f}"
        f"  probe_context_gate_mean={gate_mean}"
        f"  skipped={sampler.skipped_positions}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10
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
### >>> Self_EvolveRec-BLOCK-START: Load all four trained routes in the generated predictor
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
    ### >>> Self_EvolveRec-BLOCK-START: Require complete predictor initialization and explicit artifacts
    if len(_models) == len(DATASETS):
        return
    for ds in DATASETS:
        for filename in (f"{ds}_meta.pkl", f"{ds}_model.pt"):
            path = os.path.join(_HERE, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing required dataset artifact: {path}")
    _models.clear()
    _metas.clear()
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly if artifacts disappear during loading
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model or metadata for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the context-gated directional architecture at inference
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            use_recent_residual=meta.get("use_recent_residual", False),
            recent_window=meta.get("recent_window", 3),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Prevent explicit dataset requests from silently using another category
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
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


