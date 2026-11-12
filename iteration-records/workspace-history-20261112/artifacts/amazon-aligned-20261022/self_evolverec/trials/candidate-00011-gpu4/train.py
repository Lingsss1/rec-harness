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

### >>> Self_EvolveRec-BLOCK-START: Track bounded pooling trials and checkpoint selection
import random
import time
import shutil
import gc
from collections import defaultdict

### <<< Self_EvolveRec-BLOCK-END

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
### >>> Self_EvolveRec-BLOCK-START: Match the host training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host dataset directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four dataset routes and prespecify pooling comparisons
DATASETS = list(DEFAULT_DATASETS)
POOLING_TRIALS = ("baseline", "max", "mean2", "second")
TRIAL_SECONDS = 10800.0
SEARCH_SECONDS = 43200.0

random.seed(42)
### <<< Self_EvolveRec-BLOCK-END
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


### >>> Self_EvolveRec-BLOCK-START: Cache sampling metadata while preserving baseline sampling draws
# DEBUG: Replace the undefined import-time placeholder with the sampler used by training.
def sample_batch(
    user_train,
    itemnum,
    batch_size,
    maxlen,
    sampling_state=None,
):
    if itemnum < 1 or batch_size < 1 or maxlen < 1:
        raise ValueError("itemnum, batch_size and maxlen must be positive")
    if sampling_state is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
        rated_by_user = {u: set(user_train[u]) for u in users}
    else:
        users, rated_by_user = sampling_state
    if not users:
        raise ValueError("No training users have at least two interactions")

    sequences = np.zeros((batch_size, maxlen), dtype=np.int32)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)

    for row in range(batch_size):
        user = int(np.random.choice(users))
        history = user_train[user]
        rated = rated_by_user[user]
        # DEBUG: Prevent rejection sampling from hanging on a saturated catalog.
        if len(rated) >= itemnum and all(i in rated for i in range(1, itemnum + 1)):
            raise ValueError(f"Training user {user} has no unseen negative items")

        next_item = history[-1]
        position = maxlen - 1
        for item in reversed(history[:-1]):
            sequences[row, position] = item
            positives[row, position] = next_item
            negative = np.random.randint(1, itemnum + 1)
            while negative in rated:
                negative = np.random.randint(1, itemnum + 1)
            negatives[row, position] = negative
            next_item = item
            position -= 1
            if position < 0:
                break

    # DEBUG: Match the three B x L arrays expected by train_one_dataset.
    return sequences, positives, negatives


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
        ### >>> Self_EvolveRec-BLOCK-START: Configure a learned nonnegative recent-match residual
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        pooling: str = "second",
    ):
        super().__init__()
        if pooling not in ("baseline", "max", "mean2", "second"):
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
        self.pooling = pooling
        self.residual_gate = nn.Parameter(torch.tensor(-3.0))
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
            attn_out, _ = self.attention_layers[i](
                queries, seq_emb, seq_emb, attn_mask=attention_mask
            )
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        return self.last_layernorm(seq_emb)

    ### >>> Self_EvolveRec-BLOCK-START: Share causal distinct-item cosine scoring across training and inference
    def recent_context(
        self, history_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ):
        """Return normalized raw embeddings and masks for positional windows.

        Windows contain at most sixteen positions, including the query position.
        Duplicate removal keeps the latest occurrence without backfilling.
        """
        if history_ids.ndim != 2 or history_ids.shape[1] == 0:
            raise ValueError("history_ids must have shape B x L with L > 0")
        length = history_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(length, device=history_ids.device)
        position_ids = torch.as_tensor(
            position_ids, dtype=torch.long, device=history_ids.device
        ).reshape(-1)
        if bool(((position_ids < 0) | (position_ids >= length)).any()):
            raise ValueError("Causal positions must index the supplied history")
        offsets = torch.arange(-15, 1, device=history_ids.device)
        indices = position_ids[:, None] + offsets[None, :]
        ids = history_ids[:, indices.clamp_min(0)]
        valid = (indices >= 0).unsqueeze(0) & ids.ne(0)
        later = torch.triu(
            torch.ones(16, 16, device=history_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        duplicated_later = (
            ids.unsqueeze(-1).eq(ids.unsqueeze(-2)) & valid.unsqueeze(-2) & later
        ).any(dim=-1)
        valid = valid & ~duplicated_later
        # Normalize once per source position instead of once per overlapping window.
        # Gathering preserves gradients through every selected history occurrence.
        source_embeddings = torch.nn.functional.normalize(
            self.item_emb(history_ids), dim=-1, eps=1e-8
        )
        embeddings = source_embeddings[:, indices.clamp_min(0)]
        return embeddings, valid

    def shared_score(
        self,
        features: torch.Tensor,
        items: torch.Tensor,
        history_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        recent_state=None,
    ) -> torch.Tensor:
        """Score aligned training targets or final-state candidate matrices.

        B x L x D features pair with B x L targets or B x L x C targets.
        B x D final features pair with shared C or per-user B x C candidates.
        recent_state is an explicit request-local cache, never mutable model state.
        """
        final_only = features.ndim == 2
        if final_only:
            features = features.unsqueeze(1)
            if items.ndim == 1:
                items = items.unsqueeze(0).expand(features.shape[0], -1)
            if items.ndim != 2 or items.shape[0] != features.shape[0]:
                raise ValueError("Final candidates must have shape C or B x C")
            items = items.unsqueeze(1)
            aligned = False
        else:
            if features.ndim != 3:
                raise ValueError("Features must have shape B x D or B x L x D")
            aligned = items.ndim == 2
            if aligned:
                items = items.unsqueeze(-1)
            if items.ndim != 3 or items.shape[:2] != features.shape[:2]:
                raise ValueError(
                    "Sequence candidates must align with feature positions"
                )

        candidate_embeddings = self.item_emb(items)
        scores = (features.unsqueeze(-2) * candidate_embeddings).sum(dim=-1)
        if self.pooling != "baseline":
            if recent_state is None:
                if history_ids is None:
                    raise ValueError("Residual scoring requires explicit history_ids")
                if position_ids is None and final_only:
                    position_ids = torch.tensor(
                        [history_ids.shape[1] - 1], device=history_ids.device
                    )
                recent_state = self.recent_context(history_ids, position_ids)
            recent_embeddings, valid = recent_state
            if recent_embeddings.shape[:2] != features.shape[:2]:
                raise ValueError("Recent history must align with feature positions")
            candidate_unit = torch.nn.functional.normalize(
                candidate_embeddings, dim=-1, eps=1e-8
            )
            similarities = torch.einsum(
                "btcd,btwd->btcw", candidate_unit, recent_embeddings
            )
            similarities = similarities.masked_fill(~valid.unsqueeze(-2), float("-inf"))
            largest_two = similarities.topk(2, dim=-1).values
            enough = valid.sum(dim=-1).ge(2).unsqueeze(-1).unsqueeze(-1)
            # Replace invalid order statistics before arithmetic, avoiding infinities
            # in both forward values and backward paths for short histories.
            largest_two = torch.where(
                enough, largest_two, torch.zeros_like(largest_two)
            )
            if self.pooling == "max":
                residual = largest_two[..., 0]
            elif self.pooling == "mean2":
                residual = largest_two.mean(dim=-1)
            else:
                residual = largest_two[..., 1]
            scores = (
                scores + torch.nn.functional.softplus(self.residual_gate) * residual
            )
        if final_only:
            return scores[:, 0, :]
        return scores.squeeze(-1) if aligned else scores

    def predict_score(
        self,
        features: torch.Tensor,
        items: torch.Tensor,
        history_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        recent_state=None,
    ) -> torch.Tensor:
        return self.shared_score(
            features, items, history_ids, position_ids, recent_state
        )

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        features = self.log2feats(seq)
        if pos is None or neg is None:
            return features
        recent = None if self.pooling == "baseline" else self.recent_context(seq)
        return (
            self.shared_score(features, pos, history_ids=seq, recent_state=recent),
            self.shared_score(features, neg, history_ids=seq, recent_state=recent),
        )

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        features = self.log2feats(seq)[:, -1, :]
        return self.predict_score(features, candidates, history_ids=seq)

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
    ### >>> Self_EvolveRec-BLOCK-START: Make validation candidates deterministic without changing training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Fail clearly when unique validation negatives are impossible
            neg_pool: set[int] = set()
            interacted = set(user_train[u]) | set(pos_items)
            available = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available < 99:
                raise ValueError(
                    f"Validation user {u} has only {available} unseen candidates; "
                    "the pinned protocol requires 99 unique negatives"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Keep validation sampling local to evaluation seed 42
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


### >>> Self_EvolveRec-BLOCK-START: Yield synchronized validation checkpoints for macro selection
class TrialBudgetExceeded(RuntimeError):
    pass


def check_budget(deadline: float):
    if time.monotonic() >= deadline:
        raise TrialBudgetExceeded("Trial wall-time budget exhausted")


def train_one_dataset(
    dataset_name: str,
    device: torch.device,
    data_dir: str,
    pooling: str,
    deadline: float,
):
    """Yield every ten epochs and at the configured final epoch.

    The coordinator advances every category to the same checkpoint before
    selecting anything. No category-specific best checkpoints are combined.
    """
    check_budget(deadline)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError(f"No trainable histories for {dataset_name}")
    sampling_state = (users, {u: set(user_train[u]) for u in users})
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        pooling=pooling,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()
    meta = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "pooling": pooling,
    }
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            check_budget(deadline)
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                sampling_state=sampling_state,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            positive, negative = model(seq_t, pos_t, neg_t)
            valid = pos_t.ne(0)
            loss = bce_criterion(positive[valid], torch.ones_like(positive[valid]))
            loss = loss + bce_criterion(
                negative[valid], torch.zeros_like(negative[valid])
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            check_budget(deadline)
            metrics = evaluate_val(
                model, user_train, user_valid, itemnum, device, MAXLEN
            )
            check_budget(deadline)
            print(
                json.dumps(
                    {
                        "pooling": pooling,
                        "dataset": dataset_name,
                        "epoch": epoch,
                        "loss": total_loss / NUM_BATCHES,
                        "validation": metrics,
                        "gate": float(
                            torch.nn.functional.softplus(model.residual_gate).detach()
                        ),
                    }
                ),
                flush=True,
            )
            # Suspended dataset generators otherwise retain dense gradient buffers.
            optimizer.zero_grad(set_to_none=True)
            yield epoch, model, meta, metrics


def verify_saved_model(dataset_name, directory, device, deadline):
    """Check checkpoint reload and candidate chunking on a public train history."""
    check_budget(deadline)
    with open(os.path.join(directory, f"{dataset_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    # Verification must not alter subsequent training initialization or dropout.
    # SASRec initializes on CPU, so only the CPU RNG needs to be forked here.
    with torch.random.fork_rng(devices=[]):
        model = SASRec(**meta).to(device).eval()
    model.load_state_dict(
        torch.load(
            os.path.join(directory, f"{dataset_name}_model.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    histories = _read_split(os.path.join(DATA_DIR, f"{dataset_name}_train.txt"))
    history = next(items for items in histories.values() if items)
    seq = torch.zeros(1, meta["maxlen"], dtype=torch.long, device=device)
    tail = history[-meta["maxlen"] :]
    seq[0, -len(tail) :] = torch.tensor(tail, device=device)
    ids = torch.arange(1, min(meta["itemnum"], 4097) + 1, device=device)
    with torch.no_grad():
        direct = model.predict(seq, ids.unsqueeze(0))
        features = model.log2feats(seq)[:, -1, :]
        recent = (
            None
            if model.pooling == "baseline"
            else model.recent_context(
                seq, torch.tensor([seq.shape[1] - 1], device=device)
            )
        )
        chunked = torch.cat(
            [
                model.predict_score(
                    features, chunk.unsqueeze(0), history_ids=seq, recent_state=recent
                )
                for chunk in ids.split(4096)
            ],
            dim=-1,
        )
        torch.testing.assert_close(direct, chunked, rtol=1e-4, atol=1e-5)
    return seq, ids, direct.cpu()


### <<< Self_EvolveRec-BLOCK-END


# DEBUG: Connect the diagnostic adapter to real checkpoint models and public
# training batches. No reviews or unavailable item attributes are fabricated.
def diagnose_checkpoint(dataset_name, model, device, directory, deadline):
    from types import SimpleNamespace
    from diagnosis_tools import DiagnosisProbe

    check_budget(deadline)
    user_train, _, itemnum = load_dataset(dataset_name, DATA_DIR)
    users = [u for u, items in user_train.items() if len(items) >= 2]
    sampling_state = (users, {u: set(user_train[u]) for u in users})

    def batches():
        for _ in range(10):
            check_budget(deadline)
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                model.maxlen,
                sampling_state=sampling_state,
            )
            yield None, seq, pos, neg

    view = SimpleNamespace(
        model=model,
        item_emb=model.item_emb,
        log2feats=model.log2feats,
        predict_score=model.predict_score,
        user_train=user_train,
        dataset_name=dataset_name,
        diagnosis_deadline=deadline,
        # This training loader does not load title/description metadata.
        # The audit explicitly reports lexical evidence as unavailable.
        meta_data={},
        review_data={},
    )
    result = DiagnosisProbe(view, device).run_full_diagnosis(batches())
    with open(os.path.join(directory, f"{dataset_name}_diagnosis.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    check_budget(deadline)


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
### >>> Self_EvolveRec-BLOCK-START: Keep all four predictor routes consistent with training
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
    # DEBUG: Publish loaded routes only after all four models load successfully;
    # otherwise a failed load leaves a partial cache that bypasses later retries.
    loaded_models = {}
    loaded_metas = {}
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Report incomplete predictor artifacts explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint or metadata for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        loaded_metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            ### >>> Self_EvolveRec-BLOCK-START: Restore the trained pooling architecture and gate
            maxlen=meta["maxlen"],
            pooling=meta["pooling"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        loaded_models[ds] = m
    # DEBUG: Commit the complete dataset routing cache atomically.
    _metas.update(loaded_metas)
    _models.update(loaded_models)


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit dataset routes instead of scoring with another model
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f"Unknown or unloaded dataset route: {ds}")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Reuse final features and recent embeddings across candidate chunks
    if not candidates:
        return []
    with torch.no_grad():
        features = model.log2feats(seq_t)[:, -1, :]
        recent = None if model.pooling == "baseline" else model.recent_context(
            seq_t, torch.tensor([maxlen - 1], device=_device)
        )
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor(
                [candidates[start:start + 4096]], dtype=torch.long, device=_device
            )
            values = model.predict_score(
                features, cand_t, history_ids=seq_t, recent_state=recent
            )
            scores.extend(values[0].cpu().tolist())
    return scores
    ### <<< Self_EvolveRec-BLOCK-END
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written → {predict_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────

### >>> Self_EvolveRec-BLOCK-START: Compare four pooling trials using complete macro checkpoints and fixed ties


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    search_started = time.monotonic()
    search_deadline = search_started + SEARCH_SECONDS
    records = []
    winner = None
    winner_score = -float("inf")
    winner_epoch = None

    def persist_records():
        with open(os.path.join(working_dir, "search_log.json"), "w") as f:
            json.dump(
                {
                    "selection_metric": "macro validation HR@10",
                    "trial_order": list(POOLING_TRIALS),
                    "records": records,
                    "selected_pooling": winner,
                    "selected_epoch": winner_epoch,
                    "selected_macro_hr10": None if winner is None else winner_score,
                    "elapsed_wall_seconds": time.monotonic() - search_started,
                    "timing_policy": "single-GPU wall-time limits; includes CPU work",
                },
                f,
                indent=2,
            )

    for pooling in POOLING_TRIALS:
        if time.monotonic() >= search_deadline:
            break
        trial_started = time.monotonic()
        deadline = min(search_deadline, trial_started + TRIAL_SECONDS)
        trial_dir = os.path.join(working_dir, "trials", pooling)
        os.makedirs(trial_dir, exist_ok=True)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)
        generators = [
            train_one_dataset(ds, device, DATA_DIR, pooling, deadline)
            for ds in DATASETS
        ]
        checkpoint = []
        best_trial_score = -float("inf")
        # DEBUG: Run the bounded diagnostic once per category per trial, at
        # the first completed checkpoint, rather than leaving it unreachable.
        diagnosed_categories = set()
        try:
            while True:
                checkpoint = []
                for generator in generators:
                    check_budget(deadline)
                    checkpoint.append(next(generator))
                epochs = {entry[0] for entry in checkpoint}
                if len(epochs) != 1:
                    raise RuntimeError("Dataset checkpoint epochs do not agree")
                epoch = checkpoint[0][0]
                category_scores = {
                    ds: entry[3]["HR@10"] for ds, entry in zip(DATASETS, checkpoint)
                }
                # Equal hit totals must tie despite floating-point summation order.
                macro = round(sum(category_scores.values()) / len(DATASETS), 12)
                records.append(
                    {
                        "pooling": pooling,
                        "epoch": epoch,
                        "category_hr10": category_scores,
                        "macro_hr10": macro,
                        "trial_wall_seconds": time.monotonic() - trial_started,
                        "search_wall_seconds": time.monotonic() - search_started,
                    }
                )
                # Strict comparison retains the earliest exact within-trial tie.
                if macro > best_trial_score:
                    best_trial_score = macro
                    for ds, (_, model, meta, _) in zip(DATASETS, checkpoint):
                        check_budget(deadline)
                        model.eval()
                        path = os.path.join(trial_dir, f"{ds}_model.pt")
                        torch.save(model.state_dict(), path)
                        with open(os.path.join(trial_dir, f"{ds}_meta.pkl"), "wb") as f:
                            pickle.dump(meta, f)
                        # Compare the loaded model against the original trained module.
                        seq, ids, reloaded = verify_saved_model(
                            ds, trial_dir, device, deadline
                        )
                        with torch.no_grad():
                            original = model.predict(seq, ids.unsqueeze(0)).cpu()
                        torch.testing.assert_close(
                            original, reloaded, rtol=1e-4, atol=1e-5
                        )
                    # Trial order implements baseline/max/mean2/second exact ties.
                    if macro > winner_score:
                        for ds in DATASETS:
                            for suffix in ("model.pt", "meta.pkl"):
                                name = f"{ds}_{suffix}"
                                shutil.copyfile(
                                    os.path.join(trial_dir, name),
                                    os.path.join(working_dir, name),
                                )
                        winner = pooling
                        winner_score = macro
                        winner_epoch = epoch
                        write_predict_script(working_dir)
                persist_records()
                print(json.dumps(records[-1]), flush=True)
                # DEBUG: Persist the verified winner before spending the remaining
                # trial allowance on diagnosis. Probe failures remain logged by
                # the trial handler, and diagnosis restores training RNG/modes.
                for ds, (_, model, _, _) in zip(DATASETS, checkpoint):
                    if ds not in diagnosed_categories:
                        diagnose_checkpoint(ds, model, device, trial_dir, deadline)
                        diagnosed_categories.add(ds)
                if epoch == NUM_EPOCHS:
                    break
        except StopIteration:
            records.append({"pooling": pooling, "status": "finished"})
        except TrialBudgetExceeded as exc:
            records.append(
                {
                    "pooling": pooling,
                    "status": "budget_truncated",
                    "reason": str(exc),
                    "trial_wall_seconds": time.monotonic() - trial_started,
                    "partial_checkpoint_discarded": True,
                }
            )
        except Exception as exc:
            print(
                f"Pooling trial {pooling} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            records.append(
                {
                    "pooling": pooling,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "trial_wall_seconds": time.monotonic() - trial_started,
                }
            )
        finally:
            for generator in generators:
                generator.close()
            generators.clear()
            checkpoint.clear()
            # Drop loop-held references before starting the next independent trial.
            for name in ("model", "original", "seq", "ids", "reloaded"):
                if name == "model":
                    model = None
                elif name == "original":
                    original = None
                elif name == "seq":
                    seq = None
                elif name == "ids":
                    ids = None
                else:
                    reloaded = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            persist_records()
    if winner is None:
        raise RuntimeError(
            "No verified complete four-dataset checkpoint; see search_log.json"
        )
    print(
        f"Selected pooling={winner} epoch={winner_epoch} "
        f"macro validation HR@10={winner_score:.6f}",
        flush=True,
    )


### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


