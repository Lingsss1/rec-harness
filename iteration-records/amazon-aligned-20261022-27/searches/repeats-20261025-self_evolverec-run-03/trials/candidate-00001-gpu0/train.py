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
### >>> Self_EvolveRec-BLOCK-START: Match the benchmark epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Explicit architecture and objective ablation controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]
# Each host trial runs one cell across all four categories. These controls do
# not launch additional training runs or claim that the ablation has been run.
ABLATION_CELLS = {
    "baseline_bce": (False, "bce"),
    "residual_bce": (True, "bce"),
    "baseline_softmax": (False, "candidate_softmax"),
    "residual_softmax": (True, "candidate_softmax"),
}
ABLATION_CELL = os.environ.get("SASREC_CELL", "residual_softmax")
if ABLATION_CELL not in ABLATION_CELLS:
    raise ValueError(f"Unknown SASREC_CELL: {ABLATION_CELL}")
USE_RECENT_RESIDUAL, TRAIN_OBJECTIVE = ABLATION_CELLS[ABLATION_CELL]
RECENT_WINDOW = 5
TRAIN_NEGATIVES = 64 if TRAIN_OBJECTIVE == "candidate_softmax" else 1
SOFTMAX_CHUNK_SIZE = 512
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four independent dataset routes
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


### >>> Self_EvolveRec-BLOCK-START: Exact uniform training-history exclusion with independent sampler streams
class TrainingSampler:
    def __init__(self, user_train, itemnum, seed=42):
        self.user_train = user_train
        self.itemnum = itemnum
        self.user_rng = np.random.default_rng(seed)
        self.negative_rng = np.random.default_rng(seed + 1)
        self.users = []
        self.exclusions = {}
        self.skipped_no_eligible = 0
        for user, history in user_train.items():
            blocked = np.asarray(sorted(set(history)), dtype=np.int64)
            if blocked.size and (blocked[0] < 1 or blocked[-1] > itemnum):
                raise ValueError("Training item outside category-local item IDs")
            available = itemnum - len(blocked)
            if available == 0:
                self.skipped_no_eligible += 1
                continue
            if len(history) < 2:
                continue
            self.users.append(user)
            # Invert the complement's ranks without allocating a full catalogue
            # per user. Repeated boundaries are intentional for adjacent IDs.
            boundaries = blocked - 1 - np.arange(len(blocked), dtype=np.int64)
            self.exclusions[user] = (available, boundaries)
        print(f"  sampling: skipped_no_eligible={self.skipped_no_eligible}", flush=True)
        if not self.users:
            raise ValueError(
                "No training users with transitions and eligible negatives"
            )

    def batch(self, batch_size, maxlen, num_negatives=1):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
        user_indices = self.user_rng.integers(len(self.users), size=batch_size)
        for row, user_index in enumerate(user_indices):
            user = self.users[int(user_index)]
            history = self.user_train[user]
            length = min(maxlen, len(history) - 1)
            seq[row, -length:] = history[-length - 1 : -1]
            pos[row, -length:] = history[-length:]
            available, boundaries = self.exclusions[user]
            ranks = self.negative_rng.integers(available, size=(length, num_negatives))
            neg[row, -length:, :] = (
                ranks + np.searchsorted(boundaries, ranks, side="right") + 1
            )
        if num_negatives == 1:
            return seq, pos, neg[..., 0]
        return seq, pos, neg


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    # Keep diagnostic streams separate from training, but advance them between
    # batches instead of recreating the same seed and returning identical data.
    cache = getattr(sample_batch, "_samplers", None)
    if cache is None:
        cache = {}
        sample_batch._samplers = cache
    key = (id(user_train), itemnum)
    if key not in cache:
        cache[key] = TrainingSampler(user_train, itemnum, seed=2042)
    return cache[key].batch(batch_size, maxlen)


def candidate_softmax_loss(model, seq, pos, neg):
    from torch.utils.checkpoint import checkpoint

    valid = pos != 0
    features = model.log2feats(seq)[valid]
    positive = pos[valid]
    negative = neg[valid]
    count = features.shape[0]
    if not count:
        raise ValueError("Training batch has no positive transitions")

    def chunk_loss(states, positive_ids, negative_ids, embedding_weight):
        candidates = torch.cat((positive_ids[:, None], negative_ids), dim=1)
        embeddings = torch.nn.functional.embedding(candidates, embedding_weight)
        # Avoid materializing an additional chunk-by-65-by-hidden product.
        logits = torch.bmm(embeddings, states.unsqueeze(-1)).squeeze(-1)
        return (torch.logsumexp(logits, dim=-1) - logits[:, 0]).sum()

    # Recompute candidate embeddings during backward instead of retaining a
    # B*L*65*d activation. Preserve duplicate draws and the exact position mean.
    loss = features.new_zeros(())
    for start in range(0, count, SOFTMAX_CHUNK_SIZE):
        end = start + SOFTMAX_CHUNK_SIZE
        loss = loss + checkpoint(
            chunk_loss,
            features[start:end],
            positive[start:end],
            negative[start:end],
            model.item_emb.weight,
            use_reentrant=False,
            # This checkpointed function contains no stochastic operations.
            preserve_rng_state=False,
        )
    return loss / count


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
        ### >>> Self_EvolveRec-BLOCK-START: Persistable recent-history architecture controls
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        use_recent_residual: bool = False,
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

        ### >>> Self_EvolveRec-BLOCK-START: Small vector-gated recent-history residual
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.use_recent_residual = use_recent_residual
        self.recent_window = int(recent_window)
        if self.recent_window < 1:
            raise ValueError("recent_window must be positive")
        if use_recent_residual:
            # Residual construction must not shift the backbone's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.recent_norm = nn.LayerNorm(hidden_units, eps=1e-8)
                self.recent_projection = nn.Linear(
                    hidden_units, hidden_units, bias=False
                )
                self.recent_gate = nn.Linear(2 * hidden_units, hidden_units)
                nn.init.eye_(self.recent_projection.weight)
                nn.init.zeros_(self.recent_gate.weight)
                nn.init.constant_(self.recent_gate.bias, -3.0)
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
            attn_out, _ = self.attention_layers[i](
                queries, seq_emb, seq_emb, attn_mask=attention_mask
            )
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        ### >>> Self_EvolveRec-BLOCK-START: Causal padding-masked raw embedding window mean
        states = self.last_layernorm(seq_emb)
        if not self.use_recent_residual:
            return states
        valid = (~timeline_mask).unsqueeze(-1)
        raw = self.item_emb(seq) * valid
        cumulative = torch.cat(
            (raw.new_zeros(B, 1, self.hidden_units), raw.cumsum(dim=1)), dim=1
        )
        counts = valid.to(raw.dtype)
        cumulative_counts = torch.cat(
            (counts.new_zeros(B, 1, 1), counts.cumsum(dim=1)), dim=1
        )
        ends = torch.arange(1, M + 1, device=seq.device)
        starts = (ends - self.recent_window).clamp_min(0)
        window_sum = cumulative[:, ends] - cumulative[:, starts]
        window_count = cumulative_counts[:, ends] - cumulative_counts[:, starts]
        recent = window_sum / window_count.clamp_min(1.0)
        recent = recent * valid
        gate = torch.sigmoid(self.recent_gate(torch.cat((states, recent), dim=-1)))
        residual = gate * self.recent_projection(self.recent_norm(recent))
        return states + residual * valid
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
    ### >>> Self_EvolveRec-BLOCK-START: Stable local validation randomness independent of training
    validation_rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = validation_rng.sample(val_users, num_users)
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound unique validation candidate sampling
            if itemnum - len(interacted) < 99:
                raise ValueError("Fewer than 99 eligible validation negatives")
            while len(neg_pool) < 99:
                ni = validation_rng.randint(1, itemnum)
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore validation caller mode
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
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    ### >>> Self_EvolveRec-BLOCK-START: Paired initialization and sampling for each category and ablation cell
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Input train/validation splits are consumed unchanged; no preprocessing here.
    sampler = TrainingSampler(user_train, itemnum, seed=42)
    print(
        f"  cell={ABLATION_CELL} negatives={TRAIN_NEGATIVES} "
        f"checkpoints=every_5_epochs max_epochs={NUM_EPOCHS}; "
        "checkpoint ties retain the earlier epoch; local validation is provisional",
        flush=True,
    )
    model_config = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "use_recent_residual": USE_RECENT_RESIDUAL,
        "recent_window": RECENT_WINDOW,
    }
    model = SASRec(**model_config).to(device)
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even at zero hits
    best_hr10 = -1.0
    best_epoch = 0
    validation_history = []
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Release previous gradients before constructing the next loss
            optimizer.zero_grad(set_to_none=True)
            seq, pos, neg = sampler.batch(BATCH_SIZE, MAXLEN, TRAIN_NEGATIVES)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            if TRAIN_OBJECTIVE == "candidate_softmax":
                loss = candidate_softmax_loss(model, seq_t, pos_t, neg_t)
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                indices = pos_t != 0
                pos_labels = torch.ones_like(pos_logits[indices])
                neg_labels = torch.zeros_like(neg_logits[indices])
                loss = bce_criterion(pos_logits[indices], pos_labels)
                loss = loss + bce_criterion(neg_logits[indices], neg_labels)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Backpropagate after gradients were cleared before forward
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the final epoch including runs shorter than five epochs
        evaluate_checkpoint = epoch % 5 == 0 or epoch == NUM_EPOCHS
        ### <<< Self_EvolveRec-BLOCK-END
        if evaluate_checkpoint:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Record every inspected local validation checkpoint
            hr10 = val_m["HR@10"]
            validation_history.append({"epoch": epoch, **val_m})
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
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    ### >>> Self_EvolveRec-BLOCK-START: Reject runs without a newly selected checkpoint
    if best_epoch == 0:
        raise RuntimeError(
            "No checkpoint selected in this run; require positive SASREC_EPOCHS "
            "and finite validation HR@10. Existing checkpoints cannot be reused."
        )
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Persist architecture and verify trained-checkpoint reconstruction
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    # Training-derived inputs only; do not consume the training sampler's streams.
    verify_sampler = TrainingSampler(user_train, itemnum, seed=1042)
    verify_seq, verify_pos, verify_neg = verify_sampler.batch(2, MAXLEN)
    verify_seq = torch.as_tensor(verify_seq, dtype=torch.long, device=device)
    verify_candidates = torch.as_tensor(
        np.stack((verify_pos[:, -1], verify_neg[:, -1]), axis=1),
        dtype=torch.long,
        device=device,
    )
    # Reconstruction must not perturb subsequent category initialization streams.
    with torch.random.fork_rng(
        devices=[device.index or 0] if device.type == "cuda" else []
    ):
        restored = SASRec(**model_config).to(device)
    restored.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    restored.eval()
    with torch.no_grad():
        reference_scores = model.predict(verify_seq, verify_candidates)
        restored_scores = restored.predict(verify_seq, verify_candidates)
        torch.testing.assert_close(
            reference_scores, restored_scores, rtol=1e-5, atol=1e-6
        )
    del restored

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                **model_config,
                "architecture_version": "sasrec_gated_recent_v1",
                "cell": ABLATION_CELL,
                "objective": TRAIN_OBJECTIVE,
                "num_negatives": TRAIN_NEGATIVES,
                "gate_initial_bias": -3.0,
                "projection_initialization": "identity_bias_free",
                "recent_normalization_eps": 1e-8,
                "training_seed": 42,
                "sampler_user_seed": 42,
                "sampler_negative_seed": 43,
                "preprocessing_seed": 0,
                "item_mapping": "identity_category_local_input_ids_1_through_itemnum",
                "optimizer": {
                    "name": "Adam",
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "betas": (0.9, 0.98),
                },
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "max_epochs": NUM_EPOCHS,
                "best_epoch": best_epoch,
                "skipped_no_eligible": sampler.skipped_no_eligible,
                "validation_history": validation_history,
                "selection_note": "Local checkpoint validation; host macro HR@10 selects trials",
            },
            f,
        )
    ### <<< Self_EvolveRec-BLOCK-END

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
    ### >>> Self_EvolveRec-BLOCK-START: Require all four trained category routes
    for ds in DEFAULT_DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing trained artifacts for {ds}")
    ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved gated residual architecture at inference
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            use_recent_residual=meta.get("use_recent_residual", False),
            recent_window=meta.get("recent_window", 5),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Enforce explicit dataset routing while preserving the legacy signature
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset route: {dataset_name}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    else:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    if not candidates:
        return []
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


