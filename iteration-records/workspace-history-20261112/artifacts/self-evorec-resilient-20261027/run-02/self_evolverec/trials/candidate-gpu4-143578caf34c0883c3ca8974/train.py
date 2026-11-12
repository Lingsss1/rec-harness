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
### >>> Self_EvolveRec-BLOCK-START: Respect the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Configure recent-support fusion and sampled classification
DATA_DIR = os.environ["GAGC_DATA_DIR"]
# Each invocation executes one configuration; the host measures its validation score.
# Ablations are baseline, objective_only, fusion_only, and combined.
CONFIGURATION = os.environ.get("SASREC_CONFIGURATION", "combined")
if CONFIGURATION not in ("baseline", "objective_only", "fusion_only", "combined"):
    raise ValueError(f"Unknown configuration: {CONFIGURATION}")
USE_SAMPLED_CE = CONFIGURATION in ("objective_only", "combined")
USE_RECENT_FUSION = CONFIGURATION in ("fusion_only", "combined")
RECENT_WINDOW = 5
SAMPLED_NEGATIVES = 64
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


### >>> Self_EvolveRec-BLOCK-START: Uniform unique sequence-shared negatives from training-only exclusions
class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.itemnum = itemnum
        self.rng = random.Random(42)
        self.exclusions = {
            u: set(items) for u, items in user_train.items() if len(items) >= 2
        }
        self.users = [u for u, seen in self.exclusions.items() if len(seen) < itemnum]
        skipped = len(self.exclusions) - len(self.users)
        print(f"  Training sequences without eligible negatives: {skipped}", flush=True)
        if not self.users:
            raise ValueError("No training users have eligible negative items")
        self.eligible_cache = {}

    def negatives(self, user, count):
        seen = self.exclusions[user]
        count = min(count, self.itemnum - len(seen))
        chosen = set()
        result = []
        # Bound rejection work, including users with almost exhaustive histories.
        for _ in range(max(128, 8 * count)):
            item = self.rng.randint(1, self.itemnum)
            if item not in seen and item not in chosen:
                chosen.add(item)
                result.append(item)
                if len(result) == count:
                    return result
        eligible = self.eligible_cache.get(user)
        if eligible is None:
            eligible = tuple(i for i in range(1, self.itemnum + 1) if i not in seen)
            self.eligible_cache[user] = eligible
        # Restart with an exact uniform sample; do not favor rejection survivors.
        return self.rng.sample(eligible, count)


def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, sampler=None
):
    sampler = sampler if sampler is not None else TrainingSampler(user_train, itemnum)
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        u = sampler.rng.choice(sampler.users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(SAMPLED_NEGATIVES if USE_SAMPLED_CE else maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if not USE_SAMPLED_CE:
                # Original BCE uses independently drawn negatives at each position.
                neg[idx] = sampler.negatives(u, 1)[0]
            nxt = item
            idx -= 1
            if idx == -1:
                break
        if USE_SAMPLED_CE:
            sampled = sampler.negatives(u, SAMPLED_NEGATIVES)
            neg[: len(sampled)] = sampled
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


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
        ### >>> Self_EvolveRec-BLOCK-START: Serialize explicit fusion configuration
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_fusion: bool = False,
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

        ### >>> Self_EvolveRec-BLOCK-START: Zero-initialized residual preserves the SASRec backbone
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.recent_fusion = recent_fusion
        self.recent_window = int(recent_window)
        if self.recent_window < 1:
            raise ValueError("recent_window must be positive")
        if recent_fusion:
            self.recent_gate = nn.Linear(2 * hidden_units, 1)
            self.recent_projection = nn.Linear(hidden_units, hidden_units)
            nn.init.zeros_(self.recent_gate.weight)
            nn.init.constant_(self.recent_gate.bias, -2.0)
            nn.init.zeros_(self.recent_projection.weight)
            nn.init.zeros_(self.recent_projection.bias)
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

        ### >>> Self_EvolveRec-BLOCK-START: Causal nonpadding recent-support residual without renormalization
        features = self.last_layernorm(seq_emb)
        if self.recent_fusion:
            observed = (seq != 0).unsqueeze(-1)
            embeddings = self.item_emb(seq) * observed
            prefix = torch.cat(
                (torch.zeros_like(embeddings[:, :1]), embeddings.cumsum(dim=1)), dim=1
            )
            counts = observed.to(embeddings.dtype)
            count_prefix = torch.cat(
                (torch.zeros_like(counts[:, :1]), counts.cumsum(dim=1)), dim=1
            )
            end = torch.arange(1, M + 1, device=seq.device)
            start = (end - self.recent_window).clamp_min(0)
            recent_sum = prefix[:, end] - prefix[:, start]
            recent_count = count_prefix[:, end] - count_prefix[:, start]
            recent = recent_sum / recent_count.clamp_min(1.0)
            gate = torch.sigmoid(
                self.recent_gate(torch.cat((features, recent), dim=-1))
            )
            features = features + gate * self.recent_projection(recent)
        return features
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
    ### >>> Self_EvolveRec-BLOCK-START: Stable local validation RNG independent of training draws
    # This is a local checkpoint diagnostic; the independent host evaluation
    # remains authoritative for cross-trial selection.
    validation_rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = validation_rng.sample(val_users, num_users)
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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives with explicit catalog checks
            if itemnum - len(interacted) < 99:
                raise ValueError(
                    "Validation requires 99 eligible unique negative candidates"
                )
            attempts = 0
            while len(neg_pool) < 99 and attempts < 10000:
                ni = validation_rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
                attempts += 1
            if len(neg_pool) < 99:
                eligible = [i for i in range(1, itemnum + 1) if i not in interacted]
                neg_pool = set(validation_rng.sample(eligible, 99))
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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible category initialization and training-only sampler
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    sampler = TrainingSampler(user_train, itemnum)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        ### >>> Self_EvolveRec-BLOCK-START: Activate and verify baseline-preserving fusion
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        recent_fusion=USE_RECENT_FUSION,
        recent_window=RECENT_WINDOW,
    ).to(device)
    if USE_RECENT_FUSION:
        model.eval()
        probe = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
        probe[0, -min(MAXLEN, 5) :] = 1
        candidates = torch.tensor([[1, itemnum]], dtype=torch.long, device=device)
        with torch.no_grad():
            fused_scores = model.predict(probe, candidates)
            model.recent_fusion = False
            backbone_scores = model.predict(probe, candidates)
            model.recent_fusion = True
            torch.testing.assert_close(fused_scores, backbone_scores, rtol=0, atol=0)
    # Added modules must not shift the backbone's training dropout RNG stream.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint
    best_hr10 = -1.0
    ### <<< Self_EvolveRec-BLOCK-END
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Refresh eligible negatives for every sequence draw
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, sampler
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            ### >>> Self_EvolveRec-BLOCK-START: Reuse sequence-shared negative embeddings with exact sampled-CE gradients
            optimizer.zero_grad(set_to_none=True)
            indices = pos_t != 0
            if not indices.any():
                raise ValueError("Training batch contains no valid next-item targets")
            if USE_SAMPLED_CE:
                features = model(seq_t)
                # Embed each sequence's alternatives once, rather than once
                # per valid timestep. Chunk sequences to bound score memory.
                valid_count = indices.sum()
                loss_sum = features.new_zeros(())
                for start in range(0, seq_t.size(0), 16):
                    stop = start + 16
                    valid = indices[start:stop]
                    h = features[start:stop]
                    positive_scores = (h * model.item_emb(pos_t[start:stop])).sum(
                        dim=-1, keepdim=True
                    )
                    negative_ids = neg_t[start:stop]
                    negative_embeddings = model.item_emb(negative_ids)
                    negative_scores = torch.bmm(
                        h, negative_embeddings.transpose(1, 2)
                    ).masked_fill(negative_ids[:, None, :] == 0, -torch.inf)
                    logits = torch.cat((positive_scores, negative_scores), dim=-1)
                    # Summation over empty selections contributes zero.
                    selected = logits[valid]
                    loss_sum = loss_sum + nn.functional.cross_entropy(
                        selected,
                        torch.zeros(selected.size(0), dtype=torch.long, device=device),
                        reduction="sum",
                    )
                loss = loss_sum / valid_count
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                )
                loss = loss + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite training loss in {dataset_name}, epoch {epoch}"
                )
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        if epoch % 5 == 0:
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist the trained residual architecture
                "maxlen": MAXLEN,
                "recent_fusion": USE_RECENT_FUSION,
                "recent_window": RECENT_WINDOW,
                "configuration": CONFIGURATION,
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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four routes in the generated predictor
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
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category artifact is missing
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
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct fused features in the generated predictor
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            recent_fusion=meta.get("recent_fusion", False),
            recent_window=meta.get("recent_window", 5),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routing while retaining the legacy signature
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset route: {dataset_name}")
        ds = dataset_name
    else:
        ds = DEFAULT_DATASETS[0]
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


