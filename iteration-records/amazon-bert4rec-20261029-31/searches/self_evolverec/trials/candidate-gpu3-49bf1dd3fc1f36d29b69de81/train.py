"""Per-dataset BERT4Rec baseline for Amazon Reviews.

Trains one independent BERT4Rec checkpoint per dataset while sharing one default
hyper-parameter profile across datasets. The generated predict.py routes by the
optional dataset_name keyword used by the RecHarness Amazon Reviews harness.
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

### >>> Self_EvolveRec-BLOCK-START: Fixed benchmark data and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]

SEED = 42
### <<< Self_EvolveRec-BLOCK-END
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _read_split(path: str) -> dict[int, list[int]]:
    data: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]["itemnum"])
    itemnum = 0
    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        if not os.path.isfile(path):
            continue
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


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    # Cloze training: masked targets are not exposed as unmasked shifted inputs.
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        items = history[-maxlen:]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        seq[-len(items) :] = items
        eligible = list(range(maxlen - len(items), maxlen))
        selected = [i for i in eligible if random.random() < 0.2]
        if not selected:
            selected = [random.choice(eligible)]
        if random.random() < 0.5 and maxlen - 1 not in selected:
            selected.append(maxlen - 1)
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError("no unobserved negative item available")
        for i in selected:
            pos[i] = seq[i]
            seq[i] = itemnum + 1  # reserved mask token, never a recommendation
            item = random.randint(1, itemnum)
            while item in interacted:
                item = random.randint(1, itemnum)
            neg[i] = item
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


def _history_to_tensor(
    history: list[int], maxlen: int, device: torch.device
) -> torch.Tensor:
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    return torch.tensor([seq_arr], dtype=torch.long, device=device)


def evaluate_val(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible local validation independent of training randomness
    rng = random.Random(SEED)
    previous_training = model.training
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Fixed local candidates and explicit negative-pool validation
            candidates = [target]
            excluded_real = sum(1 for item in interacted if 1 <= item <= itemnum)
            if itemnum - excluded_real < 99:
                raise RuntimeError(
                    f"Validation user {user} has fewer than 99 eligible negatives"
                )
            while len(candidates) < 100:
                neg_item = rng.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = _history_to_tensor(history, maxlen, device)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += 1.0 if rank < 10 else 0.0
            hr20 += 1.0 if rank < 20 else 0.0
            ndcg10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            ndcg20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's evaluation mode
    n = float(len(val_users))
    model.train(previous_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


### >>> Self_EvolveRec-BLOCK-START: Stable BPR arithmetic with explicit masked-target checks
def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
        raise ValueError(
            "Expected matching B x L sequence, positive and negative tensors"
        )
    valid = pos.ne(0)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Every example requires a supervised target")
    if not bool(seq[valid].eq(model.mask_token).all()):
        raise ValueError("Every supervised input position must contain MASK")
    if not bool(
        ((pos[valid] >= 1) & (pos[valid] <= model.itemnum)).all()
        and ((neg[valid] >= 1) & (neg[valid] <= model.itemnum)).all()
        and pos[valid].ne(neg[valid]).all()
    ):
        raise ValueError(
            "Supervised positives and negatives must be distinct real items"
        )
    # Preserve the original pooled-position BPR objective and dot-product head.
    # Softplus avoids the epsilon floor that suppresses badly misranked gradients.
    features = model.encode_sequence(seq)[valid].float()
    positive = (features * model.item_emb(pos[valid]).float()).sum(-1)
    negative = (features * model.item_emb(neg[valid]).float()).sum(-1)
    return torch.nn.functional.softplus(negative - positive).mean()


### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("BERT4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("BERT4REC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("BERT4REC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("BERT4REC_HEADS", "2"))
DROPOUT_RATE = float(os.environ.get("BERT4REC_DROPOUT", "0.2"))
LR = float(os.environ.get("BERT4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("BERT4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("BERT4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the benchmark training limits
NUM_BATCHES = 200
NUM_EPOCHS = max(
    1,
    min(
        200,
        int(os.environ.get("BERT4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("BERT4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("BERT4REC_PATIENCE", "20"))


### >>> Self_EvolveRec-BLOCK-START: Learned logarithmic distance bias inside bidirectional BERT4Rec
class LogLocalityEncoderLayer(nn.TransformerEncoderLayer):
    """Original projections and residual blocks with a per-head distance penalty."""

    def __init__(self, *args, locality_maxlen, **kwargs):
        super().__init__(*args, **kwargs)
        if locality_maxlen < 2:
            raise ValueError("Locality requires maxlen >= 2")
        self.locality_maxlen = int(locality_maxlen)
        initial_strength = 0.1
        raw = initial_strength + math.log(-math.expm1(-initial_strength))
        self.locality_raw = nn.Parameter(torch.full((self.self_attn.num_heads,), raw))
        positions = torch.arange(self.locality_maxlen, dtype=torch.float32)
        distance = (positions[:, None] - positions[None, :]).abs()
        self.register_buffer(
            "locality_distance",
            torch.log1p(distance) / math.log(self.locality_maxlen),
            persistent=False,
        )

    def _local_attention(self, x, src_mask, src_key_padding_mask, zero_bias=False):
        batch, length, _ = x.shape
        if not 1 <= length <= self.locality_maxlen:
            raise ValueError("Sequence exceeds checkpoint locality maxlen")
        strength = torch.nn.functional.softplus(self.locality_raw.float())
        if zero_bias:
            strength = strength * 0.0
        bias = (
            -strength[None, :, None, None]
            * self.locality_distance[None, None, :length, :length]
        ).expand(batch, -1, -1, -1)
        if src_mask is not None:
            extra = src_mask.to(device=x.device)
            if extra.dtype == torch.bool:
                extra = torch.zeros_like(extra, dtype=x.dtype).masked_fill(
                    extra, float("-inf")
                )
            if extra.ndim == 2:
                extra = extra[None, None]
            elif extra.ndim == 3:
                extra = extra.reshape(batch, self.self_attn.num_heads, length, length)
            else:
                raise ValueError("Unsupported attention mask shape")
            bias = bias + extra
        if src_key_padding_mask is not None:
            padding = src_key_padding_mask.to(device=x.device)
            if padding.dtype == torch.bool:
                padding = torch.zeros_like(padding, dtype=x.dtype).masked_fill(
                    padding, float("-inf")
                )
            if bool(torch.isneginf(padding).all(dim=1).any()):
                raise ValueError("Attention requires at least one visible key")
            bias = bias + padding[:, None, None, :]
        bias = bias.to(dtype=x.dtype).reshape(
            batch * self.self_attn.num_heads, length, length
        )
        # MultiheadAttention retains its original Q/K/V/output projections and
        # standard scaling. A floating additive mask also avoids its fused
        # inference path bypassing the trainable distance penalty.
        attended = self.self_attn(
            x, x, x, attn_mask=bias, need_weights=False, is_causal=False
        )[0]
        return self.dropout1(attended)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
        *,
        zero_bias=False,
    ):
        # Explicit residual computation prevents TransformerEncoderLayer's
        # evaluation fast path from silently bypassing custom attention.
        if is_causal:
            raise ValueError("BERT4Rec locality attention must remain bidirectional")
        x = src
        if self.norm_first:
            x = x + self._local_attention(
                self.norm1(x), src_mask, src_key_padding_mask, zero_bias
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._local_attention(x, src_mask, src_key_padding_mask, zero_bias)
            )
            x = self.norm2(x + self._ff_block(x))
        return x


def verify_locality_layer(layer):
    """Small deterministic output/gradient check; no optimizer or RNG changes."""
    device = layer.locality_raw.device
    generator = torch.Generator(device="cpu").manual_seed(4242)
    length = min(8, layer.locality_maxlen)
    source = (
        torch.randn((2, length, layer.self_attn.embed_dim), generator=generator)
        .to(device)
        .requires_grad_(True)
    )
    probe = torch.randn(source.shape, generator=generator).to(device)
    padding = torch.zeros((2, length), dtype=torch.bool, device=device)
    padding[0, 0] = True
    probe = probe.masked_fill(padding[..., None], 0.0)
    previous = layer.training
    parameters = tuple(layer.self_attn.parameters())
    try:
        layer.eval()
        with torch.enable_grad():
            # Calling the base implementation provides the actual original
            # attention path with the same projections and residual weights.
            reference = nn.TransformerEncoderLayer.forward(
                layer, source, src_key_padding_mask=padding
            )
            actual = layer(source, src_key_padding_mask=padding, zero_bias=True)
            torch.testing.assert_close(
                actual[~padding], reference[~padding], atol=1e-5, rtol=1e-4
            )
            reference_grad = torch.autograd.grad(
                (reference * probe).sum(), (source, *parameters)
            )
            actual_grad = torch.autograd.grad(
                (actual * probe).sum(), (source, *parameters)
            )
            for expected, observed in zip(reference_grad, actual_grad):
                torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-4)
            enabled = layer(source, src_key_padding_mask=padding)
            gradient = torch.autograd.grad((enabled * probe).sum(), layer.locality_raw)[
                0
            ]
            if not bool(
                torch.isfinite(enabled).all() and torch.isfinite(gradient).all()
            ):
                raise FloatingPointError("Nonfinite locality output or gradient")
            if not bool(gradient.abs().sum() > 0):
                raise RuntimeError("Locality parameter receives no gradient")
    finally:
        layer.train(previous)


class BERT4RecModel(nn.Module):
    ### <<< Self_EvolveRec-BLOCK-END
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        ### >>> Self_EvolveRec-BLOCK-START: Install locality in every existing encoder layer
        encoder_layer = LogLocalityEncoderLayer(
            locality_maxlen=maxlen,
            d_model=hidden_units,
            ### <<< Self_EvolveRec-BLOCK-END
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        ### >>> Self_EvolveRec-BLOCK-START: Preserve dense slots and padding holes
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks, enable_nested_tensor=False
        )
        ### <<< Self_EvolveRec-BLOCK-END
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Validate attention inputs without exposing a synthetic PAD key
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[0] == 0 or not 1 <= seq.shape[1] <= self.maxlen:
            raise ValueError(
                "Expected nonempty B x L sequences within checkpoint maxlen"
            )
        if seq.dtype != torch.long:
            raise ValueError("Sequence IDs must be int64")
        if not bool(((seq >= 0) & (seq <= self.mask_token)).all()):
            raise ValueError("Sequence contains an unknown token ID")
        padding = seq.eq(0)
        if bool(padding.all(dim=1).any()):
            raise ValueError(
                "Every encoded row requires a visible token or reserved MASK"
            )
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (~padding).long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        return self.norm(self.encoder(x, src_key_padding_mask=padding))

    ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    ### >>> Self_EvolveRec-BLOCK-START: Consistent singleton and batched appended-MASK scoring
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] != self.maxlen:
            raise ValueError("Expected nonempty B x configured maxlen histories")
        if candidates.ndim not in (1, 2):
            raise ValueError("Candidates must have shape C or B x C")
        if candidates.ndim == 2 and candidates.shape[0] != seq.shape[0]:
            raise ValueError("Candidate and history batch sizes differ")
        if not bool(((candidates >= 1) & (candidates <= self.itemnum)).all()):
            raise ValueError("Recommendation candidates must be real catalog items")
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        # Under left padding the appended MASK is the final nonpadding slot.
        user_emb = self.encode_sequence(prediction_seq)[:, -1]
        cand_emb = self.item_emb(candidates)
        if candidates.ndim == 1:
            scores = user_emb @ cand_emb.transpose(0, 1)
        else:
            scores = torch.einsum("bd,bcd->bc", user_emb, cand_emb)
        return scores[0] if seq.shape[0] == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Independent reproducible category initialization
def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = BERT4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    ### >>> Self_EvolveRec-BLOCK-START: Verify locality integration before unchanged optimization
    model.dataset_name = dataset_name
    for layer in model.encoder.layers:
        verify_locality_layer(layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on nonfinite optimization
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"[{dataset_name}] nonfinite loss at epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += loss_value
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
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
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Diagnose and export the selected checkpoint consistently
    if best_epoch == 0:
        raise RuntimeError(f"[{dataset_name}] no validation checkpoint was selected")
    selected_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state, strict=True)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, selected_state[name]):
            raise RuntimeError(f"[{dataset_name}] checkpoint mismatch at {name}")
    del selected_state
    model.eval()
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                ### >>> Self_EvolveRec-BLOCK-START: Record the single-network locality architecture
                "dropout_rate": DROPOUT_RATE,
                "architecture_version": "bert4rec-log-locality-v1",
                "locality_formula": "-softplus(raw_h)*log1p(abs(i-j))/log(maxlen)",
                "locality_initial_strength": 0.1,
                "position_convention": "fixed absolute dense slots, left padding",
                "pad_token": 0,
                "mask_token": itemnum + 1,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset BERT4Rec predict function."""
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
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"bert4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    ### >>> Self_EvolveRec-BLOCK-START: Reliable canonical locality-model import in the predictor writer
    import sys

    if spec is None or spec.loader is None:
        raise ImportError("Cannot import the current trial's BERT4Rec training module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.BERT4RecModel
    ### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail clearly on incomplete category artifacts
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing BERT4Rec artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct and strictly load the trained locality network
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
        ).to(_device)
        if meta.get("architecture_version") != "bert4rec-log-locality-v1":
            raise ValueError("Checkpoint does not describe the locality architecture")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=_device, weights_only=True), strict=True
        )
        ### <<< Self_EvolveRec-BLOCK-END
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Require unambiguous category routing
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required for four-category prediction")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"Missing BERT4Rec checkpoint for {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {predict_path}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset BERT4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


