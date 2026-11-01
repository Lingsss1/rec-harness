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

### >>> Self_EvolveRec-BLOCK-START: Required data boundary and fixed training seed
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


### >>> Self_EvolveRec-BLOCK-START: Count-matched candidate CE and BPR with unchanged corruption
# The supplied objective is single-negative BPR, not BCE.
# Separate runs: bpr/K=1, bpr/K=16, ce/K=16. Never combine checkpoints.
# K=1 CE equals BPR, so that configuration need not be trained twice.
# The unchanged host baseline remains the external benchmark anchor;
# purpose-separated sampling below defines the matched experimental controls.
# Preserve terminal allocation, endpoint distribution and reconstruction policy.
TERMINAL_MASK_PROBABILITY = 0.5
ENDPOINT_HALF_LIFE = 10.0
RECONSTRUCTION_MASK_PROBABILITY = 0.15
TRAIN_OBJECTIVE = os.environ.get("BERT4REC_OBJECTIVE", "ce").lower()
TRAIN_NEGATIVES = int(os.environ.get("BERT4REC_NEGATIVES", "16"))
if TRAIN_OBJECTIVE not in {"bpr", "ce"}:
    raise ValueError("BERT4REC_OBJECTIVE must be bpr or ce")
if TRAIN_NEGATIVES not in {1, 16}:
    raise ValueError("BERT4REC_NEGATIVES must be 1 or 16")


def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    num_negatives: int = 1,
    rngs=None,
):
    # Default output remains three B x L arrays for diagnostic adapters.
    # Only the explicit training caller requests B x L x K negatives.
    if maxlen < 2 or batch_size < 1 or num_negatives < 1:
        raise ValueError("require maxlen >= 2, positive batch_size and negatives")
    window_rng, corruption_rng, negative_rng = (
        rngs if rngs is not None else (random, random, random)
    )
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[window_rng.choice(users)]
        # Preserve the supplied zero-based endpoint distribution.
        # P(t) is proportional to 2 ** (-(n - 1 - t) / 10).
        endpoint = window_rng.choices(
            range(1, len(history)),
            weights=[
                2.0 ** (-(len(history) - 1 - t) / ENDPOINT_HALF_LIFE)
                for t in range(1, len(history))
            ],
            k=1,
        )[0]
        terminal = corruption_rng.random() < TERMINAL_MASK_PROBABILITY
        items = history[max(0, endpoint + 1 - maxlen) : endpoint + 1]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros((maxlen, num_negatives), dtype=np.int64)
        seq[-len(items) :] = items
        eligible = list(range(maxlen - len(items), maxlen))
        if terminal:
            # Only the endpoint is supervised; its occurrence is replaced below.
            # No suffix or shifted target is exposed to the bidirectional encoder.
            selected = [maxlen - 1]
        else:
            selected = [
                i
                for i in eligible
                if corruption_rng.random() < RECONSTRUCTION_MASK_PROBABILITY
            ]
            if not selected:
                selected = [corruption_rng.choice(eligible)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError("no unobserved negative item available")
        for i in selected:
            pos[i] = seq[i]
            seq[i] = itemnum + 1
            # Exactly one stream draw per target makes later target seeds
            # independent of K and of rejection counts. Each target then uses
            # independent uniform rejection draws WITH replacement, excluding
            # the same full training history as the supplied sampler.
            target_rng = random.Random(negative_rng.getrandbits(64))
            for j in range(num_negatives):
                item = target_rng.randint(1, itemnum)
                while item in interacted:
                    item = target_rng.randint(1, itemnum)
                neg[i, j] = item
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg[:, 0] if num_negatives == 1 else neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible validation without consuming training randomness
    rng = random.Random(SEED)
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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives with bounded sampling
            available_count = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available_count < 99:
                raise RuntimeError(
                    f"validation user {user} has only {available_count} eligible negatives; "
                    "99 unique negatives are required"
                )
            candidates = [target]
            if available_count < 198:
                available = [
                    item for item in range(1, itemnum + 1) if item not in interacted
                ]
                candidates.extend(rng.sample(available, 99))
            else:
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
    n = float(len(val_users))
    model.train()
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    ### >>> Self_EvolveRec-BLOCK-START: Bounded detached sigmoid weighting of uniform negatives
    valid = pos != 0
    if not bool(valid.any()):
        raise RuntimeError("masked-target batch has no supervised positions")
    if not bool((seq[valid] == model.mask_token).all()):
        raise RuntimeError("every supervised target must be hidden by MASK")
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != pos.shape or neg.shape[-1] < 1:
        raise ValueError("negatives must have shape B x L or B x L x K")
    targets, negatives = pos[valid], neg[valid]
    if not bool(((targets >= 1) & (targets <= model.itemnum)).all()):
        raise ValueError("positive target outside the real-item catalog")
    if not bool(((negatives >= 1) & (negatives <= model.itemnum)).all()):
        raise ValueError("negative outside the real-item catalog")
    if bool((negatives == targets[:, None]).any()):
        raise ValueError("positive sampled as a negative")

    features = model.log2feats(seq)[valid]
    positive_scores = model.predict_score(features, targets).float()
    negative_scores = model.predict_score(features, negatives).float()
    alpha = float(getattr(model, "_negative_weight_alpha", 0.0))
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("negative weighting alpha must be finite and in [0, 1]")
    if alpha and (TRAIN_OBJECTIVE != "ce" or negatives.shape[-1] != 16):
        raise ValueError("ramped weighting requires candidate CE with K=16")
    if not bool(
        torch.isfinite(positive_scores).all() & torch.isfinite(negative_scores).all()
    ):
        raise FloatingPointError("Nonfinite masked-target candidate logits")

    # Detached, mean-one weights lie in [1/(1+alpha), 1+alpha].
    # This changes allocation and aggregate pressure; it does not suppress
    # extreme negatives or identify false negatives. Candidate draws stay uniform.
    with torch.no_grad():
        differences = negative_scores.detach() - positive_scores.detach()[:, None]
        raw_weights = 1.0 + alpha * torch.sigmoid(differences)
        weights = raw_weights / raw_weights.mean(dim=-1, keepdim=True)
        log_weights = weights.log()
    if TRAIN_OBJECTIVE == "ce":
        # At alpha=0, execute ordinary candidate CE directly. With active
        # weighting, log(w) changes the denominator with detached weights.
        weighted_negatives = (
            negative_scores + log_weights if alpha > 0.0 else negative_scores
        )
        logits = torch.cat((positive_scores[:, None], weighted_negatives), dim=1)
        target_losses = torch.nn.functional.cross_entropy(
            logits,
            torch.zeros(len(targets), dtype=torch.long, device=seq.device),
            reduction="none",
        )
    else:
        target_losses = torch.nn.functional.softplus(
            negative_scores - positive_scores[:, None]
        ).mean(dim=1)
    loss = target_losses.mean()

    if getattr(model, "_capture_negative_audit", False):
        with torch.no_grad():
            p = positive_scores.detach()
            n = negative_scores.detach()
            log_mass_uniform = torch.logsumexp(n, dim=-1)
            log_mass_weighted = torch.logsumexp(n + log_weights, dim=-1)
            allocation = torch.softmax(n + log_weights, dim=-1)
            reference = torch.softmax(n, dim=-1)

            def allocation_summary(values):
                entropy = -(values * values.clamp_min(1e-30).log()).sum(-1)
                ordered = values.sort(dim=-1, descending=True).values
                return {
                    "mean_entropy": float(entropy.mean()),
                    "mean_effective_sample_size": float(
                        values.square().sum(-1).reciprocal().mean()
                    ),
                    "mean_top1_share": float(ordered[:, 0].mean()),
                    "mean_top4_share": float(ordered[:, :4].sum(-1).mean()),
                }

            sorted_negatives = negatives.sort(dim=-1).values
            unique_counts = (sorted_negatives[:, 1:] != sorted_negatives[:, :-1]).sum(
                -1
            ) + 1
            model._negative_audit_records.append(
                {
                    "epoch_zero_based": int(model._training_epoch_zero_based),
                    "batch_index": int(model._training_batch_index),
                    "batch_digests": dict(model._current_batch_digests),
                    "objective": TRAIN_OBJECTIVE,
                    "alpha": alpha,
                    "beta": 1.0,
                    "minimum_weight": float(weights.min()),
                    "maximum_weight": float(weights.max()),
                    "maximum_mean_weight_error": float(
                        (weights.mean(dim=-1) - 1.0).abs().max()
                    ),
                    "uniform_aggregate_negative_pressure": float(
                        torch.sigmoid(log_mass_uniform - p).mean()
                    ),
                    "negative_over_positive_rate": float(
                        (n > p[:, None]).float().mean()
                    ),
                    "targets": len(targets),
                    "negative_count": negatives.shape[-1],
                    "denominator": len(targets),
                    "reduction": "pooled_valid_target_mean",
                    "actual_loss": float(loss.detach()),
                    "mean_unique_negatives": float(unique_counts.float().mean()),
                    "mean_duplicate_negatives": float(
                        (negatives.shape[-1] - unique_counts).float().mean()
                    ),
                    "mean_log_negative_mass_increase": float(
                        (log_mass_weighted - log_mass_uniform).mean()
                    ),
                    "mean_log_weighted_mass_minus_positive": float(
                        (log_mass_weighted - p).mean()
                    ),
                    "mean_ce_aggregate_negative_pressure": (
                        float(torch.sigmoid(log_mass_weighted - p).mean())
                        if TRAIN_OBJECTIVE == "ce"
                        else None
                    ),
                    "weighted_allocation": allocation_summary(allocation),
                    "uniform_reference_allocation": allocation_summary(reference),
                    "mean_positive_score": float(p.mean()),
                    "negative_score_std": float(n.std(unbiased=False)),
                    "mean_within_target_negative_score_std": float(
                        n.std(dim=-1, unbiased=False).mean()
                    ),
                    "mean_feature_norm": float(
                        features.detach().float().norm(dim=-1).mean()
                    ),
                    "mean_positive_embedding_norm": float(
                        model.item_emb(targets).detach().float().norm(dim=-1).mean()
                    ),
                    "known_training_positive_collisions": None,
                    "collision_basis": "Full-history exclusion is the sampler policy; this summary has no sampled-user provenance to independently measure history collisions. Null means unmeasured, not zero.",
                    "scope": "Same-forward summaries only; no event-provenance or gradient certification. Captures may precede the selected checkpoint.",
                }
            )
    return loss
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
### >>> Self_EvolveRec-BLOCK-START: Enforce the matched training budget
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    int(os.environ.get("BERT4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
)
if NUM_EPOCHS < 1:
    raise ValueError("The declared training epoch count must be positive")
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("BERT4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("BERT4REC_PATIENCE", "20"))


### >>> Self_EvolveRec-BLOCK-START: Independent reproducible masked-target measurement


def measure_masked_target_diagnostics(model, user_train, itemnum, device):
    """Measure training margins without consuming production sampling streams."""
    modes = [(module, module.training) for module in model.modules()]
    diagnostic_rngs = tuple(
        random.Random(SEED + purpose) for purpose in (41000, 42000, 43000)
    )
    inspected = selected_count = invalid_count = 0
    margins = []
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(20):
                sequence, positive, negative = sample_batch(
                    user_train,
                    itemnum,
                    BATCH_SIZE,
                    model.maxlen,
                    num_negatives=1,
                    rngs=diagnostic_rngs,
                )
                seq = torch.as_tensor(sequence, dtype=torch.long, device=device)
                pos = torch.as_tensor(positive, dtype=torch.long, device=device)
                neg = torch.as_tensor(negative, dtype=torch.long, device=device)
                selected = pos != 0
                valid = (
                    selected
                    & (seq == model.mask_token)
                    & (pos >= 1)
                    & (pos <= itemnum)
                    & (neg >= 1)
                    & (neg <= itemnum)
                    & (neg != pos)
                )
                inspected += 1
                selected_count += int(selected.sum())
                invalid_count += int((selected & ~valid).sum())
                if bool(valid.any()):
                    features = model.log2feats(seq)[valid]
                    values = (
                        model.predict_score(features, pos[valid])
                        - model.predict_score(features, neg[valid])
                    ).float()
                    if not bool(torch.isfinite(values).all()):
                        raise RuntimeError("Nonfinite masked-target diagnostic margins")
                    margins.append(values.cpu())
        values = torch.cat(margins) if margins else torch.empty(0)
        count = int(values.numel())
        metrics = {
            "inspected_batches": inspected,
            "selected_positions": selected_count,
            "invalid_target_positions": invalid_count,
            "valid_masked_targets": count,
            "pos_neg_margin_mean": float(values.mean()) if count else None,
            "pos_neg_margin_p10": (
                float(values.kthvalue(max(1, math.ceil(0.1 * count))).values)
                if count
                else None
            ),
            "pos_neg_neg_beats_pos_rate": (
                float((values < 0).float().mean()) if count else None
            ),
            "pos_neg_small_margin_rate": (
                float((values < 0.5).float().mean()) if count else None
            ),
            "unavailable_reason": None if count else "No valid masked targets measured",
        }
        return {
            "metrics": metrics,
            "metric_definitions": {
                "inspected_batches": "Twenty disposable one-negative batches using independent seed-42-derived local sampling streams.",
                "selected_positions": "Positions with nonzero positive labels.",
                "invalid_target_positions": "Selected positions failing reserved-mask, catalog-label, catalog-negative or positive/negative inequality checks.",
                "valid_masked_targets": "Number of audited positions contributing measured margins.",
                "pos_neg_margin_mean": "Mean positive-minus-negative logit on training-exposed masked targets; score-scale sensitive, not validation performance.",
                "pos_neg_margin_p10": "Nearest-rank tenth percentile of those margins.",
                "pos_neg_neg_beats_pos_rate": "Fraction of those margins below zero; invariant to common positive score rescaling.",
                "pos_neg_small_margin_rate": "Fraction of those margins below 0.5; score-scale sensitive.",
                "unavailable_reason": "Measurement limitation when coverage is zero; zero coverage is not a passing result.",
            },
            "scope": "Selected category checkpoint, independent training-sample instrumentation. Does not certify original event provenance, fresh-process predictor parity or held-out usefulness. Existing diagnosis_tools APIs remain unchanged.",
        }
    finally:
        for module, training in modes:
            module.training = training


### <<< Self_EvolveRec-BLOCK-END


class BERT4RecModel(nn.Module):
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        padding = seq == 0
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    ### >>> Self_EvolveRec-BLOCK-START: Shared BERT4Rec feature and diagnostic scoring interfaces
    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = torch.as_tensor(
            sequence, dtype=torch.long, device=self.item_emb.weight.device
        )
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        items = torch.as_tensor(
            items, dtype=torch.long, device=self.item_emb.weight.device
        )
        embeddings = self.item_emb(items)
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.ndim == features.ndim and items.shape[:-1] == features.shape[:-1]:
            return torch.matmul(embeddings, features.unsqueeze(-1)).squeeze(-1)
        raise ValueError(
            f"Unsupported feature/item shapes: {tuple(features.shape)}, {tuple(items.shape)}"
        )

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Consistent multi-negative forward and batched appended-mask inference
    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        device = self.item_emb.weight.device
        seq = torch.as_tensor(seq, dtype=torch.long, device=device)
        candidates = torch.as_tensor(candidates, dtype=torch.long, device=device)
        if seq.ndim != 2 or seq.shape[0] < 1 or not 1 <= seq.shape[1] <= self.maxlen:
            raise ValueError("Expected nonempty B x L histories with L <= maxlen")
        if candidates.ndim not in (1, 2):
            raise ValueError("Candidates must have shape K or B x K")
        if candidates.ndim == 2 and candidates.shape[0] != seq.shape[0]:
            raise ValueError("Per-user candidates must match the history batch size")

        # Retain the final L-1 history slots and append the reserved MASK.
        # Left padding remains left padding, including for empty histories.
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        features = self.log2feats(prediction_seq)[:, -1, :]
        # The appended MASK is the final nonpadding position, even when the
        # history is short. Never use nonpadding-count minus one here.
        embeddings = self.item_emb(candidates)
        if candidates.ndim == 1:
            scores = torch.matmul(features, embeddings.transpose(0, 1))
        else:
            scores = torch.bmm(embeddings, features.unsqueeze(-1)).squeeze(-1)
        return scores[0] if seq.shape[0] == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Initialize every independent category model with seed 42
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Reproducible matched sampling streams and configuration record
    # Reset independently for each category, just as model seed 42 is reset.
    # User/window and corruption draws never depend on negative count/objective.
    training_rngs = tuple(
        random.Random(SEED + purpose) for purpose in (1000, 2000, 3000)
    )
    print(
        f"  [{dataset_name}] objective={TRAIN_OBJECTIVE} K={TRAIN_NEGATIVES} "
        "negative_policy=uniform_with_replacement_full_train_history_exclusion "
        "reduction=mean_targets; BPR_inner=mean_negatives; CE=no_logQ "
        "checkpoint=first_strict_validation_HR10_improvement",
        flush=True,
    )
    with open(
        os.path.join(working_dir, f"{dataset_name}_training_config.json"), "w"
    ) as f:
        json.dump(
            {
                "seed": SEED,
                "objective": TRAIN_OBJECTIVE,
                "negatives": TRAIN_NEGATIVES,
                "sampling_stream_seeds": [SEED + p for p in (1000, 2000, 3000)],
                "replacement": True,
                "exclusions": "full supplied user training history",
                "target_reduction": "pooled valid-target mean",
                "terminal_probability": TERMINAL_MASK_PROBABILITY,
                "endpoint_half_life": ENDPOINT_HALF_LIFE,
                "reconstruction_probability": RECONSTRUCTION_MASK_PROBABILITY,
                "eval_every": EVAL_EVERY,
                "patience": PATIENCE,
                "checkpoint_rule": "first strict validation HR@10 improvement",
            },
            f,
            indent=2,
        )
    best_hr10 = -1.0
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Validated warm-start curriculum and bounded weighting measurements
    if EVAL_EVERY < 1 or PATIENCE < 1:
        raise ValueError("Evaluation interval and patience must be positive")
    if MAXLEN < 2 or BATCH_SIZE < 1:
        raise ValueError("Require maxlen >= 2 and a positive training batch size")
    default_weighting = (
        "ramp" if TRAIN_OBJECTIVE == "ce" and TRAIN_NEGATIVES == 16 else "uniform"
    )
    weighting_mode = os.environ.get(
        "BERT4REC_NEGATIVE_WEIGHTING", default_weighting
    ).lower()
    if weighting_mode not in {"uniform", "ramp"}:
        raise ValueError("BERT4REC_NEGATIVE_WEIGHTING must be uniform or ramp")
    if weighting_mode == "ramp" and (TRAIN_OBJECTIVE != "ce" or TRAIN_NEGATIVES != 16):
        raise ValueError(
            "ramp requires BERT4REC_OBJECTIVE=ce and BERT4REC_NEGATIVES=16"
        )
    warm_epochs = math.ceil(0.2 * NUM_EPOCHS)

    def alpha_at_epoch(epoch_zero_based):
        if not 0 <= epoch_zero_based < NUM_EPOCHS:
            raise ValueError("Epoch outside the declared weighting schedule")
        if (
            weighting_mode == "uniform"
            or epoch_zero_based < warm_epochs
            or warm_epochs == NUM_EPOCHS
        ):
            return 0.0
        return (epoch_zero_based - warm_epochs + 1) / (NUM_EPOCHS - warm_epochs)

    capture_epochs = sorted(
        {1, NUM_EPOCHS}
        | ({warm_epochs + 1} if warm_epochs < NUM_EPOCHS else set())
        | {warm_epochs + math.ceil((NUM_EPOCHS - warm_epochs) / 2)}
    )
    model._negative_audit_records = []
    active_weighting_steps = 0
    cumulative_alpha = 0.0
    print(
        f"  [{dataset_name}] negative_weighting={weighting_mode}; "
        f"warm_epochs={warm_epochs}; declared_epochs={NUM_EPOCHS}; "
        "linear post-warmup alpha to 1; detached mean-one sigmoid weights; "
        "joint allocation/pressure/schedule intervention",
        flush=True,
    )
    for epoch in range(1, NUM_EPOCHS + 1):
        model._training_epoch_zero_based = epoch - 1
        model._negative_weight_alpha = alpha_at_epoch(epoch - 1)
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            model._training_batch_index = _
            active_weighting_steps += int(model._negative_weight_alpha > 0.0)
            cumulative_alpha += model._negative_weight_alpha
            # At most four predeclared same-forward summaries; no extra forward.
            model._capture_negative_audit = epoch in capture_epochs and _ == 0
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Explicit training-only multi-negative sampling
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                num_negatives=TRAIN_NEGATIVES,
                rngs=training_rngs,
            )
            if model._capture_negative_audit:
                # Compare these bounded digests across separately executed
                # controls. Computing them consumes no sampling/dropout RNG.
                import hashlib

                example_digest = hashlib.sha256(
                    seq.tobytes() + pos.tobytes()
                ).hexdigest()
                negative_digest = hashlib.sha256(neg.tobytes()).hexdigest()
                model._current_batch_digests = {
                    "sequence_and_labels_sha256": example_digest,
                    "negative_ids_sha256": negative_digest,
                }
                print(
                    f"  [{dataset_name}] epoch={epoch} batch={_} "
                    f"examples_sha256={example_digest} "
                    f"negatives_sha256={negative_digest} "
                    f"supervised_targets={int((pos != 0).sum())}",
                    flush=True,
                )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Reject nonfinite training loss before updating parameters
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(
                    f"Nonfinite loss for {dataset_name}, epoch={epoch}, batch={_}, "
                    f"alpha={model._negative_weight_alpha}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Reject nonfinite gradients and measure bounded clipping diagnostics
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            if model._capture_negative_audit:
                audit = model._negative_audit_records[-1]
                audit["batch_parameter_gradient_norm_before_clipping"] = float(
                    gradient_norm.detach()
                )
                audit["gradient_clip_threshold"] = 5.0
                audit["gradient_clipped"] = bool(gradient_norm > 5.0)
                audit["gradient_measurement_scope"] = (
                    "Actual pooled batch parameter-gradient norm before clipping; "
                    "per-target gradients and optimizer-update norms are unmeasured"
                )
            optimizer.step()
            ### <<< Self_EvolveRec-BLOCK-END
            total_loss += float(loss.item())

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

    ### >>> Self_EvolveRec-BLOCK-START: Record bounded weighting exposure and selected-checkpoint margin coverage
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    model._capture_negative_audit = False
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    measured_diagnostics = measure_masked_target_diagnostics(
        model, user_train, itemnum, device
    )
    with open(
        os.path.join(
            working_dir, f"{dataset_name}_negative_weighting_diagnostics.json"
        ),
        "w",
    ) as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "training_seed": SEED,
                "weighting_mode": weighting_mode,
                "schedule": "W=ceil(0.2*E); alpha=0 for e<W, otherwise (e-W+1)/(E-W); e is zero-based",
                "warm_epochs": warm_epochs,
                "declared_epochs": NUM_EPOCHS,
                "beta": 1.0,
                "maximum_alpha": (
                    1.0
                    if weighting_mode == "ramp" and warm_epochs < NUM_EPOCHS
                    else 0.0
                ),
                "weight_formula": "raw=1+alpha*sigmoid(negative-positive); weight=raw/mean(raw)",
                "weight_bounds": [0.5, 2.0],
                "weight_derivatives": "detached",
                "target_reduction": "sum target losses / number of valid masked targets",
                "negative_policy": "uniform with replacement; full supplied training-history exclusion",
                "capture_policy": "first batch of predeclared initial, first active, mid-ramp and final epochs; at most four",
                "planned_capture_epochs": capture_epochs,
                "missing_capture_epochs": [
                    planned for planned in capture_epochs if planned > epoch
                ],
                "active_weighting_steps": active_weighting_steps,
                "cumulative_prescribed_alpha": cumulative_alpha,
                "selected_checkpoint_active_steps": NUM_BATCHES
                * sum(alpha_at_epoch(e) > 0.0 for e in range(best_epoch)),
                "selected_checkpoint_cumulative_alpha": NUM_BATCHES
                * sum(alpha_at_epoch(e) for e in range(best_epoch)),
                "executed_maximum_alpha": alpha_at_epoch(epoch - 1),
                "selected_checkpoint_alpha": (
                    alpha_at_epoch(best_epoch - 1) if best_epoch > 0 else 0.0
                ),
                "executed_epochs": epoch,
                "selected_epoch": best_epoch,
                "captures": model._negative_audit_records,
                "masked_target_diagnostics": measured_diagnostics,
                "fresh_process_predictor_parity": {
                    "available": False,
                    "reason": "No fresh-process predictor runner was executed by this instrumentation.",
                },
            },
            handle,
            indent=2,
            allow_nan=False,
        )
    print(
        f"  [{dataset_name}] selected-checkpoint training diagnostic: "
        f"batches={measured_diagnostics['metrics']['inspected_batches']} "
        f"valid_masks={measured_diagnostics['metrics']['valid_masked_targets']} "
        f"invalid_positions={measured_diagnostics['metrics']['invalid_target_positions']}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BERT4RecModel


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
        ### >>> Self_EvolveRec-BLOCK-START: Require a trained checkpoint for every configured category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing BERT4Rec metadata or checkpoint for {ds}: "
                f"{meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved architecture before evaluation
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Explicit category routing without silent cross-category predictions
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unloaded dataset: {dataset_name}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when multiple categories are loaded")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Validate predictor inputs and preserve appended-mask preprocessing
    if any(item < 1 or item > model.itemnum for item in candidates):
        raise ValueError("Candidates must contain real catalog IDs, excluding PAD and MASK")
    if any(item < 1 or item > model.itemnum for item in history):
        raise ValueError("History must contain real catalog item IDs")
    if len(candidates) == 0:
        return []
    maxlen = int(meta["maxlen"])
    seq_arr = [0] * maxlen
    # score_candidates reserves the final slot by shifting this raw history
    # once, retaining exactly its final maxlen-1 observed events.
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
    ### <<< Self_EvolveRec-BLOCK-END
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


