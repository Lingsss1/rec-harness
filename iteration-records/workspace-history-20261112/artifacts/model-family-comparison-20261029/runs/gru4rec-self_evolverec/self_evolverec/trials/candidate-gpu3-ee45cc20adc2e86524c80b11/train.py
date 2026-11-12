"""Per-dataset GRU4Rec baseline for Amazon Reviews.

Trains one independent GRU4Rec checkpoint per dataset while sharing one default
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

### >>> Self_EvolveRec-BLOCK-START: Preserve required data boundary and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]

SEED = 42
### <<< Self_EvolveRec-BLOCK-END
# DEBUG: Reduced-precision CUDA matrix multiplication can produce different
# GRU states for singleton and packed mixed-length batches. Require full FP32
# precision for both recurrent execution and candidate scoring; retain the
# existing strict execution checks, padding semantics and checkpoint weights.
# The generated predictor imports this module before constructing its models,
# so training, ModelView diagnostics and checkpoint inference share this policy.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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


### >>> Self_EvolveRec-BLOCK-START: Cache fixed training histories while preserving sampler order and random draws
_SAMPLER_CACHE = {}


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    if itemnum < 1 or batch_size < 1 or maxlen < 1:
        raise ValueError("itemnum, batch_size and maxlen must be positive")
    # The supplied training splits remain immutable throughout a category fit.
    # Preserve user order, random.choice and rejection-sampling draws exactly.
    if (
        _SAMPLER_CACHE.get("source") is not user_train
        or _SAMPLER_CACHE.get("itemnum") != itemnum
    ):
        users = [u for u, items in user_train.items() if len(items) >= 2]
        if not users:
            raise RuntimeError("no users with at least two training interactions")
        exclusions = {}
        for user in users:
            observed = set(user_train[user])
            if any(item < 1 or item > itemnum for item in observed):
                raise ValueError(f"Invalid training item ID for user {user}")
            if len(observed) >= itemnum:
                raise ValueError(
                    f"Training user {user} has no eligible negative; "
                    "the unchanged rejection sampler cannot sample this user"
                )
            exclusions[user] = observed
        _SAMPLER_CACHE.clear()
        _SAMPLER_CACHE.update(
            source=user_train, itemnum=itemnum, users=users, exclusions=exclusions
        )

    users = _SAMPLER_CACHE["users"]
    batch_seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    batch_pos = np.zeros_like(batch_seq)
    batch_neg = np.zeros_like(batch_seq)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq, pos, neg = batch_seq[row], batch_pos[row], batch_neg[row]
        nxt = history[-1]
        idx = maxlen - 1
        interacted = _SAMPLER_CACHE["exclusions"][user]
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
    return batch_seq, batch_pos, batch_neg


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
    ### >>> Self_EvolveRec-BLOCK-START: Make local checkpoint validation reproducible without consuming training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Use fixed unique validation negatives and reject impossible candidate counts
            available = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available < 99:
                raise ValueError(
                    f"Validation user {user} has only {available} unseen negatives; "
                    "99 unique negatives are required"
                )
            candidates = [target]
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
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the fixed training budget
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))


class GRU4RecModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure fixed chrono-inspired e-folding horizons without extra parameters
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        architecture_variant: str = "heterogeneous",
        history_window: int = 128,
        tau_min: float = 2.0,
        tau_max: float = 32.0,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        ### >>> Self_EvolveRec-BLOCK-START: Initialize exact nominal e-folding retention with a probability-matched uniform control
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        if architecture_variant not in {
            "baseline",
            "uniform",
            "uniform_mean",
            "heterogeneous",
        }:
            raise ValueError(f"Unsupported initialization: {architecture_variant}")
        if history_window < 1:
            raise ValueError("history_window must be positive")
        if not (
            math.isfinite(tau_min)
            and math.isfinite(tau_max)
            and 1.0 < tau_min <= tau_max
        ):
            raise ValueError("Invalid nominal e-folding horizon bounds")
        self.architecture_config = {
            "itemnum": itemnum,
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "architecture_variant": architecture_variant,
            "history_window": history_window,
            "tau_min": float(tau_min),
            "tau_max": float(tau_max),
        }

    @torch.no_grad()
    def initialize_retention_horizons(self) -> None:
        # Fresh training initialization only; checkpoint reconstruction does not
        # call this method. Deterministic operations consume no RNG draws.
        variant = self.architecture_config["architecture_variant"]
        if variant == "baseline":
            return
        if (
            not isinstance(self.gru, nn.GRU)
            or not self.gru.bias
            or self.gru.bidirectional
        ):
            raise ValueError("Expected a biased, unidirectional PyTorch GRU")
        width = self.gru.hidden_size
        reference = self.gru.bias_ih_l0
        horizons = torch.linspace(
            math.log(self.architecture_config["tau_min"]),
            math.log(self.architecture_config["tau_max"]),
            steps=width,
            device=reference.device,
            dtype=torch.float64,
        ).exp()
        # Chrono-inspired geometric e-folding prior: r**k = exp(-k/tau).
        # This differs from original chrono retention r = 1 - 1/T.
        retention = torch.exp(-horizons.reciprocal())
        if variant in {"uniform", "uniform_mean"}:
            retention = retention.mean().expand_as(retention)
        total_bias = retention.log() - torch.log1p(-retention)
        for layer in range(self.gru.num_layers):
            # PyTorch chunks are reset/update/new; update z carries old state.
            b_ih = getattr(self.gru, f"bias_ih_l{layer}")
            b_hh = getattr(self.gru, f"bias_hh_l{layer}")
            b_ih[width : 2 * width].copy_(total_bias.to(b_ih.dtype))
            b_hh[width : 2 * width].zero_()
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Execute ordinary GRU on real tokens and share padding-safe features and scoring
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        emb = self.dropout(self.item_emb(seq))
        batch, length = seq.shape
        output = emb * 0.0
        if batch == 0 or length == 0:
            return output
        valid = seq.ne(0)
        lengths = valid.sum(dim=1)
        active = lengths.gt(0).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            return output

        # Compact real items chronologically; never pack left padding directly
        # and never introduce a synthetic recurrent step for an empty row.
        active_valid = valid.index_select(0, active)
        active_emb = emb.index_select(0, active)
        compact_positions = active_valid.long().cumsum(dim=1) - 1
        rows, original_positions = active_valid.nonzero(as_tuple=True)
        compact_indices = compact_positions[rows, original_positions]
        # Transfer lengths once and allocate recurrent buffers only to the
        # longest real history. External output coordinates remain unchanged.
        active_lengths_cpu = lengths.index_select(0, active).detach().cpu()
        compact_width = int(active_lengths_cpu.max().item())
        compact = active_emb.new_zeros(
            active.size(0), compact_width, self.hidden_units
        ).index_put((rows, compact_indices), active_emb[rows, original_positions])
        packed = nn.utils.rnn.pack_padded_sequence(
            compact,
            active_lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Restore the external B x L coordinates for training and ModelView.
        return output.index_put(
            (active[rows], original_positions), unpacked[rows, compact_indices]
        )

    def _readout(self, out: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return out.new_zeros(seq.size(0), self.hidden_units)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1)
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp_min(0)]
        return torch.where(last[:, None].ge(0), features, torch.zeros_like(features))

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self._readout(self.log2feats(seq), seq)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        item_features = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, item_features.transpose(0, 1))
        if item_features.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * item_features).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    @torch.inference_mode()
    def validate_execution(self) -> None:
        device = self.item_emb.weight.device
        a, b = 1, min(2, self.itemnum)
        seq = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, a], [0, a, 0, b], [a, b, a, b], [0, a, 0, 0]],
            dtype=torch.long,
            device=device,
        )
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:

            def compare(actual, expected):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

            expected_last = torch.tensor([-1, 3, 3, 3, 1], device=device)
            positions = torch.arange(4, device=device).expand_as(seq)
            torch.testing.assert_close(
                positions.masked_fill(seq.eq(0), -1).amax(1),
                expected_last,
                atol=0,
                rtol=0,
            )
            features = self.log2feats(seq)
            independent = torch.stack(
                [
                    (
                        features[row, int(index)]
                        if index >= 0
                        else features.new_zeros(self.hidden_units)
                    )
                    for row, index in enumerate(expected_last.tolist())
                ]
            )
            torch.testing.assert_close(
                self._readout(features, seq), independent, atol=0, rtol=0
            )
            compare(self.log2feats(seq), features)
            compare(self.encode(seq), independent)
            padded = torch.cat((torch.zeros_like(seq[:, :2]), seq), dim=1)
            compare(self.encode(padded), independent)
            candidates = torch.tensor([a, b], dtype=torch.long, device=device)
            scores = self.predict_score(independent, candidates)
            compare(self.score_candidates(seq, candidates), scores)
            compare(
                self.score_candidates(seq, candidates.expand(seq.size(0), -1)),
                scores,
            )
            item_ids = candidates.view(1, 1, -1).expand(seq.size(0), seq.size(1), -1)
            position_scores = self.predict_score(features, item_ids)
            compare(position_scores, self.predict_score(features, candidates))
            aligned = seq.clamp_min(1)
            compare(
                self.predict_score(features, aligned),
                (features * self.item_emb(aligned)).sum(-1),
            )
            for row in range(seq.size(0)):
                compare(self.encode(seq[row : row + 1])[0], independent[row])
                compare(
                    self.score_candidates(seq[row : row + 1], candidates), scores[row]
                )
                real = seq[row][seq[row].ne(0)].unsqueeze(0)
                compare(self.encode(real)[0], independent[row])
                if real.size(1):
                    # Recommendation features are raw GRU outputs here.
                    raw, hidden = self.gru(self.item_emb(real))
                    compare(raw[0, -1], hidden[-1, 0])
                    compare(hidden[-1, 0], independent[row])
                for position in range(seq.size(1)):
                    compare(
                        position_scores[row, position],
                        self.predict_score(features[row, position], candidates),
                    )
            compare(self.log2feats(seq[:1]), torch.zeros_like(features[:1]))
            compare(self.encode(seq[:, :0]), torch.zeros_like(independent))
            compare(features[seq.eq(0)], torch.zeros_like(features[seq.eq(0)]))
        finally:
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Use the same readout and scorer for batched and single-history predictions
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Audit isolated update-bias initialization and record training-only horizon support
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        architecture_variant=os.environ.get("GRU4REC_ARCHITECTURE", "heterogeneous"),
        history_window=MAXLEN,
        tau_min=2.0,
        tau_max=32.0,
    ).to(device)
    lengths = np.asarray(
        [sum(int(item) != 0 for item in history) for history in user_train.values()],
        dtype=np.int64,
    )
    initialization_report = {
        "variant": model.architecture_config["architecture_variant"],
        "description": "chrono-inspired geometric e-folding initialization",
        "tau_min": 2.0,
        "tau_max": 32.0,
        "training_users": int(lengths.size),
        "history_length_quantiles": np.quantile(
            lengths, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        ).tolist(),
        "history_quantile_probabilities": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
        "retained_sequence_width": MAXLEN,
        "fraction_histories_at_least_2": float(np.mean(lengths >= 2)),
        "fraction_histories_at_least_32": float(np.mean(lengths >= 32)),
        "fraction_supervised_prefixes_at_least_32": float(
            np.mean(np.minimum(np.maximum(lengths - 1, 0), MAXLEN) >= 32)
        ),
    }
    model.validate_execution()
    # Snapshot this fresh baseline, rather than fitting or retaining a second
    # network. Verify that only existing update-bias slices can change.
    # Keep the audit snapshot on CPU so it does not duplicate the item table
    # in GPU memory. This does not change initialization or random streams.
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.initialize_retention_horizons()
    variant = model.architecture_config["architecture_variant"]
    width = model.gru.hidden_size
    for name, parameter in model.named_parameters():
        parameter = parameter.detach().cpu()
        before = initial[name]
        if variant != "baseline" and name.startswith(
            ("gru.bias_ih_l", "gru.bias_hh_l")
        ):
            torch.testing.assert_close(
                parameter[:width], before[:width], rtol=0, atol=0
            )
            torch.testing.assert_close(
                parameter[2 * width :], before[2 * width :], rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    if not torch.equal(cpu_rng, torch.random.get_rng_state()):
        raise AssertionError("Retention initialization consumed CPU RNG")
    if cuda_rng is not None and any(
        not torch.equal(before, after)
        for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all())
    ):
        raise AssertionError("Retention initialization consumed CUDA RNG")
    del initial
    initialization_report["unchanged_parameter_audit_passed"] = True
    initialization_report["rng_preserved"] = True
    initialization_report["bias_only_retention_by_layer"] = [
        torch.sigmoid(
            getattr(model.gru, f"bias_ih_l{layer}")[width : 2 * width].double()
            + getattr(model.gru, f"bias_hh_l{layer}")[width : 2 * width].double()
        )
        .detach()
        .cpu()
        .tolist()
        for layer in range(model.gru.num_layers)
    ]
    print(
        f"  [{dataset_name}] initialization={json.dumps(initialization_report)}",
        flush=True,
    )
    model.validate_execution()
    model.dataset_name = dataset_name
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
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
                ### >>> Self_EvolveRec-BLOCK-START: Save one validation-selected checkpoint with initialization provenance
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture_config": model.architecture_config,
                        "initialization_report": initialization_report,
                        "selected_epoch": best_epoch,
                        "validation_hr10": best_hr10,
                    },
                    ckpt_path,
                )
                ### <<< Self_EvolveRec-BLOCK-END
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Restore the selected checkpoint before host diagnostics and verify serialized scores
    if best_epoch == 0:
        raise RuntimeError(f"No validation-selected checkpoint for {dataset_name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if checkpoint["architecture_config"] != model.architecture_config:
        raise RuntimeError(
            f"Selected checkpoint architecture mismatch for {dataset_name}"
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.validate_execution()
    # A temporary reconstruction uses exactly the same selected weights; it is
    # neither independently fitted nor used to combine recommendation scores.
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        restored = GRU4RecModel(**checkpoint["architecture_config"]).to(device)
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        restored.eval()
        fixture = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [1, min(2, itemnum), 1]],
            dtype=torch.long,
            device=device,
        )
        candidates = torch.arange(1, min(itemnum, 16) + 1, device=device)
        with torch.inference_mode():
            reference = model.score_candidates(fixture, candidates)
            actual = restored.score_candidates(fixture, candidates)
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
            print(
                f"  [{dataset_name}] checkpoint reload score max_abs_error="
                f"{float((actual - reference).abs().max()):.9g}; "
                "atol=1e-5 rtol=1e-4; emitted predictor uses this same model class",
                flush=True,
            )
        del restored
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Record inherited training settings and retention architecture
                "dropout_rate": DROPOUT_RATE,
                "architecture_config": model.architecture_config,
                "training_config": {
                    "seed": SEED,
                    "max_epochs": NUM_EPOCHS,
                    "batches_per_epoch": NUM_BATCHES,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "optimizer": "Adam",
                    "loss": "baseline_sampled_bpr",
                    "eval_every": EVAL_EVERY,
                    "patience": PATIENCE,
                    "best_epoch": best_epoch,
                    "gradient_clip_norm": 5.0,
                },
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
    script = r'''"""Per-dataset GRU4Rec predict function."""
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
        f"gru4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


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
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing metadata or checkpoint for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Load the shared ordinary-GRU implementation without reinitializing trained update biases
        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=True)
        config = checkpoint["architecture_config"]
        if config != meta["architecture_config"]:
            raise RuntimeError(f"Checkpoint/metadata architecture mismatch for {ds}")
        model = ModelClass(**config).to(_device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.dataset_name = ds
        model.eval()
        model.validate_execution()
        ### <<< Self_EvolveRec-BLOCK-END
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing category checkpoints instead of silently scoring with another category
    _ensure_loaded()
    if dataset_name is not None:
        if dataset_name not in _models:
            raise RuntimeError(f"No trained GRU4Rec checkpoint for {dataset_name!r}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when routing multiple category models")
    model = _models[ds]
    meta = _metas[ds]
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
    ### >>> Self_EvolveRec-BLOCK-START: Verify emitted routing and preprocessing against selected checkpoint scores
    print(f"predict.py written -> {predict_path}", flush=True)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "current_trial_predictor_verification", predict_path
    )
    exported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exported)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # The emitted loader imports train.py, which seeds global RNGs.
            # Preserve those streams around the entire verification operation.
            ModelClass = exported._load_model_class()
            for ds in DEFAULT_DATASETS:
                meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
                checkpoint_path = os.path.join(working_dir, f"{ds}_model.pt")
                with open(meta_path, "rb") as handle:
                    metadata = pickle.load(handle)
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=True
                )
                if checkpoint["architecture_config"] != metadata["architecture_config"]:
                    raise RuntimeError(f"Export architecture mismatch for {ds}")
                model = ModelClass(**checkpoint["architecture_config"]).to(device)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                # Reconstruct only this selected checkpoint. This temporary
                # instance is never trained or combined with another predictor.
                exported._models = {ds: model}
                exported._metas = {ds: metadata}
                exported._device = device
                width = int(metadata["maxlen"])
                itemnum = int(metadata["itemnum"])
                histories = [
                    [],
                    [1],
                    [1, min(2, itemnum), 1],
                    [1 + index % itemnum for index in range(width + 3)],
                ]
                candidates = list(range(1, min(itemnum, 16) + 1))
                candidate_tensor = torch.tensor(
                    candidates, dtype=torch.long, device=device
                )
                maximum_error = 0.0
                with torch.inference_mode():
                    for history in histories:
                        # Independently construct LEFT-padded inputs, including
                        # an over-window history to exercise export truncation.
                        sequence = torch.zeros(
                            1, width, dtype=torch.long, device=device
                        )
                        retained = history[-width:]
                        if retained:
                            sequence[0, -len(retained) :] = torch.tensor(
                                retained, dtype=torch.long, device=device
                            )
                        reference = model.score_candidates(sequence, candidate_tensor)
                        actual = torch.tensor(
                            exported.predict(0, history, candidates, dataset_name=ds),
                            dtype=reference.dtype,
                            device=device,
                        )
                        torch.testing.assert_close(
                            actual, reference, atol=1e-5, rtol=1e-4
                        )
                        maximum_error = max(
                            maximum_error, float((actual - reference).abs().max())
                        )
                print(
                    f"  [{ds}] emitted predictor score max_abs_error="
                    f"{maximum_error:.9g}; atol=1e-5 rtol=1e-4",
                    flush=True,
                )
                exported._models.clear()
                exported._metas.clear()
                del model, checkpoint
    finally:
        exported._models.clear()
        exported._metas.clear()
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    # Static review: initialization is invoked before optimizer construction;
    # fixed 2–32 e-folding horizons and probability-mean control are preserved.
    # Checked checkpoint reconstruction, left padding, empty/singleton inputs,
    # truncation, category routing, RNG restoration and single-checkpoint use.
    # Verification failures propagate. No training or validation gain is claimed.
    ### <<< Self_EvolveRec-BLOCK-END


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### >>> Self_EvolveRec-BLOCK-START: Require every benchmark category before training
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError(
            "This benchmark requires all four Amazon categories exactly once"
        )
    scores = []
    ### <<< Self_EvolveRec-BLOCK-END
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


