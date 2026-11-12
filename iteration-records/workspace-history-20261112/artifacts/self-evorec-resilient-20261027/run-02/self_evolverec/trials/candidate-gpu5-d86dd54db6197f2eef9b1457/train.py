"""Per-dataset SASRec — trains one model per Amazon Reviews dataset."""

### >>> Self_EvolveRec-BLOCK-START: Activate cross-fitted diagnostics and export their required module
from __future__ import annotations

import itertools
import json

import torch
import torch.nn.functional as F


class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)

    def _sequence_folds(self):
        existing = getattr(self.model, "training_sequence_folds", {})
        if existing:
            return existing
        histories = self.view.user_train
        histories = histories.get("History", histories)
        histories = {
            int(u): [int(i) for i in history] for u, history in histories.items()
        }
        import numpy as np

        users = sorted(histories)
        permutation = np.random.RandomState(42).permutation(len(users))
        folds = {users[int(index)]: rank % 5 for rank, index in enumerate(permutation)}
        result = {}
        for user, history in histories.items():
            if len(history) < 2:
                continue
            key = tuple(history[:-1][-self.model.maxlen :])
            fold = folds[user]
            previous = result.get(key, fold)
            result[key] = fold if previous == fold else -1
        return result

    def run_full_diagnosis(self, train_loader):
        model = self.model
        required = ("score", "transition_evidence", "recent_contexts", "log2feats")
        if any(not callable(getattr(model, name, None)) for name in required):
            raise TypeError(
                "Diagnosis requires the trained transition-aware SASRec model"
            )
        was_training = model.training
        model.eval()
        metrics = {}
        definitions = {}
        collected = {
            "raw_dot_margin": [],
            "complete_score_margin": [],
            "context_coverage": [],
            "positive_edge_coverage": [],
            "absolute_residual": [],
            "deployment_training_residual_MAE": [],
            "singleton_user_support_margin": [],
            "multiple_user_support_margin": [],
        }
        sequence_folds = self._sequence_folds()
        unresolved_rows = 0
        sampled_positions = 0
        try:
            with torch.no_grad():
                for _, sequence, positive, negative in itertools.islice(
                    train_loader, 4
                ):
                    sequence = torch.as_tensor(
                        sequence, dtype=torch.long, device=self.device
                    )
                    positive = torch.as_tensor(
                        positive, dtype=torch.long, device=self.device
                    )
                    negative = torch.as_tensor(
                        negative, dtype=torch.long, device=self.device
                    )
                    fold_list = []
                    for row in sequence.detach().cpu().tolist():
                        key = tuple(int(i) for i in row if i != 0)
                        fold_list.append(sequence_folds.get(key, -1))
                    folds = torch.tensor(
                        fold_list, dtype=torch.long, device=self.device
                    )
                    unresolved_rows += int((folds < 0).sum())
                    valid = (positive != 0) & (folds[:, None] >= 0)
                    locations = valid.nonzero(as_tuple=False)
                    if not locations.numel():
                        continue
                    if locations.shape[0] > 512:
                        indices = torch.linspace(
                            0, locations.shape[0] - 1, 512, device=self.device
                        ).long()
                        locations = locations[indices]
                    b, t = locations.unbind(dim=1)
                    features = model.log2feats(sequence)[b, t]
                    contexts = model.recent_contexts(sequence)[b, t]
                    candidates = torch.stack((positive[b, t], negative[b, t]), dim=1)
                    selectors = folds[b]
                    raw = (model.item_emb(candidates) * features.unsqueeze(1)).sum(-1)
                    complete = model.score(features, contexts, candidates, selectors)
                    residual, coverage, support = model.transition_evidence(
                        contexts, candidates, selectors
                    )
                    deployment_residual, _, _ = model.transition_evidence(
                        contexts, candidates
                    )
                    margin = complete[:, 0] - complete[:, 1]
                    collected["raw_dot_margin"].append(raw[:, 0] - raw[:, 1])
                    collected["complete_score_margin"].append(margin)
                    collected["context_coverage"].append(coverage)
                    collected["positive_edge_coverage"].append(
                        (support[:, 0] > 0).float()
                    )
                    collected["absolute_residual"].append(residual.abs().reshape(-1))
                    collected["deployment_training_residual_MAE"].append(
                        (deployment_residual - residual).abs().reshape(-1)
                    )
                    collected["singleton_user_support_margin"].append(
                        margin[support[:, 0] == 1]
                    )
                    collected["multiple_user_support_margin"].append(
                        margin[support[:, 0] >= 2]
                    )
                    sampled_positions += int(locations.shape[0])
                descriptions = {
                    "raw_dot_margin": "Mean positive-minus-negative embedding dot-product score before transition evidence.",
                    "complete_score_margin": "Mean positive-minus-negative complete score using each training user's excluded-fold graph.",
                    "context_coverage": "Mean recency-weighted share of context sources with retained outgoing edges in the user-excluding graph.",
                    "positive_edge_coverage": "Fraction of sampled positive targets with a retained edge from at least one recent context item.",
                    "absolute_residual": "Mean absolute pooled transition residual before multiplication by the learned gate, across positive and negative candidates.",
                    "deployment_training_residual_MAE": "Mean absolute difference between deployment-average residual and user-excluding residual on sampled training positions; this is a training-side distribution-shift diagnostic.",
                    "singleton_user_support_margin": "Mean complete margin where the maximum distinct-other-user support among retained positive-target context edges is exactly one.",
                    "multiple_user_support_margin": "Mean complete margin where at least one retained positive-target context edge has support from two or more distinct other training users.",
                }
                unavailable = {}
                for name, parts in collected.items():
                    parts = [part.reshape(-1) for part in parts if part.numel()]
                    definitions[name] = descriptions[name]
                    if parts:
                        values = torch.cat(parts).float()
                        metrics[name] = float(values.mean())
                        metrics[name + "_count"] = int(values.numel())
                        definitions[name + "_count"] = (
                            "Number of scalar observations used for " + name + "."
                        )
                    else:
                        unavailable[name] = (
                            "No eligible observations in the bounded diagnostic sample."
                        )
                embedding = model.item_emb.weight[1:]
                if embedding.shape[0] >= 2:
                    generator = torch.Generator(device="cpu").manual_seed(42)
                    pairs = torch.randint(
                        0, embedding.shape[0], (1024, 2), generator=generator
                    )
                    pairs = pairs[pairs[:, 0] != pairs[:, 1]].to(embedding.device)
                    if pairs.numel():
                        cosine = F.cosine_similarity(
                            embedding[pairs[:, 0]], embedding[pairs[:, 1]], dim=-1
                        )
                        metrics["embedding_cosine_mean"] = float(cosine.mean())
                        metrics["embedding_cosine_std"] = float(
                            cosine.std(unbiased=False)
                        )
                        definitions["embedding_cosine_mean"] = (
                            "Mean cosine similarity of seed-42 sampled distinct nonpadding item pairs; alone this does not establish embedding collapse."
                        )
                        definitions["embedding_cosine_std"] = (
                            "Population standard deviation of the sampled pair cosine similarities."
                        )
                metrics["sampled_positions"] = sampled_positions
                metrics["unresolved_training_rows"] = unresolved_rows
                definitions["sampled_positions"] = (
                    "Number of supervised positions measured from at most four batches and 512 positions per batch."
                )
                definitions["unresolved_training_rows"] = (
                    "Rows excluded because their exact sampled training history could not be assigned unambiguously to a user fold."
                )
                unavailable.update(
                    {
                        "category_breakdown": "Item category metadata is unavailable in this matched task.",
                        "review_preferences": "Review ratings and text are unavailable.",
                        "simulator_traits": "Diversity and conformity traits are unobserved.",
                    }
                )
                return {
                    "metrics": metrics,
                    "metric_definitions": definitions,
                    "unavailable": unavailable,
                    "limitations": [
                        "Only public training histories contribute to transition graphs.",
                        "Diagnostics use a bounded training sample and do not establish validation improvement.",
                        "Distinct-user support is per directed edge, not a union of users across context edges.",
                        "Validation JSON records constituent-fold versus deployment score shift and HR@10 on fixed validation candidates.",
                        "The host independently computes the benchmark selection metric.",
                    ],
                }
        finally:
            model.train(was_training)


def diagnosis_interpreter_prompt(raw_diagnosis):
    return (
        "Interpret only the measured diagnostics and their supplied definitions below. "
        "Separate raw embedding margins from complete transition-aware margins. "
        "Discuss coverage, residual magnitude, user-support strata, and measured graph distribution shift. "
        "Do not infer benchmark improvement from training margins or diagnose severe collapse from "
        "a cosine mean alone. Respect unavailable fields and sampling limitations. "
        "Do not invent review preferences, category metadata, simulator traits, or test results. "
        "Keep uncertain recommendations separate from explicit implementation mismatches.\n\n"
        + json.dumps(raw_diagnosis, indent=2, ensure_ascii=False, allow_nan=False)
    )


### <<< Self_EvolveRec-BLOCK-END
"""Architecture (aligned with LLM-SRec SeqRec/sasrec, adapted for the RecHarness raw .txt data format):
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

### >>> Self_EvolveRec-BLOCK-START: Keep the future import exclusively at the beginning of the module
import json
import math
import os
import pickle
import random

### <<< Self_EvolveRec-BLOCK-END

### >>> Self_EvolveRec-BLOCK-START: Count training-only directed transitions and distinct-user support
from collections import Counter, defaultdict

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
### >>> Self_EvolveRec-BLOCK-START: Match the host epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the matched benchmark data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four matched categories
_requested_datasets = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]
if set(_requested_datasets) != set(DEFAULT_DATASETS):
    raise ValueError("This matched task requires all four Amazon categories")
DATASETS = DEFAULT_DATASETS.copy()

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


### >>> Self_EvolveRec-BLOCK-START: Retain internal user identities for user-excluding graph selection
def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No training users have two interactions")
    batch_seq, batch_pos, batch_neg, batch_users = [], [], [], []
    ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Return fold-routing identities only to the training workflow
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
        batch_users.append(u)
    result = (np.array(batch_seq), np.array(batch_pos), np.array(batch_neg))
    return (*result, batch_users) if return_users else result


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


### >>> Self_EvolveRec-BLOCK-START: Build five user-disjoint shrunk directed-transition graphs
TRANSITION_CONFIG = {
    "window": 5,
    "beta": 0.5,
    "alpha": 10.0,
    "tau": 5.0,
    "minimum_count": 2,
    "neighbors": 64,
    "clip": 3.0,
    "folds": 5,
    "seed": 42,
}


def _transition_graphs(user_train, itemnum):
    users = sorted(user_train)
    permutation = np.random.RandomState(42).permutation(len(users))
    user_folds = {users[int(index)]: rank % 5 for rank, index in enumerate(permutation)}
    total, support = Counter(), Counter()
    excluded = [Counter() for _ in range(5)]
    excluded_support = [Counter() for _ in range(5)]
    source = np.zeros(itemnum + 1, dtype=np.int64)
    destination = np.zeros(itemnum + 1, dtype=np.int64)
    fold_source = np.zeros((5, itemnum + 1), dtype=np.int64)
    fold_destination = np.zeros((5, itemnum + 1), dtype=np.int64)
    for user in users:
        history = [int(i) for i in user_train[user]]
        if any(i < 1 or i > itemnum for i in history):
            raise ValueError("Training item outside the original item mapping")
        pairs = list(zip(history, history[1:]))
        fold = user_folds[user]
        counts = Counter(pairs)
        total.update(counts)
        excluded[fold].update(counts)
        support.update(set(pairs))
        excluded_support[fold].update(set(pairs))
        for (i, j), count in counts.items():
            source[i] += count
            destination[j] += count
            fold_source[fold, i] += count
            fold_destination[fold, j] += count

    graphs = []
    for fold in range(5):
        # All contributions of every user in this fold are removed before
        # smoothing, support thresholds, shrinkage, or pruning.
        n = source - fold_source[fold]
        d = destination - fold_destination[fold]
        assert np.all(n >= 0) and np.all(d >= 0)
        q = (d.astype(np.float64) + 1.0) / (float(d.sum()) + itemnum)
        rows = defaultdict(list)
        for (i, j), total_count in total.items():
            count = total_count - excluded[fold].get((i, j), 0)
            assert count >= 0
            if count < 2:
                continue
            probability = (count + 10.0 * q[j]) / (float(n[i]) + 10.0)
            lift = float(np.clip(np.log(probability / q[j]), -3.0, 3.0))
            weight = count / (count + 5.0) * lift
            users_support = support[(i, j)] - excluded_support[fold].get((i, j), 0)
            assert users_support >= 1
            if weight != 0.0:
                rows[i].append((j, weight, users_support))
        entries = []
        active_rows = torch.zeros(itemnum + 1, dtype=torch.bool)
        for i, edges in rows.items():
            edges.sort(key=lambda edge: (-abs(edge[1]), edge[0]))
            retained = edges[:64]
            active_rows[i] = bool(retained)
            entries.extend(
                (i * (itemnum + 1) + j, weight, count) for j, weight, count in retained
            )
        entries.sort(key=lambda edge: edge[0])
        graphs.append(
            (
                torch.tensor([e[0] for e in entries], dtype=torch.long),
                torch.tensor([e[1] for e in entries], dtype=torch.float32),
                torch.tensor([e[2] for e in entries], dtype=torch.long),
                active_rows,
            )
        )
    return graphs, user_folds


class SASRec(nn.Module):
    ### <<< Self_EvolveRec-BLOCK-END
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

        ### >>> Self_EvolveRec-BLOCK-START: Serialize transition estimators, fix padded contexts, average deployment coverage, and bound compatibility scoring memory
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.gate_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.gate_weight = nn.Parameter(torch.zeros(hidden_units))
        self.gate_coverage = nn.Parameter(torch.zeros(()))
        self.gate_bias = nn.Parameter(torch.tensor(math.log(0.01 / 0.99)))
        for fold in range(5):
            self.register_buffer(
                f"transition_keys_{fold}", torch.empty(0, dtype=torch.long)
            )
            self.register_buffer(f"transition_values_{fold}", torch.empty(0))
            self.register_buffer(
                f"transition_support_{fold}", torch.empty(0, dtype=torch.long)
            )
            self.register_buffer(
                f"transition_rows_{fold}", torch.zeros(itemnum + 1, dtype=torch.bool)
            )
        self.user_folds = {}
        self.training_sequence_folds = {}
        self._bound_features = None
        self._bound_sequence = None

    def configure_transitions(self, user_train):
        graphs, self.user_folds = _transition_graphs(user_train, self.itemnum)
        device = self.item_emb.weight.device
        for fold, graph in enumerate(graphs):
            for name, tensor in zip(("keys", "values", "support", "rows"), graph):
                setattr(self, f"transition_{name}_{fold}", tensor.to(device))
        self.training_sequence_folds = {}
        for user, history in user_train.items():
            if len(history) < 2:
                continue
            key = tuple(int(i) for i in history[:-1][-self.maxlen :])
            fold = self.user_folds[user]
            previous = self.training_sequence_folds.get(key, fold)
            self.training_sequence_folds[key] = fold if previous == fold else -1

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Registered sparse buffers have data-dependent lengths. Resize before
        # the ordinary strict loader copies their checkpoint contents.
        for fold in range(5):
            for kind in ("keys", "values", "support", "rows"):
                name = f"transition_{kind}_{fold}"
                key = prefix + name
                if key in state_dict:
                    current = getattr(self, name)
                    setattr(
                        self,
                        name,
                        torch.empty(
                            state_dict[key].shape,
                            dtype=state_dict[key].dtype,
                            device=current.device,
                        ),
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def recent_contexts(self, sequence):
        # Zero-valued padding contributes nothing. Unlike scatter_, additive
        # scatter cannot overwrite an earlier item at an internal padding rank.
        batch, length = sequence.shape
        ranks = (sequence != 0).long().cumsum(dim=1)
        compressed = sequence.new_zeros(batch, length + 1)
        compressed.scatter_add_(1, ranks, sequence)
        offsets = torch.arange(5, device=sequence.device)
        indices = ranks.unsqueeze(-1) - offsets
        contexts = compressed.gather(1, indices.clamp_min(0).reshape(batch, -1))
        contexts = contexts.reshape(batch, length, 5)
        return contexts.masked_fill(indices <= 0, 0)

    def _edge_lookup(self, fold, query):
        keys = getattr(self, f"transition_keys_{fold}")
        values = getattr(self, f"transition_values_{fold}")
        support = getattr(self, f"transition_support_{fold}")
        if keys.numel() == 0:
            return (
                torch.zeros_like(query, dtype=self.item_emb.weight.dtype),
                torch.zeros_like(query),
            )
        indices = torch.searchsorted(keys, query.contiguous())
        safe = indices.clamp_max(keys.numel() - 1)
        matched = (indices < keys.numel()) & (keys[safe] == query)
        return (
            torch.where(matched, values[safe], 0.0),
            torch.where(matched, support[safe], 0),
        )

    def transition_evidence(self, context_ids, candidate_ids, graph_selector=None):
        if context_ids.ndim != 2 or context_ids.shape[1] != 5:
            raise ValueError("Contexts must have shape positions x 5")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != context_ids.shape[0]:
            raise ValueError("Candidates must have shape positions x candidates")
        count = context_ids.shape[0]
        decay = torch.exp(
            -0.5
            * torch.arange(
                5, device=context_ids.device, dtype=self.item_emb.weight.dtype
            )
        )
        weights = (context_ids != 0).to(decay.dtype) * decay
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        residual = weights.new_zeros(candidate_ids.shape)
        coverage = weights.new_zeros(count)
        support = torch.zeros_like(candidate_ids)
        if graph_selector is not None:
            selectors = torch.as_tensor(
                graph_selector, device=context_ids.device, dtype=torch.long
            )
            if selectors.ndim == 0:
                selectors = selectors.expand(count)
            if selectors.shape != (count,) or bool(
                ((selectors < 0) | (selectors >= 5)).any()
            ):
                raise ValueError("Each position needs a valid user-excluding fold")
        else:
            selectors = None
        # Average source-row coverage with the same equal fold weights used
        # for deployment evidence, rather than using a denser union graph.
        for fold in range(5):
            selected = (
                torch.ones(count, dtype=torch.bool, device=context_ids.device)
                if selectors is None
                else selectors == fold
            )
            if not bool(selected.any()):
                continue
            contexts = context_ids[selected]
            candidates = candidate_ids[selected]
            query = contexts.unsqueeze(-1) * (self.itemnum + 1) + candidates.unsqueeze(
                1
            )
            values, users_support = self._edge_lookup(fold, query)
            values = values.masked_fill(contexts.unsqueeze(-1) == 0, 0.0)
            evidence = (weights[selected].unsqueeze(-1) * values).sum(dim=1)
            rows = getattr(self, f"transition_rows_{fold}")[contexts] & (contexts != 0)
            if selectors is None:
                # Exactly average the five estimators, including missing zeros.
                residual = residual + evidence / 5.0
                coverage = coverage + (weights[selected] * rows).sum(dim=1) / 5.0
                support = torch.maximum(support, users_support.max(dim=1).values)
            else:
                residual[selected] = evidence
                coverage[selected] = (weights[selected] * rows).sum(dim=1)
                support[selected] = users_support.max(dim=1).values
        return residual, coverage, support

    def score(self, features, context_ids, candidate_ids, graph_selector=None):
        if features.ndim != 2 or features.shape[0] != context_ids.shape[0]:
            raise ValueError(
                "Features and explicit contexts must describe the same positions"
            )
        base = (self.item_emb(candidate_ids) * features.unsqueeze(1)).sum(dim=-1)
        residual, coverage, _ = self.transition_evidence(
            context_ids, candidate_ids, graph_selector
        )
        gate_logit = (self.gate_norm(features) * self.gate_weight).sum(dim=-1)
        gate_logit = gate_logit + self.gate_coverage * coverage + self.gate_bias
        gate = 2.0 * torch.sigmoid(gate_logit)
        return base + gate.unsqueeze(-1) * residual

    def predict_score(self, features, items, context_ids=None, graph_selector=None):
        # Legacy calls must use the exact feature tensor returned by the latest
        # log2feats call. New callers should always supply explicit contexts.
        if context_ids is None:
            if features is not self._bound_features or self._bound_sequence is None:
                raise RuntimeError(
                    "Missing or stale context binding; pass context_ids explicitly"
                )
            context_ids = self.recent_contexts(self._bound_sequence)
        leading = features.shape[:-1]
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        single = tuple(items.shape) == tuple(leading)
        if single:
            items = items.unsqueeze(-1)
        elif items.ndim == 1:
            items = items.reshape(*([1] * len(leading)), -1).expand(*leading, -1)
        if tuple(items.shape[:-1]) != tuple(leading):
            raise ValueError("Candidate shape does not match feature positions")
        if tuple(context_ids.shape) != (*leading, 5):
            raise ValueError("Context shape does not match feature positions")
        if graph_selector is not None:
            graph_selector = torch.as_tensor(
                graph_selector, device=features.device, dtype=torch.long
            )
            if graph_selector.ndim:
                # A B-vector selects one graph per sequence, including every
                # supervised position in a B x L feature tensor.
                if len(leading) > 1 and graph_selector.shape == (leading[0],):
                    graph_selector = graph_selector.reshape(
                        leading[0], *([1] * (len(leading) - 1))
                    )
                graph_selector = graph_selector.expand(leading).reshape(-1)
        if items.shape[-1] == 0:
            return features.new_empty(*leading, 0)
        flat_features = features.reshape(-1, features.shape[-1])
        flat_contexts = context_ids.reshape(-1, 5)
        flat_items = items.reshape(flat_features.shape[0], items.shape[-1])
        # Bound full-catalog compatibility calls as well as model.predict.
        # Chunk both axes so B x L diagnostic calls cannot allocate a full
        # positions x catalog x hidden_units embedding tensor.
        row_results = []
        for start in range(0, flat_features.shape[0], 256):
            stop = start + 256
            selectors = graph_selector
            if selectors is not None and selectors.ndim:
                selectors = selectors[start:stop]
            candidate_results = []
            for chunk in flat_items[start:stop].split(1024, dim=1):
                candidate_results.append(
                    self.score(
                        flat_features[start:stop],
                        flat_contexts[start:stop],
                        chunk,
                        selectors,
                    )
                )
            row_results.append(torch.cat(candidate_results, dim=1))
        if not row_results:
            return (
                features.new_empty(items.shape[:-1])
                if single
                else features.new_empty(items.shape)
            )
        result = torch.cat(row_results, dim=0).reshape(*leading, items.shape[-1])
        return result.squeeze(-1) if single else result
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
                queries, seq_emb, seq_emb, attn_mask=attention_mask, need_weights=False
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        ### >>> Self_EvolveRec-BLOCK-START: Bind compatibility contexts and share complete scoring across training and inference
        features = self.last_layernorm(seq_emb)
        self._bound_features = features
        self._bound_sequence = seq.detach().clone()
        return features

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
        fold_ids=None,
    ):
        log_feats = self.log2feats(seq)
        if pos is None or neg is None:
            return log_feats
        if fold_ids is None:
            raise ValueError("Supervised training requires user-excluding fold_ids")
        valid = pos != 0
        contexts = self.recent_contexts(seq)[valid]
        selectors = fold_ids[:, None].expand_as(seq)[valid]
        candidates = torch.stack((pos[valid], neg[valid]), dim=-1)
        logits = self.score(log_feats[valid], contexts, candidates, selectors)
        pos_logits = log_feats.new_zeros(pos.shape)
        neg_logits = log_feats.new_zeros(neg.shape)
        pos_logits[valid] = logits[:, 0]
        neg_logits[valid] = logits[:, 1]
        return pos_logits, neg_logits

    def predict(
        self, seq: torch.Tensor, candidates: torch.Tensor, graph_selector=None
    ) -> torch.Tensor:
        features = self.log2feats(seq)[:, -1, :]
        contexts = self.recent_contexts(seq)[:, -1, :]
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(seq.shape[0], -1)
        if candidates.shape[0] != seq.shape[0]:
            raise ValueError("Candidates and sequences must have matching batches")
        if candidates.shape[1] == 0:
            return features.new_empty(seq.shape[0], 0)
        scores = []
        for chunk in candidates.split(1024, dim=1):
            scores.append(self.score(features, contexts, chunk, graph_selector))
        return torch.cat(scores, dim=1)
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
    ### >>> Self_EvolveRec-BLOCK-START: Freeze validation sampling without perturbing training randomness
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)

    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    fold_deployment_shift = 0.0
    fold_hits = [0.0] * 5
    ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Preserve unique validation negatives with a local evaluation RNG
            if itemnum - len(interacted) < 99:
                raise ValueError(
                    "Validation requires 99 unique unseen negative candidates"
                )
            while len(neg_pool) < 99:
                ni = rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Measure constituent-graph versus deployment score shift on validation
            features = model.log2feats(seq_t)[:, -1, :]
            contexts = model.recent_contexts(seq_t)[:, -1, :]
            deployment = model.score(features, contexts, cand_t)
            scores = deployment[0].cpu().tolist()
            constituent_scores = []
            for fold in range(5):
                fold_scores = model.score(features, contexts, cand_t, fold)
                constituent_scores.append(fold_scores)
                values = fold_scores[0].cpu().tolist()
                fold_rank = sorted(
                    range(len(candidates)), key=lambda i: -values[i]
                ).index(0)
                fold_hits[fold] += float(fold_rank < 10)
            fold_deployment_shift += float(
                (torch.stack(constituent_scores) - deployment.unsqueeze(0)).abs().mean()
            )
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)
            ### <<< Self_EvolveRec-BLOCK-END

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    model.train()
    ### >>> Self_EvolveRec-BLOCK-START: Keep HR@10 selection separate from graph-shift diagnostics
    result = {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
        "fold_deployment_score_MAE": fold_deployment_shift / n,
    }
    result.update(
        {f"constituent_fold_{f}_HR@10": hits / n for f, hits in enumerate(fold_hits)}
    )
    return result
    ### <<< Self_EvolveRec-BLOCK-END


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

    ### >>> Self_EvolveRec-BLOCK-START: Construct graph features exclusively from original training histories
    model.configure_transitions(user_train)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even when HR is zero
    best_hr10 = -1.0
    best_epoch = 0
    validation_history = []
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Route every supervised position to its user's excluded graph
            seq, pos, neg, batch_users = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, return_users=True
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            fold_ids = torch.tensor(
                [model.user_folds[u] for u in batch_users],
                dtype=torch.long,
                device=device,
            )

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t, fold_ids=fold_ids)
            ### <<< Self_EvolveRec-BLOCK-END
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

        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Record every evaluated epoch and graph-distribution diagnostic
            hr10 = val_m["HR@10"]
            validation_history.append({"epoch": epoch, **val_m})
            with open(
                os.path.join(working_dir, f"{dataset_name}_validation.json"), "w"
            ) as f:
                json.dump(
                    {
                        "selection_metric": "HR@10",
                        "seed": 42,
                        "transition_config": TRANSITION_CONFIG,
                        "evaluations": validation_history,
                    },
                    f,
                    indent=2,
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
                ### >>> Self_EvolveRec-BLOCK-START: Export architecture configuration and original item-ID semantics
                "maxlen": MAXLEN,
                "architecture": "sasrec_transition_v1",
                "transition_config": TRANSITION_CONFIG.copy(),
                "item_id_mapping": "identity: original IDs 1 through itemnum; padding 0",
                "user_folds": model.user_folds,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Verify strict checkpoint reload and candidate versus catalog scoring
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    verification_user = next(u for u, history in user_train.items() if history)
    verification_sequence = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
    history = user_train[verification_user][-MAXLEN:]
    verification_sequence[0, -len(history) :] = torch.tensor(history, device=device)
    catalog = torch.arange(1, itemnum + 1, device=device).unsqueeze(0)
    candidate_indices = torch.tensor(
        sorted(set((0, itemnum // 2, itemnum - 1))), device=device
    )
    candidates = catalog[:, candidate_indices]
    with torch.no_grad():
        before_reload = model.predict(verification_sequence, candidates)
        catalog_scores = model.predict(verification_sequence, catalog)
        torch.testing.assert_close(
            before_reload, catalog_scores[:, candidate_indices], atol=2e-5, rtol=2e-5
        )
        # Exercise the same fresh-instance, variable-buffer loading path used by
        # generated inference, rather than only reloading an initialized graph.
        restored = SASRec(
            itemnum, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE, MAXLEN
        ).to(device)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        after_reload = restored.predict(verification_sequence, candidates)
        torch.testing.assert_close(before_reload, after_reload, atol=2e-5, rtol=2e-5)
        del restored, catalog_scores
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Export executable diagnostic adapters before training begins


def write_diagnosis_script(working_dir: str) -> None:
    import inspect

    # Export the active implementations, not text hidden inside a docstring.
    # Keep an adapter beside train.py and a trial-local copy beside predict.py.
    source = (
        "from __future__ import annotations\n\n"
        "import itertools\nimport json\nimport torch\n"
        "import torch.nn.functional as F\n\n"
        + inspect.getsource(DiagnosisProbe)
        + "\n\n"
        + inspect.getsource(diagnosis_interpreter_prompt)
    )
    compile(source, "diagnosis_tools.py", "exec")
    trial_dir = os.path.dirname(os.path.abspath(__file__))
    for directory in (trial_dir, working_dir):
        path = os.path.join(directory, "diagnosis_tools.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)


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
        ### >>> Self_EvolveRec-BLOCK-START: Require checkpoints containing the complete transition architecture
        if meta.get("architecture") != "sasrec_transition_v1":
            raise ValueError(f"Unsupported checkpoint architecture for {ds}")
        _metas[ds] = meta
        m = SASRec(
        ### <<< Self_EvolveRec-BLOCK-END
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Restore graph buffers and learned gates with strict loading
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True), strict=True)
        ### <<< Self_EvolveRec-BLOCK-END
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit category routes instead of using another category's model
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"No trained model for dataset_name={dataset_name!r}")
    if dataset_name and dataset_name in _models:
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

    ### >>> Self_EvolveRec-BLOCK-START: Materialize the required diagnostic module in the training workflow
    write_diagnosis_script(working_dir)
    all_hr10: list[float] = []
    ### <<< Self_EvolveRec-BLOCK-END
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


