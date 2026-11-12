"""Per-dataset HSTU predict — routes by dataset_name kwarg."""
from __future__ import annotations

import math
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = list(DEFAULT_DATASETS)

_models: dict = {}
_metas: dict = {}
_device = None


class HSTUBlock(nn.Module):
    """Causal HSTU with finite positional bias and multiplicative U gating.

    Temporal attention is deferred: the supplied sampler and public predictor
    provide item IDs without occurrence-aligned historical timestamps.
    """

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.norm = nn.LayerNorm(hidden_units)
        self.uvqk = nn.Linear(hidden_units, hidden_units * 4)
        # Signed distances are unnecessary because attention is inclusive causal.
        # Distances beyond 128 share the final learned positional bucket.
        self.position_bias = nn.Parameter(torch.zeros(num_heads, 129))
        self.aggregate_norm = nn.LayerNorm(hidden_units)
        self.output = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        seq_emb: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, hidden_units = seq_emb.shape
        u, v, q, k = F.silu(self.uvqk(self.norm(seq_emb))).chunk(4, dim=-1)

        def split_heads(tensor):
            return tensor.reshape(
                batch_size, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        positions = torch.arange(length, device=seq_emb.device)
        distances = (positions[:, None] - positions[None, :]).clamp(0, 128)
        bias = self.position_bias[:, distances]
        similarity = torch.matmul(q, k.transpose(-2, -1))
        similarity = similarity / math.sqrt(self.head_dim)
        # HSTU uses SiLU attention rather than softmax. Mask AFTER SiLU:
        # negative infinity before SiLU would create nonfinite values.
        weights = F.silu(similarity + bias.unsqueeze(0)) / max(length, 1)
        allowed = (
            (~causal_mask)[None, None, :, :]
            & (~padding_mask)[:, None, :, None]
            & (~padding_mask)[:, None, None, :]
        )
        weights = weights.masked_fill(~allowed, 0.0)
        aggregated = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .reshape(batch_size, length, hidden_units)
        )
        update = self.output(u * self.aggregate_norm(aggregated))
        seq_emb = seq_emb + self.dropout(update)
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
    ):
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError(
                f"hidden_units={hidden_units} must be divisible by num_heads={num_heads}"
            )
        self.hidden_units = hidden_units
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(hidden_units, num_heads, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.item_emb(seq) * math.sqrt(self.hidden_units)
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        ### >>> Self_EvolveRec-BLOCK-START: Finite padding states and shared compatibility scorer
        causal_mask = torch.triu(
            torch.ones((maxlen, maxlen), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        padding_mask = seq == 0
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, padding_mask)
            seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_norm(seq_emb).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def final_features(self, sequence: torch.Tensor) -> torch.Tensor:
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        final = (
            positions.unsqueeze(0).expand_as(sequence).masked_fill(sequence == 0, -1)
        )
        final = final.amax(dim=1)
        result = features[
            torch.arange(sequence.shape[0], device=sequence.device), final.clamp_min(0)
        ]
        return result.masked_fill((final < 0).unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded() -> None:
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    working_dir = _HERE
    for dataset_name in DATASETS:
        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing category checkpoint: {dataset_name}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("architecture") != "hstu_silu_position_v1":
            raise ValueError(f"Incompatible HSTU checkpoint: {dataset_name}")
        if meta.get("temporal_inputs") is not False:
            raise ValueError("This predictor requires position-only HSTU")
        _metas[dataset_name] = meta
        model = HSTURec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[dataset_name] = model


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"Missing category model: {dataset_name}")
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        first_dataset = next(iter(_models))
        model = _models[first_dataset]
        meta = _metas[first_dataset]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    seq_arr = [0] * maxlen
    for idx, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - idx] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        user_emb = model.final_features(seq_t)[0]
        scores = model.predict_score(user_emb, cand_t).cpu().tolist()
    return scores
