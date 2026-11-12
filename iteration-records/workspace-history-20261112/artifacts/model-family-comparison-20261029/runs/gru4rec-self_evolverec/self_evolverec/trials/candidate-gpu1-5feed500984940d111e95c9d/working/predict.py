"""Per-dataset GRU4Rec predict function."""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch
import torch.nn as nn

class GRU4RecModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
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
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch_size, width = seq.shape
        if width == 0 or batch_size == 0:
            return self.item_emb.weight.new_zeros(batch_size, width, self.hidden_units)
        valid = seq.ne(0)
        lengths = valid.sum(dim=1)
        positions = torch.arange(width, device=seq.device).expand_as(seq)
        # Compact observed items in their original order. Packed recurrence never
        # processes leading padding, so GRU biases cannot alter the initial state.
        order = (positions + (~valid).long() * width).argsort(dim=1)
        embeddings = self.dropout(self.item_emb(seq))
        compact = embeddings.gather(
            1, order.unsqueeze(-1).expand(-1, -1, self.hidden_units)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        compact_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=width
        )
        out = torch.zeros_like(compact_out).scatter(
            1, order.unsqueeze(-1).expand_as(compact_out), compact_out
        )
        # The synthetic one-step input for an empty history must emit no feature.
        return out.masked_fill(~valid.unsqueeze(-1), 0.0)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return self.item_emb.weight.new_zeros(seq.size(0), self.hidden_units)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return features.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_score(self.encode(seq)[0], candidates)


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


### >>> Self_EvolveRec-BLOCK-START: Load the embedded trained GRU4Rec architecture
# DEBUG: Replace the unexpectedly indented placeholder with the missing loader.
# DEBUG: The predictor writer embeds GRU4RecModel before this function is called.
def _load_model_class():
    return GRU4RecModel
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    ### >>> Self_EvolveRec-BLOCK-START: Require every category to finish loading
    if all(ds in _models for ds in DEFAULT_DATASETS):
        return
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    ### >>> Self_EvolveRec-BLOCK-START: Preserve all four inference routes
    for ds in DEFAULT_DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
    ### <<< Self_EvolveRec-BLOCK-END
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Reject missing category artifacts instead of silently misrouting
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved architecture configuration
            dropout_rate=meta["dropout_rate"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Enforce explicit category routing when a category is supplied
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")
        ds = dataset_name
    elif _models:
        ds = next(iter(_models))
    else:
        raise RuntimeError("No GRU4Rec category models were loaded")
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
