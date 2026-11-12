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
        if num_layers < 1:
            raise ValueError("GRU4Rec requires at least one recurrent layer")
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden_units, hidden_units) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.interlayer_dropout = nn.Dropout(dropout_rate)
        # Initialize embeddings before new gate parameters consume RNG draws.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        self.retention = nn.ModuleList(
            [nn.Linear(2 * hidden_units, 1) for _ in range(num_layers)]
        )
        for gate in self.retention:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 3.0)

    def _recur(self, seq: torch.Tensor, bypass_retention: bool = False):
        valid = seq.ne(0)
        emb = self.dropout(self.item_emb(seq))
        states = [
            emb.new_zeros(seq.size(0), self.hidden_units)
            for _ in range(self.num_layers)
        ]
        # Skip shared leading padding while preserving embedding-dropout draws,
        # full output alignment, and a separate state for each recurrent layer.
        occupied_columns = valid.any(dim=0).nonzero(as_tuple=True)[0]
        first = (
            int(occupied_columns[0].item()) if occupied_columns.numel() else seq.size(1)
        )
        outputs = [emb.new_zeros(seq.size(0), self.hidden_units)] * first
        for t in range(first, seq.size(1)):
            active = valid[:, t, None]
            x = emb[:, t]
            for layer, cell in enumerate(self.cells):
                previous = states[layer]
                if bypass_retention:
                    recurrent = previous
                else:
                    alpha = torch.sigmoid(
                        self.retention[layer](torch.cat((previous, x), dim=-1))
                    )
                    recurrent = alpha * previous
                proposed = cell(x, recurrent)
                states[layer] = torch.where(active, proposed, previous)
                x = states[layer]
                if layer + 1 < self.num_layers:
                    x = self.interlayer_dropout(x)
            outputs.append(torch.where(active, states[-1], torch.zeros_like(x)))
        features = (
            torch.stack(outputs, dim=1)
            if outputs
            else emb.new_zeros(seq.size(0), 0, self.hidden_units)
        )
        return features, torch.stack(states, dim=0)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        return self._recur(seq)[0]

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
        feature = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return feature * last.ge(0).unsqueeze(-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_score(self.encode(seq)[0], candidates)

    @torch.inference_mode()
    def audit_cell_parity(self):
        # A temporary copied-weight reference checks implementation semantics only.
        # It is never trained, saved, or used to produce deployed scores.
        reference = nn.GRU(
            self.hidden_units,
            self.hidden_units,
            self.num_layers,
            batch_first=True,
            dropout=self.interlayer_dropout.p if self.num_layers > 1 else 0.0,
        ).to(device=self.item_emb.weight.device, dtype=self.item_emb.weight.dtype)
        reference.eval()
        for layer, cell in enumerate(self.cells):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(reference, f"{name}_l{layer}").copy_(getattr(cell, name))
        prior_modes = [(module, module.training) for module in self.modules()]
        self.eval()
        max_difference = 0.0
        ranking_checks = []
        try:
            device = self.item_emb.weight.device
            candidates = torch.arange(1, min(self.itemnum, 16) + 1, device=device)
            for history in ([], [1], [1, min(2, self.itemnum), 1]):
                for width in (3, 6, 10):
                    seq = torch.zeros(1, width, dtype=torch.long, device=device)
                    if history:
                        seq[0, -len(history) :] = torch.tensor(history, device=device)
                    actual, states = self._recur(seq, bypass_retention=True)
                    expected = torch.zeros_like(actual)
                    expected_states = torch.zeros_like(states)
                    if history:
                        compact = seq[:, -len(history) :]
                        with torch.backends.cudnn.flags(enabled=False):
                            output, expected_states = reference(self.item_emb(compact))
                        expected[:, -len(history) :] = output
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(
                        states, expected_states, atol=1e-5, rtol=1e-4
                    )
                    a = self.predict_score(actual[0, -1], candidates)
                    b = self.predict_score(expected[0, -1], candidates)
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                    max_difference = max(
                        max_difference, float((actual - expected).abs().max())
                    )
                    ranking_checks.append(
                        bool(
                            torch.equal(
                                a.argsort(descending=True), b.argsort(descending=True)
                            )
                        )
                    )
            return {
                "bypass_parity": "passed",
                "atol": 1e-5,
                "rtol": 1e-4,
                "max_feature_abs_difference": max_difference,
                "ranking_agreement": all(ranking_checks),
                "ranking_note": "Numerically tied scores can have different orders",
            }
        finally:
            for module, training in prior_modes:
                module.training = training


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


### >>> Self_EvolveRec-BLOCK-START: Load the model definition embedded by the predictor writer
def _load_model_class():
    return GRU4RecModel
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    ### >>> Self_EvolveRec-BLOCK-START: Require every category checkpoint before completing predictor loading
    if all(ds in _models for ds in DATASETS):
        return
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on missing category artifacts
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the identical checkpoint architecture before evaluation mode
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve category routing without silently substituting another trained model
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
