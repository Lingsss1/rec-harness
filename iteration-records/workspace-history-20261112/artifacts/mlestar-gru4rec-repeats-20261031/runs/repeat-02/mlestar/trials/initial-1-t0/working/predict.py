from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        out, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return out

    def encode(self, seq):
        if not torch.is_tensor(seq):
            seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            seq = seq.to(device=self.item_emb.weight.device, dtype=torch.long)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        emb = self.item_emb(items)
        if emb.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * emb).sum(dim=-1)
        return (features * emb).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        candidates = torch.as_tensor(candidates, dtype=torch.long, device=features.device)
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).T
        else:
            scores = (features[:, None, :] * self.item_emb(candidates)).sum(-1)
        return scores[0] if features.size(0) == 1 else scores


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load(dataset_name):
    if dataset_name not in _DATASETS:
        raise ValueError("Unknown dataset: " + str(dataset_name))
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + "_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta["itemnum"], meta["hidden_units"], meta["num_layers"], meta["dropout_rate"]).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + "_model.pt"), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
        _metas[dataset_name] = meta
    return _models[dataset_name], _metas[dataset_name]


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    model, meta = _load(ds)
    maxlen = int(meta["maxlen"])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    ids = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, ids).cpu().tolist()
    return [float(value) for value in scores]
