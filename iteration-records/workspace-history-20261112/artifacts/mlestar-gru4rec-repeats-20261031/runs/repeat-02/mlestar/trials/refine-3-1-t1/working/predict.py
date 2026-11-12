from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        return self.gru(self.dropout(self.item_emb(sequence)))[0]

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1).clamp_min(0)
        result = features[torch.arange(sequence.size(0), device=sequence.device), last]
        return result * sequence.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def forward(self, sequence, pos, neg):
        features = self.log2feats(sequence)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, sequence, candidates):
        features = self.encode(sequence)
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).T
        else:
            scores = self.predict_score(features, candidates)
        return scores.squeeze(0)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in _DATASETS:
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, ds + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _models:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
