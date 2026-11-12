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
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        else:
            items = items.to(device=features.device, dtype=torch.long)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
