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
        embeddings = self.dropout(self.item_emb(sequence))
        output, _ = self.gru(embeddings)
        return output

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def encode(self, seq):
        output = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        return output[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        output = self.log2feats(seq)
        positives = (output * self.item_emb(pos)).sum(-1)
        if neg.ndim == seq.ndim + 1:
            negatives = (output.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        else:
            negatives = (output * self.item_emb(neg)).sum(-1)
        return positives, negatives

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                         meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    _ensure_loaded(ds)
    model = _models[ds]
    meta = _metas[ds]
    maxlen = int(meta['maxlen'])
    recent = list(history[-maxlen:])
    sequence = [0] * (maxlen - len(recent)) + recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(value) for value in scores.cpu().tolist()]
