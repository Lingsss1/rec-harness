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
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        return self.gru(self.dropout(self.item_emb(sequence)))[0]

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.size(0), device=sequence.device), last]

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == seq.ndim + 1:
            negative = (features.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        else:
            negative = (features * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, sequence, candidates):
        return torch.matmul(self.item_emb(candidates), self.encode(sequence)[0])


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for dataset in DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                             meta['num_layers'], meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else DATASETS[0]
    if dataset not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _models[dataset]
    meta = _metas[dataset]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    history = list(history[-maxlen:])
    if history:
        sequence[-len(history):] = history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
