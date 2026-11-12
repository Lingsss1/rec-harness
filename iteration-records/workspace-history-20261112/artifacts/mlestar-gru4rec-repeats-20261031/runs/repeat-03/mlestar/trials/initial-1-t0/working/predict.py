from __future__ import annotations
import os
import pickle
import torch
from torch import nn

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
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1).clamp_min(0)
        return features[torch.arange(sequence.size(0), device=sequence.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(-1)
        if items.ndim == 1:
            return features @ embeddings.T
        if items.ndim == features.ndim:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        raise ValueError('Unsupported feature/item shapes')

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == 3:
            negative = torch.bmm(features, self.item_emb(neg[:, -1, :]).transpose(1, 2))
        else:
            negative = (features * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        model.load_state_dict(torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True))
        model.eval()
        _models[dataset_name] = model
        _metas[dataset_name] = meta
    return _models[dataset_name], _metas[dataset_name]


def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model, meta = _load(ds)
    maxlen = meta['maxlen']
    history = list(history)[-maxlen:]
    sequence = [0] * (maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand)
    return scores.cpu().tolist()
