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
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long,
                                       device=self.item_emb.weight.device)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.size(0), device=sequence.device), last]

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long,
                                    device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError('Unsupported feature/item shapes')

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).T
        else:
            scores = self.predict_score(features, candidates)
        return scores[0] if scores.shape[0] == 1 else scores


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name in _models:
        return _models[dataset_name], _metas[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                        meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta
    return model, meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model, meta = _load(dataset_name)
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t)
    return [float(score) for score in scores.cpu().tolist()]
