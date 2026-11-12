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
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        embedded = self.dropout(self.item_emb(sequence))
        output, _ = self.gru(embedded)
        return output

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values
        result = features[torch.arange(sequence.size(0), device=sequence.device), last.clamp(min=0)]
        return result * last.ge(0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        embedded = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embedded.transpose(0, 1))
        return (features.unsqueeze(-2) * embedded).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if seq.size(0) == 1 else scores


_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(itemnum=meta['itemnum'],
                             hidden_units=meta['hidden_units'],
                             num_layers=meta['num_layers'],
                             dropout_rate=meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset_name: {ds}')
    if not candidates:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_device)
    items_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, items_t)
    return [float(s) for s in scores.cpu().tolist()]
