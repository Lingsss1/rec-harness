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
        self.input_residual = nn.Linear(hidden_units, hidden_units, bias=False)
        self.residual_gate = nn.Parameter(torch.full((hidden_units,), -2.0))
        self.history_residual = nn.Linear(hidden_units, hidden_units, bias=False)
        self.history_gate = nn.Parameter(torch.full((hidden_units,), -2.0))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.eye_(self.input_residual.weight)
            nn.init.eye_(self.history_residual.weight)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch_size, width = sequence.shape
        mask = sequence.ne(0)
        lengths = mask.sum(dim=1)
        positions = torch.arange(width, device=sequence.device).expand_as(sequence)
        # Compact left padding before recurrence, keeping every feature causal.
        offsets = width - lengths
        compact_indices = (positions + offsets[:, None]).clamp(max=width - 1)
        compact = sequence.gather(1, compact_indices)
        compact_mask = positions < lengths[:, None]
        compact = compact.masked_fill(~compact_mask, 0)
        embedded = self.item_emb(compact)
        out, _ = self.gru(self.dropout(embedded))
        out = out + torch.sigmoid(self.residual_gate) * self.input_residual(embedded)
        # Each prefix mean includes only observed items up to this position.
        observed = embedded * compact_mask.unsqueeze(-1)
        counts = compact_mask.cumsum(dim=1).clamp(min=1).unsqueeze(-1)
        history_mean = observed.cumsum(dim=1) / counts.to(embedded.dtype)
        out = out + torch.sigmoid(self.history_gate) * self.history_residual(history_mean)
        # Restore the original positions for causal per-position training.
        restore_indices = (positions - offsets[:, None]).clamp(min=0)
        out = out.gather(1, restore_indices.unsqueeze(-1).expand(batch_size, width, self.hidden_units))
        return out * mask.unsqueeze(-1)

    def encode(self, seq):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        pos_scores = self.predict_score(features, pos)
        if neg.ndim == pos.ndim + 1:
            neg_scores = self.predict_score(features.unsqueeze(-2), neg)
        else:
            neg_scores = self.predict_score(features, neg)
        return pos_scores, neg_scores

    def score_candidates(self, seq, candidates):
        candidates = torch.as_tensor(candidates, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.encode(seq)
        if candidates.ndim == 1:
            if features.size(0) == 1:
                return self.item_emb(candidates) @ features[0]
            return features @ self.item_emb(candidates).transpose(0, 1)
        return self.predict_score(features.unsqueeze(1), candidates)


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


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset_name = dataset_name or DATASETS[0]
    _ensure_loaded(dataset_name)
    if len(candidates) == 0:
        return []
    model = _models[dataset_name]
    maxlen = _metas[dataset_name]['maxlen']
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
