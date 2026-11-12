from __future__ import annotations
import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
            for i in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        # Retain original absolute indices when empty leading columns are trimmed.
        positions = torch.arange(self.maxlen - length + 1, self.maxlen + 1, device=seq.device)
        positions = positions.unsqueeze(0).expand(batch, -1) * seq.ne(0).long()
        mask = seq.ne(0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if features.ndim == embeddings.ndim and features.shape[:-1] == embeddings.shape[:-1]:
            return (features * embeddings).sum(-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        negative = torch.bmm(features, self.item_emb(neg).transpose(1, 2))
        return positive, negative

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(1).values.clamp_min(0)
        final = features[torch.arange(seq.shape[0], device=seq.device), last]
        embeddings = self.item_emb(candidates)
        if candidates.ndim == 1:
            scores = final @ embeddings.t()
        else:
            scores = (final.unsqueeze(1) * embeddings).sum(-1)
        return scores[0] if seq.shape[0] == 1 else scores



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in _DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = NextItNetModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if len(candidates) == 0:
        return []
    model = _models[dataset]
    maxlen = int(_metas[dataset]['maxlen'])
    history = list(history)[-maxlen:]
    sequence = [0] * (maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
