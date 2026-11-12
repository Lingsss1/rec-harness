from __future__ import annotations
import math
import json
import os
import torch
import torch.nn as nn

class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

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
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        mask = seq.ne(0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        emb = self.item_emb(items)
        if emb.ndim == features.ndim:
            return (features * emb).sum(-1)
        if emb.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * emb).sum(-1)
        if features.ndim == 1 and emb.ndim == 2:
            return emb @ features
        raise ValueError('Incompatible feature and item shapes')

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(1).values.clamp(min=0)
        rows = torch.arange(seq.size(0), device=seq.device)
        final = features[rows, last]
        emb = self.item_emb(candidates)
        if candidates.ndim == 1:
            result = final @ emb.transpose(0, 1)
        else:
            result = (final.unsqueeze(1) * emb).sum(-1)
        return result[0] if seq.size(0) == 1 else result



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
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.json')) as f:
        meta = json.load(f)
    model = NextItNetModel(**meta).to(_device)
    checkpoint = os.path.join(_HERE, dataset_name + '_model.pt')
    model.load_state_dict(torch.load(checkpoint, map_location=_device, weights_only=True))
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    _ensure_loaded(ds)
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    suffix = list(history[-maxlen:])
    if suffix:
        sequence[-len(suffix):] = suffix
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    scores = _models[ds].score_candidates(seq, cand)
    return [float(s) for s in scores.cpu().tolist()]
