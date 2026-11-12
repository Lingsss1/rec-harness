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
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self._causal_conv(self.conv1, self.norm1(x).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        y = self._causal_conv(self.conv2, self.norm2(y).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128, dilations=None):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 1]
        if len(dilations) != num_blocks:
            raise ValueError('Dilation count must equal residual block count')
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, d, kernel_size, dropout_rate) for d in dilations
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(1, seq.size(1) + 1, device=seq.device).unsqueeze(0)
        positions = positions * (seq != 0).long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        x = x * mask
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x) * mask

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if features.ndim == 1:
            return torch.matmul(embeddings, features)
        if features.ndim == 2 and embeddings.ndim == 2:
            return features @ embeddings.transpose(0, 1)
        if features.ndim == 2 and embeddings.ndim == 3:
            return (features.unsqueeze(1) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 and candidates.ndim == 1 else scores



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None


def _load(dataset_name):
    global _device
    if dataset_name in _models:
        return _models[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = NextItNetModel(**meta['model_kwargs']).to(_device)
    checkpoint = os.path.join(_HERE, dataset_name + '_model.pt')
    model.load_state_dict(torch.load(checkpoint, map_location=_device, weights_only=True))
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(ds)
    sequence = [0] * model.maxlen
    recent = list(history[-model.maxlen:])
    if recent:
        sequence[-len(recent):] = recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
