from __future__ import annotations
import os
import json
import math
import torch
from torch import nn
import torch.nn.functional as F

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = F.relu(self.dropout1(outputs))
        outputs = self.dropout2(self.conv2(outputs))
        return inputs + outputs.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1, dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0.0, 1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen, dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device)
        mask = seq.ne(0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions)) * mask
        for norm, attn, ffn_norm, ffn in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            q = norm(x)
            attended = attn(q, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)[0]
            x = ffn(ffn_norm(q + attended)) * mask
        return self.last_layernorm(x)

    def forward(self, seq, pos=None, neg=None):
        features = self.log2feats(seq)
        if pos is None or neg is None:
            return features
        positive = (features * self.item_emb(pos)).sum(-1)
        negative = (features * self.item_emb(neg)).sum(-1)
        return positive, negative

    def predict(self, seq, candidates):
        final = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * final.unsqueeze(1)).sum(-1)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(name):
    global _device
    if name in _models:
        return
    if _device is None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, name + '_meta.json')) as handle:
        meta = json.load(handle)
    model = SASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[name] = model
    _metas[name] = meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    _ensure_loaded(name)
    meta = _metas[name]
    maxlen = meta['maxlen']
    hist = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in history[-maxlen:]]
    seq = torch.tensor([[0] * (maxlen - len(hist)) + hist], dtype=torch.long, device=_device)
    raw = torch.tensor([candidates], dtype=torch.long, device=_device)
    valid = (raw > 0) & (raw <= meta['itemnum'])
    scores = _models[name].predict(seq, raw.clamp(0, meta['itemnum']))
    return scores.masked_fill(~valid, float('-inf'))[0].cpu().tolist()
