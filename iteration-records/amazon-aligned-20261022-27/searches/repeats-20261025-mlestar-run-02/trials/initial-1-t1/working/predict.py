from __future__ import annotations
import os
import math
import pickle
import torch
from torch import nn
from torch.nn import functional as F

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.dropout1(self.conv1(x.transpose(1, 2)))
        y = self.dropout2(self.conv2(F.relu(y)))
        return x + y.transpose(1, 2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.3, maxlen=128):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
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
        self.register_buffer('causal_mask', torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        mask = (seq != 0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions)) * mask
        for norm, attention, ffn_norm, ffn in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            q = norm(x)
            y, _ = attention(q, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)
            x = q + y
            x = ffn(ffn_norm(x)) * mask
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        z = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * z.unsqueeze(1)).sum(-1)



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _load(dataset):
    global _device
    if dataset in _models:
        return _models[dataset], _metas[dataset]
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = SASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta
    return model, meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model, meta = _load(dataset)
    maxlen = meta['maxlen']
    hist = list(history[-maxlen:])
    hist = [int(i) if 0 < int(i) <= meta['itemnum'] else 0 for i in hist]
    seq = torch.tensor([[0] * (maxlen - len(hist)) + hist], dtype=torch.long, device=_device)
    valid = [0 < int(i) <= meta['itemnum'] for i in candidates]
    ids = [int(i) if ok else 0 for i, ok in zip(candidates, valid)]
    with torch.inference_mode():
        c = torch.tensor([ids], dtype=torch.long, device=_device)
        scores = model.predict(seq, c)[0].cpu().tolist()
    return [float(s) if ok else -1e30 for s, ok in zip(scores, valid)]
