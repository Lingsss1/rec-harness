from __future__ import annotations
import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSTUBlock(nn.Module):
    """Causal UVQK transduction with SiLU attention and multiplicative gating."""
    def __init__(self, hidden_units, num_heads, dropout_rate, maxlen):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.maxlen = maxlen
        self.input_norm = nn.LayerNorm(hidden_units)
        self.uvqk = nn.Linear(hidden_units, 4 * hidden_units)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * maxlen - 1))
        self.attention_norm = nn.LayerNorm(hidden_units)
        self.output = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        positions = torch.arange(maxlen)
        relative = positions[:, None] - positions[None, :] + maxlen - 1
        self.register_buffer('relative_index', relative, persistent=False)
        self.register_buffer('causal', torch.ones(maxlen, maxlen, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, valid):
        batch, length, width = x.shape
        u, v, q, k = F.silu(self.uvqk(self.input_norm(x))).chunk(4, dim=-1)
        def split_heads(t):
            return t.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        bias = self.relative_bias[:, self.relative_index[:length, :length]]
        weights = F.silu(torch.matmul(q, k.transpose(-1, -2)) + bias.unsqueeze(0)) / float(self.maxlen)
        allowed = self.causal[:length, :length][None, None, :, :]
        allowed = allowed & valid[:, None, None, :] & valid[:, None, :, None]
        weights = weights * allowed.to(weights.dtype)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        update = self.output(u * self.attention_norm(attended))
        return (x + self.dropout(update)) * valid.unsqueeze(-1).to(x.dtype)

class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('Hidden size must be divisible by number of heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            HSTUBlock(hidden_units, num_heads, dropout_rate, maxlen)
            for _ in range(num_blocks)
        ])
        self.last_norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        valid = sequence.ne(0)
        length = sequence.shape[1]
        positions = torch.arange(1, length + 1, device=sequence.device)[None, :] * valid.long()
        x = self.item_emb(sequence) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.emb_dropout(x) * valid.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, valid)
        return self.last_norm(x) * valid.unsqueeze(-1).to(x.dtype)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def final_features(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        last = torch.where(sequence.ne(0), positions, -1).amax(dim=1)
        result = features[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return result * last.ge(0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)


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
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    if meta['architecture'] != 'native_silu_hstu_v1':
        raise ValueError('Architecture mismatch')
    model = HSTURec(**meta['config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = os.environ.get('GAGC_DATASET')
        if dataset_name is None:
            configured = [s.strip() for s in os.environ.get('GAGC_DATASETS', '').split(',') if s.strip()]
            if len(configured) == 1:
                dataset_name = configured[0]
            else:
                raise ValueError('dataset_name is required for four-category routing')
    model = _load(dataset_name)
    seq = [0] * model.maxlen
    tail = [int(i) for i in history[-model.maxlen:]]
    if any(i < 0 or i > model.itemnum for i in tail):
        raise ValueError('History item outside dataset vocabulary')
    if tail:
        seq[-len(tail):] = tail
    with torch.inference_mode():
        features = model.final_features(torch.tensor([seq], dtype=torch.long, device=_device))
        items = torch.tensor(candidates, dtype=torch.long, device=_device)
        valid = (items >= 1) & (items <= model.itemnum)
        scores = model.predict_score(features, items.clamp(0, model.itemnum))[0]
        scores = scores.masked_fill(~valid, float('-inf'))
        return scores.float().cpu().tolist()
