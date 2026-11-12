from __future__ import annotations
import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSTUBlock(nn.Module):
    """HSTU sequential transduction with native SiLU attention and U gating."""

    def __init__(self, hidden_units, num_heads, dropout_rate, maxlen):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.maxlen = maxlen
        self.norm = nn.LayerNorm(hidden_units)
        self.uvqk = nn.Linear(hidden_units, 4 * hidden_units)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * maxlen - 1))
        self.attn_norm = nn.LayerNorm(hidden_units)
        self.output = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        offsets = torch.arange(maxlen)[:, None] - torch.arange(maxlen)[None, :]
        self.register_buffer('relative_index', offsets + maxlen - 1, persistent=False)
        self.register_buffer('causal', torch.ones(maxlen, maxlen, dtype=torch.bool).tril(), persistent=False)
        nn.init.xavier_uniform_(self.uvqk.weight)
        nn.init.zeros_(self.uvqk.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, seq_emb, valid):
        batch, length, hidden = seq_emb.shape
        u, v, q, k = F.silu(self.uvqk(self.norm(seq_emb))).chunk(4, dim=-1)
        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        bias = self.relative_bias[:, self.relative_index[:length, :length]]
        logits = torch.matmul(q, k.transpose(-1, -2)) + bias.unsqueeze(0)
        weights = F.silu(logits) / float(self.maxlen)
        allowed = self.causal[:length, :length][None, None, :, :]
        allowed = allowed & valid[:, None, :, None] & valid[:, None, None, :]
        weights = weights * allowed.to(weights.dtype)
        attended = torch.matmul(self.dropout(weights), v)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        update = self.output(u * self.attn_norm(attended))
        result = seq_emb + self.dropout(update)
        return result * valid.unsqueeze(-1).to(result.dtype)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
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
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        valid = sequence.ne(0)
        length = sequence.shape[1]
        position = torch.arange(1, length + 1, device=sequence.device).unsqueeze(0)
        position = position * valid.long()
        features = self.item_emb(sequence) * math.sqrt(self.hidden_units)
        features = self.emb_dropout(features + self.pos_emb(position))
        features = features * valid.unsqueeze(-1).to(features.dtype)
        for block in self.blocks:
            features = block(features, valid)
        return self.last_norm(features) * valid.unsqueeze(-1).to(features.dtype)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        else:
            items = items.to(device=features.device, dtype=torch.long)
        if features.ndim == 3:
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(-1, -2))
        return (features.unsqueeze(-2) * embeddings).sum(-1)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
            meta = pickle.load(stream)
        if meta['family'] != 'hstu' or meta['architecture'] != 'native_silu_uvqk_v1':
            raise ValueError('Incompatible HSTU architecture')
        model = HSTURec(**meta['config']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
    return _models[dataset_name]


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model = _load(dataset_name)
    history = [int(i) if 0 < int(i) <= model.itemnum else 0 for i in history[-model.maxlen:]]
    sequence = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    ids = torch.tensor(candidates, dtype=torch.long, device=_device)
    supported = (ids > 0) & (ids <= model.itemnum)
    safe_ids = ids.clamp(0, model.itemnum)
    with torch.inference_mode():
        # Left-padded histories end at the final position.
        features = model.log2feats(seq)[:, -1, :]
        scores = model.predict_score(features, safe_ids).squeeze(0)
        scores = scores.masked_fill(~supported, -1e9)
    return scores.cpu().tolist()
