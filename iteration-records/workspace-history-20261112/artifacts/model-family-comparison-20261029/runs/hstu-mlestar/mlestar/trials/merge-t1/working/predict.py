from __future__ import annotations
import os
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4), nn.SiLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate))

    def forward(self, seq_emb, attention_mask, padding_mask):
        normed = self.norm(seq_emb)
        attended, _ = self.attn(normed, normed, normed, attn_mask=attention_mask, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask[..., None], 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.3, maxlen=128, temperature=0.05, normalize_scores=True):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.temperature = float(temperature)
        self.normalize_scores = bool(normalize_scores)
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([HSTUBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch_size, length = seq.shape
        padding = seq.eq(0)
        positions = torch.arange(1, length + 1, device=seq.device)[None, :].expand(batch_size, -1)
        positions = positions.masked_fill(padding, 0)
        hidden = self.item_emb(seq) * math.sqrt(self.hidden_units)
        hidden = self.emb_dropout(hidden + self.pos_emb(positions))
        hidden = hidden.masked_fill(padding[..., None], 0.0)
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        allowed = causal[None, :, :] & (~padding[:, None, :])
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device)[None, :, :]
        allowed = allowed | (padding[:, :, None] & diagonal)
        attention_mask = (~allowed).repeat_interleave(self.num_heads, dim=0)
        for block in self.blocks:
            hidden = block(hidden, attention_mask, padding)
        return self.last_norm(hidden).masked_fill(padding[..., None], 0.0)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if self.normalize_scores:
            features = F.normalize(features, dim=-1, eps=1e-8)
            embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(-1, -2)) / self.temperature
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1) / self.temperature

    def readout(self, seq, features=None):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        if features is None:
            features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device)[None, :].expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        selected = features[torch.arange(seq.shape[0], device=seq.device), last.clamp(min=0)]
        return selected.masked_fill(last.lt(0).unsqueeze(-1), 0.0)



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
            meta = pickle.load(stream)
        model = HSTURec(**meta['config']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
    return _models[dataset_name]


def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if len(candidates) == 0:
        return []
    model = _load(dataset_name)
    history = [int(item) for item in history][-model.maxlen:]
    candidates = [int(item) for item in candidates]
    if any(item < 0 or item > model.itemnum for item in history):
        raise ValueError('History item outside dataset catalog')
    if any(item <= 0 or item > model.itemnum for item in candidates):
        raise ValueError('Candidate outside dataset catalog')
    seq = torch.zeros((1, model.maxlen), dtype=torch.long, device=_device)
    if history:
        seq[0, -len(history):] = torch.tensor(history, dtype=torch.long, device=_device)
    with torch.inference_mode():
        features = model.readout(seq)
        scores = model.predict_score(features, candidates)[0]
    return scores.cpu().tolist()
