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

    def forward(self, seq_emb, causal_mask, padding_mask):
        batch, length, _ = seq_emb.shape
        # Every padding query may attend to itself, avoiding all-masked rows.
        blocked = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=seq_emb.device).unsqueeze(0)
        blocked = blocked & ~(diagonal & padding_mask.unsqueeze(-1))
        blocked = blocked.unsqueeze(1).expand(batch, self.num_heads, length, length)
        blocked = blocked.reshape(batch * self.num_heads, length, length)
        normed = self.norm(seq_emb)
        attended, _ = self.attn(normed, normed, normed, attn_mask=blocked, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.3, maxlen=128, temperature=0.05,
                 normalize_embeddings=True):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
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
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch, length = sequence.shape
        padding = sequence.eq(0)
        positions = torch.arange(1, length + 1, device=sequence.device).unsqueeze(0).expand(batch, -1)
        positions = positions.masked_fill(padding, 0)
        x = self.item_emb(sequence) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.emb_dropout(x).masked_fill(padding.unsqueeze(-1), 0.0)
        causal = torch.ones(length, length, dtype=torch.bool, device=sequence.device).triu(1)
        for block in self.blocks:
            x = block(x, causal, padding)
        return self.last_norm(x).masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def readout(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1)
        selected = features[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return selected.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if self.normalize_embeddings:
            features = F.normalize(features, p=2, dim=-1, eps=1e-8)
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        if items.ndim == 1:
            scores = torch.matmul(features, embeddings.transpose(-1, -2))
        else:
            scores = (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return scores / self.temperature



_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
            meta = pickle.load(stream)
        model = HSTURec(**meta['model_config']).to(_device)
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
    history = [int(item) for item in history if 0 < int(item) <= model.itemnum][-model.maxlen:]
    sequence = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    valid = items.gt(0) & items.le(model.itemnum)
    with torch.inference_mode():
        features = model.readout(seq)[0]
        scores = model.predict_score(features, items.clamp(0, model.itemnum))
        scores = scores.masked_fill(~valid, -1e9)
    return scores.cpu().tolist()
