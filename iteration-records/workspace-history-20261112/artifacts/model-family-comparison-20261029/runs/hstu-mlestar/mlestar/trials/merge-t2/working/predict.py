from __future__ import annotations
import os
import math
import pickle
import torch
from torch import nn
from torch.nn import functional as F

class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(nn.Linear(hidden_units, hidden_units * 4), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_units * 4, hidden_units), nn.Dropout(dropout_rate))

    def forward(self, x, attention_mask, valid):
        z = self.norm(x)
        attended, _ = self.attn(z, z, z, attn_mask=attention_mask, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        x = x + self.dropout(value * F.silu(gate))
        x = x + self.ffn(self.ffn_norm(x))
        return x * valid


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('Hidden width must be divisible by heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([HSTUBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch, length = seq.shape
        valid_bool = seq.ne(0)
        valid = valid_bool.unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0) * valid_bool
        x = self.emb_dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)) * valid
        causal = torch.ones((length, length), dtype=torch.bool, device=seq.device).tril()
        allowed = causal.unsqueeze(0) & valid_bool.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device).unsqueeze(0)
        allowed = (allowed & valid_bool.unsqueeze(2)) | (diagonal & ~valid_bool.unsqueeze(2))
        attention_mask = (~allowed).repeat_interleave(self.num_heads, dim=0)
        for block in self.blocks:
            x = block(x, attention_mask, valid)
        return self.last_norm(x) * valid

    def forward(self, sequence):
        return self.log2feats(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if features.ndim == embeddings.ndim - 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def readout(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        feats = self.log2feats(sequence)
        indices = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0).expand_as(sequence)
        last = indices.masked_fill(sequence.eq(0), -1).amax(dim=1)
        result = feats[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return result * last.ge(0).unsqueeze(-1)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _load(name):
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as stream:
            config = pickle.load(stream)
        model = HSTURec(**config).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[name] = model
    return _models[name]

@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model = _load(dataset_name)
    if len(candidates) == 0:
        return []
    history = [int(item) for item in history if 0 < int(item) <= model.itemnum][-model.maxlen:]
    seq = [0] * (model.maxlen - len(history)) + history
    sequence = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.as_tensor(candidates, dtype=torch.long, device=_device)
    legal = (items > 0) & (items <= model.itemnum)
    scores = model.predict_score(model.readout(sequence)[0], items.clamp(0, model.itemnum))
    return scores.masked_fill(~legal, -1e9).cpu().tolist()
