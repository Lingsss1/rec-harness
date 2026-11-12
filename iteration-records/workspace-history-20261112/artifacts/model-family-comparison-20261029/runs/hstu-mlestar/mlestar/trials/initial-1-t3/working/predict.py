from __future__ import annotations
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSTUBlock(nn.Module):
    """Causal attention, SiLU-gated residual update, and residual FFN."""
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
        # An otherwise fully masked padded query may attend only to itself.
        # Real queries can attend only to nonpadding causal keys.
        blocked = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=seq_emb.device).unsqueeze(0)
        blocked = blocked & ~(padding_mask.unsqueeze(2) & diagonal)
        blocked = blocked.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        blocked = blocked.reshape(batch * self.num_heads, length, length)
        normed = self.norm(seq_emb)
        attended, _ = self.attn(normed, normed, normed, attn_mask=blocked, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.hidden_units = hidden_units
        self.itemnum = itemnum
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

    def forward(self, seq):
        batch, length = seq.shape
        padding_mask = seq.eq(0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions.masked_fill(padding_mask, 0)
        features = self.item_emb(seq) * math.sqrt(self.hidden_units)
        features = self.emb_dropout(features + self.pos_emb(positions))
        features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=seq.device).triu(1)
        for block in self.blocks:
            features = block(features, causal_mask, padding_mask)
        return self.last_norm(features).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.forward(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        if features.ndim == 3:
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name not in _models:
        path = os.path.join(_HERE, dataset_name + '_model.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = HSTURec(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(_device).eval()
        _models[dataset_name] = model
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model = _load(dataset_name)
    if not candidates:
        return []
    history = [int(item) for item in history if 0 < int(item) <= model.itemnum][-model.maxlen:]
    sequence = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    if bool(((items < 1) | (items > model.itemnum)).any()):
        raise ValueError('Candidate outside category item vocabulary')
    features = model.log2feats(seq)[:, -1, :]
    return model.predict_score(features, items)[0].float().cpu().tolist()
