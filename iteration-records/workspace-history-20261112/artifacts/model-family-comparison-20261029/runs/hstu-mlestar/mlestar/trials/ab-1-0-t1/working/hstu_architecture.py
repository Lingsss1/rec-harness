import math
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
            nn.Linear(hidden_units, hidden_units * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

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
        # Padded query rows attend only to themselves, avoiding all-masked attention.
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
        if features is None:
            features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device)[None, :].expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        selected = features[torch.arange(seq.shape[0], device=seq.device), last.clamp(min=0)]
        return selected * last.ge(0).unsqueeze(-1)

