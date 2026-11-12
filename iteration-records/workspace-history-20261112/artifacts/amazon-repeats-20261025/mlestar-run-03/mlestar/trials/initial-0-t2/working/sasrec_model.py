import math
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

    def forward(self, inputs):
        x = self.conv1(inputs.transpose(1, 2))
        x = F.relu(self.dropout1(x))
        x = self.dropout2(self.conv2(x)).transpose(1, 2)
        return inputs + x


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate,
                                  batch_first=True) for _ in range(num_blocks)])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.forward_layers = nn.ModuleList([
            PointWiseFeedForward(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen,
                             dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        keep = seq.ne(0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions)) * keep
        for norm, attn, fnorm, ff in zip(self.attention_layernorms,
                self.attention_layers, self.forward_layernorms, self.forward_layers):
            q = norm(x)
            attended = attn(q, x, x, attn_mask=self.causal_mask[:length, :length],
                            need_weights=False)[0]
            x = ff(fnorm(q + attended)) * keep
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)

