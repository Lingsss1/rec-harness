import math
import torch
from torch import nn
from torch.nn import functional as F

class RecurrentMemory(nn.Module):
    def __init__(self, itemnum, hidden=96, maxlen=64, dropout=0.25):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.embedding = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.input_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.key_projection = nn.Linear(hidden, hidden, bias=False)
        self.value_projection = nn.Linear(hidden, hidden, bias=False)
        self.memory_projection = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(3 * hidden, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.03)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        raw = self.embedding(seq)
        x = self.input_norm(raw) * mask
        recurrent, _ = self.gru(self.dropout(x))
        # Retrieve relevant prefix items using jointly trained causal attention.
        queries = self.query_projection(recurrent)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        causal = torch.ones(
            seq.shape[1], seq.shape[1], dtype=torch.bool, device=seq.device
        ).tril()
        attention_mask = causal.unsqueeze(0) & mask.transpose(1, 2) & mask
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(queries.shape[-1])
        attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        # Padded queries have no allowed keys; use finite logits before softmax
        # and then zero their weights so neither outputs nor gradients see NaNs.
        attention_scores = attention_scores.masked_fill(~mask, 0.0)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(~attention_mask, 0.0)
        memory = torch.matmul(attention_weights, values) * mask
        memory = torch.tanh(self.memory_projection(memory))
        gate = torch.sigmoid(self.gate(torch.cat([recurrent, x, memory], dim=-1)))
        h = self.output_norm(x + recurrent + gate * memory)
        return self.dropout(h)

    def forward(self, seq, positives, negatives):
        h = self.encode(seq)
        pos = (h * self.embedding(positives)).sum(-1)
        pos = pos + self.item_bias(positives).squeeze(-1)
        # The same negative set is reused across positions within a history.
        neg = torch.einsum("blh,bkh->blk", h, self.embedding(negatives))
        neg = neg + self.item_bias(negatives).squeeze(-1).unsqueeze(1)
        return pos, neg

    def predict(self, seq, candidates):
        h = self.encode(seq)
        lengths = seq.ne(0).sum(1).clamp_min(1)
        final = h[torch.arange(seq.shape[0], device=seq.device), lengths - 1]
        scores = torch.einsum("bh,bkh->bk", final, self.embedding(candidates))
        return scores + self.item_bias(candidates).squeeze(-1)

