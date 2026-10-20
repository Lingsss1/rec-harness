import math
import torch
import torch.nn as nn

MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 0.001
WEIGHT_DECAY = 0.0
BATCH_SIZE = 128


class NextItNetBlock(nn.Module):
    def __init__(
        self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            hidden_units, 2 * hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.residual_scale = nn.Parameter(torch.full((hidden_units,), 0.1))
        with torch.no_grad():
            self.conv1.bias.zero_()
            self.conv1.bias[hidden_units:].fill_(1.0)
            self.conv2.bias.zero_()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        return conv(nn.functional.pad(x, (pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(0).any(dim=-1, keepdim=True).to(x.dtype)
        y = (self.norm1(x) * mask).transpose(1, 2)
        value, gate = self._causal_conv(self.conv1, y).chunk(2, dim=1)
        y = (self.activation(value) * torch.sigmoid(gate)).transpose(1, 2)
        y = self.dropout(y) * mask
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return (x + self.residual_scale * self.dropout(y)) * mask


class NextItNetModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList(
            [
                NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                for i in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq.ne(0)
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = valid.long().cumsum(dim=1).clamp(max=self.maxlen)
        positions = positions * valid.long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.dropout(x + self.pos_emb(positions)) * mask
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x) * mask

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return (pos_scores, neg_scores)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        last = positions.expand_as(seq).masked_fill(seq.eq(0), -1)
        last = last.max(dim=1).values.clamp(min=0)
        user_emb = out[torch.arange(seq.size(0), device=seq.device), last]
        cand_emb = self.item_emb(candidates)
        if candidates.dim() == 1:
            scores = torch.matmul(user_emb, cand_emb.transpose(0, 1))
            return scores.squeeze(0) if seq.size(0) == 1 else scores
        return (cand_emb * user_emb.unsqueeze(1)).sum(dim=-1)
