import math
import torch
import torch.nn as nn

MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 0.001
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128


class NextItNetBlock(nn.Module):

    def __init__(
        self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.local_depthwise = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, groups=hidden_units, padding=0
        )
        self.local_pointwise = nn.Conv1d(hidden_units, hidden_units, 1)
        self.gate = nn.Conv1d(hidden_units, hidden_units, 1)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.residual_scale = nn.Parameter(torch.full((hidden_units,), 0.1))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.0)

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        return conv(nn.functional.pad(x, (pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm1(x).transpose(1, 2)
        dilated = self._causal_conv(self.conv1, y)
        local = self.local_pointwise(self._causal_conv(self.local_depthwise, y))
        gate = torch.sigmoid(self.gate(y))
        y = (gate * dilated + (1.0 - gate) * local).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return residual + self.residual_scale * self.dropout(y)


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

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(seq) * self.hidden_units**0.5
        mask = (seq != 0).unsqueeze(-1).to(dtype=x.dtype)
        x = self.dropout(x) * mask
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
        last = (
            positions.expand_as(seq)
            .masked_fill(seq == 0, -1)
            .max(dim=1)
            .values.clamp(min=0)
        )
        user_emb = out[0, last[0]]
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)
