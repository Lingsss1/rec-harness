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
    def __init__(self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        bottleneck = max(1, hidden_units // 2)
        self.conv1 = nn.Conv1d(hidden_units, bottleneck, 1)
        self.conv2 = nn.Conv1d(bottleneck, bottleneck, kernel_size, dilation=dilation, padding=0)
        self.conv3 = nn.Conv1d(bottleneck, hidden_units, 1)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(bottleneck)
        self.norm3 = nn.LayerNorm(bottleneck)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        x = nn.functional.pad(x, (pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.activation(self.norm1(x)).transpose(1, 2)
        y = self.conv1(y).transpose(1, 2)
        y = self.activation(self.norm2(y)).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        y = self.dropout(y)
        y = self.activation(self.norm3(y)).transpose(1, 2)
        y = self.conv3(y).transpose(1, 2)
        return residual + self.dropout(y)

class NextItNetModel(nn.Module):
    def __init__(self, itemnum: int, hidden_units: int=64, num_blocks: int=8, kernel_size: int=3, dropout_rate: float=0.2, maxlen: int=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate) for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.Identity()
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        (batch, length) = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (seq != 0).long()
        x = self.item_emb(seq) * self.hidden_units ** 0.5 + self.pos_emb(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * (seq != 0).unsqueeze(-1).float()
        return self.norm(x)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return (pos_scores, neg_scores)

    def score_candidates(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        out = self.encode_sequence(seq)
        user_emb = out[0, -1] * (seq[0] != 0).any().to(out.dtype)
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)
