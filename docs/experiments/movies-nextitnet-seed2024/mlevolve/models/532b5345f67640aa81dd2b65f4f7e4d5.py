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
        self.local_conv = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=1, padding=0
        )
        self.conv1 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, 2 * hidden_units, kernel_size, dilation=dilation, padding=0
        )
        with torch.no_grad():
            self.conv2.bias[hidden_units:].fill_(1.0)
        self.gate = nn.Conv1d(hidden_units, hidden_units, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.residual_scale = nn.Parameter(torch.full((hidden_units,), 0.1))

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        return conv(nn.functional.pad(x, (pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        g = torch.sigmoid(self.gate(y))
        local = self._causal_conv(self.local_conv, y)
        dilated = self._causal_conv(self.conv1, y)
        y = (math.sqrt(2.0) * (g * local + (1.0 - g) * dilated)).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y)
        value, gate_logits = y.chunk(2, dim=1)
        y = (torch.tanh(value) * torch.sigmoid(gate_logits)).transpose(1, 2)
        return x + self.residual_scale.view(1, 1, -1) * self.dropout(y)


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
        with torch.no_grad():
            rng_state = torch.get_rng_state()
            self.short_conv = nn.Conv1d(
                hidden_units, hidden_units, kernel_size=5, dilation=1, padding=0
            )
            nn.init.zeros_(self.short_conv.weight)
            nn.init.zeros_(self.short_conv.bias)
            torch.set_rng_state(rng_state)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        mask = seq.ne(0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        mask = mask.to(dtype=x.dtype)
        x = self.dropout(x) * mask
        input_x = x
        for block in self.blocks:
            x = block(x) * mask
        local = self.short_conv(
            nn.functional.pad(input_x.transpose(1, 2), (4, 0))
        ).transpose(1, 2)
        x = (x + local) * mask
        return self.norm(x) * mask

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device)
        positions = positions.unsqueeze(0).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1)
        last = last.clamp(min=0)
        batch_indices = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[batch_indices, last]
        cand_emb = self.item_emb(candidates)
        if candidates.dim() == 1:
            scores = torch.matmul(user_emb, cand_emb.transpose(0, 1))
            return scores.squeeze(0) if seq.size(0) == 1 else scores
        scores = (cand_emb * user_emb.unsqueeze(1)).sum(dim=-1)
        return scores
