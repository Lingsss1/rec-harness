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
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        return conv(nn.functional.pad(x, (pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


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
=======
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
=======
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():

=======
        with torch.no_grad():
=======
        with torch.no_grad():

=======
        with torch.no_grad():
=======
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
=======
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():

=======
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def _item_vectors(self, items: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.item_emb(items), p=2, dim=-1, eps=1e-8)

    def _temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=50.0)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        valid = seq.ne(0)
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = (
            torch.arange(1, length + 1, device=seq.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        positions = positions * valid.long()
        x = self._item_vectors(seq) * math.sqrt(self.hidden_units)
        x = x + self.pos_emb(positions)
        x = self.dropout(x) * mask
        for block in self.blocks:
            x = block(x) * mask
        x = self.norm(x) * mask
        return nn.functional.normalize(x, p=2, dim=-1, eps=1e-8)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        out = self.encode_sequence(seq)
        positions = torch.arange(1, seq.size(1) + 1, device=seq.device).unsqueeze(0)
        last = (positions * seq.ne(0).long()).amax(dim=1)
        last = (last - 1).clamp(min=0)
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last]
        cand_emb = self.item_emb(candidates)
        if candidates.dim() == 1:
            scores = torch.matmul(user_emb, cand_emb.transpose(0, 1))
            if seq.size(0) == 1:
                return scores[0]
            return scores
        return (cand_emb * user_emb.unsqueeze(1)).sum(dim=-1)