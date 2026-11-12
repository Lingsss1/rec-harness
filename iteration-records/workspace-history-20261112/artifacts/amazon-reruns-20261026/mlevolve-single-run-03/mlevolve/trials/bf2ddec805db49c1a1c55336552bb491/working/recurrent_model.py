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
        self.output_norm = nn.LayerNorm(hidden)
        # One jointly trained candidate-conditioned scorer replaces the readout.
        self.scorer = nn.Sequential(
            nn.Linear(5 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
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
        x = self.input_norm(self.embedding(seq)) * mask
        recurrent, _ = self.gru(self.dropout(x))
        return self.dropout(self.output_norm(recurrent)) * mask

    def _score_chunk(self, states, keys, values, candidate_embeddings, allowed):
        # Each candidate queries contextual GRU states independently. The mask
        # includes observed positions through t, never the next-item target.
        queries = self.query_projection(candidate_embeddings)
        logits = torch.einsum("bpch,blh->bpcl", queries, keys)
        logits = logits / math.sqrt(queries.shape[-1])
        mask = allowed.unsqueeze(2)
        logits = logits.masked_fill(~mask, float("-inf"))
        logits = logits.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)
        attention = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        context = torch.einsum("bpcl,blh->bpch", attention, values)
        states = states.unsqueeze(2).expand_as(candidate_embeddings)
        features = torch.cat([context, states, candidate_embeddings, context * candidate_embeddings, states * candidate_embeddings], dim=-1)
        return self.scorer(features).squeeze(-1)

    def _score(self, h, seq, candidates, positions,
               position_chunk=8, candidate_chunk=16):
        # candidates: [batch, supervised positions, candidates per position].
        # Checkpoint each chunk so backward recomputes its attention/MLP
        # activations. Concatenation preserves gradients and loss normalization.
        from torch.utils.checkpoint import checkpoint

        if candidates.shape[-1] == 0:
            return h.new_empty(candidates.shape)
        keys = self.key_projection(h)
        values = self.value_projection(h)
        history_indices = torch.arange(seq.shape[1], device=seq.device)
        position_scores = []
        for start in range(0, positions.shape[1], position_chunk):
            stop = start + position_chunk
            selected = positions[:, start:stop]
            states = h.gather(
                1, selected.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )
            allowed = (
                (history_indices[None, None, :] <= selected.unsqueeze(-1))
                & seq.ne(0).unsqueeze(1)
                & seq.gather(1, selected).ne(0).unsqueeze(-1)
            )
            candidate_scores = []
            for offset in range(0, candidates.shape[-1], candidate_chunk):
                ids = candidates[:, start:stop, offset:offset + candidate_chunk]
                embeddings = self.embedding(ids)
                if self.training and torch.is_grad_enabled():
                    scores = checkpoint(
                        self._score_chunk, states, keys, values, embeddings,
                        allowed, use_reentrant=False,
                    )
                else:
                    scores = self._score_chunk(
                        states, keys, values, embeddings, allowed
                    )
                candidate_scores.append(scores + self.item_bias(ids).squeeze(-1))
            position_scores.append(torch.cat(candidate_scores, dim=-1))
        return torch.cat(position_scores, dim=1)

    def forward(self, seq, positives, negatives):
        h = self.encode(seq)
        positions = torch.arange(seq.shape[1], device=seq.device)
        positions = positions.unsqueeze(0).expand(seq.shape[0], -1)
        # Share the exact scoring path for positives and the reused negatives.
        candidates = torch.cat([
            positives.unsqueeze(-1),
            negatives.unsqueeze(1).expand(-1, seq.shape[1], -1),
        ], dim=-1)
        scores = self._score(h, seq, candidates, positions)
        return scores[:, :, 0], scores[:, :, 1:]

    def predict(self, seq, candidates):
        h = self.encode(seq)
        lengths = seq.ne(0).sum(1).clamp_min(1)
        positions = (lengths - 1).unsqueeze(1)
        return self._score(h, seq, candidates.unsqueeze(1), positions)[:, 0]

