"""Compact count/freshness evidence over the host-supplied causal history.

Study: compact-evidence-count-freshness-factorial-v1 in proposal.json.
This is a field-adapted normalized-attention model, not an exact DIN replica.
Original DIN already motivates preserving intensity without softmax (v4,
lines 272-277). No novelty claim is made for preserving counts.

All factorial cells retain the baseline summary h and two extra 16-wide slots.
An inactive slot repeats h. Active slots expose count C or freshness T.
Both evidence blocks are computed in every cell for matched arithmetic.
The modules have identical parameter shapes and initialization order.
Only the literal configuration flags differ across standalone files.
"""

import math
import torch
from torch import nn


CONFIG = {
    'kind': 'din',
    'embedding_dim': 16,
    'hidden': [128, 64],
    'lr': 0.001,
    'use_counts': False,
    'use_freshness': True,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        if self.config['kind'] != 'din' or self.config['embedding_dim'] != 16:
            raise ValueError('This registered factorial fixes DIN and 16-dimensional slots.')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.use_counts = self.config['use_counts']
        self.use_freshness = self.config['use_freshness']
        dim = self.config['embedding_dim']

        # Preserve baseline module construction order up to the widened head.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim + 32
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _evidence(self, batch):
        """Return group-major [B,16] count/freshness blocks and [B,4,4] counts.

        Groups: all, category match, brand match, joint match.
        Within each group: pv, cart, fav, buy. Matches use supplied buckets,
        require nonzero target values, and never include masked events.
        Counts describe only the supplied last 128 events, not lifetime counts.
        T is mean exp(-age/one day), not a decayed sum, so C and T separate
        multiplicity from conditional freshness as far as the zero convention
        permits. Empty cells give C=T=0; T can itself reveal cell presence.
        """
        mask = batch['history_mask']
        target_category = batch['static'][:, self.category].unsqueeze(1)
        target_brand = batch['static'][:, self.brand].unsqueeze(1)
        category_match = (
            mask & target_category.ne(0)
            & batch['history_category'].eq(target_category)
        )
        brand_match = (
            mask & target_brand.ne(0)
            & batch['history_brand'].eq(target_brand)
        )
        groups = torch.stack([
            mask, category_match, brand_match, category_match & brand_match
        ], dim=-1).float()
        actions = torch.stack([
            batch['history_type'].eq(action) for action in range(1, 5)
        ], dim=-1).float()
        # Batched reductions avoid materializing a [B,L,4,4] event tensor.
        group_rows = groups.transpose(1, 2)
        counts = torch.bmm(group_rows, actions)
        ages = torch.where(
            mask, batch['history_age_seconds'],
            torch.zeros_like(batch['history_age_seconds'])
        ).clamp_min(0)
        decay = torch.exp(-ages / 86400.0)
        decayed_counts = torch.bmm(group_rows, actions * decay.unsqueeze(-1))
        count_block = torch.log1p(counts) / math.log(129.0)
        freshness_block = decayed_counts / counts.clamp_min(1.0)
        return count_block.flatten(1), freshness_block.flatten(1), counts

    def forward(self, batch):
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        ages = (
            torch.log2(batch['history_age_seconds'] + 1).long() + 1
        ).clamp(0, 32)
        history = (
            self.embeddings[self.category](batch['history_category'])
            + self.embeddings[self.brand](batch['history_brand'])
            + self.action(batch['history_type']) + self.recency(ages)
        )
        mask = batch['history_mask']
        query = (fields[self.category] + fields[self.brand])[:, None, :]
        query = query.expand_as(history)
        scores = self.attention(torch.cat([
            query, history, query - history, query * history
        ], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        summary = (weights.unsqueeze(-1) * history).sum(1)

        counts, freshness, _ = self._evidence(batch)
        count_slot = counts if self.use_counts else summary
        freshness_slot = freshness if self.use_freshness else summary
        features = torch.cat(fields + [summary, count_slot, freshness_slot], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """Descriptive checks on the host's first 512 validation impressions.

        These label-free scalars are not full-population mechanism evidence.
        The host controls diagnostic timing and checkpoints.
        """
        count_block, freshness, counts = self._evidence(batch)
        group_totals = counts.sum(-1)
        present = counts.gt(0).float()
        freshness = freshness.reshape_as(counts)
        result = {
            'counts_enabled': float(self.use_counts),
            'freshness_enabled': float(self.use_freshness),
            'valid_history_count_mean': float(group_totals[:, 0].mean()),
            'empty_history_fraction': float(group_totals[:, 0].eq(0).float().mean()),
            'count_block_mean': float(count_block.mean()),
            'freshness_mean_over_occupied_cells': float(
                (freshness * present).sum() / present.sum().clamp_min(1.0)
            ),
        }
        for group, name in enumerate(('all', 'category', 'brand', 'joint')):
            result[name + '_count_mean'] = float(group_totals[:, group].mean())
            result[name + '_present_fraction'] = float(
                group_totals[:, group].gt(0).float().mean()
            )
            occupied = present[:, group]
            result[name + '_freshness_occupied_mean'] = float(
                (freshness[:, group] * occupied).sum()
                / occupied.sum().clamp_min(1.0)
            )
        for action, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            result[name + '_count_mean'] = float(counts[:, 0, action].mean())
        return result
