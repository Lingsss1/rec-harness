"""Does deterministic target evidence replace learned event retrieval?

Prospective study: scalar-evidence-summary-removal-factorial-v1.
All four cells compute identical count and freshness reductions. They retain
identical registered module shapes and construction order. Summary-off cells
skip event embedding lookup and attention, and pad the 16-wide summary slot
with zeros. Dormant action/recency/attention parameters remain registered to
control initialization; registered parameter compression is NOT claimed.

The two crossed factors are learned-summary accessibility and direct scalar
freshness versus presence. Both retain target-group counts. Presence is a
function of counts; it is not independent temporal information. This tests
replacement of this field-adapted normalized attention, not original DIN's
unnormalized intensity mechanism. No novelty or measured speed claim.
"""

import math
import torch
from torch import nn


CONFIG = {
    'kind': 'din',
    'embedding_dim': 16,
    'hidden': [128, 64],
    'lr': 0.001,
    'use_counts': True,
    'use_freshness': True,
    'use_target_groups': True,
    'use_time_values': False,
    'use_attention_summary': False,
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
        if not self.use_counts or not self.use_freshness:
            raise ValueError('Both evidence slots are fixed on in this study.')
        self.use_target_groups = self.config['use_target_groups']
        self.use_time_values = self.config['use_time_values']
        self.use_attention_summary = self.config['use_attention_summary']
        if not self.use_target_groups:
            raise ValueError('Target groups are fixed on in this study.')
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

    def _select_evidence(self, count_block, freshness, counts):
        """Keep shapes fixed while replacing information in the scalar slots.

        P=1[n>0] exposes no age value and is already derivable from counts.
        Global activity control repeats group zero (all valid events), without
        using category/brand matches in its direct scalar readout. It still
        retains target-aware learned attention in the separate summary slot.
        """
        presence = counts.gt(0).float().flatten(1)
        time_slot = freshness if self.use_time_values else presence
        if not self.use_target_groups:
            count_block = count_block[:, :4].repeat(1, 4)
            time_slot = time_slot[:, :4].repeat(1, 4)
        return count_block, time_slot

    def _summary(self, batch, fields):
        # The inactive path reads only batch size/device through static fields.
        # No padded or valid historical identity, action or age is embedded.
        if not self.use_attention_summary:
            return torch.zeros_like(fields[self.category])
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
        return (weights.unsqueeze(-1) * history).sum(1)

    def forward(self, batch):
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        summary = self._summary(batch, fields)
        count_block, freshness, counts = self._evidence(batch)
        count_slot, freshness_slot = self._select_evidence(count_block, freshness, counts)
        features = torch.cat(fields + [summary, count_slot, freshness_slot], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """Descriptive checks on the host's first 512 validation impressions.

        These label-free scalars are not full-population mechanism evidence.
        The host controls diagnostic timing and checkpoints.
        """
        count_block, freshness, counts = self._evidence(batch)
        selected_counts, selected_time = self._select_evidence(count_block, freshness, counts)
        group_totals = counts.sum(-1)
        present = counts.gt(0).float()
        freshness = freshness.reshape_as(counts)
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        summary = self._summary(batch, fields)
        branch_parameters = sum(
            p.numel() for module in (self.action, self.recency, self.attention)
            for p in module.parameters()
        )
        result = {
            'attention_summary_enabled': float(self.use_attention_summary),
            'summary_abs_mean': float(summary.abs().mean()),
            'summary_zero_fraction': float(summary.eq(0).float().mean()),
            'active_history_readout_width': 48.0 if self.use_attention_summary else 32.0,
            'dormant_event_branch_parameters': (
                0.0 if self.use_attention_summary else float(branch_parameters)
            ),
            'dormant_summary_head_weights': 0.0 if self.use_attention_summary else float(16 * self.config['hidden'][0]),
            'target_groups_enabled': float(self.use_target_groups),
            'time_values_enabled': float(self.use_time_values),
            'direct_scalar_unique_entries': 32.0 if self.use_target_groups else 8.0,
            'selected_count_mean': float(selected_counts.mean()),
            'selected_time_or_presence_mean': float(selected_time.mean()),
            'selected_freshness_mse': float((selected_time - freshness.flatten(1)).square().mean()),
            'scalar_slots_zero_fraction': float(
                torch.cat([selected_counts, selected_time], -1).eq(0).float().mean()
            ),
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
