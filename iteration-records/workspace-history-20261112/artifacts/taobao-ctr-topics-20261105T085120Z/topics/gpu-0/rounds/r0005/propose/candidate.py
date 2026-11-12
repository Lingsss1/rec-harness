"""Identity versus action/time content in a compact learned history readout.

Prospective design: compact-history-component-factorial-v1 in proposal.json.
C/T retain exact existing target statistics. A 16-wide slot receives a
masked mean of category+brand, action+log-age, both, or neither. Context-only
is a small-vocabulary learned marginal readout, not a fixed coarse histogram.
It skips identity-history embedding lookups, but the static identity tables
remain. All learned shapes/construction order match the parent. Attention is
registered but dormant; inactive action/recency parameters are also reported.
No parameter-count reduction, measured gain, or general novelty is claimed.
DIN v4 already describes embedding pooling and intensity preservation.
"""

import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'use_counts': True, 'use_freshness': True, 'use_identity_summary': False, 'use_context_summary': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        if self.config['kind'] != 'din' or self.config['embedding_dim'] != 16:
            raise ValueError('This registered factorial fixes DIN and 16-dimensional slots.')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        if not self.config['use_counts'] or not self.config['use_freshness']:
            raise ValueError('Inherited count/freshness blocks must stay enabled.')
        self.use_identity = self.config['use_identity_summary']
        self.use_context = self.config['use_context_summary']
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

    def _context_events(self, batch):
        # Match the inherited logarithmic recency buckets exactly.
        ages = (
            torch.log2(batch['history_age_seconds'] + 1).long() + 1
        ).clamp(0, 32)
        return self.action(batch['history_type']) + self.recency(ages)

    def _identity_events(self, batch):
        return (
            self.embeddings[self.category](batch['history_category'])
            + self.embeddings[self.brand](batch['history_brand'])
        )

    def _pool(self, batch, events):
        mask = batch['history_mask']
        weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        return (weights.unsqueeze(-1) * events).sum(1)

    def _full_mean(self, batch):
        # Preserve the parent's left-associated event sum and pooling algebra.
        ages = (
            torch.log2(batch['history_age_seconds'] + 1).long() + 1
        ).clamp(0, 32)
        history = (
            self.embeddings[self.category](batch['history_category'])
            + self.embeddings[self.brand](batch['history_brand'])
            + self.action(batch['history_type']) + self.recency(ages)
        )
        return self._pool(batch, history)

    def _summary(self, batch):
        if self.use_identity and self.use_context:
            return self._full_mean(batch)
        if self.use_identity:
            return self._pool(batch, self._identity_events(batch))
        if self.use_context:
            return self._pool(batch, self._context_events(batch))
        return self.output[0].weight.new_zeros((batch['static'].shape[0], 16))

    def forward(self, batch):
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        count_block, freshness, _ = self._evidence(batch)
        summary = self._summary(batch)
        features = torch.cat(fields + [summary, count_block, freshness], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """Label-free source diagnostics, host's first 512 validation rows only.

        Inactive components are evaluated here for descriptive decomposition
        checks, not in forward. These are not population mechanism evidence.
        """
        count_block, freshness, counts = self._evidence(batch)
        identity = self._pool(batch, self._identity_events(batch))
        context = self._pool(batch, self._context_events(batch))
        full = self._full_mean(batch)
        summary = self._summary(batch)
        expected = (
            identity * float(self.use_identity) + context * float(self.use_context)
        )
        mask = batch['history_mask']
        valid = mask.sum(-1)
        empty = valid.eq(0)
        scorer_parameters = sum(p.numel() for p in self.attention.parameters())
        context_parameters = self.action.weight.numel() + self.recency.weight.numel()
        result = {
            'identity_summary_enabled': float(self.use_identity),
            'context_summary_enabled': float(self.use_context),
            'registered_parameters': float(sum(p.numel() for p in self.parameters())),
            'dormant_scorer_parameters': float(scorer_parameters),
            'dormant_context_parameters': float(0 if self.use_context else context_parameters),
            'dormant_summary_head_weights': float(
                0 if (self.use_identity or self.use_context) else 16 * self.output[0].out_features
            ),
            'summary_slot_width': 16.0,
            'active_history_readout_width': float(
                48 if (self.use_identity or self.use_context) else 32
            ),
            'valid_history_count_mean': float(valid.float().mean()),
            'empty_history_fraction': float(empty.float().mean()),
            'summary_abs_mean': float(summary.abs().mean()),
            'identity_abs_mean': float(identity.abs().mean()),
            'context_abs_mean': float(context.abs().mean()),
            'full_abs_mean': float(full.abs().mean()),
            'summary_zero_fraction': float(summary.eq(0).float().mean()),
            'summary_l2_mean': float(summary.square().sum(-1).sqrt().mean()),
            'identity_l2_mean': float(identity.square().sum(-1).sqrt().mean()),
            'context_l2_mean': float(context.square().sum(-1).sqrt().mean()),
            'full_l2_mean': float(full.square().sum(-1).sqrt().mean()),
            'additive_decomposition_max_error': float((full - identity - context).abs().max()),
            'selected_component_max_error': float((summary - expected).abs().max()),
            'empty_summary_abs_sum': float(summary[empty].abs().sum()),
            'empty_count_block_abs_sum': float(count_block[empty].abs().sum()),
            'empty_freshness_abs_sum': float(freshness[empty].abs().sum()),
            'count_block_mean': float(count_block.mean()),
            'freshness_block_mean': float(freshness.mean()),
            'action_count_conservation_max_error': float((counts[:, 0].sum(-1) - valid).abs().max()),
            'invalid_masked_in_action_count': float((
                mask & (batch['history_type'].lt(1) | batch['history_type'].gt(4))
            ).float().sum()),
            'category_count_mean': float(counts[:, 1].sum(-1).mean()),
            'brand_count_mean': float(counts[:, 2].sum(-1).mean()),
            'joint_count_mean': float(counts[:, 3].sum(-1).mean()),
        }
        return result
