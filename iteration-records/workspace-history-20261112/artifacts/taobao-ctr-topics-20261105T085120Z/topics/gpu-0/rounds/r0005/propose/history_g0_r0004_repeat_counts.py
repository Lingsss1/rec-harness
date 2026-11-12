"""Fixed-width action/time readout replacement study.

The 32 inherited target count/freshness values stay fixed. The remaining
16-wide slot holds a joint action-by-age histogram, its independence
reconstruction, a repeated count block, or the inherited global mean.
All modules retain identical learned shapes and initialization order.
Histogram cells skip event embeddings and scoring; dormant parameters are
reported. Registered parameter compression and measured speedups are not
claimed. Histograms describe only the host-supplied last 128 causal events.

DIN v4 lines 232-277 already describe pooling and intensity preservation.
This is a local replacement experiment, without a priority claim. DMR's
README confirms category/brand/action histories, not item identities.
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
    'summary_mode': 'repeat',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        if self.config['kind'] != 'din' or self.config['embedding_dim'] != 16:
            raise ValueError('This study fixes the inherited 16-dimensional slots.')
        if not self.config['use_counts'] or not self.config['use_freshness']:
            raise ValueError('Target count and freshness blocks are fixed on.')
        self.mode = self.config['summary_mode']
        if self.mode not in ('joint', 'independent', 'repeat', 'mean'):
            raise ValueError('Unknown registered summary mode.')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']

        # Same shapes, order, and random initialization calls as the incumbent.
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
        """Inherited group-major C,T: all/category/brand/joint x four actions.

        Nonzero target bucket matches only. C=log1p(n)/log(129);
        T=mean(exp(-age/86400)) within occupied cells, otherwise zero.
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

    def _temporal_counts(self, batch):
        """[B,action,age] counts and independence reconstruction.

        Fixed disjoint age intervals in seconds: [0,3600), [3600,86400),
        [86400,604800), [604800,infinity). No data-dependent cut points.
        J[a,t] counts only masked-in events of action 1..4. The replacement
        I[a,t]=J[a,:].sum()*J[:,t].sum()/J.sum() preserves both marginals,
        while discarding the within-impression action/time association.
        Empty histories produce zero in every cell. Fractional I counts are
        intentional. Both arrays are computed in every variant for controls.
        """
        mask = batch['history_mask']
        ages = torch.where(
            mask, batch['history_age_seconds'],
            torch.zeros_like(batch['history_age_seconds'])
        ).clamp_min(0)
        windows = torch.stack([
            ages.lt(3600.0),
            ages.ge(3600.0) & ages.lt(86400.0),
            ages.ge(86400.0) & ages.lt(604800.0),
            ages.ge(604800.0),
        ], dim=-1).float() * mask.unsqueeze(-1).float()
        actions = torch.stack([
            batch['history_type'].eq(action) for action in range(1, 5)
        ], dim=-1).float()
        joint = torch.bmm(actions.transpose(1, 2), windows)
        action_marginal = joint.sum(-1, keepdim=True)
        age_marginal = joint.sum(1, keepdim=True)
        total = action_marginal.sum(1, keepdim=True)
        independent = action_marginal * age_marginal / total.clamp_min(1.0)
        return joint, independent

    def _mean_summary(self, batch):
        # Exact all-event uniform pooling used by g0-r0003-global_mean.
        ages = (
            torch.log2(batch['history_age_seconds'] + 1).long() + 1
        ).clamp(0, 32)
        history = (
            self.embeddings[self.category](batch['history_category'])
            + self.embeddings[self.brand](batch['history_brand'])
            + self.action(batch['history_type']) + self.recency(ages)
        )
        mask = batch['history_mask']
        weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        return (weights.unsqueeze(-1) * history).sum(1)

    def _summary(self, batch, count_block, joint, independent):
        if self.mode == 'mean':
            return self._mean_summary(batch)
        if self.mode == 'repeat':
            return count_block
        selected = joint if self.mode == 'joint' else independent
        return (torch.log1p(selected) / math.log(129.0)).flatten(1)

    def forward(self, batch):
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        count_block, freshness, _ = self._evidence(batch)
        joint, independent = self._temporal_counts(batch)
        summary = self._summary(batch, count_block, joint, independent)
        features = torch.cat(fields + [summary, count_block, freshness], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """Candidate-supplied, label-free checks on 512 host-selected rows.

        These are implementation diagnostics, not population mechanism tests.
        Dormant event parameters exclude shared static field embeddings.
        """
        count_block, freshness, counts = self._evidence(batch)
        joint, independent = self._temporal_counts(batch)
        summary = self._summary(batch, count_block, joint, independent)
        valid = batch['history_mask'].sum(-1)
        total = joint.sum((1, 2))
        scorer_parameters = sum(p.numel() for p in self.attention.parameters())
        event_parameters = (
            scorer_parameters + self.action.weight.numel() + self.recency.weight.numel()
        )
        result = {
            'joint_enabled': float(self.mode == 'joint'),
            'independent_enabled': float(self.mode == 'independent'),
            'repeat_enabled': float(self.mode == 'repeat'),
            'mean_enabled': float(self.mode == 'mean'),
            'registered_parameters': float(sum(p.numel() for p in self.parameters())),
            'dormant_event_parameters': float(
                scorer_parameters if self.mode == 'mean' else event_parameters
            ),
            'active_history_readout_width': 48.0,
            'summary_abs_mean': float(summary.abs().mean()),
            'summary_zero_fraction': float(summary.eq(0).float().mean()),
            'count_block_mean': float(count_block.mean()),
            'freshness_block_mean': float(freshness.mean()),
            'empty_history_fraction': float(valid.eq(0).float().mean()),
            'valid_history_count_mean': float(valid.float().mean()),
            'histogram_conservation_max_error': float((total - valid).abs().max()),
            'action_marginal_max_error': float(
                (joint.sum(-1) - independent.sum(-1)).abs().max()
            ),
            'age_marginal_max_error': float(
                (joint.sum(1) - independent.sum(1)).abs().max()
            ),
            'action_count_reference_max_error': float(
                (joint.sum(-1) - counts[:, 0]).abs().max()
            ),
            'empty_histogram_abs_sum': float(joint[valid.eq(0)].abs().sum()),
            'association_l1_per_event_mean': float(
                ((joint - independent).abs().sum((1, 2)) / total.clamp_min(1)).mean()
            ),
        }
        for index, name in enumerate(('under_hour', 'hour_to_day', 'day_to_week', 'week_plus')):
            result[name + '_fraction_mean'] = float(
                (joint[:, :, index].sum(-1) / total.clamp_min(1)).mean()
            )
        return result
