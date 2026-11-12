"""Action/age contingencies added to the inherited true-match mean pools.

Four standalone cells cross observed/factorized action-age counts with
semantic/same-cardinality rank grouping of ONLY the added count tables.
Factorization preserves raw groupwise action and age margins. All cells keep
the ancestor's learned pools, counts and allocation order; no event is dropped.
Age boundaries are fixed seconds, never fitted. No global novelty claim.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'rank', 'histogram_pairing': 'independent'}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.joint_mode = self.config['joint_mode']
        self.statistic_mode = self.config['statistic_mode']
        self.grouping_mode = self.config['grouping_mode']
        if self.joint_mode not in ('observed', 'independent'):
            raise ValueError('Unknown joint_mode')
        if self.statistic_mode not in ('count', 'rate'):
            raise ValueError('Unknown statistic_mode')
        if self.grouping_mode not in ('match', 'rank', 'occupied_global'):
            raise ValueError('Unknown grouping_mode')
        # This factorial intervenes only on the added count table. The learned
        # means must stay in the incumbent's true-match groups in all cells.
        if self.grouping_mode != 'match':
            raise ValueError('This study fixes learned pooling to match')
        self.histogram_grouping = self.config['histogram_grouping']
        self.histogram_pairing = self.config['histogram_pairing']
        if self.histogram_grouping not in ('match', 'rank'):
            raise ValueError('Unknown histogram_grouping')
        if self.histogram_pairing not in ('observed', 'independent'):
            raise ValueError('Unknown histogram_pairing')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Exact ancestor allocation up to the wider output layer. Dormant
        # attention is retained in every variant; it is never executed.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + 4 * dim + 4 + 64
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _match_counts(self, batch):
            mask = batch['history_mask']
            target_c = batch['static'][:, self.category:self.category + 1]
            target_b = batch['static'][:, self.brand:self.brand + 1]
            hc = batch['history_category']
            hb = batch['history_brand']
            # Zero represents padding/missing; missing-to-missing is never a match.
            cm = mask & (target_c > 0) & (hc > 0) & (hc == target_c)
            bm = mask & (target_b > 0) & (hb > 0) & (hb == target_b)
            n = mask.float().sum(-1)
            c = cm.float().sum(-1)
            b = bm.float().sum(-1)
            joint = (cm & bm).float().sum(-1)
            independent = c * b / n.clamp_min(1)
            return n, c, b, joint, independent

    def _statistics(self, counts):
            n, c, b, joint, independent = counts
            selected = joint if self.joint_mode == 'observed' else independent
            if self.statistic_mode == 'count':
                # Fixed scale from frozen history_limit=128, never from validation.
                # For fractional independence estimates the same log1p applies.
                return torch.log1p(torch.stack([n, c, b, selected], -1)) / math.log(129.0)
            denom = n.clamp_min(1)
            return torch.stack([
                (n > 0).float(), c / denom, b / denom, selected / denom
            ], -1)

    def _membership(self, batch):
        mask = batch['history_mask']
        target_c = batch['static'][:, self.category:self.category + 1]
        target_b = batch['static'][:, self.brand:self.brand + 1]
        hc, hb = batch['history_category'], batch['history_brand']
        cm = mask & (target_c > 0) & (hc > 0) & (hc == target_c)
        bm = mask & (target_b > 0) & (hb > 0) & (hb == target_b)
        semantic = torch.stack([
            mask & ~cm & ~bm, cm & ~bm, bm & ~cm, cm & bm
        ], dim=1).float()  # [B,4,L]
        sizes = semantic.sum(-1)
        if self.grouping_mode == 'rank':
            # Assign in supplied input order, counting only valid positions.
            # Prefix thresholds reproduce true cardinalities even with gaps.
            # This is a positional replacement, not a randomized control.
            rank = mask.long().cumsum(-1) - 1
            upper = sizes.long().cumsum(-1)
            lower = upper - sizes.long()
            membership = ((rank[:, None, :] >= lower[:, :, None]) &
                          (rank[:, None, :] < upper[:, :, None]) &
                          mask[:, None, :]).float()
        else:
            membership = semantic
        return membership, semantic, sizes

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        membership, semantic, sizes = self._membership(batch)
        mask = batch['history_mask']
        if mask.shape[1] == 0:
            pools = static.new_zeros((static.shape[0], 4, self.config['embedding_dim']))
            global_mean = pools[:, 0, :]
        else:
            ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            history = (self.embeddings[self.category](batch['history_category']) +
                       self.embeddings[self.brand](batch['history_brand']) +
                       self.action(batch['history_type']) + self.recency(ages))
            # [B,4,L] @ [B,L,d] avoids an expanded [B,4,L,d] temporary.
            # Empty groups divide by one and have an exactly zero numerator.
            pools = torch.bmm(membership, history) / sizes.clamp_min(1).unsqueeze(-1)
            n = mask.float().sum(-1, keepdim=True).clamp_min(1)
            global_mean = (history * mask.unsqueeze(-1)).sum(1) / n
            if self.grouping_mode == 'occupied_global':
                pools = global_mean[:, None, :] * (sizes > 0).unsqueeze(-1)
        counts = self._match_counts(batch)
        statistics = self._statistics(counts)
        return static, pools, counts, statistics, membership, semantic, sizes, global_mean

    def _histogram(self, batch, semantic, sizes):
        """Return raw [B,group,action,age] tables and the selected readout.

        No learned values enter this calculation. Every variant computes BOTH
        tables, so the pairing switch changes information rather than a neural
        event-scoring path. The log transform is applied after replacement;
        only RAW margins, not sums of log-transformed cells, are preserved.
        """
        mask = batch['history_mask']
        if self.histogram_grouping == 'rank':
            rank = mask.long().cumsum(-1) - 1
            upper = sizes.long().cumsum(-1)
            lower = upper - sizes.long()
            membership = ((rank[:, None, :] >= lower[:, :, None]) &
                          (rank[:, None, :] < upper[:, :, None]) &
                          mask[:, None, :]).float()
        else:
            membership = semantic
        # Contract: valid events have type 1..4; masked positions have no mass.
        action_id = (batch['history_type'] - 1).clamp(0, 3)
        age = batch['history_age_seconds']
        age_id = ((age >= 3600.0).long() + (age >= 86400.0).long() +
                  (age >= 604800.0).long())
        cell = (4 * action_id + age_id)[:, None, :].expand(-1, 4, -1)
        observed = membership.new_zeros((mask.shape[0], 4, 16))
        observed = observed.scatter_add(2, cell, membership)
        observed = observed.reshape(mask.shape[0], 4, 4, 4)
        action_margin = observed.sum(-1)
        age_margin = observed.sum(-2)
        expected = (action_margin.unsqueeze(-1) * age_margin.unsqueeze(-2) /
                    sizes.clamp_min(1)[:, :, None, None])
        selected = observed if self.histogram_pairing == 'observed' else expected
        features = torch.log1p(selected.flatten(1)) / math.log(129.0)
        return features, observed, expected, membership

    def forward(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        features = torch.cat([static, pools.flatten(1), statistics, histogram], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        # Label-free exploratory diagnostics on host's first 512 impressions.
        # They check this code, not independent evidence for the mechanism.
        static, pools, counts, stats, membership, semantic, sizes, global_mean = self._represent(batch)
        n, c, b, joint, _ = counts
        expected_sizes = torch.stack([n - c - b + joint, c - joint, b - joint, joint], -1)
        weighted = (pools * sizes.unsqueeze(-1)).sum(1) / n.clamp_min(1).unsqueeze(-1)
        coverage_error = (membership.sum(1) - batch['history_mask'].float()).abs().sum(-1)
        mismatch = (membership - semantic).abs().sum((1, 2)) / (2 * n.clamp_min(1))
        rotated = dict(batch)
        rotated_static = batch['static'].clone()
        for index in (self.category, self.brand):
            rotated_static[:, index] = batch['static'][:, index].roll(1, 0)
        rotated['static'] = rotated_static
        rotated_pools = self._represent(rotated)[1]
        padded = dict(batch)
        # Perturb only masked positions to valid zero IDs/age; outputs must agree.
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[name] = torch.where(batch['history_mask'], batch[name], torch.zeros_like(batch[name]))
        padded_pools = self._represent(padded)[1]
        empty_batch = dict(batch)
        empty_batch['history_mask'] = torch.zeros_like(batch['history_mask'])
        empty_pools = self._represent(empty_batch)[1]
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'dormant_scorer_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'neural_event_scorer_executed': 0.0,
            'true_match_membership_enabled': float(self.grouping_mode == 'match'),
            'rank_membership_enabled': float(self.grouping_mode == 'rank'),
            'global_repeat_enabled': float(self.grouping_mode == 'occupied_global'),
            'empty_history_fraction': (n == 0).float().mean(),
            'visible_count_mean': n.mean(),
            'visible_count_at_limit_fraction': (n == 128).float().mean(),
            'occupied_groups_mean': (sizes > 0).float().sum(-1).mean(),
            'mixed_groups_fraction': ((sizes > 0).sum(-1) > 1).float().mean(),
            'partition_coverage_error_max': coverage_error.max(),
            'group_cardinality_error_max': (membership.sum(-1) - sizes).abs().max(),
            'group_count_identity_error_max': (sizes - expected_sizes).abs().max(),
            'membership_changed_fraction_mean': mismatch.mean(),
            'global_reconstruction_abs_error_max': (weighted - global_mean).abs().max(),
            'padding_pool_abs_error_max': (pools - padded_pools).abs().max(),
            'all_empty_pool_abs_max': empty_pools.abs().max(),
            'rotated_target_pool_rms_delta': (pools - rotated_pools).square().mean().sqrt(),
            'pool_rms': pools.square().mean().sqrt(),
            'category_match_present_fraction': (c > 0).float().mean(),
            'brand_match_present_fraction': (b > 0).float().mean(),
            'joint_match_present_fraction': (joint > 0).float().mean(),
            'target_brand_missing_fraction': (batch['static'][:, self.brand] == 0).float().mean(),
        }
        for i, name in enumerate(('neither', 'category_only', 'brand_only', 'both')):
            result[name + '_count_mean'] = sizes[:, i].mean()
            result[name + '_present_fraction'] = (sizes[:, i] > 0).float().mean()
            result[name + '_pool_rms'] = pools[:, i, :].square().mean().sqrt()
        for i, name in enumerate(('activity', 'category', 'brand', 'joint')):
            result['selected_' + name + '_mean'] = stats[:, i].mean()
        histogram, observed, expected, hist_membership = self._histogram(batch, semantic, sizes)
        padded_histogram = self._histogram(padded, semantic, sizes)[0]
        empty_membership, empty_semantic, empty_sizes = self._membership(empty_batch)
        empty_histogram = self._histogram(empty_batch, empty_semantic, empty_sizes)[0]
        joint_gap = (observed - expected).abs().sum((1, 2, 3))
        singleton_gap = ((observed - expected).abs() *
                         (sizes <= 1)[:, :, None, None]).max()
        result.update({
            'hist_true_match_enabled': float(self.histogram_grouping == 'match'),
            'hist_observed_pairing_enabled': float(self.histogram_pairing == 'observed'),
            'hist_added_parameters': float(64 * self.config['hidden'][0]) if self.config['hidden'] else 64.0,
            'hist_coverage_error_max': (hist_membership.sum(1) - batch['history_mask'].float()).abs().max() if batch['history_mask'].shape[1] else 0.0,
            'hist_group_size_error_max': (observed.sum((-1, -2)) - sizes).abs().max(),
            'hist_factorized_action_margin_error_max': (observed.sum(-1) - expected.sum(-1)).abs().max(),
            'hist_factorized_age_margin_error_max': (observed.sum(-2) - expected.sum(-2)).abs().max(),
            'hist_singleton_pairing_error_max': singleton_gap,
            'hist_association_l1_per_event_mean': (joint_gap / n.clamp_min(1)).mean(),
            'hist_nonzero_association_fraction': (joint_gap > 1e-5).float().mean(),
            'hist_occupied_cells_mean': (observed > 0).float().sum((1, 2, 3)).mean(),
            'hist_feature_rms': histogram.square().mean().sqrt(),
            'hist_padding_abs_error_max': (histogram - padded_histogram).abs().max(),
            'hist_all_empty_abs_max': empty_histogram.abs().max(),
            'hist_rank_changed_fraction_mean': ((hist_membership - semantic).abs().sum((1, 2)) / (2 * n.clamp_min(1))).mean(),
        })
        total = observed.sum().clamp_min(1)
        for i in range(4):
            result['hist_age_bin_' + str(i) + '_event_fraction'] = observed[:, :, :, i].sum() / total
            result['hist_action_' + str(i + 1) + '_event_fraction'] = observed[:, :, i, :].sum() / total
        return {key: float(value) for key, value in result.items()}
