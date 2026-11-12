"""Count/freshness and match-group complementarity factorial.

All cells retain N,C,B,J, four d-wide history slots, and two d-wide
additional slots. G replaces occupied global means with disjoint match means.
E replaces repeated global means with inherited count/freshness blocks.
Both alternatives are computed in every cell; parameter shapes and module
construction order are identical. Counts describe only supplied last <=128
bucket-coded events. Global novelty is unverified. No scorer is executed.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'use_evidence': False}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.joint_mode = self.config['joint_mode']
        self.statistic_mode = self.config['statistic_mode']
        self.grouping_mode = self.config['grouping_mode']
        self.use_evidence = self.config['use_evidence']
        if self.config['embedding_dim'] != 16 or not isinstance(self.use_evidence, bool):
            raise ValueError('The factorial fixes 16-wide slots and a boolean switch.')
        if self.joint_mode not in ('observed', 'independent'):
            raise ValueError('Unknown joint_mode')
        if self.statistic_mode not in ('count', 'rate'):
            raise ValueError('Unknown statistic_mode')
        if self.grouping_mode not in ('match', 'rank', 'occupied_global'):
            raise ValueError('Unknown grouping_mode')
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
        width = len(meta['fields']) * dim + 4 * dim + 4 + 32
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
            occupied_global = global_mean[:, None, :] * (sizes > 0).unsqueeze(-1)
            switch = float(self.grouping_mode == 'occupied_global')
            pools = (1.0 - switch) * pools + switch * occupied_global
        counts = self._match_counts(batch)
        statistics = self._statistics(counts)
        return static, pools, counts, statistics, membership, semantic, sizes, global_mean

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
        static, pools, _, statistics, _, _, _, global_mean = self._represent(batch)
        counts, freshness, _ = self._evidence(batch)
        switch = float(self.use_evidence)
        count_slot = switch * counts + (1.0 - switch) * global_mean
        freshness_slot = switch * freshness + (1.0 - switch) * global_mean
        features = torch.cat([static, pools.flatten(1), statistics,
                              count_slot, freshness_slot], dim=-1)
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
        cb, fb, cells = self._evidence(batch)
        pcb, pfb, _ = self._evidence(padded)
        ecb, efb, _ = self._evidence(empty_batch)
        totals = torch.stack([n, c, b, joint], -1)
        fresh_cells = fb.reshape_as(cells)
        present = cells.gt(0).float()
        result.update({
            'evidence_enabled': float(self.use_evidence),
            'action_count_identity_error_max': (cells.sum(-1) - totals).abs().max(),
            'count_block_min': cb.min(),
            'count_block_max': cb.max(),
            'freshness_min': fb.min(),
            'freshness_max': fb.max(),
            'empty_cell_freshness_abs_max': (fresh_cells * (1.0 - present)).abs().max(),
            'padding_evidence_abs_error_max': torch.cat([cb-pcb, fb-pfb], -1).abs().max(),
            'all_empty_evidence_abs_max': torch.cat([ecb, efb], -1).abs().max(),
            'freshness_occupied_mean': (fresh_cells * present).sum() / present.sum().clamp_min(1),
            'all_empty_logits_finite_fraction': torch.isfinite(self.forward(empty_batch)).float().mean(),
        })
        return {key: float(value) for key, value in result.items()}

