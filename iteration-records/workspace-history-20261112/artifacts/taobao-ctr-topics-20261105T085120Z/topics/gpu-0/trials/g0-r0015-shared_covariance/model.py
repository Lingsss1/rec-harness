"""Prospective content-statistic x gradient-route factorial.

Inherit the round-13 covariance/marginal-product model and every allocated
parameter. The sole new training intervention is a value-preserving detach
at the selected content summary before its existing 256->32 projection.
Backbone and context-gate gradients remain active. All four content summaries
are still computed. Attached arms replay the prior forward/training graph;
detached arms remove only the direct content-summary embedding gradient.
No new inputs, loss, optimizer, parameters, or general novelty claim.
Host measures predictive utility and cost; local Torch execution unavailable.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'detailed_counts': True, 'continuous_freshness': True, 'context_mode': 'multiply', 'evidence_mode': 'detailed', 'gate_normalization': 'rms', 'evidence_partition': 'exclusive', 'moment_mode': 'second', 'content_mode': 'covariance', 'content_gradient': 'detached', 'content_readout': 'shared'}


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
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Exact ancestor allocation, including the output layer. Dormant
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
        width = len(meta['fields']) * dim + 4 * dim + 4
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)
        # Append only after all incumbent modules, preserving seeded backbone.
        if (self.config['embedding_dim'] != 16 or
                self.joint_mode != 'observed' or
                self.statistic_mode != 'count' or self.grouping_mode != 'match'):
            raise ValueError('This study fixes the incumbent representation.')
        self.detailed_counts = self.config['detailed_counts']
        self.continuous_freshness = self.config['continuous_freshness']
        if not isinstance(self.detailed_counts, bool) or not isinstance(self.continuous_freshness, bool):
            raise ValueError('Factor switches must be boolean.')
        self.correction = nn.Sequential(
            nn.Linear(32, 32, bias=False), nn.ReLU(),
            nn.Linear(32, 1, bias=False)
        )
        nn.init.zeros_(self.correction[-1].weight)
        # Appended after the ancestor correction to preserve its seeded tensors.
        self.context_projection = nn.Linear(4 * dim + 4, 32, bias=False)
        self.context_mode = self.config['context_mode']
        self.evidence_mode = self.config['evidence_mode']
        if self.context_mode != 'multiply':
            raise ValueError('This study fixes multiplicative context.')
        self.gate_normalization = self.config['gate_normalization']
        if self.gate_normalization not in ('raw', 'rms'):
            raise ValueError('Unknown gate_normalization')
        if self.evidence_mode not in ('detailed', 'redundant'):
            raise ValueError('Unknown evidence_mode')
        if not self.detailed_counts:
            raise ValueError('This study fixes detailed counts.')
        if self.evidence_mode != 'detailed' or self.gate_normalization != 'rms':
            raise ValueError('This study fixes detailed evidence and RMS gating.')
        self.evidence_partition = self.config['evidence_partition']
        if self.evidence_partition not in ('exclusive', 'cumulative'):
            raise ValueError('Unknown evidence_partition')
        if self.config['hidden'] != [128, 64]:
            raise ValueError('This study fixes inherited hidden widths.')
        if not self.continuous_freshness:
            raise ValueError('This study retains mean continuous freshness.')
        self.moment_mode = self.config['moment_mode']
        if self.moment_mode not in ('second', 'square'):
            raise ValueError('Unknown moment_mode')
        # Append after every ancestor module. Random construction consumes RNG
        # identically in all four cells; zero initialization preserves function.
        self.moment_projection = nn.Linear(16, 32, bias=False)
        nn.init.zeros_(self.moment_projection.weight)
        self.content_mode = self.config['content_mode']
        if self.content_mode not in ('joint', 'factorized', 'covariance', 'uniform'):
            raise ValueError('Unknown content_mode')
        if self.evidence_partition != 'exclusive' or self.moment_mode != 'second':
            raise ValueError('This study fixes the incumbent exclusive second moment.')
        # Construction after every incumbent tensor preserves its seeded values.
        self.content_projection = nn.Linear(16 * dim, 32, bias=False)
        nn.init.zeros_(self.content_projection.weight)
        # No module construction or RNG use after the inherited allocation.
        self.content_gradient = self.config['content_gradient']
        if self.content_gradient not in ('attached', 'detached'):
            raise ValueError('Unknown content_gradient')
        if self.content_mode not in ('covariance', 'factorized'):
            raise ValueError('This factorial fixes the two matched content signals.')
        self.content_readout = self.config['content_readout']
        if self.content_readout not in ('shared', 'bypass'):
            raise ValueError('Unknown content_readout')
        if self.content_gradient != 'detached':
            raise ValueError('This study fixes the direct content detach.')



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

    def _evidence_bases(self, batch):
        """Both [B,4,4] bases: raw action counts and one-day decayed sums.

        The cumulative order is all/category/brand/joint; the exclusive order
        is neither/category-only/brand-only/both. Zero target and history
        buckets never match. Only the supplied causal prefix is visible.
        Both bases are reduced directly to avoid subtractive cancellation
        in small exclusive cells. Both are computed in every variant.
        """
        mask = batch['history_mask']
        target_c = batch['static'][:, self.category:self.category + 1]
        target_b = batch['static'][:, self.brand:self.brand + 1]
        hc, hb = batch['history_category'], batch['history_brand']
        cm = mask & (target_c > 0) & (hc > 0) & (hc == target_c)
        bm = mask & (target_b > 0) & (hb > 0) & (hb == target_b)
        cumulative = torch.stack([mask, cm, bm, cm & bm], dim=-1).float().transpose(1, 2)
        exclusive = torch.stack([
            mask & ~cm & ~bm, cm & ~bm, bm & ~cm, cm & bm
        ], dim=1).float()
        actions = torch.stack([
            batch['history_type'].eq(action) for action in range(1, 5)
        ], dim=-1).float()
        ages = torch.where(mask, batch['history_age_seconds'],
                           torch.zeros_like(batch['history_age_seconds'])).clamp_min(0)
        decay = torch.exp(-ages / 86400.0)
        decayed_actions = actions * decay.unsqueeze(-1)
        cumulative_counts = torch.bmm(cumulative, actions)
        cumulative_decay = torch.bmm(cumulative, decayed_actions)
        exclusive_counts = torch.bmm(exclusive, actions)
        exclusive_decay = torch.bmm(exclusive, decayed_actions)
        second_actions = actions * decay.square().unsqueeze(-1)
        exclusive_second = torch.bmm(exclusive, second_actions)
        cumulative_second = self._to_cumulative(exclusive_second)
        return (cumulative_counts, cumulative_decay, exclusive_counts,
                exclusive_decay, cumulative_second, exclusive_second)

    def _encoded_evidence(self, batch):
        nc, dc, ne, de, sc, se = self._evidence_bases(batch)
        use = float(self.evidence_partition == 'exclusive')
        counts = use * ne + (1.0 - use) * nc
        decayed = use * de + (1.0 - use) * dc
        squared_decay = use * se + (1.0 - use) * sc
        denom = counts.clamp_min(1.0)
        count_block = torch.log1p(counts) / math.log(129.0)
        mean = decayed / denom
        second = squared_decay / denom
        return count_block.flatten(1), mean.flatten(1), counts, second.flatten(1)

    def _evidence(self, batch):
        cb, mean, cells, _ = self._encoded_evidence(batch)
        return cb, mean, cells

    @staticmethod
    def _to_cumulative(exclusive):
        # Works for raw counts and decayed sums. Do not sum cell means.
        n0, n1, n2, n3 = exclusive.unbind(dim=1)
        return torch.stack([n0 + n1 + n2 + n3, n1 + n3,
                            n2 + n3, n3], dim=1)

    @staticmethod
    def _to_exclusive(cumulative):
        n, c, b, j = cumulative.unbind(dim=1)
        return torch.stack([n - c - b + j, c - j, b - j, j], dim=1)

    def _correction_inputs(self, batch):
        counts, mean, cells, second = self._encoded_evidence(batch)
        # E[d^2] - E[d]^2 is variance for occupied cells. Both are in [0,1],
        # identical at singleton cells, and zero at empty cells. No clamping
        # of moments: diagnostics expose roundoff and identity failures.
        square = mean.square()
        use = float(self.moment_mode == 'second')
        selected = use * second + (1.0 - use) * square
        inputs = torch.cat([counts, mean, selected], -1)
        return inputs, counts, mean, cells, cells.gt(0).float().flatten(1)

    def _temporal_content(self, batch):
        # Group order: neither/category-only/brand-only/both, then actions 1..4.
        membership = self._membership(batch)[1]
        actions = torch.stack([batch['history_type'].eq(a) for a in range(1, 5)], dim=1).float()
        cell_mask = (membership[:, :, None, :] * actions[:, None, :, :]).reshape(
            membership.shape[0], 16, membership.shape[-1])
        counts = cell_mask.sum(-1)
        content = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']))
        ages = torch.where(batch['history_mask'], batch['history_age_seconds'],
                           torch.zeros_like(batch['history_age_seconds'])).clamp_min(0)
        decay = torch.exp(-ages / 86400.0)
        weighted = cell_mask * decay[:, None, :]
        denom = counts.clamp_min(1).unsqueeze(-1)
        uniform = torch.bmm(cell_mask, content) / denom
        joint = torch.bmm(weighted, content) / denom
        mean_decay = weighted.sum(-1, keepdim=True) / denom
        factorized = uniform * mean_decay
        covariance = joint - factorized
        return joint, factorized, covariance, uniform, counts

    def _raw_content_inputs(self, batch):
        joint, factorized, covariance, uniform, _ = self._temporal_content(batch)
        # Original feature arithmetic and all four reductions are unchanged.
        return (float(self.content_mode == 'joint') * joint +
                float(self.content_mode == 'factorized') * factorized +
                float(self.content_mode == 'covariance') * covariance +
                float(self.content_mode == 'uniform') * uniform).flatten(1)

    def _route_content(self, values):
        # This changes only backward connectivity, never numeric feature values.
        # Do not detach context or incumbent pools: those are separate paths.
        if self.content_gradient == 'detached':
            return values.detach()
        return values

    def _content_inputs(self, batch):
        return self._route_content(self._raw_content_inputs(batch))

    def _selected_inputs(self, batch, statistics):
        return torch.cat([self._correction_inputs(batch)[0], self._content_inputs(batch)], -1)

    def _gate(self, context):
        # Per-impression normalization across 32 projected coordinates. Fixed
        # epsilon; no fitted statistics, affine parameters or batch dependence.
        raw = self.context_projection(context)
        scale = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / scale
        use = float(self.gate_normalization == 'rms')
        selected = use * normalized + (1.0 - use) * raw
        return torch.tanh(selected), raw, normalized, scale

    def _readout_parts(self, inputs):
        scalar = (self.correction[0](inputs[:, :32]) +
                  self.moment_projection(inputs[:, 32:48]))
        content = self.content_projection(inputs[:, 48:])
        shared = self.correction[1](scalar + content)
        bypass = self.correction[1](scalar) + content
        return scalar, content, shared, bypass

    def _readout(self, inputs, context):
        _, _, shared, bypass = self._readout_parts(inputs)
        # Same parameters, constructor and initial function in every cell.
        # Both forms execute; backward connectivity and measured cost can differ.
        hidden = bypass if self.content_readout == 'bypass' else shared
        modulation = self._gate(context)[0]
        product = hidden * (1.0 + modulation)
        additive = hidden + modulation
        use = float(self.context_mode == 'multiply')
        mixed = use * product + (1.0 - use) * additive
        correction = self.correction[2](mixed).squeeze(-1)
        return correction, hidden, modulation

    def _logits(self, batch):
        static, pools, _, statistics, _, _, _, _ = self._represent(batch)
        context = torch.cat([pools.flatten(1), statistics], dim=-1)
        features = torch.cat([static, context], dim=-1)
        base = self.output(features).squeeze(-1)
        inputs = self._selected_inputs(batch, statistics)
        correction = self._readout(inputs, context)[0]
        return base + correction, base, correction

    def forward(self, batch):
        return self._logits(batch)[0]

    def _swap_cell_ages(self, batch):
        # Diagnostic only: exchange first/last ages within each action/match cell.
        # All content, masks, cell margins, age multisets and incumbent pools stay
        # fixed. This is an invariance check, not a scored intervention.
        membership = self._membership(batch)[1].bool()
        positions = torch.arange(batch['history_mask'].shape[1], device=batch['static'].device)[None, :]
        original = batch['history_age_seconds']
        swapped = original
        length = original.shape[1]
        if length == 0:
            return dict(batch)
        for group in range(4):
            for action in range(1, 5):
                valid = membership[:, group, :] & batch['history_type'].eq(action)
                first = torch.where(valid, positions, length).min(-1, keepdim=True).values
                last = torch.where(valid, positions, -1).max(-1, keepdim=True).values
                occupied = first < length
                early = original.gather(1, first.clamp(0, length - 1))
                late = original.gather(1, last.clamp(0, length - 1))
                swapped = torch.where(occupied & positions.eq(first), late, swapped)
                swapped = torch.where(occupied & positions.eq(last), early, swapped)
        result = dict(batch)
        result['history_age_seconds'] = swapped
        return result

    def diagnostics(self, batch):
        # First 512 validation impressions only; label-free implementation checks.
        logits, base, correction = self._logits(batch)
        joint, fact, cov, uniform, cells = self._temporal_content(batch)
        swapped = self._swap_cell_ages(batch)
        sj, sf, scov, su, sn = self._temporal_content(swapped)
        rep = self._represent(batch)
        swap_rep = self._represent(swapped)
        selected = self._selected_inputs(batch, rep[3])
        context = torch.cat([rep[1].flatten(1), rep[3]], -1)
        inherited = self._correction_inputs(batch)[0]
        inherited_swap = self._correction_inputs(swapped)[0]
        projected = self.content_projection(selected[:, 48:])
        without = torch.cat([inherited, torch.zeros_like(selected[:, 48:])], -1)
        removed = self._readout(without, context)[0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero_length = dict(batch)
        reverse = dict(batch)
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero_length[key] = batch[key][:, :0]
            reverse[key] = batch[key].flip(1)
            if key != 'history_mask':
                padded[key] = torch.where(batch['history_mask'], batch[key], torch.zeros_like(batch[key]))
        zero_age = dict(batch)
        zero_age['history_age_seconds'] = torch.zeros_like(batch['history_age_seconds'])
        zj, zf, zc, zu, _ = self._temporal_content(zero_age)
        singleton = cells.eq(1).unsqueeze(-1).float()
        multi = cells.gt(1).float()
        absent = cells.eq(0).unsqueeze(-1).float()
        empty_selected = self._selected_inputs(empty, rep[3])
        expected_cells = self._encoded_evidence(batch)[2].flatten(1)
        raw_content = self._raw_content_inputs(batch)
        routed_content = self._route_content(raw_content)
        # Host calls under no_grad. These check values/flags, not training
        # Jacobians or measured gradient conflict. Source semantics define route.
        scalar, content, shared, bypass = self._readout_parts(selected)
        scalar_hidden = self.correction[1](scalar)
        gate = 1.0 + self._gate(context)[0]
        scalar_only = self.correction[2](scalar_hidden * gate).squeeze(-1)
        bypass_only = self.correction[2](bypass * gate).squeeze(-1)
        direct_content = self.correction[2](content * gate).squeeze(-1)
        selected_hidden = bypass if self.content_readout == 'bypass' else shared
        result = {
            'readout_bypass': float(self.content_readout == 'bypass'),
            'scalar_preactivation_rms': scalar.square().mean().sqrt(),
            'scalar_relu_active_fraction': scalar.gt(0).float().mean(),
            'shared_relu_flip_fraction': (scalar.gt(0) != (scalar + content).gt(0)).float().mean(),
            'content_projected_negative_fraction': content.lt(0).float().mean(),
            'shared_bypass_hidden_gap_rms': (shared - bypass).square().mean().sqrt(),
            'bypass_hidden_identity_error_max': (bypass - scalar_hidden - content).abs().max(),
            'bypass_logit_identity_error_max': (bypass_only - scalar_only - direct_content).abs().max(),
            'zero_content_route_identity_error_max': (
                self.correction[1](scalar + torch.zeros_like(content)) -
                (scalar_hidden + torch.zeros_like(content))).abs().max(),
            'readout_reconstruction_error_max': (correction -
                self.correction[2](selected_hidden * gate).squeeze(-1)).abs().max(),
            'content_gradient_detached': float(self.content_gradient == 'detached'),
            'route_value_identity_error_max': (raw_content - routed_content).abs().max(),
            'route_projection_identity_error_max': (self.content_projection(raw_content) -
                                                     self.content_projection(routed_content)).abs().max(),
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'content_projection_parameters': float(self.content_projection.weight.numel()),
            'content_projection_weight_rms': self.content_projection.weight.square().mean().sqrt(),
            'content_mode_joint': float(self.content_mode == 'joint'),
            'content_mode_factorized': float(self.content_mode == 'factorized'),
            'content_mode_covariance': float(self.content_mode == 'covariance'),
            'content_mode_uniform': float(self.content_mode == 'uniform'),
            'cell_count_identity_error_max': (cells - expected_cells).abs().max(),
            'occupied_cells_mean': cells.gt(0).float().sum(-1).mean(),
            'multievent_cells_mean': multi.sum(-1).mean(),
            'joint_rms': joint.square().mean().sqrt(),
            'factorized_rms': fact.square().mean().sqrt(),
            'uniform_rms': uniform.square().mean().sqrt(),
            'covariance_rms': cov.square().mean().sqrt(),
            'multievent_covariance_rms': ((cov.square().sum(-1) * multi).sum() /
                                        (multi.sum().clamp_min(1) * cov.shape[-1])).sqrt(),
            'covariance_identity_error_max': (joint - fact - cov).abs().max(),
            'singleton_covariance_error_max': (cov * singleton).abs().max(),
            'empty_cell_joint_error_max': (joint * absent).abs().max(),
            'empty_cell_factorized_error_max': (fact * absent).abs().max(),
            'zero_age_joint_uniform_error_max': (zj - zu).abs().max(),
            'zero_age_factor_uniform_error_max': (zf - zu).abs().max(),
            'zero_age_covariance_error_max': zc.abs().max(),
            'swap_age_changed_fraction': (swapped['history_age_seconds'] != batch['history_age_seconds']).float().mean(),
            'swap_cell_count_error_max': (cells - sn).abs().max(),
            'swap_uniform_error_max': (uniform - su).abs().max(),
            'swap_factorized_error_max': (fact - sf).abs().max(),
            'swap_incumbent_pools_error_max': (rep[1] - swap_rep[1]).abs().max(),
            'swap_incumbent_statistics_error_max': (rep[3] - swap_rep[3]).abs().max(),
            'swap_incumbent_evidence_error_max': (inherited - inherited_swap).abs().max(),
            'swap_joint_change_rms': (joint - sj).square().mean().sqrt(),
            'swap_covariance_change_rms': (cov - scov).square().mean().sqrt(),
            'swap_selected_change_rms': (selected[:, 48:] - self._content_inputs(swapped)).square().mean().sqrt(),
            'swap_base_error_max': (base - self._logits(swapped)[1]).abs().max(),
            'swap_logit_change_rms': (logits - self.forward(swapped)).square().mean().sqrt(),
            'content_projected_rms': projected.square().mean().sqrt(),
            'content_head_removal_effect_rms': (correction - removed).square().mean().sqrt(),
            'empty_inputs_error_max': empty_selected.abs().max(),
            'empty_correction_error_max': self._logits(empty)[2].abs().max(),
            'zero_length_empty_error_max': (self.forward(zero_length) - self.forward(empty)).abs().max(),
            'padding_logit_error_max': (logits - self.forward(padded)).abs().max(),
            'reverse_logit_error_max': (logits - self.forward(reverse)).abs().max(),
            'logits_finite_fraction': torch.isfinite(logits).float().mean(),
            'correction_rms': correction.square().mean().sqrt(),
            'additivity_error_max': (logits - base - correction).abs().max(),
        }
        return {key: float(value) for key, value in result.items()}
