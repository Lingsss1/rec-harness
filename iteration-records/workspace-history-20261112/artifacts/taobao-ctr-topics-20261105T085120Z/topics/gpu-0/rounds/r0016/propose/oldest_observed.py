"""Prospective temporal-second-moment x evidence-partition factorial.

Retain count and mean one-day decay, append E[decay**2] per action/match
cell, and compare with the information-redundant E[decay]**2 replacement.
The two features coincide for singleton and constant-age cells. Their gap
is within-cell variance in decay space, not event order or a burst detector.
Each standalone cell allocates the same extra 16->32 projection, initialized
to zero after every inherited module so the initial function is unchanged.
Both moments/bases are computed in every variant; cost requires host measures.
No novelty, local predictive improvement, or full DIN reproduction is claimed.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'detailed_counts': True, 'continuous_freshness': True, 'context_mode': 'multiply', 'evidence_mode': 'detailed', 'gate_normalization': 'rms', 'evidence_partition': 'exclusive', 'moment_mode': 'second', 'extreme_direction': 'oldest', 'extreme_source': 'observed'}


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
        self.extreme_direction = self.config['extreme_direction']
        self.extreme_source = self.config['extreme_source']
        if self.extreme_direction not in ('recent', 'oldest'):
            raise ValueError('Unknown extreme_direction')
        if self.extreme_source not in ('observed', 'bound'):
            raise ValueError('Unknown extreme_source')
        if self.evidence_partition != 'exclusive' or self.moment_mode != 'second':
            raise ValueError('This study fixes the exclusive second-moment incumbent.')
        # Appended after every ancestor module; same RNG path in all arms.
        # Zero projection preserves the inherited initial function exactly.
        self.extreme_projection = nn.Linear(16, 32, bias=False)
        nn.init.zeros_(self.extreme_projection.weight)



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

    def _extreme_features(self, batch, cells, mean, second):
        """Group-major [B,16] observed extrema and moment-only bounds.

        For n observations, population variance v implies every observation
        lies within mean +/- sqrt((n-1)*v). Intersect with [0,1]. A max is
        the most recent decay; a min is the oldest visible decay. Empty
        cells use zero for all features, not the reduction sentinel.
        """
        n = cells.flatten(1)
        present = n.gt(0).to(mean.dtype)
        variance = (second - mean.square()).clamp_min(0.0)
        radius = ((n - 1.0).clamp_min(0.0) * variance).sqrt()
        upper = (mean + radius).clamp(0.0, 1.0) * present
        lower = (mean - radius).clamp(0.0, 1.0) * present
        mask = batch['history_mask']
        if mask.shape[1] == 0:
            maximum = torch.zeros_like(mean)
            minimum = torch.zeros_like(mean)
        else:
            membership = self._membership(batch)[0].bool()
            ages = torch.where(mask, batch['history_age_seconds'],
                               torch.zeros_like(batch['history_age_seconds'])).clamp_min(0.0)
            decay = torch.exp(-ages / 86400.0)[:, None, :]
            maxima, minima = [], []
            # Avoid constructing a [B,4,4,L] temporary. Action-major loop,
            # stack as [B,match,action], same flatten order as evidence.
            for action in range(1, 5):
                members = membership & batch['history_type'].eq(action)[:, None, :]
                values = decay.expand_as(membership)
                maxima.append(values.masked_fill(~members, 0.0).amax(-1))
                minima.append(values.masked_fill(~members, 1.0).amin(-1))
            maximum = torch.stack(maxima, dim=-1).flatten(1) * present
            minimum = torch.stack(minima, dim=-1).flatten(1) * present
        recent = float(self.extreme_direction == 'recent')
        observed = recent * maximum + (1.0 - recent) * minimum
        bound = recent * upper + (1.0 - recent) * lower
        use_observed = float(self.extreme_source == 'observed')
        selected = use_observed * observed + (1.0 - use_observed) * bound
        return selected, maximum, minimum, upper, lower

    def _correction_inputs(self, batch):
        counts, mean, cells, second = self._encoded_evidence(batch)
        # Frozen incumbent input coordinates, followed by one new block.
        selected = self._extreme_features(batch, cells, mean, second)[0]
        inputs = torch.cat([counts, mean, second, selected], -1)
        return inputs, counts, mean, cells, cells.gt(0).float().flatten(1)

    def _selected_inputs(self, batch, statistics):
        return self._correction_inputs(batch)[0]

    def _gate(self, context):
        # Per-impression normalization across 32 projected coordinates. Fixed
        # epsilon; no fitted statistics, affine parameters or batch dependence.
        raw = self.context_projection(context)
        scale = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / scale
        use = float(self.gate_normalization == 'rms')
        selected = use * normalized + (1.0 - use) * raw
        return torch.tanh(selected), raw, normalized, scale

    def _readout(self, inputs, context):
        preactivation = (self.correction[0](inputs[:, :32]) +
                         self.moment_projection(inputs[:, 32:48]) +
                         self.extreme_projection(inputs[:, 48:64]))
        hidden = self.correction[1](preactivation)
        modulation = self._gate(context)[0]
        # Evaluate both forms in every cell. Equal parameters and arithmetic;
        # measured end-to-end cost, not this source claim, decides efficiency.
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

    def diagnostics(self, batch):
        # Candidate-supplied, label-free checks on the host's first 512
        # impressions. Feature/head sensitivity is not predictive attribution.
        logits, base, correction = self._logits(batch)
        _, pools, totals, stats, _, _, sizes, _ = self._represent(batch)
        context = torch.cat([pools.flatten(1), stats], -1)
        inputs, cb, mean, cells, presence = self._correction_inputs(batch)
        _, _, _, second = self._encoded_evidence(batch)
        selected, maximum, minimum, upper, lower = self._extreme_features(
            batch, cells, mean, second)
        n = cells.flatten(1)
        singleton = n.eq(1).float()
        pair = n.eq(2).float()
        large = n.ge(3).float()
        occupied = presence.sum().clamp_min(1.0)
        recent = float(self.extreme_direction == 'recent')
        observed = recent * maximum + (1.0 - recent) * minimum
        bound = recent * upper + (1.0 - recent) * lower
        scalar = self.correction[0](inputs[:, :32]) + self.moment_projection(inputs[:, 32:48])
        projected = self.extreme_projection(selected)
        gate = self._gate(context)[0]
        reconstructed = self.correction[2](self.correction[1](scalar + projected) * (1.0 + gate)).squeeze(-1)
        no_extreme = self.correction[2](self.correction[1](scalar) * (1.0 + gate)).squeeze(-1)
        observed_inputs = torch.cat([inputs[:, :48], observed], -1)
        bound_inputs = torch.cat([inputs[:, :48], bound], -1)
        observed_correction = self._readout(observed_inputs, context)[0]
        bound_correction = self._readout(bound_inputs, context)[0]
        padded = dict(batch)
        reverse = dict(batch)
        zero_length = dict(batch)
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[name] = torch.where(batch['history_mask'], batch[name], torch.zeros_like(batch[name]))
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[name] = batch[name].flip(1)
            zero_length[name] = batch[name][:, :0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        empty_logits, _, empty_correction = self._logits(empty)
        empty_inputs = self._correction_inputs(empty)[0]
        zero_age = dict(batch)
        zero_age['history_age_seconds'] = torch.zeros_like(batch['history_age_seconds'])
        _, zero_mean, zero_cells, zero_second = self._encoded_evidence(zero_age)
        zsel, zmax, zmin, zupper, zlower = self._extreme_features(
            zero_age, zero_cells, zero_mean, zero_second)
        # Never reduce an empty event axis: all reduced feature axes have 16
        # entries, even if the supplied history has zero length.
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'extreme_projection_parameters': float(self.extreme_projection.weight.numel()),
            'extreme_projection_weight_rms': self.extreme_projection.weight.square().mean().sqrt(),
            'recent_direction': recent,
            'observed_source': float(self.extreme_source == 'observed'),
            'logits_finite_fraction': torch.isfinite(logits).float().mean(),
            'empty_logits_finite_fraction': torch.isfinite(empty_logits).float().mean(),
            'empty_history_fraction': totals[0].eq(0).float().mean(),
            'visible_count_mean': totals[0].mean(),
            'visible_at_limit_fraction': totals[0].eq(128).float().mean(),
            'occupied_cells_mean': presence.sum(-1).mean(),
            'singleton_occupied_fraction': singleton.sum() / occupied,
            'pair_occupied_fraction': pair.sum() / occupied,
            'large_occupied_fraction': large.sum() / occupied,
            'cell_count_alignment_error_max': (cells.sum(-1) - sizes).abs().max(),
            'maximum_minimum_order_violation_max': (minimum - maximum).clamp_min(0).max(),
            'mean_above_maximum_violation_max': (mean - maximum).clamp_min(0).max(),
            'mean_below_minimum_violation_max': (minimum - mean).clamp_min(0).max(),
            'upper_bound_violation_max': (maximum - upper).clamp_min(0).max(),
            'lower_bound_violation_max': (lower - minimum).clamp_min(0).max(),
            'variance_roundoff_floor_max': (mean.square() - second).clamp_min(0).max(),
            'recent_gap_rms': (maximum - upper).square().mean().sqrt(),
            'oldest_gap_rms': (minimum - lower).square().mean().sqrt(),
            'selected_gap_rms': (observed - bound).square().mean().sqrt(),
            'large_recent_gap_rms': (((maximum - upper).square() * large).sum() / large.sum().clamp_min(1.0)).sqrt(),
            'large_oldest_gap_rms': (((minimum - lower).square() * large).sum() / large.sum().clamp_min(1.0)).sqrt(),
            'large_selected_gap_gt1e3_fraction': ((observed - bound).abs().gt(1e-3).float() * large).sum() / large.sum().clamp_min(1.0),
            'singleton_extreme_bound_error_max': torch.maximum(((maximum - upper) * singleton).abs().max(), ((minimum - lower) * singleton).abs().max()),
            'pair_extreme_bound_error_max': torch.maximum(((maximum - upper) * pair).abs().max(), ((minimum - lower) * pair).abs().max()),
            'upper_saturated_occupied_fraction': (upper.ge(1.0).float() * presence).sum() / occupied,
            'lower_saturated_occupied_fraction': (lower.le(0.0).float() * presence).sum() / occupied,
            'empty_cell_extreme_error_max': torch.maximum((maximum * (1.0 - presence)).abs().max(), (minimum * (1.0 - presence)).abs().max()),
            'selected_feature_min': selected.min(),
            'selected_feature_max': selected.max(),
            'selected_observed_error_max': (selected - observed).abs().max(),
            'selected_bound_error_max': (selected - bound).abs().max(),
            'zero_age_extreme_presence_error_max': torch.stack([(v - presence).abs().max() for v in (zsel, zmax, zmin, zupper, zlower)]).amax(),
            'padding_logit_error_max': (logits - self.forward(padded)).abs().max(),
            'reverse_logit_error_max': (logits - self.forward(reverse)).abs().max(),
            'empty_inputs_error_max': empty_inputs.abs().max(),
            'empty_correction_error_max': empty_correction.abs().max(),
            'zero_length_empty_error_max': (self.forward(zero_length) - empty_logits).abs().max(),
            'readout_reconstruction_error_max': (reconstructed - correction).abs().max(),
            'projected_extreme_rms': projected.square().mean().sqrt(),
            'head_removal_effect_rms': (correction - no_extreme).square().mean().sqrt(),
            'head_observed_bound_effect_rms': (observed_correction - bound_correction).square().mean().sqrt(),
            'base_logit_rms': base.square().mean().sqrt(),
            'correction_rms': correction.square().mean().sqrt(),
            'additivity_error_max': (logits - base - correction).abs().max(),
        }
        return {key: float(value) for key, value in result.items()}
