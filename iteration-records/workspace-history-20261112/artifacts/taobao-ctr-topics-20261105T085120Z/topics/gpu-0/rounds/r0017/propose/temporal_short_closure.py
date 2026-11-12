"""Fixed-clock freshness with a two-moment distributional replacement.

Four-arm scale x source factorial, retaining the frozen oldest_bound model.
Actual mean decay at seven days or six hours is compared to the same decay
moment under a two-point distribution on {a,1} matching the existing one-day
mean and second moment. The closure is a control, not an optimal estimator.
All four models compute both actual features and both closures. One appended
zero-initialized projection preserves the inherited initial logits and RNG
prefix. This represents only the supplied causal history, not older events.
No local performance or novelty claim; training/scoring belong to the host.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'detailed_counts': True, 'continuous_freshness': True, 'context_mode': 'multiply', 'evidence_mode': 'detailed', 'gate_normalization': 'rms', 'evidence_partition': 'exclusive', 'moment_mode': 'second', 'extreme_direction': 'oldest', 'extreme_source': 'bound', 'temporal_scale': 'short', 'temporal_source': 'closure'}


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
        self.temporal_scale = self.config['temporal_scale']
        self.temporal_source = self.config['temporal_source']
        if self.temporal_scale not in ('long', 'short'):
            raise ValueError('Unknown temporal_scale')
        if self.temporal_source not in ('observed', 'closure'):
            raise ValueError('Unknown temporal_source')
        if self.extreme_direction != 'oldest' or self.extreme_source != 'bound':
            raise ValueError('Retain the frozen oldest_bound branch.')
        # Append after every inherited allocation; matched constructor RNG path.
        self.temporal_projection = nn.Linear(16, 32, bias=False)
        nn.init.zeros_(self.temporal_projection.weight)



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

    def _temporal_features(self, batch, cells, mean, second):
        """Same cell layout as inherited evidence: match-major/action-minor.

        x=exp(-age/86400). New powers q=1/7 and q=4 correspond to seven
        days and six hours. Actual averages use ages directly to avoid
        exponentiating an already rounded/underflowed one-day decay.

        For feasible m=E[x], s=E[x*x], a=(m-s)/(1-m), w=(m-a)/(1-a),
        a distribution with masses w at 1 and (1-w) at a matches m,s.
        The degenerate m=1 case uses a=1,w=0. Feasibility repair and
        closure arithmetic use float64 on just [B,16] tensors; incumbent
        statistics and its bound are unchanged. This is one arbitrary
        admissible distribution, not a reconstruction of observed ages.
        """
        n = cells.flatten(1)
        present = n.gt(0).to(mean.dtype)
        mask = batch['history_mask']
        if mask.shape[1] == 0:
            long_actual = torch.zeros_like(mean)
            short_actual = torch.zeros_like(mean)
        else:
            membership = self._membership(batch)[0]
            actions = torch.stack([
                batch['history_type'].eq(action) for action in range(1, 5)
            ], dim=-1).float()
            ages = torch.where(mask, batch['history_age_seconds'],
                               torch.zeros_like(batch['history_age_seconds'])).clamp_min(0.0)
            long_decay = torch.exp(-ages / 604800.0)
            short_decay = torch.exp(-ages / 21600.0)
            long_actual = (torch.bmm(membership, actions * long_decay.unsqueeze(-1)) /
                           cells.clamp_min(1.0)).flatten(1)
            short_actual = (torch.bmm(membership, actions * short_decay.unsqueeze(-1)) /
                            cells.clamp_min(1.0)).flatten(1)
        m = mean.double().clamp(0.0, 1.0)
        s = torch.maximum(torch.minimum(second.double(), m), m.square())
        a = ((m - s) / (1.0 - m).clamp_min(1e-12)).clamp_min(0.0)
        a = torch.minimum(a, m)
        a = torch.where(m.eq(1.0), torch.ones_like(a), a)
        weight = ((m - a) / (1.0 - a).clamp_min(1e-12)).clamp(0.0, 1.0)
        long_closure = (weight + (1.0 - weight) * a.pow(1.0 / 7.0)).to(mean.dtype) * present
        short_closure = (weight + (1.0 - weight) * a.pow(4.0)).to(mean.dtype) * present
        use_long = float(self.temporal_scale == 'long')
        actual = use_long * long_actual + (1.0 - use_long) * short_actual
        closure = use_long * long_closure + (1.0 - use_long) * short_closure
        use_actual = float(self.temporal_source == 'observed')
        selected = use_actual * actual + (1.0 - use_actual) * closure
        return (selected, long_actual, long_closure, short_actual,
                short_closure, a, weight, m, s)

    def _correction_inputs(self, batch):
        counts, mean, cells, second = self._encoded_evidence(batch)
        # Preserve all four incumbent blocks; append only the selected clock.
        extreme = self._extreme_features(batch, cells, mean, second)[0]
        temporal = self._temporal_features(batch, cells, mean, second)[0]
        inputs = torch.cat([counts, mean, second, extreme, temporal], -1)
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
                         self.extreme_projection(inputs[:, 48:64]) +
                         self.temporal_projection(inputs[:, 64:80]))
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
        # Label-free, candidate-supplied measurements on the host's first 512
        # validation impressions. Sensitivity is not predictive attribution.
        logits, base, correction = self._logits(batch)
        _, pools, totals, stats, _, _, sizes, _ = self._represent(batch)
        context = torch.cat([pools.flatten(1), stats], -1)
        inputs, _, mean, cells, presence = self._correction_inputs(batch)
        second = inputs[:, 32:48]
        (selected, long_actual, long_closure, short_actual,
         short_closure, a, weight, m, s) = self._temporal_features(batch, cells, mean, second)
        n = cells.flatten(1)
        occupied = presence.sum().clamp_min(1.0)
        singleton = n.eq(1).float()
        use_long = float(self.temporal_scale == 'long')
        actual = use_long * long_actual + (1.0 - use_long) * short_actual
        closure = use_long * long_closure + (1.0 - use_long) * short_closure
        scalar = (self.correction[0](inputs[:, :32]) +
                  self.moment_projection(inputs[:, 32:48]) +
                  self.extreme_projection(inputs[:, 48:64]))
        projected = self.temporal_projection(selected)
        gate = self._gate(context)[0]
        reconstructed = self.correction[2](self.correction[1](scalar + projected) * (1.0 + gate)).squeeze(-1)
        without = self.correction[2](self.correction[1](scalar) * (1.0 + gate)).squeeze(-1)
        actual_inputs = torch.cat([inputs[:, :64], actual], -1)
        closure_inputs = torch.cat([inputs[:, :64], closure], -1)
        actual_correction = self._readout(actual_inputs, context)[0]
        closure_correction = self._readout(closure_inputs, context)[0]
        padded, reverse, zero_length = dict(batch), dict(batch), dict(batch)
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
        _, zm, zc, zs = self._encoded_evidence(zero_age)
        zfeatures = self._temporal_features(zero_age, zc, zm, zs)[:5]
        zero_presence = zc.gt(0).float().flatten(1)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'temporal_projection_parameters': float(self.temporal_projection.weight.numel()),
            'temporal_projection_weight_rms': self.temporal_projection.weight.square().mean().sqrt(),
            'long_clock_selected': use_long,
            'observed_source_selected': float(self.temporal_source == 'observed'),
            'logits_finite_fraction': torch.isfinite(logits).float().mean(),
            'empty_logits_finite_fraction': torch.isfinite(empty_logits).float().mean(),
            'empty_history_fraction': totals[0].eq(0).float().mean(),
            'visible_count_mean': totals[0].mean(),
            'visible_at_limit_fraction': totals[0].eq(128).float().mean(),
            'occupied_cells_mean': presence.sum(-1).mean(),
            'singleton_occupied_fraction': singleton.sum() / occupied,
            'cell_count_alignment_error_max': (cells.sum(-1) - sizes).abs().max(),
            'selected_feature_min': selected.min(),
            'selected_feature_max': selected.max(),
            'long_actual_closure_gap_rms': (long_actual - long_closure).square().mean().sqrt(),
            'short_actual_closure_gap_rms': (short_actual - short_closure).square().mean().sqrt(),
            'selected_actual_closure_gap_rms': (actual - closure).square().mean().sqrt(),
            'long_gap_gt1e3_occupied_fraction': ((long_actual - long_closure).abs().gt(1e-3).float() * presence).sum() / occupied,
            'short_gap_gt1e3_occupied_fraction': ((short_actual - short_closure).abs().gt(1e-3).float() * presence).sum() / occupied,
            'singleton_long_gap_max': ((long_actual - long_closure) * singleton).abs().max(),
            'singleton_short_gap_max': ((short_actual - short_closure) * singleton).abs().max(),
            'long_actual_below_mean_violation_max': (mean - long_actual).clamp_min(0.0).max(),
            'short_actual_above_second_violation_max': (short_actual - second).clamp_min(0.0).max(),
            'long_closure_below_mean_violation_max': (mean - long_closure).clamp_min(0.0).max(),
            'short_closure_above_second_violation_max': (short_closure - second).clamp_min(0.0).max(),
            'closure_first_moment_error_max': (weight + (1.0 - weight) * a - m).abs().max(),
            'closure_second_moment_error_max': (weight + (1.0 - weight) * a.square() - s).abs().max(),
            'closure_mean_repair_max': (m - mean.double()).abs().max(),
            'closure_second_repair_max': (s - second.double()).abs().max(),
            'closure_support_min': a.min(),
            'closure_support_max': a.max(),
            'closure_weight_min': weight.min(),
            'closure_weight_max': weight.max(),
            'selected_actual_error_max': (selected - actual).abs().max(),
            'selected_closure_error_max': (selected - closure).abs().max(),
            'long_actual_occupied_mean': (long_actual * presence).sum() / occupied,
            'long_closure_occupied_mean': (long_closure * presence).sum() / occupied,
            'short_actual_occupied_mean': (short_actual * presence).sum() / occupied,
            'short_closure_occupied_mean': (short_closure * presence).sum() / occupied,
            'empty_cell_temporal_error_max': torch.stack([(v * (1.0 - presence)).abs().max() for v in (selected, long_actual, long_closure, short_actual, short_closure)]).amax(),
            'zero_age_temporal_presence_error_max': torch.stack([(v - zero_presence).abs().max() for v in zfeatures]).amax(),
            'padding_logit_error_max': (logits - self.forward(padded)).abs().max(),
            'reverse_logit_error_max': (logits - self.forward(reverse)).abs().max(),
            'empty_inputs_error_max': empty_inputs.abs().max(),
            'empty_correction_error_max': empty_correction.abs().max(),
            'zero_length_empty_error_max': (self.forward(zero_length) - empty_logits).abs().max(),
            'readout_reconstruction_error_max': (reconstructed - correction).abs().max(),
            'projected_temporal_rms': projected.square().mean().sqrt(),
            'temporal_head_removal_effect_rms': (correction - without).square().mean().sqrt(),
            'actual_closure_head_effect_rms': (actual_correction - closure_correction).square().mean().sqrt(),
            'incumbent_lower_bound_zero_occupied_fraction': (inputs[:, 48:64].le(0).float() * presence).sum() / occupied,
            'base_logit_rms': base.square().mean().sqrt(),
            'correction_rms': correction.square().mean().sqrt(),
            'additivity_error_max': (logits - base - correction).abs().max(),
        }
        return {key: float(value) for key, value in result.items()}
