"""Prospective evidence-basis x freshness factorial.

The inherited backbone uses exclusive match pools but its correction uses
cumulative all/category/brand/joint cells. Replace only the evidence basis
with neither/category-only/brand-only/both cells, crossed with continuous
one-day freshness versus cell presence. Raw counts and decayed sums in the
two bases are invertible linear transforms; this adds no events or information
in exact arithmetic. A gain would support this finite-width reparameterization,
not discovery of new information or a priority claim. All cells allocate the
same inherited modules and calculate both bases before selection. Only host
execution can establish numerical behavior, predictive utility, or cost.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'detailed_counts': True, 'continuous_freshness': True, 'context_mode': 'multiply', 'evidence_mode': 'detailed', 'gate_normalization': 'rms', 'evidence_partition': 'cumulative'}


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
        return cumulative_counts, cumulative_decay, exclusive_counts, exclusive_decay

    def _evidence(self, batch):
        nc, dc, ne, de = self._evidence_bases(batch)
        use = float(self.evidence_partition == 'exclusive')
        counts = use * ne + (1.0 - use) * nc
        decayed = use * de + (1.0 - use) * dc
        count_block = torch.log1p(counts) / math.log(129.0)
        freshness_block = decayed / counts.clamp_min(1.0)
        return count_block.flatten(1), freshness_block.flatten(1), counts

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
        counts, freshness, cells = self._evidence(batch)
        presence = cells.gt(0).float().flatten(1)
        # Replacement retains exact cell occupancy. It equals log-count at n=1.
        presence_count = torch.log1p(presence) / math.log(129.0)
        c = float(self.detailed_counts)
        t = float(self.continuous_freshness)
        selected_counts = c * counts + (1.0 - c) * presence_count
        selected_freshness = t * freshness + (1.0 - t) * presence
        inputs = torch.cat([selected_counts, selected_freshness], -1)
        return inputs, counts, freshness, cells, presence

    def _selected_inputs(self, batch, statistics):
        detailed = self._correction_inputs(batch)[0]
        # [N,C,B,J] repeated eight times; all information is already in the
        # unchanged backbone. Same width/range, lower information rank.
        redundant = statistics.repeat(1, 8)
        use = float(self.evidence_mode == 'detailed')
        return use * detailed + (1.0 - use) * redundant

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
        hidden = self.correction[1](self.correction[0](inputs))
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
        # Label-free checks on the host's frozen first-512 validation batch.
        # These verify implementation identities, not predictive attribution.
        logits, base, correction = self._logits(batch)
        static, pools, totals, statistics, membership, _, sizes, global_mean = self._represent(batch)
        context = torch.cat([pools.flatten(1), statistics], dim=-1)
        inputs, cb, fb, cells, presence = self._correction_inputs(batch)
        nc, dc, ne, de = self._evidence_bases(batch)
        n, c, b, j, _ = totals
        expected_cumulative = torch.stack([n, c, b, j], dim=-1)
        expected_exclusive = torch.stack([n - c - b + j, c - j, b - j, j], dim=-1)
        cumulative_inputs = torch.cat([
            (torch.log1p(nc) / math.log(129.0)).flatten(1),
            (dc / nc.clamp_min(1.0)).flatten(1)
        ], dim=-1)
        exclusive_inputs = torch.cat([
            (torch.log1p(ne) / math.log(129.0)).flatten(1),
            (de / ne.clamp_min(1.0)).flatten(1)
        ], dim=-1)
        padded = dict(batch)
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[name] = torch.where(batch['history_mask'], batch[name], torch.zeros_like(batch[name]))
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        empty_logits, _, empty_correction = self._logits(empty)
        empty_inputs = self._correction_inputs(empty)[0]
        zero_length = dict(batch)
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero_length[name] = batch[name][:, :0]
        zero_age = dict(batch)
        zero_age['history_age_seconds'] = torch.zeros_like(batch['history_age_seconds'])
        zero_cb, zero_fb, _ = self._evidence(zero_age)
        reversed_batch = dict(batch)
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reversed_batch[name] = batch[name].flip(1)
        selected = self._selected_inputs(batch, statistics)
        current, hidden, modulation = self._readout(selected, context)
        no_context = self._readout(selected, torch.zeros_like(context))[0]
        no_evidence = self._readout(torch.zeros_like(selected), context)[0]
        neither = self._readout(torch.zeros_like(selected), torch.zeros_like(context))[0]
        interaction = current - no_context - no_evidence + neither
        gate, raw, normalized, scale = self._gate(context)
        fresh_slot = inputs[:, 16:]
        weighted = (pools * sizes.unsqueeze(-1)).sum(1) / n.clamp_min(1).unsqueeze(-1)
        # Count and mean freshness encode decayed sums jointly, subject to
        # finite precision and exponential underflow. Check the reconstruction.
        recovered_counts = torch.expm1(cb.reshape_as(cells) * math.log(129.0))
        recovered_decay = recovered_counts * fb.reshape_as(cells)
        use = float(self.evidence_partition == 'exclusive')
        selected_decay = use * de + (1.0 - use) * dc
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'correction_parameters': float(sum(p.numel() for p in self.correction.parameters()) + sum(p.numel() for p in self.context_projection.parameters())),
            'exclusive_evidence_enabled': use,
            'continuous_freshness_enabled': float(self.continuous_freshness),
            'empty_history_fraction': (n == 0).float().mean(),
            'visible_count_mean': n.mean(),
            'visible_count_at_limit_fraction': (n == 128).float().mean(),
            'occupied_cells_mean': presence.sum(-1).mean(),
            'exclusive_occupied_cells_mean': ne.gt(0).float().sum((1, 2)).mean(),
            'cumulative_occupied_cells_mean': nc.gt(0).float().sum((1, 2)).mean(),
            'exclusive_multi_event_cell_fraction': (ne > 1).float().mean(),
            'cumulative_multi_event_cell_fraction': (nc > 1).float().mean(),
            'partition_coverage_error_max': (membership.sum(1) - batch['history_mask'].float()).abs().max(),
            'cumulative_action_count_identity_error_max': (nc.sum(-1) - expected_cumulative).abs().max(),
            'exclusive_action_count_identity_error_max': (ne.sum(-1) - expected_exclusive).abs().max(),
            'exclusive_pool_alignment_error_max': (ne.sum(-1) - sizes).abs().max(),
            'count_to_cumulative_error_max': (self._to_cumulative(ne) - nc).abs().max(),
            'count_to_exclusive_error_max': (self._to_exclusive(nc) - ne).abs().max(),
            'decay_to_cumulative_error_max': (self._to_cumulative(de) - dc).abs().max(),
            'decay_to_exclusive_error_max': (self._to_exclusive(dc) - de).abs().max(),
            'encoded_count_reconstruction_error_max': (recovered_counts - cells).abs().max(),
            'encoded_decay_reconstruction_error_max': (recovered_decay - selected_decay).abs().max(),
            'global_reconstruction_abs_error_max': (weighted - global_mean).abs().max(),
            'basis_input_difference_rms': (exclusive_inputs - cumulative_inputs).square().mean().sqrt(),
            'count_block_min': cb.min(),
            'count_block_max': cb.max(),
            'freshness_min': fb.min(),
            'freshness_max': fb.max(),
            'freshness_occupied_mean': fb.sum() / presence.sum().clamp_min(1),
            'freshness_occupied_zero_fraction': ((fb == 0).float() * presence).sum() / presence.sum().clamp_min(1),
            'empty_cell_freshness_abs_max': (fb * (1.0 - presence)).abs().max(),
            'fresh_slot_freshness_error_max': (fresh_slot - fb).abs().max(),
            'fresh_slot_presence_error_max': (fresh_slot - presence).abs().max(),
            'zero_age_freshness_presence_error_max': (zero_fb - presence).abs().max(),
            'zero_age_count_invariance_error_max': (zero_cb - cb).abs().max(),
            'padding_logit_abs_error_max': (logits - self.forward(padded)).abs().max(),
            'event_reverse_logit_abs_error_max': (logits - self.forward(reversed_batch)).abs().max(),
            'all_empty_correction_inputs_abs_max': empty_inputs.abs().max(),
            'all_empty_correction_abs_max': empty_correction.abs().max(),
            'zero_length_empty_logit_error_max': (self.forward(zero_length) - empty_logits).abs().max(),
            'all_empty_logits_finite_fraction': torch.isfinite(empty_logits).float().mean(),
            'logits_finite_fraction': torch.isfinite(logits).float().mean(),
            'base_logit_rms': base.square().mean().sqrt(),
            'correction_mean': correction.mean(),
            'correction_rms': correction.square().mean().sqrt(),
            'additivity_error_max': (logits - base - correction).abs().max(),
            'selected_input_rms': selected.square().mean().sqrt(),
            'hidden_rms': hidden.square().mean().sqrt(),
            'context_rms': context.square().mean().sqrt(),
            'gate_raw_preactivation_rms': raw.square().mean().sqrt(),
            'gate_normalized_preactivation_rms': normalized.square().mean().sqrt(),
            'gate_scale_min': scale.min(),
            'gate_scale_max': scale.max(),
            'gate_selected_saturated_fraction': (gate.abs() > 0.95).float().mean(),
            'head_interaction_rms': interaction.square().mean().sqrt(),
            'zero_evidence_correction_rms': no_evidence.square().mean().sqrt(),
        }
        return {key: float(value) for key, value in result.items()}
