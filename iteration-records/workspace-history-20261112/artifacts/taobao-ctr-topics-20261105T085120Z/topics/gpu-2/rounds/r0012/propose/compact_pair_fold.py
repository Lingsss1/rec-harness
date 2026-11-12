"""Compact raw-summary replacement of frozen joint_identity.

Inherited scalar formulas and sorted identity tables are copied verbatim.
History tensors never enter learned embedding lookups in this model. Capacity
and conditional pair-folding controls are declared in CONFIG. No I/O, optimizer,
loss change or training is performed here. Host execution remains pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'capacity_mode': 'matched', 'pairing': 'fold'}

def _fold_brand(model, batch):
    """Partial deterministic pairing disruption, conditional on true match.

    Permute complete-support brands in input-rank order: even sources followed
    by odd sources. Every source is used once. Category, action, age and masks
    stay fixed; missing brands and all ineligible positions stay fixed too.
    This is not an independent draw and may introduce rank artifacts.
    """
    c, b, mask = batch['history_category'], batch['history_brand'], batch['history_mask']
    length = mask.shape[1]
    if length == 0:
        return dict(batch)
    semantic = model._membership(batch)[1]
    eligible = mask & (c > 0) & (b > 0)
    positions = torch.arange(length, device=b.device)[None, :].expand(b.shape[0], -1)
    changed = b
    for g in range(4):
        member = semantic[:, g, :].bool() & eligible
        count = member.long().sum(-1, keepdim=True)
        ordered = torch.where(member, positions, positions + length).argsort(dim=-1)
        rank = (member.long().cumsum(-1) - 1).clamp_min(0)
        even_count = (count + 1) // 2
        source_rank = torch.where(rank < even_count, 2 * rank,
                                  2 * (rank - even_count) + 1).clamp(0, length - 1)
        source_position = ordered.gather(1, source_rank)
        changed = torch.where(member, b.gather(1, source_position), changed)
    result = dict(batch)
    result['history_brand'] = changed
    return result


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.category, self.brand = meta['category_field_index'], meta['brand_field_index']
        dim = self.config['embedding_dim']
        self.capacity_mode, self.pairing = self.config['capacity_mode'], self.config['pairing']
        if self.capacity_mode not in ('narrow', 'matched') or self.pairing not in ('intact', 'fold'):
            raise ValueError('Unknown capacity/pairing control')
        if self.config['hidden'] != [128, 64] or dim != 16:
            raise ValueError('This study freezes the reference dimensions')
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']])
        self.category_vocab = int(meta['buckets'][self.category])
        self.brand_vocab = int(meta['buckets'][self.brand])
        if self.category_vocab * self.brand_vocab > (2**63-1)//4:
            raise ValueError('Packed identity key would overflow int64')
        self.joint_mode, self.statistic_mode, self.grouping_mode = 'observed', 'count', 'match'
        self.histogram_grouping, self.histogram_pairing, self.identity_mode = 'match', 'observed', 'joint'
        static_width = len(meta['fields']) * dim
        input_width = static_width + 68
        parent_width = static_width + 11 * dim + 68
        parent_output = 128 * (parent_width + 1) + 64 * 129 + 65
        self.reference_active_nonstatic = parent_output + 38 * dim + 464
        # Compact output parameter count is h*(input_width+65)+129.
        # Match also the reference's active action/recency parameters. No
        # dormant tables, attention, padding weights or dummy parameters.
        numerator = parent_output + 38 * dim - 129
        self.first_hidden = (numerator + (input_width+65)//2)//(input_width+65) if self.capacity_mode == 'matched' else 128
        self.output = nn.Sequential(nn.Linear(input_width, self.first_hidden), nn.ReLU(),
                                    nn.Linear(self.first_hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.identity_head = nn.Sequential(nn.Linear(28, 16, bias=False), nn.ReLU(),
                                           nn.Linear(16, 1, bias=False))
        nn.init.zeros_(self.identity_head[-1].weight)

    def _nonidentity_features(self, batch):
        static = torch.cat([embedding(batch['static'][:, i]) for i, embedding in enumerate(self.embeddings)], dim=-1)
        _, semantic, sizes = self._membership(batch)
        statistics = self._statistics(self._match_counts(batch))
        histogram = self._histogram(batch, semantic, sizes)[0]
        return torch.cat([static, statistics, histogram], dim=-1)

    def forward(self, batch):
        effective = self._pair_batch(batch)
        base = self.output(self._nonidentity_features(effective)).squeeze(-1)
        tables = self._identity_tables(effective)
        return base + self.identity_head(self._identity_features(tables)).squeeze(-1)

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

    @staticmethod
    def _sorted_distribution(identity, group, eligible, n):
        """Conditional Shannon entropy and with-replacement collision mass.

        Sort BxL packed identity/group keys, then aggregate run counts in BxL
        storage. This never builds a BxLxL training tensor or a vocabulary table.
        Zero category/brand values are excluded from ALL three distributions.
        """
        if identity.shape[1] == 0:
            return torch.zeros_like(n), torch.zeros_like(n)
        key = torch.where(eligible, identity * 4 + group,
                          torch.full_like(identity, -1))
        ordered = key.sort(dim=-1).values
        valid = ordered >= 0
        start = valid.clone()
        start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
        run = (start.long().cumsum(-1) - 1).clamp_min(0)
        mass = n.new_zeros(identity.shape).scatter_add(1, run, valid.float())
        run_group = torch.zeros_like(identity).scatter_add(
            1, run, torch.where(start, ordered.remainder(4), torch.zeros_like(ordered)))
        sum_square = torch.zeros_like(n).scatter_add(1, run_group, mass.square())
        sum_xlogx = torch.zeros_like(n).scatter_add(
            1, run_group, mass * mass.clamp_min(1).log())
        denom = n.clamp_min(1)
        entropy = torch.where(n > 0, (denom.log() - sum_xlogx / denom).clamp_min(0),
                              torch.zeros_like(n))
        collision = sum_square / denom.square()
        return entropy, collision

    def _identity_tables(self, batch, semantic=None):
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        # Complete-pair support shared by category, brand and pair summaries.
        # Missing/missing is never interpreted as a repeated product identity.
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        membership = semantic * eligible[:, None, :].float()
        n = membership.sum(-1)
        hc, cc = self._sorted_distribution(c, group, eligible, n)
        hb, cb = self._sorted_distribution(b, group, eligible, n)
        hj, cj = self._sorted_distribution(c * self.brand_vocab + b, group, eligible, n)
        # Analytic product of the two empirical marginals on the SAME support.
        # Self combinations are included. No random shuffle or derangement.
        hi, ci = hc + hb, cc * cb
        scale = math.log(129.0)  # fixed supplied history limit, not fitted
        common = torch.stack([torch.log1p(n) / scale, hc / scale, hb / scale, cc, cb], -1)
        joint = torch.stack([hj / scale, cj], -1)
        independent = torch.stack([hi / scale, ci], -1)
        return common, joint, independent, n, (hc, cc, hb, cb, hj, cj), eligible, group

    def _identity_features(self, tables, mode=None):
        mode = self.identity_mode if mode is None else mode
        common, joint, independent = tables[:3]
        selected = joint if mode == 'joint' else independent
        if mode == 'marginal':
            selected = torch.zeros_like(joint)
        features = torch.cat([common, selected], -1).flatten(1)
        return torch.zeros_like(features) if mode == 'zero' else features

    @staticmethod
    def _dense_distribution(identity, group, eligible, n):
        """Independent quadratic reference, DIAGNOSTICS ONLY, never training."""
        if identity.shape[1] == 0:
            return torch.zeros_like(n), torch.zeros_like(n)
        same = ((identity[:, :, None] == identity[:, None, :]) &
                (group[:, :, None] == group[:, None, :]) &
                eligible[:, :, None] & eligible[:, None, :])
        per_event = same.sum(-1).float()
        membership = torch.stack([(group == g) & eligible for g in range(4)], 1).float()
        sum_square = torch.bmm(membership, per_event[:, :, None]).squeeze(-1)
        sum_xlogx = torch.bmm(membership, per_event.clamp_min(1).log()[:, :, None]).squeeze(-1)
        denom = n.clamp_min(1)
        entropy = torch.where(n > 0, (denom.log() - sum_xlogx / denom).clamp_min(0),
                              torch.zeros_like(n))
        return entropy, sum_square / denom.square()

    def _pair_batch(self, batch):
        return _fold_brand(self, batch) if self.pairing == 'fold' else batch

    def _raw_summaries(self, batch):
        _, semantic, sizes = self._membership(batch)
        stats = self._statistics(self._match_counts(batch))
        histogram = self._histogram(batch, semantic, sizes)[0]
        return torch.cat([stats, histogram], dim=-1), self._identity_tables(batch, semantic)

    def diagnostics(self, batch):
        # Candidate-supplied, first 512 validation rows only. No population
        # causal inference or timing claim is made from these values.
        raw, tables = self._raw_summaries(batch)
        folded = _fold_brand(self, batch)
        fr, ft = self._raw_summaries(folded)
        sem = self._membership(batch)[1]
        fsem = self._membership(folded)[1]
        n, eligible, group = tables[3], tables[5], tables[6]
        c, b = batch['history_category'], batch['history_brand']
        refs = [self._dense_distribution(k, group, eligible, n)
                for k in (c, b, c * self.brand_vocab + b)]
        error = torch.stack([(a - z).abs().max() for a, z in
                             zip(tables[4], [v for pair in refs for v in pair])]).max()
        selected = self._pair_batch(batch)
        selected_tables = self._identity_tables(selected)
        selected_features = self._identity_features(selected_tables)
        reverse = {k: (v.flip(1) if k.startswith('history_') else v) for k, v in batch.items()}
        reversed_tables = self._identity_tables(reverse)
        active_reverse = self._identity_features(self._identity_tables(self._pair_batch(reverse)))
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {k: (v[:, :0] if k.startswith('history_') else v) for k, v in batch.items()}
        er, et = self._raw_summaries(empty)
        zr, zt = self._raw_summaries(zero)
        padded = dict(batch)
        for k in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[k] = torch.where(batch['history_mask'], batch[k], torch.ones_like(batch[k]))
        pr, pt = self._raw_summaries(padded)
        static_count = sum(x.weight.numel() for x in self.embeddings)
        total = sum(x.numel() for x in self.parameters())
        dormant = sum(x.numel() for x in self.attention.parameters()) if self.capacity_mode == 'reference' else 0
        mask = batch['history_mask']
        fb = folded['history_brand']
        # Exact sorted complete-support multisets, not just their moments.
        key = torch.where(eligible, b * 4 + group, torch.full_like(b, -1))
        fkey = torch.where(eligible, fb * 4 + group, torch.full_like(b, -1))
        marginal_equal = (key.sort(-1).values == fkey.sort(-1).values).all()
        def max_error(x, y):
            return (x-y).abs().max()
        result = {
            'allocated_parameters': total,
            'active_parameters': total - dormant,
            'static_embedding_parameters': static_count,
            'active_nonstatic_parameters': total - dormant - static_count,
            'reference_active_nonstatic_target': self.reference_active_nonstatic,
            'active_nonstatic_parameter_gap': total - dormant - static_count - self.reference_active_nonstatic,
            'actual_first_hidden': self.first_hidden,
            'compact_enabled': float(self.capacity_mode != 'reference'),
            'fold_enabled': float(self.pairing == 'fold'),
            'raw_summary_coordinates': raw.shape[1],
            'identity_coordinates': selected_features.shape[1],
            'sorted_vs_dense_raw_error_max': error,
            'fold_brand_multiset_error': (~marginal_equal).float(),
            'fold_membership_error_max': max_error(sem, fsem),
            'fold_complete_counts_error_max': max_error(n, ft[3]),
            'fold_nonidentity_summary_error_max': max_error(raw, fr),
            'fold_nonidentity_model_features_error_max': max_error(self._nonidentity_features(batch), self._nonidentity_features(folded)),
            'fold_common_identity_error_max': max_error(tables[0], ft[0]),
            'fold_independent_identity_error_max': max_error(tables[2], ft[2]),
            'fold_joint_identity_delta_max': max_error(tables[1], ft[1]),
            'fold_action_input_error_max': max_error(batch['history_type'], folded['history_type']),
            'fold_age_input_error_max': max_error(batch['history_age_seconds'], folded['history_age_seconds']),
            'fold_static_input_error_max': max_error(batch['static'], folded['static']),
            'fold_ineligible_brand_changes': ((b != fb) & ~eligible).float().sum(),
            'fold_changed_eligible_fraction': ((b != fb) & eligible).float().sum() / eligible.float().sum().clamp_min(1),
            'fold_changed_joint_row_fraction': ((tables[1]-ft[1]).abs().flatten(1).amax(-1) > 1e-6).float().mean(),
            'raw_sequence_reversal_identity_error_max': max_error(self._identity_features(tables), self._identity_features(reversed_tables)),
            'active_fold_sequence_reversal_identity_delta': max_error(selected_features, active_reverse),
            'padding_summary_error_max': max_error(raw, pr),
            'padding_identity_error_max': max_error(self._identity_features(tables), self._identity_features(pt)),
            'all_empty_summary_abs_max': er.abs().max(),
            'all_empty_identity_abs_max': self._identity_features(et).abs().max(),
            'zero_length_summary_abs_max': zr.abs().max(),
            'zero_length_identity_abs_max': self._identity_features(zt).abs().max(),
            'all_empty_identity_head_abs_max': self.identity_head(self._identity_features(et)).abs().max(),
            'visible_count_mean': mask.float().sum(-1).mean(),
            'complete_pair_count_mean': n.sum(-1).mean(),
            'empty_history_fraction': (~mask.any(-1)).float().mean(),
            'visible_count_at_limit_fraction': (mask.sum(-1) == 128).float().mean(),
            'same_checkpoint_identity_residual_rms': self.identity_head(selected_features).square().mean().sqrt(),
            'same_checkpoint_fold_residual_delta_rms': (self.identity_head(self._identity_features(tables))-self.identity_head(self._identity_features(ft))).square().mean().sqrt(),
            'identity_head_last_weight_norm': torch.linalg.vector_norm(self.identity_head[-1].weight),
        }
        for g, name in enumerate(('neither', 'category_only', 'brand_only', 'both')):
            result[name+'_complete_count_mean'] = n[:, g].mean()
            result[name+'_at_least_three_fraction'] = (n[:, g] >= 3).float().mean()
        # Pair collision witness; unlike reversal, folding mixes this coupling.
        if self.category_vocab > 3 and self.brand_vocab > 3:
            device = batch['static'].device
            witness = {'static': batch['static'][:1].clone(),
                       'history_category': torch.tensor([[1,1,2,2]], device=device),
                       'history_brand': torch.tensor([[1,1,2,2]], device=device),
                       'history_type': torch.ones((1,4), dtype=torch.long, device=device),
                       'history_age_seconds': torch.zeros((1,4), device=device),
                       'history_mask': torch.ones((1,4), dtype=torch.bool, device=device)}
            witness['static'][:, self.category] = 3
            witness['static'][:, self.brand] = 3
            wt = self._identity_tables(witness)
            wft = self._identity_tables(_fold_brand(self, witness))
            result['witness_applicable'] = 1.0
            result['witness_entropy_increase_nats'] = wft[4][4][0,0]-wt[4][4][0,0]
            result['witness_collision_decrease'] = wt[4][5][0,0]-wft[4][5][0,0]
            result['witness_common_error_max'] = max_error(wt[0], wft[0])
        else:
            result['witness_applicable'] = 0.0
        return {k: float(v) for k, v in result.items()}
