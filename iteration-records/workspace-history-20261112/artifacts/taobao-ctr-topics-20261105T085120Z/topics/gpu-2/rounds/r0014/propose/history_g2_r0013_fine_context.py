"""First-order history restoration using counts before learned context projection.

Four cells cross shared category/brand mean restoration with fine/coarse age.
Fine context equals the reference action+log2-age mean by histogram algebra;
coarse context re-encodes existing action/coarse-age counts with a fixed map.
No independent content-context moments, global norm readout, attention, I/O,
optimizer changes, loss hooks, or data selection. Host execution is pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'content_mode': 'off', 'recency_mode': 'fine'}

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
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        if dim != 16 or self.config['hidden'] != [128, 64]:
            raise ValueError('Frozen reference dimensions required')
        self.content_mode = self.config['content_mode']
        self.recency_mode = self.config['recency_mode']
        if self.content_mode not in ('pooled', 'off') or self.recency_mode not in ('fine', 'coarse'):
            raise ValueError('Unknown factorial cell')
        # Same allocation order and parameter shapes in every factorial cell.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.category_vocab = int(meta['buckets'][self.category])
        self.brand_vocab = int(meta['buckets'][self.brand])
        if self.category_vocab * self.brand_vocab > (2**63 - 1)//4:
            raise ValueError('Packed identity key would overflow int64')
        self.joint_mode, self.statistic_mode, self.grouping_mode = 'observed', 'count', 'match'
        self.histogram_grouping, self.histogram_pairing, self.identity_mode = 'match', 'observed', 'joint'
        static_width = len(meta['fields']) * dim
        self.input_width = static_width + 4 * dim + 68
        parent_width = static_width + 11 * dim + 68
        parent_output = 128 * (parent_width + 1) + 64 * 129 + 65
        self.reference_active_nonstatic = parent_output + 38 * dim + 464
        # Output has h*(input_width+65)+129 parameters. Count-match the
        # parent INCLUDING active action/recency and identity-head parameters.
        # At F=18 this gives h=158 and +214 active nonstatic parameters (0.276%).
        fixed = 38 * dim + 464 + 129
        denominator = self.input_width + 65
        self.first_hidden = max(1, (self.reference_active_nonstatic - fixed + denominator//2)//denominator)
        self.output = nn.Sequential(
            nn.Linear(self.input_width, self.first_hidden), nn.ReLU(),
            nn.Linear(self.first_hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.identity_head = nn.Sequential(
            nn.Linear(28, 16, bias=False), nn.ReLU(), nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.identity_head[-1].weight)
        # A frozen, data-independent map. For each original coarse interval,
        # assign equal mass to each log2 bin intersecting that interval. This
        # is a coarse-count re-encoding, NOT a fitted estimate of fine ages.
        # Bin 32 includes all larger ages because the reference clamps at 32.
        edges = (0.0, 3600.0, 86400.0, 604800.0, float('inf'))
        mixture = torch.zeros(4, 33)
        for coarse in range(4):
            for fine in range(1, 33):
                low = float(2**(fine - 1) - 1)
                high = float(2**fine - 1) if fine < 32 else float('inf')
                if low < edges[coarse + 1] and high > edges[coarse]:
                    mixture[coarse, fine] = 1.0
        mixture = mixture / mixture.sum(-1, keepdim=True)
        self.register_buffer('coarse_to_fine', mixture)

    def _context_tables(self, batch, semantic, observed):
        # No per-event learned context embeddings enter forward. All counts
        # are sample-local and use exactly the supplied visible history.
        sizes = semantic.sum(-1)
        action_counts = observed.sum(-1)
        # observed is [B,group,action,coarse_age].
        coarse_counts = observed.sum(-2)
        fine_counts = semantic.new_zeros((semantic.shape[0], 4, 33))
        if batch['history_mask'].shape[1] != 0:
            fine = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            fine_counts = fine_counts.scatter_add(2, fine[:, None, :].expand(-1, 4, -1), semantic)
        reconstructed = torch.matmul(coarse_counts, self.coarse_to_fine)
        return fine_counts, reconstructed, action_counts, sizes

    def _context_mean(self, tables, mode=None):
        mode = self.recency_mode if mode is None else mode
        fine, reconstructed, actions, sizes = tables
        selected = fine if mode == 'fine' else reconstructed
        return (torch.matmul(actions, self.action.weight[1:5]) +
                torch.matmul(selected, self.recency.weight)) / sizes.clamp_min(1).unsqueeze(-1)

    def _content_mean(self, batch, semantic, sizes):
        if batch['history_mask'].shape[1] == 0:
            return semantic.new_zeros((semantic.shape[0], 4, self.config['embedding_dim']))
        content = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']))
        return torch.bmm(semantic, content) / sizes.clamp_min(1).unsqueeze(-1)

    def _features(self, batch):
        static = torch.cat([e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)], -1)
        _, semantic, sizes = self._membership(batch)
        statistics = self._statistics(self._match_counts(batch))
        histogram, observed = self._histogram(batch, semantic, sizes)[:2]
        context = self._context_mean(self._context_tables(batch, semantic, observed))
        # Both levels retain active context and all first-layer input columns.
        # Shared static embeddings remain active when history content is off.
        if self.content_mode == 'pooled':
            context = context + self._content_mean(batch, semantic, sizes)
        return torch.cat([static, context.flatten(1), statistics, histogram], -1)

    def forward(self, batch):
        base = self.output(self._features(batch)).squeeze(-1)
        identity = self._identity_features(self._identity_tables(batch))
        return base + self.identity_head(identity).squeeze(-1)

    def _direct_context(self, batch, semantic, sizes):
        # Diagnostic reference only; this event-level path is never trained.
        if batch['history_mask'].shape[1] == 0:
            return semantic.new_zeros((semantic.shape[0], 4, self.config['embedding_dim']))
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        events = self.action(batch['history_type']) + self.recency(ages)
        return torch.bmm(semantic, events) / sizes.clamp_min(1).unsqueeze(-1)

    def diagnostics(self, batch):
        # At most 64 finite scalars, first 512 validation rows only. Source
        # provenance/invariance checks do not measure population mechanisms.
        _, semantic, sizes = self._membership(batch)
        hist, observed = self._histogram(batch, semantic, sizes)[:2]
        tables = self._context_tables(batch, semantic, observed)
        fine_mean = self._context_mean(tables, 'fine')
        coarse_mean = self._context_mean(tables, 'coarse')
        direct = self._direct_context(batch, semantic, sizes)
        content = self._content_mean(batch, semantic, sizes)
        identity = self._identity_tables(batch, semantic)
        c, b = batch['history_category'], batch['history_brand']
        dense = [self._dense_distribution(key, identity[6], identity[5], identity[3])
                 for key in (c, b, c * self.brand_vocab + b)]
        dense_error = torch.stack([(x-y).abs().max() for x, y in
                                  zip(identity[4], [v for pair in dense for v in pair])]).max()
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'], batch[key], torch.ones_like(batch[key]))
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {k: (v[:, :0] if k.startswith('history_') else v) for k, v in batch.items()}
        reverse = {k: (v.flip(1) if k.startswith('history_') else v) for k, v in batch.items()}
        folded = _fold_brand(self, batch)
        _, fs, fn = self._membership(folded)
        ft = self._identity_tables(folded, fs)
        features = self._features(batch)
        static_width = len(self.embeddings) * self.config['embedding_dim']
        total = sum(p.numel() for p in self.parameters())
        static_count = sum(e.weight.numel() for e in self.embeddings)
        def error(x, y):
            return (x-y).abs().max()
        result = {
            'allocated_parameters': total,
            'active_parameters': total,
            'static_embedding_parameters': static_count,
            'active_nonstatic_parameters': total-static_count,
            'reference_active_nonstatic_target': self.reference_active_nonstatic,
            'active_nonstatic_parameter_gap': total-static_count-self.reference_active_nonstatic,
            'actual_first_hidden': self.first_hidden,
            'feature_width': self.input_width,
            'content_enabled': float(self.content_mode == 'pooled'),
            'fine_recency_enabled': float(self.recency_mode == 'fine'),
            'coarse_map_row_sum_error': error(self.coarse_to_fine.sum(-1), torch.ones_like(self.coarse_to_fine.sum(-1))),
            'fine_count_mass_error': error(tables[0].sum(-1), sizes),
            'coarse_reconstruction_mass_error': error(tables[1].sum(-1), sizes),
            'action_count_mass_error': error(tables[2].sum(-1), sizes),
            'histogram_vs_event_context_error': error(fine_mean, direct),
            'fine_coarse_context_delta_rms': (fine_mean-coarse_mean).square().mean().sqrt(),
            'content_mean_rms': content.square().mean().sqrt(),
            'context_mean_rms': self._context_mean(tables).square().mean().sqrt(),
            'sorted_vs_dense_identity_error': dense_error,
            'padding_feature_error': error(features, self._features(padded)),
            'padding_identity_error': error(self._identity_features(identity), self._identity_features(self._identity_tables(padded))),
            'sequence_reversal_feature_error': error(features, self._features(reverse)),
            'sequence_reversal_identity_error': error(self._identity_features(identity), self._identity_features(self._identity_tables(reverse))),
            'all_empty_history_feature_abs_max': self._features(empty)[:, static_width:].abs().max(),
            'zero_length_history_feature_abs_max': self._features(zero)[:, static_width:].abs().max(),
            'all_empty_identity_abs_max': self._identity_features(self._identity_tables(empty)).abs().max(),
            'zero_length_identity_abs_max': self._identity_features(self._identity_tables(zero)).abs().max(),
            'fold_membership_error': error(semantic, fs),
            'fold_content_mean_error': error(content, self._content_mean(folded, fs, fn)),
            'fold_nonidentity_features_error': error(features, self._features(folded)),
            'fold_common_identity_error': error(identity[0], ft[0]),
            'fold_joint_identity_delta': error(identity[1], ft[1]),
            'visible_count_mean': sizes.sum(-1).mean(),
            'empty_history_fraction': (~batch['history_mask'].any(-1)).float().mean(),
            'history_at_limit_fraction': (batch['history_mask'].sum(-1) == 128).float().mean(),
            'identity_residual_rms': self.identity_head(self._identity_features(identity)).square().mean().sqrt(),
        }
        # Two histories with identical original summaries, but distinct fine
        # ages within the first coarse bin, witness the new information.
        witness = {
            'static': batch['static'][:1],
            'history_category': torch.ones((1, 2), device=c.device, dtype=c.dtype),
            'history_brand': torch.ones((1, 2), device=b.device, dtype=b.dtype),
            'history_type': torch.ones((1, 2), device=b.device, dtype=b.dtype),
            'history_age_seconds': torch.tensor([[1., 1.]], device=c.device),
            'history_mask': torch.ones((1, 2), device=c.device, dtype=torch.bool),
        }
        other = dict(witness)
        other['history_age_seconds'] = torch.tensor([[1., 1024.]], device=c.device)
        def witness_parts(x):
            _, s, n = self._membership(x)
            hh, obs = self._histogram(x, s, n)[:2]
            return hh, self._context_tables(x, s, obs)
        wh, wt = witness_parts(witness)
        oh, ot = witness_parts(other)
        result['witness_original_histogram_error'] = error(wh, oh)
        result['witness_fine_count_l1_delta'] = (wt[0]-ot[0]).abs().sum()
        result['witness_coarse_reconstruction_error'] = error(wt[1], ot[1])
        for g, name in enumerate(('neither', 'category_only', 'brand_only', 'both')):
            result[name+'_count_mean'] = sizes[:, g].mean()
        return {k: float(v) for k, v in result.items()}


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
