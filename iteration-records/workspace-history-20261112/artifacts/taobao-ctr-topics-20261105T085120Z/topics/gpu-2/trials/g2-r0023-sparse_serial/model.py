"""Shared match partition and packed field-count execution.

2x2 controls preserve full native count arithmetic, all visible events,
parameter allocation and the learned readouts of g2-r0021-full_native.
Only execution sharing/packing is changed; host equality and cost are pending.
No I/O, randomness, optimizer/loss change or external input is introduced.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'content_mode': 'off', 'recency_mode': 'fine', 'identity_mode': 'independent', 'recency_localization': 'local', 'recency_weighting': 'log', 'identity_estimator': 'sketch', 'sketch_bins': 256, 'sketch_salt': 0, 'use_entropy': True, 'use_collision': True, 'summary_mode': 'local', 'summary_layout': 'sparse', 'constant_arithmetic': 'native', 'share_partition': True, 'pack_fields': False}

class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.summary_mode = self.config['summary_mode']
        if self.summary_mode not in ('local', 'global', 'pooled'):
            raise ValueError('Unknown summary mode')
        self.summary_layout = self.config['summary_layout']
        self.constant_arithmetic = self.config['constant_arithmetic']
        if self.summary_mode != 'local':
            raise ValueError('This study fixes local conditioning')
        if self.summary_layout not in ('full', 'sparse'):
            raise ValueError('Unknown summary layout')
        if self.constant_arithmetic not in ('native', 'analytic'):
            raise ValueError('Unknown constant arithmetic')
        self.use_entropy = self.config['use_entropy']
        self.use_collision = self.config['use_collision']
        if type(self.use_entropy) is not bool or type(self.use_collision) is not bool:
            raise ValueError('Statistic switches must be booleans')
        if not (self.use_entropy and self.use_collision):
            raise ValueError('Both statistic families are fixed in this study')
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
        self.histogram_grouping, self.histogram_pairing = 'match', 'observed'
        self.identity_mode = self.config['identity_mode']
        self.recency_localization = self.config['recency_localization']
        self.recency_weighting = self.config['recency_weighting']
        self.identity_estimator = self.config['identity_estimator']
        self.sketch_bins = self.config['sketch_bins']
        self.sketch_salt = self.config['sketch_salt']
        if self.identity_estimator != 'sketch':
            raise ValueError('Unknown identity estimator')
        if self.sketch_bins != 256 or self.sketch_salt != 0:
            raise ValueError('Unregistered sketch configuration')
        if self.identity_mode != 'independent':
            raise ValueError('This study fixes product-of-marginals identity')
        if self.recency_weighting != 'log':
            raise ValueError('Unknown recency weighting')
        if self.identity_mode not in ('joint', 'independent'):
            raise ValueError('Unknown identity mode')
        if self.recency_localization != 'local':
            raise ValueError('Unknown recency localization')
        if self.content_mode != 'off' or self.recency_mode != 'fine':
            raise ValueError('This study fixes content off and fine recency')
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


    def _age_ids(self, batch):
        age = batch['history_age_seconds']
        fine = (torch.log2(age + 1).long() + 1).clamp(0, 32)
        coarse = ((age >= 3600.0).long() + (age >= 86400.0).long() +
                  (age >= 604800.0).long())
        return fine, coarse


    def _fine_counts(self, batch, semantic):
        fine, _ = self._age_ids(batch)
        counts = semantic.new_zeros((semantic.shape[0], 4, 33))
        return counts.scatter_add(2, fine[:, None, :].expand(-1, 4, -1), semantic)


    def _factorized_recency(self, batch, observed):
        # Raw counts, per impression. g=target-match group, s=coarse age,
        # k=log2 age. E[g,s,k] = N[g,s] N[s,k] / N[s]. This keeps N[g,s]
        # and N[s,k], including bins that straddle a coarse-age boundary.
        # No fitted distribution, sequence truncation or foreign-user input.
        fine, coarse = self._age_ids(batch)
        mask = batch['history_mask'].float()
        global_counts = observed.new_zeros((mask.shape[0], 132))
        global_counts = global_counts.scatter_add(1, 33 * coarse + fine, mask)
        global_counts = global_counts.reshape(mask.shape[0], 4, 33)
        group_coarse = observed.sum(-2)
        coarse_mass = global_counts.sum(-1)
        mixture = group_coarse / coarse_mass.clamp_min(1)[:, None, :]
        # No [B,group,coarse,fine] forward temporary is required.
        selected = torch.bmm(mixture, global_counts)
        return selected, global_counts, group_coarse, mixture


    @staticmethod
    def _recency_weights(counts, mode):
        # Sum over the supplied bins, not events outside the visible prefix.
        # Both modes have total mass one for nonempty groups and zero for
        # empty groups. Unit mass removes a trivial added-intensity contrast.
        if mode == 'log':
            mass = torch.log1p(counts)
            return mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
        if mode == 'mean':
            return counts / counts.sum(-1, keepdim=True).clamp_min(1)
        raise ValueError('Unknown recency weighting')


    def _context_from_counts(self, counts, observed, sizes):
        if self.recency_weighting == 'mean':
            # Preserve the parent arithmetic, including its rounding order.
            return (torch.matmul(observed.sum(-1), self.action.weight[1:5]) +
                    torch.matmul(counts, self.recency.weight)) / sizes.clamp_min(1).unsqueeze(-1)
        action_mean = (torch.matmul(observed.sum(-1), self.action.weight[1:5]) /
                       sizes.clamp_min(1).unsqueeze(-1))
        recency_mean = torch.matmul(self._recency_weights(counts, 'log'), self.recency.weight)
        return action_mean + recency_mean


    def _features(self, batch):
        static = torch.cat([e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)], -1)
        _, semantic, sizes = self._membership(batch)
        statistics = self._statistics(self._match_counts(batch))
        histogram, observed = self._histogram(batch, semantic, sizes)[:2]
        if self.recency_localization == 'local':
            counts = self._fine_counts(batch, semantic)
        else:
            counts = self._factorized_recency(batch, observed)[0]
        context = self._context_from_counts(counts, observed, sizes)
        return torch.cat([static, context.flatten(1), statistics, histogram], -1)




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


    def _bucket_ids(self, identity, bins=None, salt=None):
        bins = self.sketch_bins if bins is None else bins
        salt = self.sketch_salt if salt is None else salt
        # Modulo before multiplication bounds intermediates below 2**62.
        # No fitted hash, random draws, vocabulary-sized table or label input.
        multiplier, offset = ((1103515245, 12345) if salt == 0 else (1664525111, 1013904223))
        prime = 2147483647
        return ((identity.remainder(prime) * multiplier + offset)
                .remainder(prime).remainder(bins))


    def _bucket_table(self, identity, group, eligible, n, bins=None, salt=None):
        bins = self.sketch_bins if bins is None else bins
        hashed = self._bucket_ids(identity, bins, salt)
        key = group * bins + hashed
        mass = n.new_zeros((identity.shape[0], 4 * bins))
        mass = mass.scatter_add(1, key, eligible.float())
        return mass.reshape(identity.shape[0], 4, bins), hashed


    def _bucket_distribution(self, identity, group, eligible, n, full=False):
        # Count coarsened identities once. Disabled statistics incur no mass
        # reduction or entropy logarithms. Support-only skips hashing entirely.
        zero = torch.zeros_like(n)
        need_entropy = full or self.use_entropy
        need_collision = full or self.use_collision
        if not (need_entropy or need_collision):
            return zero, zero
        mass, _ = self._bucket_table(identity, group, eligible, n)
        denom = n.clamp_min(1)
        if need_entropy:
            xlogx = (mass * mass.clamp_min(1).log()).sum(-1)
            entropy = torch.where(n > 0,
                                  (denom.log() - xlogx / denom).clamp_min(0), zero)
        else:
            entropy = zero
        collision = mass.square().sum(-1) / denom.square() if need_collision else zero
        return entropy, collision

    def _global_distribution(self, identity, eligible, total):
        """One BxK table, independent of target-match group cardinality."""
        hashed = self._bucket_ids(identity)
        mass = total.new_zeros((identity.shape[0], self.sketch_bins))
        mass = mass.scatter_add(1, hashed, eligible.float())
        denom = total.clamp_min(1)
        xlogx = (mass * mass.clamp_min(1).log()).sum(-1)
        entropy = torch.where(total > 0,
                              (denom.log() - xlogx / denom).clamp_min(0),
                              torch.zeros_like(total))
        collision = mass.square().sum(-1) / denom.square()
        return entropy, collision

    @staticmethod
    def _pack_identity(raw, n):
        # raw: Hcategory,Ccategory,Hbrand,Cbrand,Hproduct,Cproduct.
        hc, cc, hb, cb, hi, ci = raw.unbind(-1)
        scale = math.log(129.0)
        common = torch.stack([torch.log1p(n)/scale, hc/scale, hb/scale, cc, cb], -1)
        product = torch.stack([hi/scale, ci], -1)
        return common, product

    @staticmethod
    def _constant_mask(n, category):
        # Codes0,1,2,3: neither,category-only,brand-only,both.
        values = [False, True, False, True] if category else [False, False, True, True]
        return n.new_tensor(values).bool()[None, :]

    def _field_statistics(self, identity, group, eligible, n, category,
                          layout, arithmetic):
        constant = self._constant_mask(n, category)
        if layout == 'full':
            # Preserve the parent's exact full-table computation.
            h, c = self._bucket_distribution(identity, group, eligible, n)
            if arithmetic == 'native':
                return h, c
        else:
            # For category only groups0,2 vary; for brand only0,1 vary.
            # No information is merged between those two distributions.
            if category:
                own_match = group.remainder(2).bool()
                other_match = group // 2
                counts = torch.stack([n[:, 0], n[:, 2]], -1)
            else:
                own_match = (group // 2).bool()
                other_match = group.remainder(2)
                counts = torch.stack([n[:, 0], n[:, 1]], -1)
            residual = eligible & ~own_match
            hashed = self._bucket_ids(identity)
            mass = n.new_zeros((n.shape[0], 2 * self.sketch_bins))
            mass = mass.scatter_add(1, other_match * self.sketch_bins + hashed,
                                    residual.float())
            mass = mass.reshape(n.shape[0], 2, self.sketch_bins)
            denom = counts.clamp_min(1)
            xlogx = (mass * mass.clamp_min(1).log()).sum(-1)
            h = torch.where(counts > 0,
                            (denom.log() - xlogx / denom).clamp_min(0),
                            torch.zeros_like(counts))
            c = mass.square().sum(-1) / denom.square()
            if category:
                h = torch.stack([h[:, 0], h[:, 0], h[:, 1], h[:, 1]], -1)
                c = torch.stack([c[:, 0], c[:, 0], c[:, 1], c[:, 1]], -1)
            else:
                h = torch.stack([h[:, 0], h[:, 1], h[:, 0], h[:, 1]], -1)
                c = torch.stack([c[:, 0], c[:, 1], c[:, 0], c[:, 1]], -1)
        if arithmetic == 'native':
            # One occupied bin of mass n. Keep floating operations rather than
            # substituting real-arithmetic H=0. Bitwise identity with the full
            # reduction must be tested on the host: it is NOT guaranteed here.
            denom = n.clamp_min(1)
            constant_h = torch.where(n > 0,
                (denom.log() - (n * n.clamp_min(1).log()) / denom).clamp_min(0),
                torch.zeros_like(n))
            constant_c = n.square() / denom.square()
        else:
            constant_h = torch.zeros_like(n)
            constant_c = (n > 0).float()
        h = torch.where(constant, constant_h, h) * (n > 0).float()
        c = torch.where(constant, constant_c, c) * (n > 0).float()
        return h, c

    def _identity_tables(self, batch, semantic=None, layout=None, arithmetic=None):
        layout = self.summary_layout if layout is None else layout
        arithmetic = self.constant_arithmetic if arithmetic is None else arithmetic
        if layout not in ('full', 'sparse') or arithmetic not in ('native', 'analytic'):
            raise ValueError('Unknown factorial cell')
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        hc, cc = self._field_statistics(c, group, eligible, n, True, layout, arithmetic)
        hb, cb = self._field_statistics(b, group, eligible, n, False, layout, arithmetic)
        raw = torch.stack([hc, cc, hb, cb, hc+hb, cc*cb], -1)
        common, product = self._pack_identity(raw, n)
        return common, product, product, n, tuple(raw.unbind(-1)), eligible, group

    def _dense_reference(self, batch, mode):
        """Quadratic equality-count oracle; diagnostics ONLY, never forward."""
        _, semantic, _ = self._membership(batch)
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        if mode == 'global':
            oracle_n = torch.zeros_like(n)
            oracle_n[:, 0] = n.sum(-1)
            oracle_group = torch.zeros_like(group)
        else:
            oracle_n, oracle_group = n, group
        hc, cc = self._dense_distribution(self._bucket_ids(c), oracle_group, eligible, oracle_n)
        hb, cb = self._dense_distribution(self._bucket_ids(b), oracle_group, eligible, oracle_n)
        raw = torch.stack([hc, cc, hb, cb, hc+hb, cc*cb], -1)
        if mode == 'global':
            raw = raw[:, :1, :].expand(-1, 4, -1) * (n > 0).float()[:, :, None]
        elif mode == 'pooled':
            # Independent explicit group sum, versus vectorized implementation.
            numerator = sum(raw[:, g, :] * n[:, g:g+1] for g in range(4))
            shared = numerator / n.sum(-1, keepdim=True).clamp_min(1)
            raw = shared[:, None, :] * (n > 0).float()[:, :, None]
        # Independently assemble input ordering to also test the packing helper.
        hc, cc, hb, cb, hi, ci = raw.unbind(-1)
        scale = math.log(129.0)
        return torch.stack([torch.log1p(n)/scale, hc/scale, hb/scale,
                            cc, cb, hi/scale, ci], -1).flatten(1)

    def _shared_features(self, batch, semantic, sizes):
        # All sums contain <=128 integer counts, exactly representable in float32.
        # Reuse the same true-match partition for every consumer in this forward.
        static = torch.cat([e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)], -1)
        n = sizes.sum(-1)
        c = sizes[:, 1] + sizes[:, 3]
        b = sizes[:, 2] + sizes[:, 3]
        joint = sizes[:, 3]
        counts = (n, c, b, joint, c * b / n.clamp_min(1))
        statistics = self._statistics(counts)
        histogram, observed = self._histogram(batch, semantic, sizes)[:2]
        fine_counts = self._fine_counts(batch, semantic)
        context = self._context_from_counts(fine_counts, observed, sizes)
        return torch.cat([static, context.flatten(1), statistics, histogram], -1)

    def _packed_identity_tables(self, batch, semantic=None):
        # Pack independent field consumers, never combine their distributions.
        # Still B x 2 x 4 x 256 cells, with the same last-axis reduction and
        # native entropy cancellation as the full-table reference.
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        identity = torch.stack([c, b], 1)
        hashed = self._bucket_ids(identity)
        key = group[:, None, :] * self.sketch_bins + hashed
        mass = n.new_zeros((n.shape[0], 2, 4 * self.sketch_bins))
        weights = eligible[:, None, :].expand(-1, 2, -1).float()
        mass = mass.scatter_add(2, key, weights)
        mass = mass.reshape(n.shape[0], 2, 4, self.sketch_bins)
        denom = n.clamp_min(1)[:, None, :]
        xlogx = (mass * mass.clamp_min(1).log()).sum(-1)
        h = torch.where(n[:, None, :] > 0,
                        (denom.log() - xlogx / denom).clamp_min(0),
                        torch.zeros_like(xlogx))
        collision = mass.square().sum(-1) / denom.square()
        hc, hb = h[:, 0, :], h[:, 1, :]
        cc, cb = collision[:, 0, :], collision[:, 1, :]
        raw = torch.stack([hc, cc, hb, cb, hc + hb, cc * cb], -1)
        common, product = self._pack_identity(raw, n)
        return common, product, product, n, tuple(raw.unbind(-1)), eligible, group

    def _sparse_packed_identity_tables(self, batch, semantic=None):
        """Pack only the two varying groups per field; restore native constants.

        Group order is neither/category-only/brand-only/both. Category varies
        in groups 0,2; brand varies in groups 0,1. The other groups have one
        occupied hash bin whenever their complete-pair support is nonzero.
        No varying distributions, events, or field identities are merged.
        """
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        category_match = group.remainder(2).bool()
        brand_match = (group // 2).bool()
        identity = torch.stack([c, b], 1)
        hashed = self._bucket_ids(identity)
        other_match = torch.stack([group // 2, group.remainder(2)], 1)
        residual = torch.stack([eligible & ~category_match,
                                eligible & ~brand_match], 1)
        key = other_match * self.sketch_bins + hashed
        mass = n.new_zeros((n.shape[0], 2, 2 * self.sketch_bins))
        mass = mass.scatter_add(2, key, residual.float())
        mass = mass.reshape(n.shape[0], 2, 2, self.sketch_bins)
        counts = torch.stack([torch.stack([n[:, 0], n[:, 2]], -1),
                              torch.stack([n[:, 0], n[:, 1]], -1)], 1)
        denom = counts.clamp_min(1)
        xlogx = (mass * mass.clamp_min(1).log()).sum(-1)
        h = torch.where(counts > 0,
                        (denom.log() - xlogx / denom).clamp_min(0),
                        torch.zeros_like(counts))
        collision = mass.square().sum(-1) / denom.square()
        hc = torch.stack([h[:, 0, 0], h[:, 0, 0],
                          h[:, 0, 1], h[:, 0, 1]], -1)
        cc = torch.stack([collision[:, 0, 0], collision[:, 0, 0],
                          collision[:, 0, 1], collision[:, 0, 1]], -1)
        hb = torch.stack([h[:, 1, 0], h[:, 1, 1],
                          h[:, 1, 0], h[:, 1, 1]], -1)
        cb = torch.stack([collision[:, 1, 0], collision[:, 1, 1],
                          collision[:, 1, 0], collision[:, 1, 1]], -1)
        denom = n.clamp_min(1)
        if self.constant_arithmetic == 'native':
            # Preserve the previous native float operations, including possible
            # tiny positive entropy residues. Analytic H=0 is a distinct cell.
            constant_h = torch.where(n > 0,
                (denom.log() - (n * n.clamp_min(1).log()) / denom).clamp_min(0),
                torch.zeros_like(n))
            constant_c = n.square() / denom.square()
        else:
            constant_h = torch.zeros_like(n)
            constant_c = (n > 0).float()
        category_constant = self._constant_mask(n, True)
        brand_constant = self._constant_mask(n, False)
        hc = torch.where(category_constant, constant_h, hc) * (n > 0).float()
        cc = torch.where(category_constant, constant_c, cc) * (n > 0).float()
        hb = torch.where(brand_constant, constant_h, hb) * (n > 0).float()
        cb = torch.where(brand_constant, constant_c, cb) * (n > 0).float()
        raw = torch.stack([hc, cc, hb, cb, hc + hb, cc * cb], -1)
        common, product = self._pack_identity(raw, n)
        return common, product, product, n, tuple(raw.unbind(-1)), eligible, group

    def _execution_parts(self, batch, shared, packed):
        if shared:
            _, semantic, sizes = self._membership(batch)
            features = self._shared_features(batch, semantic, sizes)
        else:
            semantic = None
            features = self._features(batch)
        if packed and self.summary_layout == 'sparse':
            tables = self._sparse_packed_identity_tables(batch, semantic)
        else:
            tables = (self._packed_identity_tables(batch, semantic) if packed else
                      self._identity_tables(batch, semantic))
        return features, self._identity_features(tables)


    def forward(self, batch):
        features, identity = self._execution_parts(
            batch, self.config['share_partition'], self.config['pack_fields'])
        return (self.output(features).squeeze(-1) +
                self.identity_head(identity).squeeze(-1))

    def diagnostics(self, batch):
        # All comparisons use this checkpoint. Diagnostics never enter forward.
        # The full serial native cell is the independent execution reference.
        def err(a, b):
            return (a - b).abs().max()
        def rms(a):
            return a.square().mean().sqrt()
        _, semantic, sizes = self._membership(batch)
        features = self._shared_features(batch, semantic, sizes)
        table_cells = {
            'full_serial': self._identity_tables(batch, semantic, 'full', 'native'),
            'full_packed': self._packed_identity_tables(batch, semantic),
            'sparse_serial': self._identity_tables(batch, semantic, 'sparse', 'native'),
            'sparse_packed': self._sparse_packed_identity_tables(batch, semantic),
        }
        ref_i = self._identity_features(table_cells['full_serial'])
        ref_head = self.identity_head(ref_i).squeeze(-1)
        ref_logits = self.output(features).squeeze(-1) + ref_head
        selected_name = self.summary_layout + ('_packed' if self.config['pack_fields'] else '_serial')
        selected_i = self._identity_features(table_cells[selected_name])
        actual_f, actual_i = self._execution_parts(
            batch, self.config['share_partition'], self.config['pack_fields'])
        result = {
            'allocated_parameters': sum(p.numel() for p in self.parameters()),
            'feature_width': self.input_width,
            'first_hidden': self.first_hidden,
            'sparse_layout_enabled': float(self.summary_layout == 'sparse'),
            'pack_fields_enabled': float(self.config['pack_fields']),
            'bucket_cells_total_forward': (4 if self.summary_layout == 'sparse' else 8) * self.sketch_bins,
            'selected_dense_oracle_error': err(selected_i, self._dense_reference(batch, 'local')),
            'selected_finite_logit_fraction': torch.isfinite(self.forward(batch)).float().mean(),
            'forward_feature_reference_error': err(actual_f, features),
            'forward_identity_reference_error': err(actual_i, selected_i),
            'zero_input_head_abs_max': self.identity_head(torch.zeros_like(ref_i)).abs().max(),
            'support_count_mean': table_cells['full_serial'][3].sum(-1).mean(),
        }
        for name, tables in table_cells.items():
            if name == 'full_serial':
                continue
            identity = self._identity_features(tables)
            head = self.identity_head(identity).squeeze(-1)
            logits = self.output(features).squeeze(-1) + head
            result[name + '_identity_error'] = err(identity, ref_i)
            result[name + '_identity_strict_equal'] = float(torch.equal(identity, ref_i))
            result[name + '_support_error'] = err(tables[3], table_cells['full_serial'][3])
            result[name + '_head_error'] = err(head, ref_head)
            result[name + '_logit_error'] = err(logits, ref_logits)
            result[name + '_logit_strict_equal'] = float(torch.equal(logits, ref_logits))
            result[name + '_logit_replacement_rms'] = rms(logits - ref_logits)
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'], batch[key], torch.ones_like(batch[key]))
        reverse = {k: (v.flip(1) if k.startswith('history_') else v) for k, v in batch.items()}
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {k: (v[:, :0] if k.startswith('history_') else v) for k, v in batch.items()}
        static_width = len(self.embeddings) * self.config['embedding_dim']
        for name, altered in (('padding', padded), ('reverse', reverse), ('empty', empty), ('zero_length', zero)):
            af, ai = self._execution_parts(altered, True, self.config['pack_fields'])
            _, asem, asizes = self._membership(altered)
            rf = self._shared_features(altered, asem, asizes)
            ri = self._identity_features(self._identity_tables(altered, asem, 'full', 'native'))
            result[name + '_reference_feature_error'] = err(af, rf)
            result[name + '_reference_identity_error'] = err(ai, ri)
            if name in ('padding', 'reverse'):
                result[name + '_invariance_feature_error'] = err(af, features)
                result[name + '_invariance_identity_error'] = err(ai, selected_i)
            else:
                result[name + '_nonstatic_abs_max'] = af[:, static_width:].abs().max()
                result[name + '_identity_abs_max'] = ai.abs().max()
        # Exhaust all frozen support counts for single-bin distributions in
        # each semantic group. This catches native cancellation differences
        # that the fixed validation subset might not happen to exercise.
        device = batch['static'].device
        support = torch.arange(129, device=device).repeat(4)
        sweep_static = batch['static'][:1].expand(516, -1).clone()
        sweep_static[:, self.category] = support.new_tensor([2, 1, 2, 1]).repeat_interleave(129)
        sweep_static[:, self.brand] = support.new_tensor([2, 2, 1, 1]).repeat_interleave(129)
        sweep = {
            'static': sweep_static,
            'history_category': batch['history_category'].new_ones((516, 128)),
            'history_brand': batch['history_brand'].new_ones((516, 128)),
            'history_mask': torch.arange(128, device=device)[None, :] < support[:, None],
        }
        sweep_ref = self._identity_features(self._identity_tables(sweep, layout='full', arithmetic='native'))
        sweep_packed = self._identity_features(self._sparse_packed_identity_tables(sweep))
        result['constant_sweep_identity_error'] = err(sweep_packed, sweep_ref)
        result['constant_sweep_identity_strict_equal'] = float(torch.equal(sweep_packed, sweep_ref))
        result['constant_sweep_support_error'] = err(self._identity_tables(sweep, layout='full', arithmetic='native')[3].sum(-1), support.float())
        result['constant_sweep_native_entropy_max'] = sweep_ref.reshape(516, 4, 7)[:, :, 1:3].abs().max()
        return {k: float(v) for k, v in result.items()}
