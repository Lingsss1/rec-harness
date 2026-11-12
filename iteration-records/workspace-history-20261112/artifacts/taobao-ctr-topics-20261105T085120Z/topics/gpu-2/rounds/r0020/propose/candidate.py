"""Prospective own-target exclusion versus cross-field conditioning study.

Parent g2-r0019-local_replay source SHA256:
def74351635994f53504c6c9cc5a100c9482fec086e0165a9b074dca0a26b3ee.
All trainable shapes/order, static/context/histogram features, loss and optimizer
are inherited unchanged. Structural mode reconstructs constant matched-field
concentrations analytically and hashes only two varying groups per field.
Excluded mode merges unmatched groups per field. Pooled control averages their
concentrations without identity merging. Support counts and matched-field H=0,
C=1 remain local; all modes derive product coordinates from selected marginals.
No new event pairing model, item identity, labels, I/O or novelty claim.
Torch execution and measured utility/cost are host-pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'content_mode': 'off', 'recency_mode': 'fine', 'identity_mode': 'independent', 'recency_localization': 'local', 'recency_weighting': 'log', 'identity_estimator': 'sketch', 'sketch_bins': 256, 'sketch_salt': 0, 'use_entropy': True, 'use_collision': True, 'summary_mode': 'excluded'}

class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.summary_mode = self.config['summary_mode']
        if self.summary_mode not in ('structural', 'excluded', 'pooled'):
            raise ValueError('Unknown summary mode')
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


    def forward(self, batch):
        base = self.output(self._features(batch)).squeeze(-1)
        identity = self._identity_features(self._identity_tables(batch))
        return base + self.identity_head(identity).squeeze(-1)


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
    def _mass_statistics(mass, n):
        denom = n.clamp_min(1)
        entropy = torch.where(n > 0,
            (denom.log() - (mass * mass.clamp_min(1).log()).sum(-1) / denom).clamp_min(0),
            torch.zeros_like(n))
        return entropy, mass.square().sum(-1) / denom.square()

    def _field_summary(self, identity, group, eligible, n, category, mode):
        # Match code: 0=neither, 1=category only, 2=brand only, 3=both.
        # Within a field-matched group its eligible identity is constant, so
        # H=0 and C=1 exactly, even after the frozen deterministic hash.
        if category:
            own_match = group.remainder(2).bool()
            other_match = group // 2
            unmatched_n = torch.stack([n[:, 0], n[:, 2]], -1)
            constant = n.new_tensor([False, True, False, True]).bool()[None, :]
        else:
            own_match = (group // 2).bool()
            other_match = group.remainder(2)
            unmatched_n = torch.stack([n[:, 0], n[:, 1]], -1)
            constant = n.new_tensor([False, False, True, True]).bool()[None, :]
        residual = eligible & ~own_match
        hashed = self._bucket_ids(identity)
        if mode == 'excluded':
            # Drop only other-field stratification, retaining own-field target
            # exclusion. Forward allocates Bx256 cells per field, not Bx1024.
            mass = n.new_zeros((n.shape[0], self.sketch_bins))
            mass = mass.scatter_add(1, hashed, residual.float())
            h, c = self._mass_statistics(mass, unmatched_n.sum(-1))
            h = h[:, None].expand(-1, 4)
            c = c[:, None].expand(-1, 4)
        else:
            # Two varying groups suffice; no cells for constant distributions.
            mass = n.new_zeros((n.shape[0], 2 * self.sketch_bins))
            mass = mass.scatter_add(1, other_match * self.sketch_bins + hashed,
                                    residual.float())
            h, c = self._mass_statistics(mass.reshape(n.shape[0], 2, self.sketch_bins),
                                         unmatched_n)
            if mode == 'pooled':
                # Replace local concentrations by count-weighted averages.
                # No identity merging; this is not the entropy/collision of
                # the union. Product coordinates are formed AFTER replacement.
                w = unmatched_n / unmatched_n.sum(-1, keepdim=True).clamp_min(1)
                h = (h * w).sum(-1, keepdim=True).expand(-1, 4)
                c = (c * w).sum(-1, keepdim=True).expand(-1, 4)
            elif category:
                h = torch.stack([h[:, 0], h[:, 0], h[:, 1], h[:, 1]], -1)
                c = torch.stack([c[:, 0], c[:, 0], c[:, 1], c[:, 1]], -1)
            else:
                h = torch.stack([h[:, 0], h[:, 1], h[:, 0], h[:, 1]], -1)
                c = torch.stack([c[:, 0], c[:, 1], c[:, 0], c[:, 1]], -1)
        occupied = n > 0
        h = torch.where(constant, torch.zeros_like(n), h) * occupied.float()
        c = torch.where(constant, torch.ones_like(n), c) * occupied.float()
        return h, c

    def _identity_tables(self, batch, semantic=None, summary_mode=None):
        mode = self.summary_mode if summary_mode is None else summary_mode
        if mode not in ('structural', 'excluded', 'pooled'):
            raise ValueError('Unknown summary mode')
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        hc, cc = self._field_summary(c, group, eligible, n, True, mode)
        hb, cb = self._field_summary(b, group, eligible, n, False, mode)
        raw = torch.stack([hc, cc, hb, cb, hc + hb, cc * cb], -1)
        common, product = self._pack_identity(raw, n)
        return common, product, product, n, tuple(raw.unbind(-1)), eligible, group

    def _oracle_identity(self, batch, mode):
        # Equality-count oracle, diagnostic only. Independent quadratic path.
        _, semantic, _ = self._membership(batch)
        cat, brand = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (cat > 0) & (brand > 0)
        group = semantic.argmax(1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        summaries = []
        for identity, matched, columns in ((cat, group.remainder(2).bool(), (1, 3)),
                                            (brand, (group // 2).bool(), (2, 3))):
            known = n.new_zeros(n.shape).bool()
            known[:, columns[0]] = True
            known[:, columns[1]] = True
            if mode == 'excluded':
                residual = eligible & ~matched
                total = residual.float().sum(-1)
                oracle_n = torch.stack([total, torch.zeros_like(total),
                                        torch.zeros_like(total), torch.zeros_like(total)], -1)
                h, c = self._dense_distribution(self._bucket_ids(identity),
                    torch.zeros_like(group), residual, oracle_n)
                h = h[:, :1].expand_as(n)
                c = c[:, :1].expand_as(n)
                h = torch.where(known, torch.zeros_like(n), h) * (n > 0).float()
                c = torch.where(known, torch.ones_like(n), c) * (n > 0).float()
            else:
                h, c = self._dense_distribution(self._bucket_ids(identity), group, eligible, n)
                if mode == 'pooled':
                    w = n * (~known).float()
                    denom = w.sum(-1, keepdim=True).clamp_min(1)
                    h = (h * w).sum(-1, keepdim=True) / denom
                    c = (c * w).sum(-1, keepdim=True) / denom
                    h = torch.where(known, torch.zeros_like(n), h) * (n > 0).float()
                    c = torch.where(known, torch.ones_like(n), c) * (n > 0).float()
            summaries.extend([h, c])
        hc, cc, hb, cb = summaries
        scale = math.log(129.0)
        return torch.stack([torch.log1p(n)/scale, hc/scale, hb/scale,
                            cc, cb, (hc+hb)/scale, cc*cb], -1).flatten(1)

    def diagnostics(self, batch):
        # At most 48 finite scalars. No population claims from this 512-row probe.
        def err(x, y):
            return (x-y).abs().max()
        def rms(x):
            return x.square().mean().sqrt()
        modes = ('structural', 'excluded', 'pooled')
        tables = {m: self._identity_tables(batch, summary_mode=m) for m in modes}
        identities = {m: self._identity_features(tables[m]) for m in modes}
        selected = identities[self.summary_mode]
        n, eligible, group = tables[self.summary_mode][3], tables[self.summary_mode][5], tables[self.summary_mode][6]
        # Recreate the original 4-group bucket readout without elision.
        c, b = batch['history_category'], batch['history_brand']
        hc, cc = self._bucket_distribution(c, group, eligible, n)
        hb, cb = self._bucket_distribution(b, group, eligible, n)
        common, product = self._pack_identity(torch.stack([hc, cc, hb, cb, hc+hb, cc*cb], -1), n)
        original = torch.cat([common, product], -1).flatten(1)
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'], batch[key], torch.ones_like(batch[key]))
        reverse = {k: (v.flip(1) if k.startswith('history_') else v) for k, v in batch.items()}
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {k: (v[:, :0] if k.startswith('history_') else v) for k, v in batch.items()}
        features = self._features(batch)
        static_width = len(self.embeddings) * self.config['embedding_dim']
        residual = self.identity_head(selected)
        result = {
            'allocated_parameters': sum(p.numel() for p in self.parameters()),
            'identity_head_parameters': sum(p.numel() for p in self.identity_head.parameters()),
            'first_hidden': self.first_hidden,
            'feature_width': self.input_width,
            'bucket_cells_per_field_forward': 256.0 if self.summary_mode == 'excluded' else 512.0,
            'structural_vs_original_identity_error': err(identities['structural'], original),
            'structural_vs_original_head_error': err(self.identity_head(identities['structural']), self.identity_head(original)),
            'selected_vs_structural_identity_rms': rms(selected-identities['structural']),
            'excluded_vs_pooled_identity_rms': rms(identities['excluded']-identities['pooled']),
            'support_coordinate_error': err(selected.reshape(-1, 4, 7)[:, :, 0], torch.log1p(n)/math.log(129.0)),
            'same_checkpoint_residual_rms': rms(residual),
            'same_checkpoint_structural_residual_delta': rms(residual-self.identity_head(identities['structural'])),
            'padding_feature_error': err(features, self._features(padded)),
            'reversal_feature_error': err(features, self._features(reverse)),
            'padding_identity_error': err(selected, self._identity_features(self._identity_tables(padded))),
            'reversal_identity_error': err(selected, self._identity_features(self._identity_tables(reverse))),
            'empty_feature_abs_max': self._features(empty)[:, static_width:].abs().max(),
            'zero_length_feature_abs_max': self._features(zero)[:, static_width:].abs().max(),
            'empty_identity_abs_max': self._identity_features(self._identity_tables(empty)).abs().max(),
            'zero_length_identity_abs_max': self._identity_features(self._identity_tables(zero)).abs().max(),
            'zero_input_head_abs_max': self.identity_head(torch.zeros_like(selected)).abs().max(),
            'finite_logit_fraction': torch.isfinite(self.forward(batch)).float().mean(),
            'support_count_mean': n.sum(-1).mean(),
            'missing_pair_count_mean': (batch['history_mask'].float().sum(-1)-n.sum(-1)).mean(),
        }
        for m in modes:
            result[m+'_dense_oracle_error'] = err(identities[m], self._oracle_identity(batch, m))
        for g, name in enumerate(('neither', 'category_only', 'brand_only', 'both')):
            result[name+'_count_mean'] = n[:, g].mean()
        applicable = self.category_vocab > 3 and self.brand_vocab > 3
        result['witness_applicable'] = float(applicable)
        if applicable:
            device = batch['static'].device
            witness = {
                'static': batch['static'][:1].expand(2, -1).clone(),
                'history_category': torch.tensor([[1, 1, 1, 2], [1, 2, 1, 1]], device=device),
                'history_brand': torch.tensor([[3, 3, 1, 1], [3, 3, 1, 1]], device=device),
                'history_type': torch.ones((2, 4), dtype=torch.long, device=device),
                'history_age_seconds': torch.zeros((2, 4), device=device),
                'history_mask': torch.ones((2, 4), dtype=torch.bool, device=device),
            }
            witness['static'][:, self.category] = 3
            witness['static'][:, self.brand] = 3
            result['localization_witness_base_error'] = err(self._features(witness)[0], self._features(witness)[1])
            for m in modes:
                wi = self._identity_features(self._identity_tables(witness, summary_mode=m))
                result['localization_witness_'+m+'_delta'] = err(wi[0], wi[1])
            witness['history_category'] = torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2]], device=device)
            for m in modes:
                wi = self._identity_features(self._identity_tables(witness, summary_mode=m))
                result['merging_witness_'+m+'_delta'] = err(wi[0], wi[1])
        return {k: float(v) for k, v in result.items()}
