"""Event versus pooled field energy with conditional pairing replacements.

Only the appended radius changes. Parent means/counts/action-age histograms/
factorized moments, mean direction, allocation and optimization are fixed.
All energy alternatives are computed in each forward. No model I/O.
Host forward/backward execution and predictive performance remain unmeasured.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'moment_grouping': 'match', 'moment_pairing': 'independent', 'field_mode': 'contrast', 'field_grouping': 'rank', 'field_scope': 'global', 'field_layout': 'compact', 'global_direction': 'norm_collapsed', 'energy_granularity': 'observed'}


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
        # Parent means, histograms and independent moments stay fixed in
        # true-match groups; only the appended field readout is intervened on.
        if self.grouping_mode != 'match':
            raise ValueError('This study fixes learned pooling to match')
        self.histogram_grouping = self.config['histogram_grouping']
        self.histogram_pairing = self.config['histogram_pairing']
        if self.histogram_grouping not in ('match', 'rank'):
            raise ValueError('Unknown histogram_grouping')
        if self.histogram_pairing not in ('observed', 'independent'):
            raise ValueError('Unknown histogram_pairing')
        self.moment_grouping = self.config['moment_grouping']
        self.moment_pairing = self.config['moment_pairing']
        if self.moment_grouping not in ('match', 'rank'):
            raise ValueError('Unknown moment_grouping')
        if self.moment_pairing not in ('observed', 'independent'):
            raise ValueError('Unknown moment_pairing')
        if self.histogram_grouping != 'match' or self.histogram_pairing != 'observed':
            raise ValueError('Incumbent histogram must remain observed/match')
        self.field_mode = self.config['field_mode']
        self.field_grouping = self.config['field_grouping']
        if self.field_mode not in ('contrast', 'norm_collapsed'):
            raise ValueError('Unknown field_mode')
        if self.field_grouping not in ('match', 'rank'):
            raise ValueError('Unknown field_grouping')
        if self.moment_grouping != 'match' or self.moment_pairing != 'independent':
            raise ValueError('This study fixes incumbent independent/match moments')
        self.field_scope = self.config['field_scope']
        self.field_layout = self.config['field_layout']
        self.global_direction = self.config['global_direction']
        if self.field_scope not in ('rank', 'global'):
            raise ValueError('Unknown field_scope')
        if self.field_layout not in ('wide', 'compact'):
            raise ValueError('Unknown field_layout')
        if self.global_direction not in ('contrast', 'norm_collapsed'):
            raise ValueError('Unknown global_direction')
        if self.field_scope == 'rank' and (self.field_layout != 'wide' or
                                         self.global_direction != 'contrast'):
            raise ValueError('Shared reference must retain original rank readout')
        self.energy_granularity = self.config['energy_granularity']
        if self.energy_granularity not in ('pooled', 'observed', 'bundle_independent', 'field_independent'):
            raise ValueError('Unknown energy_granularity')
        if (self.field_scope, self.field_layout, self.global_direction) != ('global', 'compact', 'norm_collapsed'):
            raise ValueError('This study fixes the incumbent compact global mean direction')
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
        field_width = (12 if self.field_layout == 'wide' else 3) * dim
        width = len(meta['fields']) * dim + 4 * dim + 4 + 64 + 4 * dim + field_width
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

    def _moments(self, batch, semantic, sizes):
        """Selected readout plus raw paired/factorized diagonal cross moments.

        C = e_category + e_brand; T = e_action + e_log2_age.
        observed = mean(C*T); independent = mean(C)*mean(T).
        The independent value equals the average over all within-group C,T
        pairs, including self pairs. It preserves vector marginal means and
        removes their empirical diagonal covariance. It is not a random shuffle
        or a derangement. A fixed factor 1/2 is shared by all four cells.
        """
        mask = batch['history_mask']
        if self.moment_grouping == 'rank':
            rank = mask.long().cumsum(-1) - 1
            upper = sizes.long().cumsum(-1)
            lower = upper - sizes.long()
            membership = ((rank[:, None, :] >= lower[:, :, None]) &
                          (rank[:, None, :] < upper[:, :, None]) &
                          mask[:, None, :]).float()
        else:
            membership = semantic
        if mask.shape[1] == 0:
            zeros = sizes.new_zeros((sizes.shape[0], 4, self.config['embedding_dim']))
            return zeros, zeros, zeros, membership
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        content = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']))
        context = self.action(batch['history_type']) + self.recency(ages)
        denominator = sizes.clamp_min(1).unsqueeze(-1)
        content_mean = torch.bmm(membership, content) / denominator
        context_mean = torch.bmm(membership, context) / denominator
        observed = torch.bmm(membership, content * context) / denominator
        independent = content_mean * context_mean
        selected = observed if self.moment_pairing == 'observed' else independent
        return 0.5 * selected, observed, independent, membership

    def _reverse_context_within_match(self, batch, semantic, sizes):
        """Diagnostic intervention, never training augmentation.

        Reverse complete (action, age) bundles within each true match group.
        This preserves that group's context multiset, the incumbent mean and
        its complete action-age table, but can change content-context pairing.
        """
        length = batch['history_mask'].shape[1]
        altered = dict(batch)
        if length == 0:
            return altered
        positions = torch.arange(length, device=semantic.device).unsqueeze(0).expand(sizes.shape[0], -1)
        for g in range(4):
            member = semantic[:, g, :].bool()
            ordered = torch.where(member, positions, positions + length).argsort(-1)
            rank = member.long().cumsum(-1) - 1
            source_rank = (sizes[:, g:g+1].long() - 1 - rank).clamp(0, length - 1)
            source_position = ordered.gather(1, source_rank)
            for key in ('history_type', 'history_age_seconds'):
                altered[key] = torch.where(member, batch[key].gather(1, source_position), altered[key])
        return altered

    def _field_readout(self, batch, semantic, sizes):
        """Return selected/full/norm-only blocks, field means and membership.

        Shape of each readout is [B,4,3,d]. A Hadamard transform of the
        four field means is orthonormal; its first row is half their sum,
        already available from parent means for match membership. It does
        not restore individual events or content/context pairing.

        The control retains the norm of the three contrast rows, hence
        contrast energy is allowed information. Its direction uses only
        the summed mean. Near zero sums use a fixed coordinate direction.
        No labels, data-dependent fitted scales or random draws enter.
        """
        mask = batch['history_mask']
        # Compute the positional replacement in every cell for forward-cost
        # matching; both partitions have the same prescribed cardinalities.
        rank = mask.long().cumsum(-1) - 1
        upper = sizes.long().cumsum(-1)
        lower = upper - sizes.long()
        positional = ((rank[:, None, :] >= lower[:, :, None]) &
                      (rank[:, None, :] < upper[:, :, None]) &
                      mask[:, None, :]).float()
        membership = positional if self.field_grouping == 'rank' else semantic
        dim = self.config['embedding_dim']
        if mask.shape[1] == 0:
            means = sizes.new_zeros((sizes.shape[0], 4, 4, dim))
        else:
            ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            # Concatenate before a single bmm; no [B,group,L,field,d] expansion.
            values = torch.cat([
                self.embeddings[self.category](batch['history_category']),
                self.embeddings[self.brand](batch['history_brand']),
                self.action(batch['history_type']), self.recency(ages)
            ], dim=-1)
            pooled = torch.bmm(membership, values) / sizes.clamp_min(1).unsqueeze(-1)
            means = pooled.reshape(sizes.shape[0], 4, 4, dim)
        c, b, a, r = means.unbind(dim=2)
        contrasts = 0.5 * torch.stack([
            c + b - a - r, c - b + a - r, c - b - a + r
        ], dim=2)
        summed = 0.5 * (c + b + a + r)
        repeated = summed.unsqueeze(2).expand(-1, -1, 3, -1)
        # vector_norm has defined zero gradients, unlike sqrt(sum(x*x)) at 0.
        energy_norm = torch.linalg.vector_norm(contrasts, dim=(-2, -1), keepdim=True)
        repeated_norm = torch.linalg.vector_norm(repeated, dim=(-2, -1), keepdim=True)
        basis = torch.zeros_like(contrasts)
        basis[:, :, 0, 0] = 1.0
        direction = torch.where(repeated_norm >= 1e-8,
                                repeated / repeated_norm.clamp_min(1e-8), basis)
        collapsed = energy_norm * direction
        selected = contrasts if self.field_mode == 'contrast' else collapsed
        return selected, contrasts, collapsed, means, membership

    def _inverse_fields(self, half_sum, contrasts):
        h1, h2, h3 = contrasts.unbind(dim=2)
        return 0.5 * torch.stack([
            half_sum + h1 + h2 + h3, half_sum + h1 - h2 - h3,
            half_sum - h1 + h2 - h3, half_sum - h1 - h2 + h3
        ], dim=2)

    def _contrast_blocks(self, means):
        """Same fixed field transform and zero-safe norm control as ancestor."""
        c, b, a, r = means.unbind(dim=2)
        full = 0.5 * torch.stack([
            c + b - a - r, c - b + a - r, c - b - a + r
        ], dim=2)
        summed = 0.5 * (c + b + a + r)
        repeated = summed.unsqueeze(2).expand(-1, -1, 3, -1)
        energy = torch.linalg.vector_norm(full, dim=(-2, -1), keepdim=True)
        denominator = torch.linalg.vector_norm(repeated, dim=(-2, -1), keepdim=True)
        basis = torch.zeros_like(full)
        basis[:, :, 0, 0] = 1.0
        direction = torch.where(denominator >= 1e-8,
                                repeated / denominator.clamp_min(1e-8), basis)
        return full, energy * direction

    @staticmethod
    def _positive_root(value):
        # Avoid sqrt(0)'s undefined backward even inside an inactive where.
        # Negative roundoff is projected to zero and reported in diagnostics.
        positive = value > 0
        safe = torch.where(positive, value, torch.ones_like(value))
        return torch.where(positive, safe.sqrt(), torch.zeros_like(value))

    def _energy_readout(self, batch, semantic, sizes, means, values,
                        content_mean, context_mean):
        """Exact expectations under three empirical event distributions.

        H is the orthonormal 3-row contrast transform across c,b,a,r.
        pooled = ||H mean(v)||; observed^2 = mean(||H v_event||^2).
        bundle_independent draws (c,b) and (a,r) independently WITHIN each
        true match group, including self pairs. field_independent draws all
        four fields independently within that group. These are analytic
        product distributions, not shuffles or derangements. Global field
        means and per-group marginal distributions remain the same. The
        existing action-age histogram stays observed in EVERY variant.

        E_bundle = E_observed + cov(C,T | match)/2, C=c+b,T=a+r.
        E_fields = sum_g p_g ||H mu_g||^2 + 3/4 sum_j E Var(v_j|g).
        All E quantities here are squared radii. The observed minus pooled
        difference is trace variance of event contrasts, not field-mean
        energy. Missing field embeddings are zero observations, as inherited.
        """
        full, parent = self._contrast_blocks(means)
        pooled = torch.linalg.vector_norm(full, dim=(-2, -1)).squeeze(1)
        mask = batch['history_mask'].float()
        n = mask.sum(-1).clamp_min(1)
        dim = self.config['embedding_dim']
        if mask.shape[1] == 0:
            observed_sq = pooled * 0
            covariance_ct = pooled * 0
            bundle_sq = pooled * 0
            fields_sq = pooled * 0
            within_field_variance = pooled * 0
        else:
            c, b, a, r = values.split(dim, dim=-1)
            # ||H v||^2 uses three sums of squares, avoiding cancellation.
            event_sq = (0.25 * (c + b - a - r).square().sum(-1) +
                        0.5 * (c - b).square().sum(-1) +
                        0.5 * (a - r).square().sum(-1))
            observed_sq = (event_sq * mask).sum(-1) / n
            weights = sizes / n[:, None]
            ct = ((c + b) * (a + r)).sum(-1)
            product_mean = (weights * (content_mean * context_mean).sum(-1)).sum(-1)
            covariance_ct = (ct * mask).sum(-1) / n - product_mean
            bundle_sq = observed_sq + 0.5 * covariance_ct
            # [B,4,L] x [B,L,4d], without an event x group expansion.
            group_means = (torch.bmm(semantic, values) /
                           sizes.clamp_min(1).unsqueeze(-1)).reshape(-1, 4, 4, dim)
            group_contrasts = self._contrast_blocks(group_means)[0]
            group_energy = group_contrasts.square().sum((-2, -1))
            second_moment = (values.square().sum(-1) * mask).sum(-1) / n
            squared_group_means = group_means.square().sum((-2, -1))
            within_field_variance = second_moment - (weights * squared_group_means).sum(-1)
            fields_sq = ((weights * group_energy).sum(-1) +
                         0.75 * within_field_variance)
        repeated = (means.sum(2) * 0.5).unsqueeze(2).expand_as(full)
        denominator = torch.linalg.vector_norm(repeated, dim=(-2, -1), keepdim=True)
        basis = torch.zeros_like(full)
        basis[:, :, 0, 0] = 1.0
        unit = torch.where(denominator >= 1e-8,
                           repeated / denominator.clamp_min(1e-8), basis)
        amplitudes = {'pooled': pooled,
                      'observed': self._positive_root(observed_sq),
                      'bundle_independent': self._positive_root(bundle_sq),
                      'field_independent': self._positive_root(fields_sq)}
        blocks = {key: amplitude[:, None, None, None] * unit
                  for key, amplitude in amplitudes.items()}
        # Preserve the incumbent operation ordering exactly in the parent cell.
        blocks['pooled'] = parent
        raw = {'observed': observed_sq, 'bundle_independent': bundle_sq,
               'field_independent': fields_sq}
        return blocks, amplitudes, raw, covariance_ct, within_field_variance

    def _shared_features(self, batch, mode=None):
        mode = self.energy_granularity if mode is None else mode
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        _, semantic, sizes = self._membership(batch)
        mask = batch['history_mask']
        dim = self.config['embedding_dim']
        if mask.shape[1] == 0:
            pools = static.new_zeros((static.shape[0], 4, dim))
            moments = torch.zeros_like(pools)
            content_mean = torch.zeros_like(pools)
            context_mean = torch.zeros_like(pools)
            values = static.new_zeros((static.shape[0], 0, 4 * dim))
            means = static.new_zeros((static.shape[0], 1, 4, dim))
        else:
            ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            cv = self.embeddings[self.category](batch['history_category'])
            bv = self.embeddings[self.brand](batch['history_brand'])
            av = self.action(batch['history_type'])
            rv = self.recency(ages)
            history = cv + bv + av + rv
            denominator = sizes.clamp_min(1).unsqueeze(-1)
            pools = torch.bmm(semantic, history) / denominator
            content_mean = torch.bmm(semantic, cv + bv) / denominator
            context_mean = torch.bmm(semantic, av + rv) / denominator
            moments = 0.5 * (content_mean * context_mean)
            values = torch.cat([cv, bv, av, rv], dim=-1)
            global_denominator = mask.float().sum(-1).clamp_min(1)[:, None, None]
            field_means = torch.bmm(mask[:, None, :].float(), values) / global_denominator
            means = field_means.reshape(static.shape[0], 1, 4, dim)
        counts = self._match_counts(batch)
        statistics = self._statistics(counts)
        histogram = self._histogram(batch, semantic, sizes)[0]
        blocks, amplitudes, raw, covariance, variance = self._energy_readout(
            batch, semantic, sizes, means, values, content_mean, context_mean)
        base = torch.cat([static, pools.flatten(1), statistics, histogram,
                          moments.flatten(1)], dim=-1)
        features = torch.cat([base, blocks[mode].flatten(1)], dim=-1)
        return features, base, blocks, amplitudes, raw, covariance, variance, semantic, sizes

    def forward(self, batch):
        return self.output(self._shared_features(batch)[0]).squeeze(-1)

    def _incumbent_features(self, batch, scope=None, layout=None, direction=None,
                         radial_amplitude=None, radial_direction=None):
        scope = self.field_scope if scope is None else scope
        layout = self.field_layout if layout is None else layout
        direction = self.global_direction if direction is None else direction
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        _, semantic, sizes = self._membership(batch)
        mask = batch['history_mask']
        dim = self.config['embedding_dim']
        if mask.shape[1] == 0:
            pools = static.new_zeros((static.shape[0], 4, dim))
            moments = torch.zeros_like(pools)
            means = static.new_zeros((static.shape[0], 4 if scope == 'rank' else 1, 4, dim))
        else:
            ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            # Each history field is embedded exactly once in forward. Static
            # target embeddings are separate lookup consumers, as in ancestor.
            cv = self.embeddings[self.category](batch['history_category'])
            bv = self.embeddings[self.brand](batch['history_brand'])
            av = self.action(batch['history_type'])
            rv = self.recency(ages)
            history = cv + bv + av + rv
            denominator = sizes.clamp_min(1).unsqueeze(-1)
            pools = torch.bmm(semantic, history) / denominator
            # Keep the parent's operation ordering for both marginal means.
            content_mean = torch.bmm(semantic, cv + bv) / denominator
            context_mean = torch.bmm(semantic, av + rv) / denominator
            moments = 0.5 * (content_mean * context_mean)
            values = torch.cat([cv, bv, av, rv], dim=-1)
            if scope == 'rank':
                rank = mask.long().cumsum(-1) - 1
                upper = sizes.long().cumsum(-1)
                lower = upper - sizes.long()
                membership = ((rank[:, None, :] >= lower[:, :, None]) &
                              (rank[:, None, :] < upper[:, :, None]) &
                              mask[:, None, :]).float()
                field_means = torch.bmm(membership, values) / denominator
                means = field_means.reshape(static.shape[0], 4, 4, dim)
            else:
                membership = mask[:, None, :].float()
                denominator = mask.float().sum(-1).clamp_min(1)[:, None, None]
                field_means = torch.bmm(membership, values) / denominator
                means = field_means.reshape(static.shape[0], 1, 4, dim)
        counts = self._match_counts(batch)
        statistics = self._statistics(counts)
        histogram = self._histogram(batch, semantic, sizes)[0]
        full, collapsed = self._contrast_blocks(means)
        selected = full if direction == 'contrast' else collapsed
        if scope == 'global' and layout == 'wide':
            # Isometry R^48 -> R^192: four copies scaled by 1/sqrt(4).
            # All columns are used, but effective input rank remains <=48.
            extra = selected.expand(-1, 4, -1, -1) * 0.5
        else:
            extra = selected
        features = torch.cat([static, pools.flatten(1), statistics, histogram,
                              moments.flatten(1), extra.flatten(1)], dim=-1)
        return features, extra, means, full, collapsed, semantic, sizes

    def _legacy_base(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        moments = self._moments(batch, semantic, sizes)[0]
        return torch.cat([static, pools.flatten(1), statistics, histogram,
                          moments.flatten(1)], dim=-1)

    def _reverse_brand_within_match(self, batch, semantic, sizes):
        altered = dict(batch)
        length = batch['history_mask'].shape[1]
        if length == 0:
            return altered
        positions = torch.arange(length, device=semantic.device).unsqueeze(0).expand(sizes.shape[0], -1)
        for g in range(4):
            member = semantic[:, g, :].bool()
            ordered = torch.where(member, positions, positions + length).argsort(-1)
            rank = member.long().cumsum(-1) - 1
            source_rank = (sizes[:, g:g+1].long() - 1 - rank).clamp(0, length - 1)
            source_position = ordered.gather(1, source_rank)
            altered['history_brand'] = torch.where(
                member, batch['history_brand'].gather(1, source_position), altered['history_brand'])
        return altered

    def diagnostics(self, batch):
        # Host calls on first 512 validation impressions. These candidate-defined
        # invariance checks are not population ablations or mechanism proof.
        features, base, blocks, amps, raw, cov, var, semantic, sizes = self._shared_features(batch)
        mask = batch['history_mask']
        parent_features = torch.cat([base, blocks['pooled'].flatten(1)], dim=-1)
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(mask, batch[key], torch.zeros_like(batch[key]))
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        zero_length = {key: (value[:, :0] if key.startswith('history_') else value)
                       for key, value in batch.items()}
        reversed_batch = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reversed_batch[key] = batch[key].flip(1)
        context_reversed = self._reverse_context_within_match(batch, semantic, sizes)
        brand_reversed = self._reverse_brand_within_match(batch, semantic, sizes)
        reverse = self._shared_features(reversed_batch)
        context = self._shared_features(context_reversed)
        brand = self._shared_features(brand_reversed)
        # Independent incumbent implementation copied from energy_mean.
        # Its energy/mean branch is algebraically the inherited collapsed block;
        # all base operations, allocation and initialization order are unchanged.
        old_parent_features = self._incumbent_features(batch)[0]
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'extra_coordinates': float(blocks[self.energy_granularity].flatten(1).shape[1]),
            'history_embedding_consumers_per_field_forward': 1.0,
            'pooled_mode_enabled': float(self.energy_granularity == 'pooled'),
            'shared_parent_base_error_max': (base-self._legacy_base(batch)).abs().max(),
            'incumbent_full_feature_error_max': (parent_features-old_parent_features).abs().max(),
            'incumbent_same_checkpoint_logit_error_max': (self.output(parent_features)-self.output(old_parent_features)).abs().max(),
            'parent_cell_feature_error_max': (features-parent_features).abs().max() if self.energy_granularity == 'pooled' else 0.0,
            'parent_cell_check_applicable': float(self.energy_granularity == 'pooled'),
            'observed_minus_pooled_squared_mean': (raw['observed']-amps['pooled'].square()).mean(),
            'observed_minus_pooled_squared_min': (raw['observed']-amps['pooled'].square()).min(),
            'within_group_field_variance_mean': var.mean(),
            'within_group_field_variance_min': var.min(),
            'conditional_content_context_covariance_mean': cov.mean(),
            'bundle_energy_covariance_identity_error_max': (raw['bundle_independent']-raw['observed']-0.5*cov).abs().max(),
            'padding_feature_error_max': (features-self._shared_features(padded)[0]).abs().max(),
            'all_empty_history_feature_abs_max': self._shared_features(empty)[0][:,batch['static'].shape[1]*self.config['embedding_dim']:].abs().max(),
            'zero_length_history_feature_abs_max': self._shared_features(zero_length)[0][:,batch['static'].shape[1]*self.config['embedding_dim']:].abs().max(),
            'sequence_reversal_feature_error_max': (features-reverse[0]).abs().max(),
            'context_reversal_base_error_max': (base-context[1]).abs().max(),
            'brand_reversal_base_error_max': (base-brand[1]).abs().max(),
            'context_reversal_membership_error_max': (semantic-context[7]).abs().max() if semantic.numel() else 0.0,
            'brand_reversal_membership_error_max': (semantic-brand[7]).abs().max() if semantic.numel() else 0.0,
            'same_checkpoint_parent_logit_rms_delta': (self.output(features)-self.output(parent_features)).square().mean().sqrt(),
            'empty_history_fraction': (~mask.any(-1)).float().mean(),
            'visible_count_mean': mask.float().sum(-1).mean(),
            'visible_count_at_limit_fraction': (mask.sum(-1)==128).float().mean(),
        }
        for mode in ('pooled', 'observed', 'bundle_independent', 'field_independent'):
            result[mode+'_amplitude_mean'] = amps[mode].mean()
            result[mode+'_selected_norm_error_max'] = (torch.linalg.vector_norm(blocks[mode],dim=(1,2,3))-amps[mode]).abs().max()
            result[mode+'_context_reversal_amplitude_delta_max'] = (amps[mode]-context[3][mode]).abs().max()
            result[mode+'_brand_reversal_amplitude_delta_max'] = (amps[mode]-brand[3][mode]).abs().max()
        for mode in ('observed', 'bundle_independent', 'field_independent'):
            result[mode+'_negative_squared_radius_fraction'] = (raw[mode]<0).float().mean()
            result[mode+'_raw_squared_radius_min'] = raw[mode].min()
        return {key: float(value) for key, value in result.items()}
