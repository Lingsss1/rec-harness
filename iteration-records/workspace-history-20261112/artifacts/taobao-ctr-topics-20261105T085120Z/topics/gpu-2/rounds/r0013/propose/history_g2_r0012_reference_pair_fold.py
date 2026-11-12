"""Conditional category-brand identity concentration beyond pooled marginals.

All cells preserve the incumbent feature extractor and parameter initialization.
A bias-free residual head receives exact empirical entropy/collision summaries.
Joint versus empirical-product summaries isolate within-event identity pairing;
common marginal and zero-input controls test marginal information and the base.
No item histories, labels, I/O, new loss, optimizer or data selection are used.
Performance and Torch execution remain host-pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'moment_grouping': 'match', 'moment_pairing': 'independent', 'field_mode': 'contrast', 'field_grouping': 'rank', 'field_scope': 'global', 'field_layout': 'compact', 'global_direction': 'norm_collapsed', 'identity_mode': 'joint', 'pairing': 'fold'}


class ReferenceModel(nn.Module):
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
        # Append allocation AFTER every incumbent parameter. The old parameter
        # shapes, allocation order and scratch initialization are unchanged.
        self.identity_mode = self.config['identity_mode']
        if self.identity_mode not in ('joint', 'independent', 'marginal', 'zero'):
            raise ValueError('Unknown identity_mode')
        if (self.field_scope, self.field_layout, self.global_direction) != ('global', 'compact', 'norm_collapsed'):
            raise ValueError('This study fixes the incumbent representation')
        self.category_vocab = int(meta['buckets'][self.category])
        self.brand_vocab = int(meta['buckets'][self.brand])
        # Pair keys use the supplied categorical bucket identities, not raw
        # products/items. Bound the packed int64 key before any tensor math.
        if self.category_vocab * self.brand_vocab > (2**63 - 1) // 4:
            raise ValueError('Packed identity key would overflow int64')
        self.identity_head = nn.Sequential(
            nn.Linear(28, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        # A zero final weight preserves the incumbent initial function. In the
        # zero-input cell the whole bias-free branch stays identically zero.
        nn.init.zeros_(self.identity_head[-1].weight)

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

    def _legacy_forward(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        moments = self._moments(batch, semantic, sizes)[0]
        extra = self._field_readout(batch, semantic, sizes)[0]
        features = torch.cat([static, pools.flatten(1), statistics, histogram,
                              moments.flatten(1), extra.flatten(1)], dim=-1)
        return self.output(features).squeeze(-1)

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

    def _shared_features(self, batch, scope=None, layout=None, direction=None):
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


    def _legacy_features(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        moments = self._moments(batch, semantic, sizes)[0]
        extra = self._field_readout(batch, semantic, sizes)[0]
        return torch.cat([static, pools.flatten(1), statistics, histogram,
                          moments.flatten(1), extra.flatten(1)], dim=-1)


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

    def forward(self, batch):
        parent_features, _, _, _, _, semantic, _ = self._shared_features(batch)
        # Every cell computes identical sort/statistic alternatives before
        # selecting a readout. All event embedding consumers stay inherited.
        tables = self._identity_tables(batch, semantic)
        parent = self.output(parent_features).squeeze(-1)
        return parent + self.identity_head(self._identity_features(tables)).squeeze(-1)

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

    def _reverse_eligible_brand(self, batch, semantic, eligible):
        """Preserve match groups and eligible marginals; change pair coupling.

        Only a diagnostic intervention, never training augmentation. Excluding
        missing values preserves each group's complete-pair support exactly.
        """
        length = eligible.shape[1]
        altered = dict(batch)
        if length == 0:
            return altered
        positions = torch.arange(length, device=eligible.device)[None, :].expand(eligible.shape[0], -1)
        brand = batch['history_brand']
        for g in range(4):
            member = semantic[:, g, :].bool() & eligible
            count = member.long().sum(-1, keepdim=True)
            ordered = torch.where(member, positions, positions + length).argsort(-1)
            rank = member.long().cumsum(-1) - 1
            source = (count - 1 - rank).clamp(0, length - 1)
            index = ordered.gather(1, source)
            brand = torch.where(member, batch['history_brand'].gather(1, index), brand)
        altered['history_brand'] = brand
        return altered

    def diagnostics(self, batch):
        # First 512 validation impressions, same checkpoint. These invariance
        # and algebra measurements are not predictive population ablations.
        features, _, _, _, _, semantic, sizes = self._shared_features(batch)
        tables = self._identity_tables(batch, semantic)
        common, joint, independent, n, raw, eligible, group = tables
        selected = self._identity_features(tables)
        delta = self.identity_head(selected).squeeze(-1)
        c, b = batch['history_category'], batch['history_brand']
        identities = (c, b, c * self.brand_vocab + b)
        ref = [self._dense_distribution(k, group, eligible, n) for k in identities]
        raw_error = torch.stack([(a-b).abs().max() for a,b in zip(raw, [v for pair in ref for v in pair])]).max()
        reverse = {key: (value.flip(1) if key.startswith('history_') else value)
                   for key, value in batch.items()}
        reversed_features = self._identity_features(self._identity_tables(reverse))
        changed = self._reverse_eligible_brand(batch, semantic, eligible)
        changed_tables = self._identity_tables(changed)
        changed_parent = self._shared_features(changed)[0]
        renamed = dict(batch)
        renamed['static'] = batch['static'].clone()
        for key, index, vocab in (('history_category', self.category, self.category_vocab),
                                  ('history_brand', self.brand, self.brand_vocab)):
            values = batch[key]
            renamed[key] = torch.where(values > 0, vocab-values, values)
            values = batch['static'][:, index]
            renamed['static'][:, index] = torch.where(values > 0, vocab-values, values)
        renamed_features = self._identity_features(self._identity_tables(renamed))
        padded = dict(batch)
        for key in ('history_category','history_brand','history_type','history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'], batch[key], torch.ones_like(batch[key]))
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero_length = {key: (value[:, :0] if key.startswith('history_') else value)
                       for key, value in batch.items()}
        all_count = batch['history_mask'].float().sum(-1)
        legacy_base = self._legacy_features(batch)[:, :features.shape[1]-48]
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'identity_head_parameters': float(sum(p.numel() for p in self.identity_head.parameters())),
            'identity_input_coordinates': float(selected.shape[1]),
            'history_embedding_consumers_per_field_forward': 1.0,
            'joint_mode_enabled': float(self.identity_mode == 'joint'),
            'independent_mode_enabled': float(self.identity_mode == 'independent'),
            'marginal_mode_enabled': float(self.identity_mode == 'marginal'),
            'zero_mode_enabled': float(self.identity_mode == 'zero'),
            'sorted_vs_dense_raw_stat_error_max': raw_error,
            'shared_parent_base_error_max': (features[:, :features.shape[1]-48]-legacy_base).abs().max(),
            'sequence_reversal_identity_error_max': (selected-reversed_features).abs().max(),
            'identity_renaming_error_max': (selected-renamed_features).abs().max(),
            'padding_identity_error_max': (selected-self._identity_features(self._identity_tables(padded))).abs().max(),
            'all_empty_identity_abs_max': self._identity_features(self._identity_tables(empty)).abs().max(),
            'zero_length_identity_abs_max': self._identity_features(self._identity_tables(zero_length)).abs().max(),
            'all_empty_residual_abs_max': self.identity_head(self._identity_features(self._identity_tables(empty))).abs().max(),
            'zero_input_head_abs_max': self.identity_head(torch.zeros_like(selected)).abs().max(),
            'brand_reverse_common_error_max': (common-changed_tables[0]).abs().max(),
            'brand_reverse_independent_error_max': (independent-changed_tables[2]).abs().max(),
            'brand_reverse_parent_feature_error_max': (features-changed_parent).abs().max(),
            'brand_reverse_membership_error_max': (semantic-self._membership(changed)[1]).abs().max(),
            'brand_reverse_joint_stat_delta_max': (joint-changed_tables[1]).abs().max(),
            'mutual_information_min': (raw[0]+raw[2]-raw[4]).min(),
            'mutual_information_mean': (raw[0]+raw[2]-raw[4]).mean(),
            'joint_collision_mean': raw[5].mean(),
            'independent_collision_mean': (raw[1]*raw[3]).mean(),
            'complete_pair_count_mean': n.sum(-1).mean(),
            'visible_count_mean': all_count.mean(),
            'missing_pair_count_mean': (all_count-n.sum(-1)).mean(),
            'empty_history_fraction': (all_count == 0).float().mean(),
            'visible_count_at_limit_fraction': (all_count == 128).float().mean(),
            'same_checkpoint_residual_rms': delta.square().mean().sqrt(),
            'same_checkpoint_brand_reverse_residual_rms_delta': (delta-self.identity_head(self._identity_features(changed_tables)).squeeze(-1)).square().mean().sqrt(),
            'residual_head_last_weight_norm': torch.linalg.vector_norm(self.identity_head[-1].weight),
        }
        for g, name in enumerate(('neither','category_only','brand_only','both')):
            result[name+'_complete_pair_count_mean'] = n[:,g].mean()
            result[name+'_multiple_complete_fraction'] = (n[:,g] > 1).float().mean()
        # A deterministic information-collision witness. Target=3, two
        # non-target categories/brands; same actions, ages and field marginals.
        applicable = self.category_vocab > 3 and self.brand_vocab > 3
        result['collision_witness_applicable'] = float(applicable)
        if applicable:
            device = batch['static'].device
            witness = {'static': batch['static'][:1].expand(2,-1).clone(),
                       'history_category': torch.tensor([[1,1,2,2],[1,1,2,2]], device=device),
                       'history_brand': torch.tensor([[1,1,2,2],[1,2,1,2]], device=device),
                       'history_type': torch.ones((2,4), dtype=torch.long, device=device),
                       'history_age_seconds': torch.zeros((2,4), device=device),
                       'history_mask': torch.ones((2,4), dtype=torch.bool, device=device)}
            witness['static'][:, self.category] = 3
            witness['static'][:, self.brand] = 3
            wf = self._shared_features(witness)[0]
            wt = self._identity_tables(witness)
            result['witness_parent_feature_error_max'] = (wf[0]-wf[1]).abs().max()
            result['witness_common_error_max'] = (wt[0][0]-wt[0][1]).abs().max()
            result['witness_independent_error_max'] = (wt[2][0]-wt[2][1]).abs().max()
            result['witness_joint_entropy_gap_nats'] = wt[4][4][1,0]-wt[4][4][0,0]
            result['witness_joint_collision_gap'] = wt[4][5][0,0]-wt[4][5][1,0]
        return {key: float(value) for key,value in result.items()}


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


class Model(ReferenceModel):
    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        self.capacity_mode, self.pairing = 'reference', self.config['pairing']
        if self.pairing != 'fold':
            raise ValueError('This registered reference is the fold control')
        self.first_hidden = self.config['hidden'][0]
        self.reference_active_nonstatic = sum(p.numel() for p in self.parameters()) - sum(e.weight.numel() for e in self.embeddings) - sum(p.numel() for p in self.attention.parameters())

    def _nonidentity_features(self, batch):
        return self._shared_features(batch)[0]

    def forward(self, batch):
        return super().forward(self._pair_batch(batch))

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
