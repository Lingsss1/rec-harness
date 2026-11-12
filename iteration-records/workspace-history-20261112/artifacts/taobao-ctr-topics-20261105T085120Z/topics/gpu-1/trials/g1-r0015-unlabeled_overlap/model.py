"""Identity-context contingency on the supplied causal history.

Retains the complete joint-identity incumbent. A second zero-initialized,
bias-free residual reads empirical category-brand versus action-age coupling.
Product-marginal, common-marginal and zero-input cells share all allocations
and statistic computations. Category-brand bucket pairs are not item IDs.
Torch execution, training and measured performance remain host-pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'moment_grouping': 'match', 'moment_pairing': 'independent', 'field_mode': 'contrast', 'field_grouping': 'rank', 'field_scope': 'global', 'field_layout': 'compact', 'global_direction': 'norm_collapsed', 'identity_mode': 'joint', 'context_mode': 'joint', 'repeat_mode': 'joint', 'overlap_mode': 'unlabeled'}


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
        # Allocate only AFTER every incumbent parameter. Each registered cell
        # computes identical empirical and product alternatives before readout.
        self.context_mode = self.config['context_mode']
        if self.context_mode not in ('joint', 'independent', 'marginal', 'zero'):
            raise ValueError('Unknown context_mode')
        if self.identity_mode != 'joint':
            raise ValueError('This study retains the joint-identity incumbent')
        if self.category_vocab * self.brand_vocab > (2**63 - 1) // 64:
            raise ValueError('Packed identity-context key would overflow int64')
        self.context_head = nn.Sequential(
            nn.Linear(28, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.context_head[-1].weight)

        # Append after EVERY parent parameter so old shapes, allocation order
        # and scratch initialization remain identical. Modes allocate equally.
        self.repeat_mode = self.config['repeat_mode']
        if self.repeat_mode not in ('joint', 'independent', 'unlabeled', 'zero'):
            raise ValueError('Unknown repeat_mode')
        if self.context_mode != 'joint':
            raise ValueError('This study retains coupled_context exactly')
        self.repeat_head = nn.Sequential(
            nn.Linear(192, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.repeat_head[-1].weight)

        # Allocate only after ALL parent parameters; every mode is identical.
        self.overlap_mode = self.config['overlap_mode']
        if self.overlap_mode not in ('observed', 'product', 'unlabeled', 'zero'):
            raise ValueError('Unknown overlap_mode')
        if self.repeat_mode != 'joint':
            raise ValueError('Retain the localized-repeat parent in every cell')
        self.overlap_head = nn.Sequential(
            nn.Linear(480, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.overlap_head[-1].weight)
        # Fixed upper-triangle readout: four groups times C(16,2) = 480.
        self.register_buffer('overlap_rows', torch.tensor(
            [i for i in range(16) for j in range(i + 1, 16)], dtype=torch.long), persistent=False)
        self.register_buffer('overlap_cols', torch.tensor(
            [j for i in range(16) for j in range(i + 1, 16)], dtype=torch.long), persistent=False)

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
        tables = self._identity_tables(batch, semantic)
        context = self._context_tables(batch, tables)
        parent = self.output(parent_features).squeeze(-1)
        identity = self.identity_head(self._identity_features(tables)).squeeze(-1)
        extra = self.context_head(self._context_features(context)).squeeze(-1)
        repeat_tables = self._conditional_tables(batch, tables, context)
        repeat = self.repeat_head(self._repeat_features(repeat_tables)).squeeze(-1)
        overlap_tables = self._overlap_tables(batch, tables, context)
        overlap = self.overlap_head(self._overlap_features(overlap_tables)).squeeze(-1)
        return parent + identity + extra + repeat + overlap

    def _context_tables(self, batch, identity_tables=None):
        """X=category-brand bucket pair, Y=(action, coarse age), given group.

        All summaries share EXACTLY the parent's complete-pair support. Reuse
        its X counts; one extra BxL sort counts (X,Y), and a Bx4x16 scatter
        counts Y. No learned embedding, BxLxL or vocabulary table in forward.
        Product entropy/collision are exact for the empirical product, including
        all self combinations; entropy is not a bias-corrected population MI.
        """
        if identity_tables is None:
            identity_tables = self._identity_tables(batch)
        _, _, _, n, raw, eligible, group = identity_tables
        hx, cx = raw[4], raw[5]
        action = (batch['history_type'] - 1).clamp(0, 3)
        age = batch['history_age_seconds']
        age_bin = ((age >= 3600.0).long() + (age >= 86400.0).long() +
                   (age >= 604800.0).long())
        context = 4 * action + age_bin
        context_mass = n.new_zeros((n.shape[0], 64)).scatter_add(
            1, 16 * group + context, eligible.float()).reshape(n.shape[0], 4, 16)
        denominator = n.clamp_min(1)
        sum_xlogx = (context_mass * context_mass.clamp_min(1).log()).sum(-1)
        hy = torch.where(n > 0,
                         (denominator.log() - sum_xlogx / denominator).clamp_min(0),
                         torch.zeros_like(n))
        cy = context_mass.square().sum(-1) / denominator.square()
        identity = batch['history_category'] * self.brand_vocab + batch['history_brand']
        hxy, cxy = self._sorted_distribution(16 * identity + context, group, eligible, n)
        scale = math.log(129.0)
        common = torch.stack([torch.log1p(n) / scale, hx / scale, hy / scale, cx, cy], -1)
        joint = torch.stack([hxy / scale, cxy], -1)
        product = torch.stack([(hx + hy) / scale, cx * cy], -1)
        return common, joint, product, n, (hx, cx, hy, cy, hxy, cxy), context, context_mass

    def _context_features(self, tables, mode=None):
        mode = self.context_mode if mode is None else mode
        common, joint, product = tables[:3]
        selected = joint if mode == 'joint' else product
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

    def _conditional_tables(self, batch, ids=None, context_tables=None):
        """Conditional X distribution in every labeled context Y.

        One extra BxL sort; no BxLxL training tensor and no vocabulary array.
        Run counts provide sum_x m_xy log(m_xy) and sum_x m_xy^2 per (g,y).
        The existing parent supplies the exact same complete-pair support and
        context masses. Product replacement is p_g(X)p_g(Y), including self
        combinations, NOT a shuffle or population independence estimate.
        """
        ids = self._identity_tables(batch) if ids is None else ids
        context_tables = (self._context_tables(batch, ids) if context_tables is None
                          else context_tables)
        eligible, group = ids[5], ids[6]
        context, context_mass = context_tables[5], context_tables[6]
        batch_size, length = eligible.shape
        # All intermediates are at most BxL or Bx64; mass is float32.
        squared = context_mass.new_zeros((batch_size, 64))
        xlogx = torch.zeros_like(squared)
        counted = torch.zeros_like(squared)
        if length:
            identity = (batch['history_category'] * self.brand_vocab +
                        batch['history_brand'])
            # Same safe packing bound as inherited joint-context statistics.
            packed = 64 * identity + 4 * context + group
            key = torch.where(eligible, packed, torch.full_like(packed, -1))
            ordered = key.sort(dim=-1).values
            valid = ordered >= 0
            start = valid.clone()
            start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
            run = (start.long().cumsum(-1) - 1).clamp_min(0)
            mass = context_mass.new_zeros((batch_size, length)).scatter_add(
                1, run, valid.float())
            # Packing suffix is 4*y+g; destination is the labeled 16*g+y.
            cell = 16 * ordered.remainder(4) + ordered.div(4, rounding_mode='floor').remainder(16)
            run_cell = torch.zeros_like(ordered).scatter_add(
                1, run, torch.where(start, cell, torch.zeros_like(cell)))
            squared = squared.scatter_add(1, run_cell, mass.square())
            xlogx = xlogx.scatter_add(1, run_cell, mass * mass.clamp_min(1).log())
            counted = counted.scatter_add(1, run_cell, mass)
        squared = squared.reshape(batch_size, 4, 16)
        xlogx = xlogx.reshape(batch_size, 4, 16)
        counted = counted.reshape(batch_size, 4, 16)
        denom = context_mass.clamp_min(1)
        occupied = context_mass > 0
        entropy = torch.where(occupied, (denom.log() - xlogx / denom).clamp_min(0),
                              torch.zeros_like(context_mass))
        collision = squared / denom.square()
        count_feature = torch.log1p(context_mass) / math.log(129.0)
        observed = torch.stack([count_feature, entropy / math.log(129.0), collision], -1)
        hx, cx = ids[4][4], ids[4][5]
        independent_h = hx.unsqueeze(-1).expand_as(context_mass) * occupied.float()
        independent_c = cx.unsqueeze(-1).expand_as(context_mass) * occupied.float()
        product = torch.stack([count_feature, independent_h / math.log(129.0), independent_c], -1)
        # Canonicalize complete tuples, NOT coordinates independently. Stable
        # least-significant-first sorts implement lexicographic (n,H,C) order.
        # Ties have identical full tuples, so no context index can leak via ties.
        unlabeled = observed
        for coordinate in (2, 1, 0):
            order = unlabeled[..., coordinate].argsort(dim=2, stable=True)
            unlabeled = unlabeled.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 3))
        return observed, product, unlabeled, context_mass, entropy, collision, counted

    def _repeat_features(self, tables, mode=None):
        mode = self.repeat_mode if mode is None else mode
        # No mode-dependent statistic calculations or trainable allocations.
        selected = tables[0] if mode == 'joint' else tables[1]
        if mode == 'unlabeled':
            selected = tables[2]
        features = selected.flatten(1)
        return torch.zeros_like(features) if mode == 'zero' else features

    def _dense_conditional(self, batch, ids, context_tables):
        """Independent O(B L^2) count oracle for host diagnostics ONLY."""
        eligible, group = ids[5], ids[6]
        context, mass = context_tables[5], context_tables[6]
        c, b = batch['history_category'], batch['history_brand']
        # Compare individual identities, not packed keys, as an independent
        # guard against packing or run-destination errors.
        same = ((c[:, :, None] == c[:, None, :]) &
                (b[:, :, None] == b[:, None, :]) &
                (group[:, :, None] == group[:, None, :]) &
                (context[:, :, None] == context[:, None, :]) &
                eligible[:, :, None] & eligible[:, None, :])
        per_event = same.sum(-1).float()
        cell = 16 * group + context
        squares = mass.new_zeros((mass.shape[0], 64)).scatter_add(
            1, cell, per_event * eligible.float()).reshape_as(mass)
        logs = mass.new_zeros((mass.shape[0], 64)).scatter_add(
            1, cell, per_event.clamp_min(1).log() * eligible.float()).reshape_as(mass)
        denom = mass.clamp_min(1)
        entropy = torch.where(mass > 0, (denom.log() - logs / denom).clamp_min(0),
                              torch.zeros_like(mass))
        return entropy, squares / denom.square()

    def _overlap_tables(self, batch, ids=None, context_tables=None):
        """K_g(y,z) = sum_x p_g(x|y) p_g(x|z), including diagonals.

        Only upper OFF-diagonal entries enter the new head. The inherited
        repeat branch already supplies diagonal collisions. One BxL sort and
        BxLx16 run/context counts suffice; no BxLxL training intermediate and
        no array indexed by the whole identity vocabulary are constructed.
        All modes compute every alternative before feature selection.
        """
        ids = self._identity_tables(batch) if ids is None else ids
        context_tables = (self._context_tables(batch, ids) if context_tables is None
                          else context_tables)
        eligible, group = ids[5], ids[6]
        context, context_mass = context_tables[5], context_tables[6]
        batch_size, length = eligible.shape
        raw = context_mass.new_zeros((batch_size, 4, 16, 16))
        counted = torch.zeros_like(context_mass)
        if length:
            identity = (batch['history_category'] * self.brand_vocab +
                        batch['history_brand'])
            packed = 4 * identity + group
            key = torch.where(eligible, packed, torch.full_like(packed, -1))
            ordered, order = key.sort(dim=-1)
            valid = ordered >= 0
            start = valid.clone()
            start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
            run = (start.long().cumsum(-1) - 1).clamp_min(0)
            sorted_context = context.gather(1, order)
            counts = context_mass.new_zeros((batch_size, length * 16)).scatter_add(
                1, 16 * run + sorted_context, valid.float())
            counts = counts.reshape(batch_size, length, 16)
            run_group = torch.zeros_like(ordered).scatter_add(
                1, run, torch.where(start, ordered.remainder(4), torch.zeros_like(ordered)))
            grams, masses = [], []
            for g in range(4):
                selected = counts * (run_group == g).unsqueeze(-1).float()
                grams.append(torch.bmm(counts.transpose(1, 2), selected))
                masses.append(selected.sum(1))
            raw = torch.stack(grams, dim=1)
            counted = torch.stack(masses, dim=1)
        denominator = context_mass.unsqueeze(-1) * context_mass.unsqueeze(-2)
        gram = raw / denominator.clamp_min(1)
        occupied = context_mass > 0
        pair_present = occupied.unsqueeze(-1) & occupied.unsqueeze(-2)
        # Analytic empirical product p_g(X)p_g(Y): every occupied conditional
        # becomes p_g(X), so its inner product is the group collision C_X(g).
        # This replaces ONLY new off-diagonals. It need not be jointly
        # realizable with the retained observed parent diagonals.
        product = ids[4][5].unsqueeze(-1).unsqueeze(-1) * pair_present.float()
        observed = gram[:, :, self.overlap_rows, self.overlap_cols]
        replacement = product[:, :, self.overlap_rows, self.overlap_cols]
        # A context permutation induces an edge permutation. Sorting all
        # scalar edge values removes their labels but retains their multiset.
        unlabeled = observed.sort(dim=-1).values
        return observed, replacement, unlabeled, gram, raw, counted, context_mass

    def _overlap_features(self, tables, mode=None):
        mode = self.overlap_mode if mode is None else mode
        selected = tables[0]
        if mode == 'product':
            selected = tables[1]
        elif mode == 'unlabeled':
            selected = tables[2]
        features = selected.flatten(1)
        return torch.zeros_like(features) if mode == 'zero' else features

    def _dense_overlap(self, batch):
        """Independent O(B L^2) equality/count oracle, diagnostics ONLY.

        Recompute groups and context bins independently, compare separate
        category and brand tensors, and count ordered pairs directly. The host
        uses only the first 32 of its diagnostic impressions for this oracle.
        """
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        tc = batch['static'][:, self.category:self.category + 1]
        tb = batch['static'][:, self.brand:self.brand + 1]
        cm = (c == tc) & (tc > 0) & (c > 0)
        bm = (b == tb) & (tb > 0) & (b > 0)
        group = cm.long() + 2 * bm.long()
        age = batch['history_age_seconds']
        age_bin = torch.where(age < 3600, 0, torch.where(
            age < 86400, 1, torch.where(age < 604800, 2, 3)))
        context = 4 * (batch['history_type'] - 1).clamp(0, 3) + age_bin
        same = ((c[:, :, None] == c[:, None, :]) &
                (b[:, :, None] == b[:, None, :]) &
                (group[:, :, None] == group[:, None, :]) &
                eligible[:, :, None] & eligible[:, None, :])
        destination = (256 * group[:, :, None] + 16 * context[:, :, None] +
                       context[:, None, :])
        raw = age.new_zeros((age.shape[0], 1024)).scatter_add(
            1, destination.flatten(1), same.float().flatten(1))
        return raw.reshape(age.shape[0], 4, 16, 16)

    def diagnostics(self, batch):
        # Candidate-supplied algebra/support measurements; not independent
        # validation performance or a population-level mechanism estimate.
        ids = self._identity_tables(batch)
        contexts = self._context_tables(batch, ids)
        conditional = self._conditional_tables(batch, ids, contexts)
        tables = self._overlap_tables(batch, ids, contexts)
        observed, product, unlabeled, gram, raw, counted, mass = tables
        features = self._overlap_features(tables)
        residual = self.overlap_head(features).squeeze(-1)
        probe_size = min(32, batch['static'].shape[0])
        probe = {key: value[:probe_size] for key, value in batch.items()}
        occupied = mass > 0
        pair_present = occupied.unsqueeze(-1) & occupied.unsqueeze(-2)
        off_present = pair_present[:, :, self.overlap_rows, self.overlap_cols]
        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        norm = ids[3].clamp_min(1).square()
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'overlap_head_parameters': float(sum(p.numel() for p in self.overlap_head.parameters())),
            'overlap_input_coordinates': float(features.shape[1]),
            'observed_overlap_enabled': float(self.overlap_mode == 'observed'),
            'product_overlap_enabled': float(self.overlap_mode == 'product'),
            'unlabeled_overlap_enabled': float(self.overlap_mode == 'unlabeled'),
            'zero_overlap_enabled': float(self.overlap_mode == 'zero'),
            'dense_oracle_impressions': float(probe_size),
            'sorted_vs_dense_raw_error_max': (raw[:probe_size] - self._dense_overlap(probe)).abs().max(),
            'context_count_conservation_error_max': (counted - mass).abs().max(),
            'diagonal_vs_parent_collision_error_max': (diagonal - conditional[5]).abs().max(),
            'global_collision_reconstruction_error_max': (raw.sum(dim=(-1, -2)) / norm - ids[4][5]).abs().max(),
            'joint_context_collision_reconstruction_error_max': (raw.diagonal(dim1=-2, dim2=-1).sum(-1) / norm - contexts[4][5]).abs().max(),
            'gram_symmetry_error_max': (gram - gram.transpose(-1, -2)).abs().max(),
            'gram_nonnegative_violation_max': (-gram).clamp_min(0).max(),
            'gram_unit_bound_violation_max': (gram - 1).clamp_min(0).max(),
            'gram_cauchy_schwarz_violation_max': (gram.square() - diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)).clamp_min(0).max(),
            'absent_context_pair_overlap_abs_max': (gram * (~pair_present).float()).abs().max(),
            'complete_pair_count_mean': mass.sum(dim=(1, 2)).mean(),
            'occupied_context_pairs_per_impression_mean': off_present.float().sum(dim=(1, 2)).mean(),
            'nonzero_shared_identity_edges_per_impression_mean': ((observed > 0) & off_present).float().sum(dim=(1, 2)).mean(),
            'observed_vs_product_gap_on_occupied_edges_mean': ((observed - product).abs() * off_present.float()).sum() / off_present.float().sum().clamp_min(1),
            'same_checkpoint_overlap_residual_rms': residual.square().mean().sqrt(),
            'overlap_head_last_weight_norm': torch.linalg.vector_norm(self.overlap_head[-1].weight),
            'zero_input_added_residual_abs_max': self.overlap_head(torch.zeros_like(features)).abs().max(),
        }
        reverse = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[key] = batch[key].flip(-1)
        result['sequence_reversal_feature_error_max'] = (features - self._overlap_features(self._overlap_tables(reverse))).abs().max()
        renamed = dict(batch)
        renamed['static'] = batch['static'].clone()
        for key, field, vocab in (('history_category', self.category, self.category_vocab),
                                  ('history_brand', self.brand, self.brand_vocab)):
            renamed[key] = torch.where(batch[key] > 0, vocab - batch[key], batch[key])
            value = batch['static'][:, field]
            renamed['static'][:, field] = torch.where(value > 0, vocab - value, value)
        result['identity_renaming_feature_error_max'] = (features - self._overlap_features(self._overlap_tables(renamed))).abs().max()
        permuted = dict(batch)
        types = batch['history_type']
        permuted['history_type'] = torch.where(types > 0, 5 - types, types)
        permuted_tables = self._overlap_tables(permuted)
        result['context_permutation_unlabeled_error_max'] = (unlabeled - permuted_tables[2]).abs().max()
        result['context_permutation_observed_delta_max'] = (observed - permuted_tables[0]).abs().max()
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        missing = dict(batch)
        missing['history_brand'] = torch.zeros_like(batch['history_brand'])
        zero_length = dict(batch)
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero_length[key] = batch[key][:, :0]
            fill = batch[key].new_zeros((batch[key].shape[0], 2))
            if key != 'history_mask':
                fill.fill_(1)
            padded[key] = torch.cat([batch[key], fill], -1)
        empty_features = self._overlap_features(self._overlap_tables(empty))
        result['all_empty_feature_abs_max'] = empty_features.abs().max()
        result['all_empty_added_residual_abs_max'] = self.overlap_head(empty_features).abs().max()
        result['zero_length_feature_abs_max'] = self._overlap_features(self._overlap_tables(zero_length)).abs().max()
        result['missing_pair_feature_abs_max'] = self._overlap_features(self._overlap_tables(missing)).abs().max()
        result['masked_padding_feature_error_max'] = (features - self._overlap_features(self._overlap_tables(padded))).abs().max()
        # Block vs cycle incidence: four identities of frequency two, four
        # equally sized contexts, and identical identity/context cell masses.
        # Every parent feature is equal for arbitrary embeddings. Only which
        # context PAIRS share identities changes. No bucket ordering enters.
        applicable = self.category_vocab > 5 and self.brand_vocab > 5
        result['overlap_witness_applicable'] = float(applicable)
        if applicable:
            device = batch['static'].device
            witness = {
                'static': batch['static'][:1].expand(3, -1).clone(),
                'history_category': torch.tensor([[1,1,2,2,3,3,4,4]] * 3, device=device),
                'history_brand': torch.tensor([[1,1,2,2,3,3,4,4]] * 3, device=device),
                'history_type': torch.tensor([[1,2,1,2,3,4,3,4], [1,2,2,3,3,4,4,1], [1,3,1,3,2,4,2,4]], device=device),
                'history_age_seconds': torch.zeros((3, 8), device=device),
                'history_mask': torch.ones((3, 8), dtype=torch.bool, device=device),
            }
            witness['static'][:, self.category] = 5
            witness['static'][:, self.brand] = 5
            wp = self._shared_features(witness)[0]
            wi = self._identity_tables(witness)
            wc = self._context_tables(witness, wi)
            wr = self._conditional_tables(witness, wi, wc)
            wt = self._overlap_tables(witness, wi, wc)
            result['witness_parent_base_feature_error_max'] = (wp - wp[:1]).abs().max()
            result['witness_parent_identity_feature_error_max'] = (self._identity_features(wi) - self._identity_features(wi)[:1]).abs().max()
            result['witness_parent_context_feature_error_max'] = (self._context_features(wc) - self._context_features(wc)[:1]).abs().max()
            result['witness_parent_repeat_feature_error_max'] = (self._repeat_features(wr) - self._repeat_features(wr)[:1]).abs().max()
            result['witness_product_feature_error_max'] = (wt[1] - wt[1][:1]).abs().max()
            result['witness_observed_overlap_gap_max'] = (wt[0][0] - wt[0][1]).abs().max()
            result['witness_unlabeled_overlap_gap_max'] = (wt[2][0] - wt[2][1]).abs().max()
            # Third history relabels the first block graph: its overlap shape
            # is identical, while which context pairs share identities changes.
            result['witness_relabel_observed_gap_max'] = (wt[0][0] - wt[0][2]).abs().max()
            result['witness_relabel_unlabeled_gap_max'] = (wt[2][0] - wt[2][2]).abs().max()
            added = self.overlap_head(self._overlap_features(wt)).squeeze(-1)
            result['same_checkpoint_witness_added_logit_gap'] = (added[0] - added[1]).abs()
            result['same_checkpoint_relabel_witness_added_logit_gap'] = (added[0] - added[2]).abs()
        return {key: float(value) for key, value in result.items()}

