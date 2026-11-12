# Standalone pooled execution and gap-head compaction study.
# Ancestor implementations below are retained for provenance and controls;
# the final Model class defines the registered experiment.
# Standalone resolution-factorial module. The original gap-parent source
# below is preserved (class renamed); the final Model supplies the new readout.
"""Elapsed-time localization of adjacent category/brand recurrence.

The exact chronological-product parent is retained as _ParentModel. A new
zero-initialized residual reads gap-conditioned transition counts. The 2x2
study replaces gap/recurrence association and category/brand joint equality
separately. All modes execute the same tables and allocate equal parameters.
Histories are host-supplied causal prefixes; no items or extra events exist.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'moment_grouping': 'match', 'moment_pairing': 'independent', 'field_mode': 'contrast', 'field_grouping': 'rank', 'field_scope': 'global', 'field_layout': 'compact', 'global_direction': 'norm_collapsed', 'identity_mode': 'joint', 'context_mode': 'joint', 'repeat_mode': 'joint', 'overlap_mode': 'observed', 'transition_order': 'chronological', 'transition_pairing': 'product', 'gap_mode': 'observed', 'gap_pairing': 'product', 'gap_resolution': 'global', 'gap_channels': 'all', 'product_scope': 'pooled', 'gap_association': 'observed', 'gap_execution': 'direct', 'gap_parameterization': 'lifted'}


class _ParentModel(nn.Module):
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

        # All parent allocations and initialization precede this append.
        self.transition_order = self.config['transition_order']
        self.transition_pairing = self.config['transition_pairing']
        if self.transition_order not in ('chronological', 'exchangeable'):
            raise ValueError('Unknown transition_order')
        if self.transition_pairing not in ('observed', 'product'):
            raise ValueError('Unknown transition_pairing')
        if self.overlap_mode != 'observed':
            raise ValueError('Retain the entire observed-overlap parent')
        self.transition_head = nn.Sequential(
            nn.Linear(1024, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.transition_head[-1].weight)

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

    def _parent_forward(self, batch):
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

    def _parent_diagnostics(self, batch):
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

    def _transition_support(self, batch):
        """Singleton timestamps on complete-pair support, oldest first.

        Remove every member of each time tie, never choose an ordering by
        input index or bucket magnitude. Reorder input events with their ages
        and the result is unchanged. Filtering can bridge excluded events;
        adjacency is explicitly on this supported subsequence, not raw logs.
        All intermediates here are BxL. Ages must obey the host contract >=0.
        """
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        complete = batch['history_mask'] & (c > 0) & (b > 0)
        batch_size, length = complete.shape
        if length == 0:
            order = torch.empty_like(c)
            return complete, order, complete.long().sum(-1), complete
        ordered_age, order = torch.where(
            complete, age, torch.full_like(age, -1.0)).sort(dim=-1, descending=True)
        valid = complete.gather(1, order)
        # Adjacent-equality in sorted times identifies every tied member.
        tied = torch.zeros_like(valid)
        equal = (valid[:, 1:] & valid[:, :-1] &
                 (ordered_age[:, 1:] == ordered_age[:, :-1]))
        tied[:, 1:] |= equal
        tied[:, :-1] |= equal
        singleton = valid & ~tied
        eligible = torch.zeros_like(complete).scatter(1, order, singleton)
        # Compact singleton events while retaining their chronological order.
        position = torch.arange(length, device=c.device)[None, :].expand(batch_size, -1)
        compact = torch.where(singleton, position, position + length).argsort(dim=-1)
        order = order.gather(1, compact)
        return eligible, order, eligible.long().sum(-1), complete

    @staticmethod
    def _exchangeable_equal(identity, cell, eligible, mass, n):
        """Expected equal-identity adjacent counts under a uniform permutation.

        For a distinct ordered pair of events, adjacency probability is 1/n.
        Sum identity-cell histogram outer products, remove self-event terms,
        divide by n. Self pairs are excluded even when source cell equals
        destination cell. This is an exact finite empirical expectation.
        Sort BL keys and use BL16 storage, never BL squared or vocabulary size.
        """
        batch_size, length = identity.shape
        if length == 0:
            return mass.new_zeros((batch_size, 16, 16))
        key = torch.where(eligible, 16 * identity + cell,
                          torch.full_like(identity, -1))
        ordered, order = key.sort(dim=-1)
        valid = ordered >= 0
        identity_key = ordered.div(16, rounding_mode='floor')
        start = valid.clone()
        start[:, 1:] = valid[:, 1:] & (identity_key[:, 1:] != identity_key[:, :-1])
        run = (start.long().cumsum(-1) - 1).clamp_min(0)
        sorted_cell = cell.gather(1, order)
        counts = mass.new_zeros((batch_size, length * 16)).scatter_add(
            1, 16 * run + sorted_cell, valid.float()).reshape(batch_size, length, 16)
        gram = torch.bmm(counts.transpose(1, 2), counts)
        return (gram - torch.diag_embed(mass)).clamp_min(0) / n.clamp_min(1)[:, None, None]

    def _transition_tables(self, batch):
        """Four raw count tables for chronology and exchangeable chronology.

        Each tuple is (total edges, category-equal, brand-equal, pair-equal).
        u=4*match_group+action-1. Parent support and outputs are unmodified.
        Every variant computes both tuples and both product replacements.
        """
        eligible, order, n_int, complete = self._transition_support(batch)
        c, b = batch['history_category'], batch['history_brand']
        group = self._membership(batch)[1].argmax(dim=1)
        cell = 4 * group + (batch['history_type'] - 1).clamp(0, 3)
        batch_size, length = c.shape
        n = batch['history_age_seconds'].new_zeros((batch_size,)) + n_int.float()
        mass = n.new_zeros((batch_size, 16)).scatter_add(1, cell, eligible.float())
        temporal = []
        if length < 2:
            temporal = [n.new_zeros((batch_size, 16, 16)) for _ in range(4)]
        else:
            valid = eligible.gather(1, order)
            ordered_cell = cell.gather(1, order)
            ordered_c, ordered_b = c.gather(1, order), b.gather(1, order)
            edge = valid[:, :-1] & valid[:, 1:]
            same_c = ordered_c[:, :-1] == ordered_c[:, 1:]
            same_b = ordered_b[:, :-1] == ordered_b[:, 1:]
            destination = 16 * ordered_cell[:, :-1] + ordered_cell[:, 1:]
            for match in (torch.ones_like(edge), same_c, same_b, same_c & same_b):
                counts = n.new_zeros((batch_size, 256)).scatter_add(
                    1, destination, (edge & match).float())
                temporal.append(counts.reshape(batch_size, 16, 16))
        total = (mass.unsqueeze(-1) * mass.unsqueeze(-2) -
                 torch.diag_embed(mass)).clamp_min(0) / n.clamp_min(1)[:, None, None]
        exchangeable = [total]
        for identity in (c, b, c * self.brand_vocab + b):
            exchangeable.append(self._exchangeable_equal(identity, cell, eligible, mass, n))
        temporal_product = temporal[1] * temporal[2] / temporal[0].clamp_min(1)
        # Fractional exchangeable masses can be <1: use the positive mass,
        # not clamp_min(1), for an exact conditional product replacement.
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        exchangeable_product = exchangeable[1] * exchangeable[2] / denominator
        return tuple(temporal), tuple(exchangeable), temporal_product, exchangeable_product, n, complete, eligible

    def _transition_features(self, tables, order_mode=None, pairing=None):
        order_mode = self.transition_order if order_mode is None else order_mode
        pairing = self.transition_pairing if pairing is None else pairing
        observed = tables[0] if order_mode == 'chronological' else tables[1]
        replacement = tables[2] if order_mode == 'chronological' else tables[3]
        last = observed[3] if pairing == 'observed' else replacement
        counts = torch.stack([observed[0], observed[1], observed[2], last], dim=1)
        return torch.log1p(counts.flatten(1)) / math.log(129.0)

    def forward(self, batch):
        parent = self._parent_forward(batch)
        tables = self._transition_tables(batch)
        residual = self.transition_head(self._transition_features(tables)).squeeze(-1)
        return parent + residual

    def _dense_transition_tables(self, batch):
        """Independent quadratic event-pair oracle, DIAGNOSTICS ONLY.

        Support, temporal ranks and adjacent pairs are defined directly with
        equality/comparison matrices, not the production sorting algorithm.
        Run on at most16 impressions; never called during forward/training.
        """
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        complete = batch['history_mask'] & (c > 0) & (b > 0)
        tie_count = ((age[:, :, None] == age[:, None, :]) &
                     complete[:, :, None] & complete[:, None, :]).sum(-1)
        eligible = complete & (tie_count == 1)
        rank = ((age[:, None, :] > age[:, :, None]) & eligible[:, None, :]).sum(-1)
        # Oldest rank0 -> next newer rank1.
        adjacent = rank[:, None, :] == rank[:, :, None] + 1
        pairs = eligible[:, :, None] & eligible[:, None, :]
        distinct = age[:, :, None] != age[:, None, :]
        tc = batch['static'][:, self.category:self.category + 1]
        tb = batch['static'][:, self.brand:self.brand + 1]
        cm = (c > 0) & (tc > 0) & (c == tc)
        bm = (b > 0) & (tb > 0) & (b == tb)
        group = cm.long() + 2 * bm.long()
        cell = 4 * group + (batch['history_type'] - 1).clamp(0, 3)
        destination = (16 * cell[:, :, None] + cell[:, None, :]).flatten(1)
        same_c, same_b = c[:, :, None] == c[:, None, :], b[:, :, None] == b[:, None, :]
        n = eligible.float().sum(-1)
        temporal, exchangeable = [], []
        for match in (torch.ones_like(pairs), same_c, same_b, same_c & same_b):
            temporal.append(age.new_zeros((age.shape[0], 256)).scatter_add(
                1, destination, (pairs & adjacent & match).float().flatten(1)).reshape(-1, 16, 16))
            exchangeable.append(age.new_zeros((age.shape[0], 256)).scatter_add(
                1, destination, (pairs & distinct & match).float().flatten(1)).reshape(-1, 16, 16) /
                n.clamp_min(1)[:, None, None])
        return temporal, exchangeable, n

    def diagnostics(self, batch):
        """At most64 finite scalars; measurements are candidate-supplied."""
        tables = self._transition_tables(batch)
        temporal, exchangeable, tp, ep, n, complete, eligible = tables
        features = self._transition_features(tables)
        residual = self.transition_head(features).squeeze(-1)
        probe_size = min(16, batch['static'].shape[0])
        probe = {key: value[:probe_size] for key, value in batch.items()}
        oracle_t, oracle_e, oracle_n = self._dense_transition_tables(probe)
        target_edges = (n - 1).clamp_min(0)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'transition_head_parameters': float(sum(p.numel() for p in self.transition_head.parameters())),
            'transition_input_coordinates': float(features.shape[1]),
            'chronological_enabled': float(self.transition_order == 'chronological'),
            'observed_pairing_enabled': float(self.transition_pairing == 'observed'),
            'dense_oracle_impressions': float(probe_size),
            'complete_events_mean': complete.float().sum(-1).mean(),
            'singleton_timestamp_events_mean': n.mean(),
            'excluded_tied_events_mean': (complete.float().sum(-1) - n).mean(),
            'supported_history_fraction': (n >= 2).float().mean(),
            'chronological_edge_mass_error_max': (temporal[0].sum(dim=(-1, -2)) - target_edges).abs().max(),
            'exchangeable_edge_mass_error_max': (exchangeable[0].sum(dim=(-1, -2)) - target_edges).abs().max(),
            'dense_support_error_max': (n[:probe_size] - oracle_n).abs().max(),
            'chronological_vs_exchangeable_total_gap_mean': (temporal[0] - exchangeable[0]).abs().mean(),
            'chronological_vs_product_pair_gap_mean': (temporal[3] - tp).abs().mean(),
            'exchangeable_vs_product_pair_gap_mean': (exchangeable[3] - ep).abs().mean(),
            'chronological_repeated_pairs_mean': temporal[3].sum(dim=(-1, -2)).mean(),
            'exchangeable_repeated_pairs_mean': exchangeable[3].sum(dim=(-1, -2)).mean(),
            'transition_residual_rms': residual.square().mean().sqrt(),
            'transition_last_weight_norm': torch.linalg.vector_norm(self.transition_head[-1].weight),
            'zero_input_added_residual_abs_max': self.transition_head(torch.zeros_like(features)).abs().max(),
        }
        for label, raw, oracle in (('chronological', temporal, oracle_t),
                                   ('exchangeable', exchangeable, oracle_e)):
            result[label + '_dense_error_max'] = torch.stack([
                (x[:probe_size] - y).abs().max() for x, y in zip(raw, oracle)]).max()
            result[label + '_nonnegative_violation_max'] = torch.stack([(-x).clamp_min(0).max() for x in raw]).max()
            result[label + '_subset_violation_max'] = torch.stack([
                (raw[1] - raw[0]).clamp_min(0).max(),
                (raw[2] - raw[0]).clamp_min(0).max(),
                (raw[3] - raw[1]).clamp_min(0).max(),
                (raw[3] - raw[2]).clamp_min(0).max()]).max()
        reversed_batch = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reversed_batch[key] = batch[key].flip(-1)
        result['input_reordering_feature_error_max'] = (features - self._transition_features(
            self._transition_tables(reversed_batch))).abs().max()
        renamed = dict(batch)
        renamed['static'] = batch['static'].clone()
        for key, field, vocab in (('history_category', self.category, self.category_vocab),
                                  ('history_brand', self.brand, self.brand_vocab)):
            renamed[key] = torch.where(batch[key] > 0, vocab - batch[key], batch[key])
            value = batch['static'][:, field]
            renamed['static'][:, field] = torch.where(value > 0, vocab - value, value)
        result['identity_renaming_feature_error_max'] = (features - self._transition_features(
            self._transition_tables(renamed))).abs().max()
        empty, missing, tied = dict(batch), dict(batch), dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        missing['history_brand'] = torch.zeros_like(batch['history_brand'])
        tied['history_age_seconds'] = torch.zeros_like(batch['history_age_seconds'])
        for label, altered in (('empty', empty), ('missing', missing), ('fully_tied', tied)):
            value = self._transition_features(self._transition_tables(altered))
            result[label + '_feature_abs_max'] = value.abs().max()
        zero_length, padded = dict(batch), dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero_length[key] = batch[key][:, :0]
            fill = batch[key].new_zeros((batch[key].shape[0], 2))
            if key != 'history_mask':
                fill.fill_(1)
            padded[key] = torch.cat([batch[key], fill], -1)
        result['zero_length_feature_abs_max'] = self._transition_features(self._transition_tables(zero_length)).abs().max()
        result['masked_padding_feature_error_max'] = (features - self._transition_features(self._transition_tables(padded))).abs().max()
        applicable = self.category_vocab > 3 and self.brand_vocab > 3
        result['pairing_witness_applicable'] = float(applicable)
        if applicable:
            device = batch['static'].device
            # Same pair multiset, actions, group and both age bins; C,B edge
            # counts agree (6,4), pair-equal edges differ (4 versus3).
            witness = {
                'static': batch['static'][:1].expand(2, -1).clone(),
                'history_category': torch.tensor([[1,1,1,1,2,2,2,2], [1,1,1,1,2,2,2,2]], device=device),
                'history_brand': torch.tensor([[1,1,2,2,1,1,2,2], [1,1,2,2,2,1,1,2]], device=device),
                'history_type': torch.ones((2, 8), dtype=torch.long, device=device),
                'history_age_seconds': torch.arange(23, 15, -1, dtype=batch['history_age_seconds'].dtype, device=device)[None, :].expand(2, -1),
                'history_mask': torch.ones((2, 8), dtype=torch.bool, device=device),
            }
            witness['static'][:, self.category] = 3
            witness['static'][:, self.brand] = 3
            base = self._shared_features(witness)[0]
            wi = self._identity_tables(witness)
            wc = self._context_tables(witness, wi)
            wr = self._conditional_tables(witness, wi, wc)
            wo = self._overlap_tables(witness, wi, wc)
            wt = self._transition_tables(witness)
            result['witness_parent_base_error_max'] = (base - base[:1]).abs().max()
            for label, value in (('identity', self._identity_features(wi)),
                                  ('context', self._context_features(wc)),
                                  ('repeat', self._repeat_features(wr)),
                                  ('overlap', self._overlap_features(wo))):
                result['witness_parent_' + label + '_error_max'] = (value - value[:1]).abs().max()
            result['witness_chronological_common_error_max'] = torch.stack([
                (x - x[:1]).abs().max() for x in wt[0][:3]]).max()
            result['witness_chronological_pair_raw_gap_max'] = (wt[0][3][0] - wt[0][3][1]).abs().max()
            for order_mode, pairing, label in (
                    ('chronological', 'product', 'chronological_product'),
                    ('exchangeable', 'observed', 'exchangeable_joint'),
                    ('exchangeable', 'product', 'exchangeable_product')):
                value = self._transition_features(wt, order_mode, pairing)
                result['witness_' + label + '_error_max'] = (value - value[:1]).abs().max()
            wfeatures = self._transition_features(wt, 'chronological', 'observed')
            result['witness_chronological_joint_feature_gap_max'] = (wfeatures[0] - wfeatures[1]).abs().max()
            added = self.transition_head(self._transition_features(wt)).squeeze(-1)
            result['same_checkpoint_witness_added_logit_gap'] = (added[0] - added[1]).abs()
        return {key: float(value) for key, value in result.items()}



class _GapParentModel(_ParentModel):
    """Factorial replacement of elapsed-time and pair-equality associations."""

    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        if (self.transition_order, self.transition_pairing) != ('chronological', 'product'):
            raise ValueError('Keep the exact chronological-product parent')
        self.gap_mode = self.config['gap_mode']
        self.gap_pairing = self.config['gap_pairing']
        if self.gap_mode not in ('observed', 'independent'):
            raise ValueError('Unknown gap_mode')
        if self.gap_pairing not in ('observed', 'product'):
            raise ValueError('Unknown gap_pairing')
        # Append AFTER all parent parameters, preserving their seeded draws.
        # Four equality channels x four gap bins x 16x16 directed states.
        self.gap_head = nn.Sequential(
            nn.Linear(4096, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.gap_head[-1].weight)

    @staticmethod
    def _gap_bin(delta):
        # Fixed before execution: (0,60], (60,3600], (3600,86400], >86400 s.
        # Timestamp ties are excluded using the inherited singleton support.
        return ((delta > 60.0).long() + (delta > 3600.0).long() +
                (delta > 86400.0).long())

    @staticmethod
    def _positive_denominator(value):
        # Replacements have fractional mass: clamping to 1 is incorrect.
        return torch.where(value > 0, value, torch.ones_like(value))

    def _gap_tables(self, batch):
        """Return actual and factorized raw tables on identical support.

        raw[b,k,g,e]: k=(all,C-equal,B-equal,pair-equal), g=elapsed-time
        bin, e=16*source+destination, source/destination=4*match+action.
        Replace time association by N(g,e)*M(k,e)/N(e), preserving BOTH
        raw gap counts and raw equality counts in each directed state cell.
        Then independently replace pair equality by C(g,e)*B(g,e)/N(g,e).
        This is an empirical product, not a shuffle or independence claim.
        Every mode computes all alternatives before choosing its readout.
        Training storage is O(BL + B*4*4*256), never B*L*L.
        """
        eligible, order, n, complete = self._transition_support(batch)
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        group = self._membership(batch)[1].argmax(dim=1)
        cell = 4 * group + (batch['history_type'] - 1).clamp(0, 3)
        batch_size, length = c.shape
        if length < 2:
            raw = age.new_zeros((batch_size, 4, 4, 256))
        else:
            valid = eligible.gather(1, order)
            ordered_c, ordered_b = c.gather(1, order), b.gather(1, order)
            ordered_age, ordered_cell = age.gather(1, order), cell.gather(1, order)
            edge = valid[:, :-1] & valid[:, 1:]
            delta = ordered_age[:, :-1] - ordered_age[:, 1:]
            destination = (256 * self._gap_bin(delta) +
                           16 * ordered_cell[:, :-1] + ordered_cell[:, 1:])
            same_c = ordered_c[:, :-1] == ordered_c[:, 1:]
            same_b = ordered_b[:, :-1] == ordered_b[:, 1:]
            channels = []
            for match in (torch.ones_like(edge), same_c, same_b, same_c & same_b):
                counts = age.new_zeros((batch_size, 1024)).scatter_add(
                    1, destination, (edge & match).float())
                channels.append(counts.reshape(batch_size, 4, 256))
            raw = torch.stack(channels, dim=1)
        margin = raw.sum(dim=2, keepdim=True)
        independent = (raw[:, :1] * margin /
                       self._positive_denominator(margin[:, :1]))
        actual_product = (raw[:, 1] * raw[:, 2] /
                          self._positive_denominator(raw[:, 0]))
        independent_product = (independent[:, 1] * independent[:, 2] /
                               self._positive_denominator(independent[:, 0]))
        return raw, independent, actual_product, independent_product, n, complete, eligible

    def _gap_features(self, tables, mode=None, pairing=None):
        mode = self.gap_mode if mode is None else mode
        pairing = self.gap_pairing if pairing is None else pairing
        raw = tables[0] if mode == 'observed' else tables[1]
        product = tables[2] if mode == 'observed' else tables[3]
        pair = raw[:, 3] if pairing == 'observed' else product
        counts = torch.cat([raw[:, :3], pair[:, None]], dim=1)
        return torch.log1p(counts.flatten(1)) / math.log(129.0)

    def forward(self, batch):
        parent = super().forward(batch)
        features = self._gap_features(self._gap_tables(batch))
        return parent + self.gap_head(features).squeeze(-1)

    def _dense_gap_oracle(self, batch):
        """Independent quadratic count oracle, only on <=8 diagnostic rows.

        Define support with a timestamp equality matrix and adjacency by rank;
        do not call production sorting, membership or binning helpers here.
        """
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        complete = batch['history_mask'] & (c > 0) & (b > 0)
        ties = ((age[:, :, None] == age[:, None, :]) &
                complete[:, :, None] & complete[:, None, :]).sum(-1)
        support = complete & (ties == 1)
        rank = ((age[:, None, :] > age[:, :, None]) & support[:, None, :]).sum(-1)
        edge = (support[:, :, None] & support[:, None, :] &
                (rank[:, None, :] == rank[:, :, None] + 1))
        tc = batch['static'][:, self.category:self.category + 1]
        tb = batch['static'][:, self.brand:self.brand + 1]
        cm = (c > 0) & (tc > 0) & (c == tc)
        bm = (b > 0) & (tb > 0) & (b == tb)
        state = 4 * (cm.long() + 2 * bm.long()) + (batch['history_type'] - 1).clamp(0, 3)
        delta = age[:, :, None] - age[:, None, :]
        gap = ((delta > 60.0).long() + (delta > 3600.0).long() +
               (delta > 86400.0).long())
        destination = (256 * gap + 16 * state[:, :, None] + state[:, None, :]).flatten(1)
        same_c = c[:, :, None] == c[:, None, :]
        same_b = b[:, :, None] == b[:, None, :]
        channels = []
        for match in (torch.ones_like(edge), same_c, same_b, same_c & same_b):
            channels.append(age.new_zeros((age.shape[0], 1024)).scatter_add(
                1, destination, (edge & match).float().flatten(1)).reshape(-1, 4, 256))
        return torch.stack(channels, dim=1)

    def _gap_witness(self, batch, pair_witness=False):
        """Two realizable histories whose parent features/logits must agree.

        All ages share one log2 bin and one inherited coarse age bin.
        The first witness moves a short/long gap across equal/unequal edges,
        holding gap and recurrence margins fixed. The second inherits the
        prior pairing witness: same C and B edge margins, different pair
        equality, with every edge in the same elapsed-time bin.
        """
        device = batch['static'].device
        if pair_witness:
            c = [[1, 1, 1, 1, 2, 2, 2, 2]] * 2
            b = [[1, 1, 2, 2, 1, 1, 2, 2], [1, 1, 2, 2, 2, 1, 1, 2]]
            age = [[15000 - i for i in range(8)]] * 2
        else:
            c = b = [[1, 1, 2, 2]] * 2
            age = [[15000, 14999, 14899, 14898], [15000, 14900, 14899, 14898]]
        witness = {
            'static': batch['static'][:1].expand(2, -1).clone(),
            'history_category': torch.tensor(c, dtype=torch.long, device=device),
            'history_brand': torch.tensor(b, dtype=torch.long, device=device),
            'history_type': torch.ones((2, len(c[0])), dtype=torch.long, device=device),
            'history_age_seconds': torch.tensor(age, dtype=batch['history_age_seconds'].dtype, device=device),
            'history_mask': torch.ones((2, len(c[0])), dtype=torch.bool, device=device),
        }
        witness['static'][:, self.category] = 3
        witness['static'][:, self.brand] = 3
        return witness

    def diagnostics(self, batch):
        """Candidate-supplied checks; actual Torch execution is host-owned."""
        tables = self._gap_tables(batch)
        raw, independent, product, iproduct, n, complete, eligible = tables
        features = self._gap_features(tables)
        residual = self.gap_head(features).squeeze(-1)
        count = min(8, batch['static'].shape[0])
        probe = {key: value[:count] for key, value in batch.items()}
        temporal = self._transition_tables(batch)[0]
        margin = raw.sum(2)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'gap_head_parameters': float(sum(p.numel() for p in self.gap_head.parameters())),
            'gap_input_coordinates': float(features.shape[1]),
            'observed_gap_association_enabled': float(self.gap_mode == 'observed'),
            'observed_pairing_enabled': float(self.gap_pairing == 'observed'),
            'dense_oracle_impressions': float(count),
            'sorted_vs_dense_gap_count_error_max': (raw[:count] - self._dense_gap_oracle(probe)).abs().max(),
            'parent_temporal_margin_error_max': torch.stack([
                (margin[:, k] - temporal[k].flatten(1)).abs().max() for k in range(4)]).max(),
            'edge_mass_error_max': (raw[:, 0].sum((1, 2)) - (n.float() - 1).clamp_min(0)).abs().max(),
            'gap_replacement_margin_error_max': (independent.sum(2) - margin).abs().max(),
            'gap_replacement_total_table_error_max': (independent[:, 0] - raw[:, 0]).abs().max(),
            'raw_nonnegative_violation_max': (-raw).clamp_min(0).max(),
            'product_subset_violation_max': torch.stack([
                (product - raw[:, 1]).clamp_min(0).max(),
                (product - raw[:, 2]).clamp_min(0).max(),
                (iproduct - independent[:, 1]).clamp_min(0).max(),
                (iproduct - independent[:, 2]).clamp_min(0).max()]).max(),
            'complete_events_mean': complete.float().sum(-1).mean(),
            'singleton_events_mean': n.float().mean(),
            'excluded_tied_events_mean': (complete.float().sum(-1) - n.float()).mean(),
            'supported_history_fraction': (n >= 2).float().mean(),
            'gap_association_raw_gap_mean': (raw - independent).abs().mean(),
            'observed_vs_product_pair_gap_mean': (raw[:, 3] - product).abs().mean(),
            'gap_residual_rms': residual.square().mean().sqrt(),
            'zero_input_added_residual_abs_max': self.gap_head(torch.zeros_like(features)).abs().max(),
        }
        for gap in range(4):
            result['gap_' + str(gap) + '_edges_mean'] = raw[:, 0, gap].sum(-1).mean()
        reversed_batch = dict(batch)
        padded, zero_length = dict(batch), dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reversed_batch[key] = batch[key].flip(-1)
            zero_length[key] = batch[key][:, :0]
            fill = batch[key].new_zeros((batch[key].shape[0], 2))
            if key != 'history_mask':
                fill.fill_(1)
            padded[key] = torch.cat([batch[key], fill], dim=-1)
        result['input_reordering_feature_error_max'] = (features - self._gap_features(self._gap_tables(reversed_batch))).abs().max()
        result['masked_padding_feature_error_max'] = (features - self._gap_features(self._gap_tables(padded))).abs().max()
        result['zero_length_feature_abs_max'] = self._gap_features(self._gap_tables(zero_length)).abs().max()
        renamed = dict(batch)
        renamed['static'] = batch['static'].clone()
        for key, field, vocab in (('history_category', self.category, self.category_vocab),
                                  ('history_brand', self.brand, self.brand_vocab)):
            renamed[key] = torch.where(batch[key] > 0, vocab - batch[key], batch[key])
            value = batch['static'][:, field]
            renamed['static'][:, field] = torch.where(value > 0, vocab - value, value)
        result['identity_renaming_feature_error_max'] = (features - self._gap_features(self._gap_tables(renamed))).abs().max()
        for label in ('empty', 'missing', 'fully_tied'):
            altered = dict(batch)
            key = {'empty': 'history_mask', 'missing': 'history_brand', 'fully_tied': 'history_age_seconds'}[label]
            altered[key] = torch.zeros_like(batch[key])
            result[label + '_feature_abs_max'] = self._gap_features(self._gap_tables(altered)).abs().max()
        applicable = self.category_vocab > 3 and self.brand_vocab > 3
        result['witness_applicable'] = float(applicable)
        if applicable:
            for pair_witness, prefix in ((False, 'time_witness'), (True, 'pair_witness')):
                witness = self._gap_witness(batch, pair_witness)
                wt = self._gap_tables(witness)
                parent = super().forward(witness)
                result[prefix + '_parent_logit_error_max'] = (parent - parent[:1]).abs().max()
                for mode, pairing, label in (
                        ('observed', 'observed', 'gap_joint'),
                        ('independent', 'observed', 'independent_joint'),
                        ('observed', 'product', 'gap_product'),
                        ('independent', 'product', 'independent_product')):
                    value = self._gap_features(wt, mode, pairing)
                    result[prefix + '_' + label + '_feature_gap_max'] = (value[0] - value[1]).abs().max()
                added = self.gap_head(self._gap_features(wt)).squeeze(-1)
                result[prefix + '_same_checkpoint_added_logit_gap'] = (added[0] - added[1]).abs()
        return {key: float(value) for key, value in result.items()}


class _ResolutionParentModel(_GapParentModel):
    """Gap-input information replacement; all trainable modules are inherited.

    Scope is the gap branch only. Earlier branches still receive their full
    supplied histories. A total-only global readout has exactly four scalar
    inputs, even though the matched-cost lift and MLP remain 4096 wide.
    """
    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        self.gap_resolution = self.config['gap_resolution']
        self.gap_channels = self.config['gap_channels']
        if self.gap_resolution not in ('local', 'global'):
            raise ValueError('Unknown gap_resolution')
        if self.gap_channels not in ('all', 'total'):
            raise ValueError('Unknown gap_channels')
        if (self.gap_mode, self.gap_pairing) != ('observed', 'product'):
            raise ValueError('Retain incumbent observed-gap/product-pair construction')

    def _gap_reductions(self, tables, mode=None, pairing=None):
        mode = self.gap_mode if mode is None else mode
        pairing = self.gap_pairing if pairing is None else pairing
        raw = tables[0] if mode == 'observed' else tables[1]
        product = tables[2] if mode == 'observed' else tables[3]
        pair = raw[:, 3] if pairing == 'observed' else product
        # Product is constructed BEFORE pooling, exactly as in the parent.
        # Global/all therefore retains its aggregate, a context-derived scalar.
        counts = torch.cat([raw[:, :3], pair[:, None]], dim=1)
        return {
            'local_all': counts,
            'local_total': counts[:, :1],
            'global_all': counts.sum(dim=-1, keepdim=True),
            'global_total': counts[:, :1].sum(dim=-1, keepdim=True),
        }

    @staticmethod
    def _gap_lift(counts):
        """Isometric replication AFTER log, never averaging log cell counts.

        For replication r, each scalar appears r times divided by sqrt(r).
        Thus the lift preserves the reduced input norm and adds no information.
        All allocated first-layer columns participate for nonzero inputs.
        Lower intrinsic rank still changes effective statistical capacity.
        """
        reduced = torch.log1p(counts) / math.log(129.0)
        copies = (4 // counts.shape[1]) * (256 // counts.shape[3])
        return (reduced.expand(-1, 4, -1, 256).reshape(counts.shape[0], 4096) /
                math.sqrt(float(copies)))

    def _gap_views(self, tables, mode=None, pairing=None):
        reduced = self._gap_reductions(tables, mode, pairing)
        # Same four reductions/lifts execute in every variant. This is an
        # information study with matched readout cost, not a speed optimization.
        return {key: self._gap_lift(value) for key, value in reduced.items()}

    def _gap_features(self, tables, mode=None, pairing=None):
        return self._gap_views(tables, mode, pairing)[
            self.gap_resolution + '_' + self.gap_channels]

    def _resolution_witness(self, batch, kind):
        """Realizable probes; only the gap probe claims parent invariance."""
        device = batch['static'].device
        c = [[1, 1, 1, 1]] * 2
        actions = [[1, 1, 1, 1]] * 2
        ages = [[15000, 14999, 14899, 14898]] * 2
        if kind == 'context':
            actions = [[1, 1, 2, 2], [1, 2, 1, 2]]
        elif kind == 'recurrence':
            c = [[1, 1, 2, 2], [1, 2, 1, 2]]
        elif kind == 'gap':
            ages = [[15000, 14999, 14899, 14898],
                    [15000, 14999, 14998, 14997]]
        else:
            raise ValueError('Unknown witness')
        witness = {
            'static': batch['static'][:1].expand(2, -1).clone(),
            'history_category': torch.tensor(c, dtype=torch.long, device=device),
            'history_brand': torch.tensor(c, dtype=torch.long, device=device),
            'history_type': torch.tensor(actions, dtype=torch.long, device=device),
            'history_age_seconds': torch.tensor(
                ages, dtype=batch['history_age_seconds'].dtype, device=device),
            'history_mask': torch.ones((2, 4), dtype=torch.bool, device=device),
        }
        witness['static'][:, self.category] = 3
        witness['static'][:, self.brand] = 3
        return witness

    def diagnostics(self, batch):
        """Candidate-supplied checks on <=512 host impressions; no new labels."""
        tables = self._gap_tables(batch)
        raw, independent, product, iproduct, n, complete, eligible = tables
        reduced = self._gap_reductions(tables)
        views = self._gap_views(tables)
        key = self.gap_resolution + '_' + self.gap_channels
        features = views[key]
        residual = self.gap_head(features).squeeze(-1)
        probe_size = min(8, batch['static'].shape[0])
        probe = {key: value[:probe_size] for key, value in batch.items()}
        norms, decoded = [], []
        for name, value in reduced.items():
            compact = torch.log1p(value) / math.log(129.0)
            copies = (4 // value.shape[1]) * (256 // value.shape[3])
            norms.append((views[name].square().sum(-1) -
                          compact.square().sum((1, 2, 3))).abs().max())
            lifted = views[name].reshape(-1, 4, 4, 256)
            recovered = lifted
            if value.shape[1] == 1:
                recovered = recovered.mean(1, keepdim=True)
            if value.shape[3] == 1:
                recovered = recovered.mean(-1, keepdim=True)
            recovered = recovered * math.sqrt(float(copies))
            decoded.append((recovered - compact).abs().max())
        # Pooled total counts use exactly the original total edge channel.
        pooled_total = reduced['global_total'][:, 0, :, 0]
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'gap_head_parameters': float(sum(p.numel() for p in self.gap_head.parameters())),
            'gap_input_coordinates': float(features.shape[1]),
            'gap_unique_source_coordinates': float(reduced[key][0].numel()),
            'global_resolution_enabled': float(self.gap_resolution == 'global'),
            'total_only_enabled': float(self.gap_channels == 'total'),
            'dense_oracle_impressions': float(probe_size),
            'raw_sorted_vs_dense_oracle_error_max': (raw[:probe_size] - self._dense_gap_oracle(probe)).abs().max(),
            'incumbent_local_all_feature_error_max': (views['local_all'] - _GapParentModel._gap_features(self, tables)).abs().max(),
            'isometric_lift_norm_squared_error_max': torch.stack(norms).max(),
            'lift_inverse_error_max': torch.stack(decoded).max(),
            'pooled_edge_count_conservation_error_max': (pooled_total.sum(-1) - (n.float() - 1).clamp_min(0)).abs().max(),
            'complete_events_mean': complete.float().sum(-1).mean(),
            'singleton_events_mean': n.float().mean(),
            'excluded_tied_events_mean': (complete.float().sum(-1) - n.float()).mean(),
            'supported_history_fraction': (n >= 2).float().mean(),
            'gap_residual_rms': residual.square().mean().sqrt(),
            'gap_head_last_weight_norm': torch.linalg.vector_norm(self.gap_head[-1].weight),
            'zero_input_residual_abs_max': self.gap_head(torch.zeros_like(features)).abs().max(),
        }
        for gap in range(4):
            result['gap_' + str(gap) + '_edges_mean'] = pooled_total[:, gap].mean()
        reverse, padded, zero_length = dict(batch), dict(batch), dict(batch)
        for field in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[field] = batch[field].flip(-1)
            zero_length[field] = batch[field][:, :0]
            fill = batch[field].new_zeros((batch[field].shape[0], 2))
            if field != 'history_mask':
                fill.fill_(1)
            padded[field] = torch.cat([batch[field], fill], -1)
        for label, altered in (('input_reordering', reverse), ('masked_padding', padded)):
            value = self._gap_features(self._gap_tables(altered))
            result[label + '_feature_error_max'] = (features - value).abs().max()
        result['zero_length_feature_abs_max'] = self._gap_features(self._gap_tables(zero_length)).abs().max()
        for label, field in (('empty', 'history_mask'), ('missing', 'history_brand'), ('fully_tied', 'history_age_seconds')):
            altered = dict(batch)
            altered[field] = torch.zeros_like(batch[field])
            result[label + '_feature_abs_max'] = self._gap_features(self._gap_tables(altered)).abs().max()
        relabeled = dict(batch)
        relabeled['static'] = batch['static'].clone()
        for field, index, vocab in (('history_category', self.category, self.category_vocab),
                                     ('history_brand', self.brand, self.brand_vocab)):
            relabeled[field] = torch.where(batch[field] > 0, vocab - batch[field], batch[field])
            target = batch['static'][:, index]
            relabeled['static'][:, index] = torch.where(target > 0, vocab - target, target)
        result['identity_renaming_feature_error_max'] = (features - self._gap_features(self._gap_tables(relabeled))).abs().max()
        # Alter action and target fields: only global_total should necessarily
        # ignore these changes; full parent logits need not be invariant.
        changed = dict(batch)
        changed['static'] = torch.zeros_like(batch['static'])
        types = batch['history_type']
        changed['history_type'] = torch.where(types > 0, 5 - types, types)
        changed_views = self._gap_views(self._gap_tables(changed))
        result['global_total_target_action_invariance_error_max'] = (views['global_total'] - changed_views['global_total']).abs().max()
        applicable = self.category_vocab > 3 and self.brand_vocab > 3
        result['witness_applicable'] = float(applicable)
        if applicable:
            for kind in ('context', 'recurrence', 'gap'):
                witness = self._resolution_witness(batch, kind)
                witness_views = self._gap_views(self._gap_tables(witness))
                for name, value in witness_views.items():
                    result[kind + '_witness_' + name + '_feature_gap_max'] = (value[0] - value[1]).abs().max()
                added = self.gap_head(witness_views[key]).squeeze(-1)
                result[kind + '_witness_same_checkpoint_added_logit_gap'] = (added[0] - added[1]).abs()
                if kind == 'gap':
                    # _ParentModel is the exact pre-gap chronological product
                    # reference. All ages share inherited bins in this probe.
                    parent = _ParentModel.forward(self, witness)
                    result['gap_witness_pre_gap_parent_logit_error_max'] = (parent[0] - parent[1]).abs()
        return {name: float(value) for name, value in result.items()}


class _LiftedParentModel(_ResolutionParentModel):
    """Global recurrence factorization: pool/product order x gap association.

    Only the gap-head input changes. All inherited representations, allocations,
    initialization draws and optimization are untouched. Every cell computes
    every replacement so the study isolates information rather than speed.
    """
    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        if (self.gap_resolution, self.gap_channels) != ('global', 'all'):
            raise ValueError('Fix the global/all incumbent')
        self.product_scope = self.config.get('product_scope', 'pooled')
        self.gap_association = self.config.get('gap_association', 'observed')
        if self.product_scope not in ('context', 'pooled'):
            raise ValueError('Unknown product_scope')
        if self.gap_association not in ('observed', 'marginal'):
            raise ValueError('Unknown gap_association')

    def _association_counts(self, tables):
        # raw[b,k,g,e], k=(N,C,B,actual-pair), g=fixed gap, e=directed state.
        # Original fourth channel is sum_e C_ge*B_ge/N_ge. Pooling first
        # instead yields (sum_e C_ge)*(sum_e B_ge)/(sum_e N_ge).
        # The latter is independent of action/target partition in THIS branch.
        raw = tables[0]
        common = raw[:, :3].sum(-1, keepdim=True)
        total, category, brand = common[:, 0:1], common[:, 1:2], common[:, 2:3]
        context_product = tables[2].sum(-1, keepdim=True)[:, None]
        pooled_product = category * brand / self._positive_denominator(total)
        observed = {
            'context': torch.cat([common, context_product], dim=1),
            'pooled': torch.cat([common, pooled_product], dim=1),
        }
        result = {}
        for scope, counts in observed.items():
            # Apply AFTER selecting product scope. Preserve every selected
            # channel total and every gap's N; remove their association only.
            # This is a deterministic fractional replacement, not a claim of
            # population independence and not a newly sampled event sequence.
            margin = counts[:, 1:].sum(2, keepdim=True)
            denominator = self._positive_denominator(total.sum(2, keepdim=True))
            marginal = torch.cat([total, total * margin / denominator], dim=1)
            result[scope + '_observed'] = counts
            result[scope + '_marginal'] = marginal
        return result

    def _association_views(self, tables):
        # All modes execute these exact four lifts and the inherited tables.
        return {name: self._gap_lift(value)
                for name, value in self._association_counts(tables).items()}

    def _gap_features(self, tables, mode=None, pairing=None):
        if mode not in (None, 'observed') or pairing not in (None, 'product'):
            raise ValueError('This factorial fixes the inherited table modes')
        return self._association_views(tables)[
            self.product_scope + '_' + self.gap_association]

    def _association_witness(self, batch, kind):
        # Realizable causal histories. Target 4 is outside history identities.
        # Context: N,C,B by gap agree, but product before pooling differs.
        # Association: same N by gap and all channel totals in both scopes,
        # with recurrence placed entirely in opposite, equally sized gap bins.
        if kind == 'context':
            c = [[1, 1, 2, 2, 3]] * 2
            b = [[1, 2, 2, 3, 3]] * 2
            action = [[1, 1, 1, 1, 1], [1, 1, 2, 2, 3]]
            age = [[15000, 14999, 14998, 14997, 14996]] * 2
        elif kind == 'association':
            c = b = [[1, 1, 1, 2, 3]] * 2
            action = [[1, 1, 1, 1, 1]] * 2
            age = [[15000, 14999, 14998, 14898, 14798],
                   [15000, 14900, 14800, 14799, 14798]]
        else:
            raise ValueError('Unknown witness')
        device = batch['static'].device
        witness = {
            'static': batch['static'][:1].expand(2, -1).clone(),
            'history_category': torch.tensor(c, dtype=torch.long, device=device),
            'history_brand': torch.tensor(b, dtype=torch.long, device=device),
            'history_type': torch.tensor(action, dtype=torch.long, device=device),
            'history_age_seconds': torch.tensor(
                age, dtype=batch['history_age_seconds'].dtype, device=device),
            'history_mask': torch.ones((2, 5), dtype=torch.bool, device=device),
        }
        witness['static'][:, self.category] = 4
        witness['static'][:, self.brand] = 4
        return witness

    def diagnostics(self, batch):
        """At most64 label-free scalars; host executes at its fixed checkpoint.

        Dense oracle is independent of production sorting and runs on <=8 rows.
        Witness logit sensitivity shows representation use, not causal benefit.
        """
        tables = self._gap_tables(batch)
        counts = self._association_counts(tables)
        views = self._association_views(tables)
        key = self.product_scope + '_' + self.gap_association
        features = views[key]
        raw, _, _, _, n, complete, eligible = tables
        probe_size = min(8, batch['static'].shape[0])
        probe = {name: value[:probe_size] for name, value in batch.items()}
        legacy = _ResolutionParentModel._gap_views(self, tables)['global_all']
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'gap_head_parameters': float(sum(p.numel() for p in self.gap_head.parameters())),
            'lifted_coordinates': float(features.shape[1]),
            'pooled_product_enabled': float(self.product_scope == 'pooled'),
            'marginal_association_enabled': float(self.gap_association == 'marginal'),
            'dense_oracle_impressions': float(probe_size),
            'raw_dense_oracle_error_max': (raw[:probe_size] - self._dense_gap_oracle(probe)).abs().max(),
            'incumbent_feature_reproduction_error_max': (views['context_observed'] - legacy).abs().max(),
            'edge_conservation_error_max': (counts[key][:, 0].sum((1, 2)) - (n.float() - 1).clamp_min(0)).abs().max(),
            'complete_events_mean': complete.float().sum(-1).mean(),
            'singleton_events_mean': n.float().mean(),
            'excluded_tied_events_mean': (complete.float().sum(-1) - n.float()).mean(),
            'supported_history_fraction': (n >= 2).float().mean(),
            'context_vs_pooled_product_raw_gap_mean': (counts['context_observed'][:, 3] - counts['pooled_observed'][:, 3]).abs().mean(),
            'observed_vs_marginal_feature_gap_mean': (views[self.product_scope + '_observed'] - views[self.product_scope + '_marginal']).abs().mean(),
            'gap_residual_rms': self.gap_head(features).square().mean().sqrt(),
            'zero_input_residual_abs_max': self.gap_head(torch.zeros_like(features)).abs().max(),
        }
        for scope in ('context', 'pooled'):
            observed, marginal = counts[scope + '_observed'], counts[scope + '_marginal']
            result[scope + '_channel_margin_error_max'] = (observed.sum(2) - marginal.sum(2)).abs().max()
            result[scope + '_gap_total_error_max'] = (observed[:, 0] - marginal[:, 0]).abs().max()
            result[scope + '_nonnegative_violation_max'] = (-marginal).clamp_min(0).max()
            result[scope + '_subset_violation_max'] = (marginal[:, 1:] - marginal[:, :1]).clamp_min(0).max()
        norms = []
        for name, value in counts.items():
            compact = torch.log1p(value) / math.log(129.0)
            norms.append((views[name].square().sum(-1) - compact.square().sum((1, 2, 3))).abs().max())
        result['isometric_lift_norm_error_max'] = torch.stack(norms).max()
        # These are feature invariances. Full-model logits need not share them.
        reverse, padded, zero_length = dict(batch), dict(batch), dict(batch)
        for field in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[field] = batch[field].flip(-1)
            zero_length[field] = batch[field][:, :0]
            fill = batch[field].new_zeros((batch[field].shape[0], 2))
            if field != 'history_mask':
                fill.fill_(1)
            padded[field] = torch.cat([batch[field], fill], -1)
        for label, altered in (('reordering', reverse), ('padding', padded)):
            result[label + '_feature_error_max'] = (features - self._gap_features(self._gap_tables(altered))).abs().max()
        result['zero_length_feature_abs_max'] = self._gap_features(self._gap_tables(zero_length)).abs().max()
        for label, field in (('empty', 'history_mask'), ('missing', 'history_brand'), ('tied', 'history_age_seconds')):
            altered = dict(batch)
            altered[field] = torch.zeros_like(batch[field])
            result[label + '_feature_abs_max'] = self._gap_features(self._gap_tables(altered)).abs().max()
        changed = dict(batch)
        changed['static'] = torch.zeros_like(batch['static'])
        # Constant action merges contexts; unlike a bijective action rename it
        # can expose the inherited sum-of-products partition dependence.
        changed['history_type'] = torch.ones_like(batch['history_type'])
        changed_views = self._association_views(self._gap_tables(changed))
        result['pooled_target_action_invariance_error_max'] = torch.stack([
            (views['pooled_' + association] - changed_views['pooled_' + association]).abs().max()
            for association in ('observed', 'marginal')]).max()
        applicable = self.category_vocab > 4 and self.brand_vocab > 4
        result['witness_applicable'] = float(applicable)
        if applicable:
            for kind in ('context', 'association'):
                witness = self._association_witness(batch, kind)
                witness_counts = self._association_counts(self._gap_tables(witness))
                witness_views = {name: self._gap_lift(value) for name, value in witness_counts.items()}
                for name, value in witness_views.items():
                    result[kind + '_witness_' + name + '_gap_max'] = (value[0] - value[1]).abs().max()
                added = self.gap_head(witness_views[key]).squeeze(-1)
                result[kind + '_witness_added_logit_gap'] = (added[0] - added[1]).abs()
                if kind == 'association':
                    for scope in ('context', 'pooled'):
                        margins = witness_counts[scope + '_observed'].sum(2)
                        result[scope + '_witness_margin_error_max'] = (margins[0] - margins[1]).abs().max()
                    parent = _ParentModel.forward(self, witness)
                    result['association_witness_pre_gap_parent_logit_error_max'] = (parent[0] - parent[1]).abs()
        return {name: float(value) for name, value in result.items()}


class _CompactGapHead(nn.Module):
    """Remove duplicate columns; preserve initialization and expose Adam control.

    x_j is repeated r=256 times as x_j/16 in the old head. Write
    U_hj=mean_i W_hji, so its effective weight is A=16U. Every old W_hji
    receives the same gradient g*x_j/16. For elementwise Adam with zero
    initial state, no weight decay, no parameter-dependent clipping, and
    exact arithmetic, all duplicate W columns have identical updates.
    Their mean U therefore follows Adam on that SAME gradient. Calibrated
    mode stores U, uses A=16U, and scales its derivative to 1/16. This also
    preserves Adam's epsilon effect. Native mode stores A and uses ordinary
    gradients: its effective Adam step is about 16 times smaller. These are
    explicit parameterization controls; neither adds an input statistic.
    Float reductions/kernel choices can still change a full training path.
    """
    def __init__(self, lifted, mode):
        super().__init__()
        self.mode = mode
        mean = lifted[0].weight.detach().reshape(16, 4, 4, 256).mean(-1).reshape(16, 16)
        self.weight = nn.Parameter(mean.clone() if mode == 'calibrated' else (16.0 * mean).clone())
        # Reuse the exact allocated zero-initialized last layer, with no new RNG.
        self.last = lifted[2]

    def effective_weight(self):
        return 16.0 * self.weight if self.mode == 'calibrated' else self.weight

    def forward(self, compact):
        if self.mode == 'calibrated':
            # Value =16U; derivative =1/16. This reparameterizes only the
            # first gap layer. Host Adam/loss/lr and every other gradient stay fixed.
            detached = self.weight.detach()
            weight = 16.0 * (detached + (self.weight - detached) / 256.0)
        else:
            weight = self.weight
        hidden = torch.relu(torch.nn.functional.linear(compact, weight))
        return self.last(hidden)


class Model(_LiftedParentModel):
    """Actual pooled execution plus controlled removal of a redundant lift.

    The entire pre-gap model is inherited, including its expensive tables.
    Only the final gap branch is optimized. Dense diagnostics are excluded
    from forward and cannot supply labels, events, or additional inputs.
    """
    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        self.gap_execution = self.config.get('gap_execution', 'direct')
        self.gap_parameterization = self.config.get('gap_parameterization', 'calibrated')
        if self.gap_execution not in ('dense', 'direct'):
            raise ValueError('Unknown gap_execution')
        if self.gap_parameterization not in ('lifted', 'calibrated', 'native'):
            raise ValueError('Unknown gap_parameterization')
        if (self.product_scope, self.gap_association) != ('pooled', 'observed'):
            raise ValueError('Fix the pooled/observed incumbent information')
        if self.gap_execution == 'dense' and self.gap_parameterization != 'lifted':
            raise ValueError('The dense arm is the exact lifted reference')
        # Parent allocation consumes all original seeded draws first. Replacing
        # the module deregisters the redundant matrix; it is not kept in a buffer.
        if self.gap_parameterization != 'lifted':
            self.gap_head = _CompactGapHead(self.gap_head, self.gap_parameterization)

    def _direct_gap_counts(self, batch):
        """Compute pooled N,C,B histograms directly, then their pooled product.

        Exact same complete-pair/singleton support and adjacent subsequence as
        the ancestor. Target match and action partitions disappear only in
        this branch because sum_e raw[k,g,e] ignores them. Never compute a
        product before pooling. Storage O(BL+B*16), independent of context cells.
        """
        eligible, order, _, _ = self._transition_support(batch)
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        batch_size, length = c.shape
        if length < 2:
            return age.new_zeros((batch_size, 4, 4, 1))
        valid = eligible.gather(1, order)
        ordered_c, ordered_b = c.gather(1, order), b.gather(1, order)
        ordered_age = age.gather(1, order)
        edge = valid[:, :-1] & valid[:, 1:]
        gap = self._gap_bin(ordered_age[:, :-1] - ordered_age[:, 1:])
        same_c = ordered_c[:, :-1] == ordered_c[:, 1:]
        same_b = ordered_b[:, :-1] == ordered_b[:, 1:]
        common = []
        for match in (torch.ones_like(edge), same_c, same_b):
            common.append(age.new_zeros((batch_size, 4)).scatter_add(
                1, gap, (edge & match).float()))
        total, category, brand = common
        product = category * brand / self._positive_denominator(total)
        return torch.stack([total, category, brand, product], dim=1).unsqueeze(-1)

    def _selected_gap_features(self, batch):
        if self.gap_execution == 'dense':
            return _LiftedParentModel._gap_features(self, self._gap_tables(batch))
        counts = self._direct_gap_counts(batch)
        if self.gap_parameterization == 'lifted':
            return self._gap_lift(counts)
        return (torch.log1p(counts) / math.log(129.0)).flatten(1)

    def forward(self, batch):
        parent = _ParentModel.forward(self, batch)
        return parent + self.gap_head(self._selected_gap_features(batch)).squeeze(-1)

    def diagnostics(self, batch):
        """Label-free source-checkable measurements, never timed as inference.

        Oracle uses independent quadratic event comparisons on <=8 rows.
        Other equivalence checks use at most the host's first 512 rows.
        Algebraic equality does not establish repeated performance equivalence.
        """
        direct = self._direct_gap_counts(batch)
        tables = self._gap_tables(batch)
        dense = self._association_counts(tables)['pooled_observed']
        compact = (torch.log1p(direct) / math.log(129.0)).flatten(1)
        lifted = self._gap_lift(direct)
        selected = self._selected_gap_features(batch)
        _, _, n, complete = self._transition_support(batch)
        probe_size = min(8, batch['static'].shape[0])
        probe = {key: value[:probe_size] for key, value in batch.items()}
        oracle_raw = self._dense_gap_oracle(probe)
        common = oracle_raw[:, :3].sum(-1, keepdim=True)
        oracle = torch.cat([common, common[:, 1:2] * common[:, 2:3] /
                            self._positive_denominator(common[:, :1])], dim=1)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'gap_head_parameters': float(sum(p.numel() for p in self.gap_head.parameters())),
            'gap_input_coordinates': float(selected.shape[-1]),
            'direct_execution_enabled': float(self.gap_execution == 'direct'),
            'calibrated_parameterization_enabled': float(self.gap_parameterization == 'calibrated'),
            'native_parameterization_enabled': float(self.gap_parameterization == 'native'),
            'direct_vs_dense_count_error_max': (direct - dense).abs().max(),
            'direct_vs_independent_oracle_error_max': (direct[:probe_size] - oracle).abs().max(),
            'edge_conservation_error_max': (direct[:, 0].sum((1, 2)) - (n.float() - 1).clamp_min(0)).abs().max(),
            'lift_norm_error_max': (compact.square().sum(-1) - lifted.square().sum(-1)).abs().max(),
            'gap_residual_rms': self.gap_head(selected).square().mean().sqrt(),
            'zero_input_residual_abs_max': self.gap_head(torch.zeros_like(selected)).abs().max(),
            'complete_events_mean': complete.float().sum(-1).mean(),
            'singleton_events_mean': n.float().mean(),
            'supported_history_fraction': (n >= 2).float().mean(),
            'dense_oracle_impressions': float(probe_size),
        }
        if self.gap_parameterization == 'lifted':
            effective = self.gap_head[0].weight.reshape(16, 4, 4, 256).sum(-1).reshape(16, 16) / 16.0
            collapsed = self.gap_head[2](torch.relu(torch.nn.functional.linear(compact, effective)))
            result['same_checkpoint_projection_error_max'] = (self.gap_head(lifted) - collapsed).abs().max()
        else:
            effective = self.gap_head.effective_weight()
            expanded = (effective.reshape(16, 4, 4, 1).expand(-1, -1, -1, 256) / 16.0).reshape(16, 4096)
            dense_output = self.gap_head.last(torch.relu(torch.nn.functional.linear(lifted, expanded)))
            result['same_checkpoint_projection_error_max'] = (self.gap_head(compact) - dense_output).abs().max()
            result['stored_to_effective_weight_scale'] = 16.0 if self.gap_parameterization == 'calibrated' else 1.0
            result['first_layer_parameter_gradient_scale'] = 1.0 / 16.0 if self.gap_parameterization == 'calibrated' else 1.0
        reverse, padded, empty_length = dict(batch), dict(batch), dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[key] = batch[key].flip(-1)
            empty_length[key] = batch[key][:, :0]
            fill = batch[key].new_zeros((batch[key].shape[0], 2))
            if key != 'history_mask':
                fill.fill_(1)
            padded[key] = torch.cat([batch[key], fill], -1)
        result['reordering_count_error_max'] = (direct - self._direct_gap_counts(reverse)).abs().max()
        result['padding_count_error_max'] = (direct - self._direct_gap_counts(padded)).abs().max()
        result['zero_length_count_abs_max'] = self._direct_gap_counts(empty_length).abs().max()
        for label, key in (('empty', 'history_mask'), ('missing', 'history_brand'), ('tied', 'history_age_seconds')):
            altered = dict(batch)
            altered[key] = torch.zeros_like(batch[key])
            result[label + '_count_abs_max'] = self._direct_gap_counts(altered).abs().max()
        changed = dict(batch)
        changed['static'] = torch.zeros_like(batch['static'])
        changed['history_type'] = torch.ones_like(batch['history_type'])
        result['target_action_invariance_error_max'] = (direct - self._direct_gap_counts(changed)).abs().max()
        return {key: float(value) for key, value in result.items()}


_FrozenModel = Model


class Model(_FrozenModel):
    """Selected execution and capacity-matched transition state replacement.

    No new parameters, optimizer, labels, histories, or initialization draws.
    Full uses the exact ancestor selected representation. Action and rotated
    replace only the transition-state mapping before counting and product.
    The unchanged 1024-input head receives an isometric 64 -> 1024 lift.
    The gap branch retains the original lifted4096 parameterization.
    """
    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        self.state_mode = self.config.get('transition_state_mode', 'full')
        if self.state_mode not in ('full', 'action', 'rotated'):
            raise ValueError('Unknown transition_state_mode')
        if self.gap_parameterization != 'lifted' or self.gap_execution != 'direct':
            raise ValueError('Keep the frozen direct/lifted gap branch')

    def _selected_repeat(self, batch, ids, ctx):
        # Exact observed conditional features; omit product/unlabeled controls.
        eligible, group = ids[5], ids[6]
        context, context_mass = ctx[5], ctx[6]
        batch_size, length = eligible.shape
        squared = context_mass.new_zeros((batch_size, 64))
        xlogx = torch.zeros_like(squared)
        if length:
            identity = batch['history_category'] * self.brand_vocab + batch['history_brand']
            packed = 64 * identity + 4 * context + group
            key = torch.where(eligible, packed, torch.full_like(packed, -1))
            ordered = key.sort(dim=-1).values
            valid = ordered >= 0
            start = valid.clone()
            start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
            run = (start.long().cumsum(-1) - 1).clamp_min(0)
            mass = context_mass.new_zeros((batch_size, length)).scatter_add(1, run, valid.float())
            cell = 16 * ordered.remainder(4) + ordered.div(4, rounding_mode='floor').remainder(16)
            run_cell = torch.zeros_like(ordered).scatter_add(
                1, run, torch.where(start, cell, torch.zeros_like(cell)))
            squared = squared.scatter_add(1, run_cell, mass.square())
            xlogx = xlogx.scatter_add(1, run_cell, mass * mass.clamp_min(1).log())
        squared = squared.reshape(batch_size, 4, 16)
        xlogx = xlogx.reshape(batch_size, 4, 16)
        denom = context_mass.clamp_min(1)
        entropy = torch.where(context_mass > 0, (denom.log() - xlogx / denom).clamp_min(0),
                              torch.zeros_like(context_mass))
        collision = squared / denom.square()
        return torch.stack([torch.log1p(context_mass) / math.log(129.0),
                            entropy / math.log(129.0), collision], -1).flatten(1)

    def _selected_overlap(self, batch, ids, ctx):
        # Exact observed Gram features; omit product/unlabeled controls.
        eligible, group = ids[5], ids[6]
        context, context_mass = ctx[5], ctx[6]
        batch_size, length = eligible.shape
        raw = context_mass.new_zeros((batch_size, 4, 16, 16))
        if length:
            identity = batch['history_category'] * self.brand_vocab + batch['history_brand']
            packed = 4 * identity + group
            key = torch.where(eligible, packed, torch.full_like(packed, -1))
            ordered, order = key.sort(dim=-1)
            valid = ordered >= 0
            start = valid.clone()
            start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
            run = (start.long().cumsum(-1) - 1).clamp_min(0)
            sorted_context = context.gather(1, order)
            counts = context_mass.new_zeros((batch_size, length * 16)).scatter_add(
                1, 16 * run + sorted_context, valid.float()).reshape(batch_size, length, 16)
            run_group = torch.zeros_like(ordered).scatter_add(
                1, run, torch.where(start, ordered.remainder(4), torch.zeros_like(ordered)))
            grams = []
            for g in range(4):
                selected = counts * (run_group == g).unsqueeze(-1).float()
                grams.append(torch.bmm(counts.transpose(1, 2), selected))
            raw = torch.stack(grams, dim=1)
        denominator = context_mass.unsqueeze(-1) * context_mass.unsqueeze(-2)
        gram = raw / denominator.clamp_min(1)
        return gram[:, :, self.overlap_rows, self.overlap_cols].flatten(1)

    def _selected_parent(self, batch):
        features, _, _, _, _, semantic, _ = self._shared_features(batch)
        ids = self._identity_tables(batch, semantic)
        ctx = self._context_tables(batch, ids)
        # Preserve original addition order and all selected feature operations.
        parent = self.output(features).squeeze(-1)
        identity = self.identity_head(self._identity_features(ids)).squeeze(-1)
        context = self.context_head(self._context_features(ctx)).squeeze(-1)
        repeat = self.repeat_head(self._selected_repeat(batch, ids, ctx)).squeeze(-1)
        overlap = self.overlap_head(self._selected_overlap(batch, ids, ctx)).squeeze(-1)
        return parent + identity + context + repeat + overlap

    def _state(self, batch, mode):
        action = (batch['history_type'] - 1).clamp(0, 3)
        if mode == 'action':
            return action, 4
        group = self._membership(batch)[1].argmax(dim=1)
        if mode == 'rotated':
            # Fixed balanced replacement; no data-dependent assignment.
            return (action + group).remainder(4), 4
        return 4 * group + action, 16

    def _selected_counts(self, batch, support, mode):
        eligible, order = support[:2]
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        batch_size, length = c.shape
        state, states = self._state(batch, mode)
        temporal = [age.new_zeros((batch_size, states * states)) for _ in range(3)]
        gap_counts = [age.new_zeros((batch_size, 4)) for _ in range(3)]
        if length >= 2:
            valid = eligible.gather(1, order)
            ordered_state = state.gather(1, order)
            ordered_c, ordered_b = c.gather(1, order), b.gather(1, order)
            ordered_age = age.gather(1, order)
            edge = valid[:, :-1] & valid[:, 1:]
            same_c = ordered_c[:, :-1] == ordered_c[:, 1:]
            same_b = ordered_b[:, :-1] == ordered_b[:, 1:]
            destination = states * ordered_state[:, :-1] + ordered_state[:, 1:]
            gap = self._gap_bin(ordered_age[:, :-1] - ordered_age[:, 1:])
            for i, match in enumerate((torch.ones_like(edge), same_c, same_b)):
                weights = (edge & match).float()
                temporal[i] = temporal[i].scatter_add(1, destination, weights)
                gap_counts[i] = gap_counts[i].scatter_add(1, gap, weights)
        # Factorize AFTER choosing the state mapping. Coarsening therefore
        # changes both labeled counts and the resolution of factorization.
        total, category, brand = temporal
        product = category * brand / total.clamp_min(1)
        transition = torch.stack([total, category, brand, product], 1).reshape(
            batch_size, 4, states, states)
        total, category, brand = gap_counts
        product = category * brand / self._positive_denominator(total)
        gap = torch.stack([total, category, brand, product], 1).unsqueeze(-1)
        return transition, gap

    @staticmethod
    def _transition_lift(counts):
        features = torch.log1p(counts) / math.log(129.0)
        if counts.shape[-1] == 16:
            return features.flatten(1)
        # Old layout [channel, source_group, source_action, dest_group, dest_action].
        return (features.reshape(counts.shape[0], 4, 1, 4, 1, 4).expand(
            -1, -1, 4, -1, 4, -1).reshape(counts.shape[0], 1024) / 4.0)

    def forward(self, batch):
        parent = self._selected_parent(batch)
        transition, gap = self._selected_counts(batch, self._transition_support(batch), self.state_mode)
        residual = self.transition_head(self._transition_lift(transition)).squeeze(-1)
        return parent + residual + self.gap_head(self._gap_lift(gap)).squeeze(-1)

    def _quadratic_counts(self, batch, mode):
        # Independent support and adjacency oracle, diagnostics on <=8 rows.
        c, b = batch['history_category'], batch['history_brand']
        age = batch['history_age_seconds']
        complete = batch['history_mask'] & (c > 0) & (b > 0)
        ties = ((age[:, :, None] == age[:, None, :]) &
                complete[:, :, None] & complete[:, None, :]).sum(-1)
        eligible = complete & (ties == 1)
        rank = ((age[:, None, :] > age[:, :, None]) & eligible[:, None, :]).sum(-1)
        edge = (eligible[:, :, None] & eligible[:, None, :] &
                (rank[:, None, :] == rank[:, :, None] + 1))
        tc, tb = batch['static'][:, self.category:self.category+1], batch['static'][:, self.brand:self.brand+1]
        cm = (c > 0) & (tc > 0) & (c == tc)
        bm = (b > 0) & (tb > 0) & (b == tb)
        group = cm.long() + 2 * bm.long()
        action = (batch['history_type'] - 1).clamp(0, 3)
        state = 4 * group + action if mode == 'full' else action
        if mode == 'rotated':
            state = (action + group).remainder(4)
        states = 16 if mode == 'full' else 4
        destination = (states * state[:, :, None] + state[:, None, :]).flatten(1)
        counts = []
        for match in (torch.ones_like(edge), c[:, :, None] == c[:, None, :], b[:, :, None] == b[:, None, :]):
            counts.append(age.new_zeros((c.shape[0], states * states)).scatter_add(
                1, destination, (edge & match).float().flatten(1)))
        total, category, brand = counts
        product = category * brand / total.clamp_min(1)
        return torch.stack([total, category, brand, product], 1).reshape(c.shape[0], 4, states, states)

    def diagnostics(self, batch):
        probe = {k: v[:min(8, v.shape[0])] for k, v in batch.items()}
        support = self._transition_support(probe)
        full, gap = self._selected_counts(probe, support, 'full')
        selected, _ = self._selected_counts(probe, support, self.state_mode)
        parent_features, _, _, _, _, semantic, _ = self._shared_features(probe)
        ids = self._identity_tables(probe, semantic)
        ctx = self._context_tables(probe, ids)
        legacy_transition = self._transition_features(self._transition_tables(probe))
        repeat = self._repeat_features(self._conditional_tables(probe, ids, ctx))
        overlap = self._overlap_features(self._overlap_tables(probe, ids, ctx))
        small = torch.log1p(selected).flatten(1) / math.log(129.0)
        lifted = self._transition_lift(selected)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'transition_head_parameters': float(sum(p.numel() for p in self.transition_head.parameters())),
            'transition_intrinsic_coordinates': float(selected[0].numel()),
            'full_transition_feature_error_max': (self._transition_lift(full) - legacy_transition).abs().max(),
            'shared_gap_count_error_max': (gap - self._direct_gap_counts(probe)).abs().max(),
            'repeat_feature_error_max': (self._selected_repeat(probe, ids, ctx) - repeat).abs().max(),
            'overlap_feature_error_max': (self._selected_overlap(probe, ids, ctx) - overlap).abs().max(),
            'parent_logit_error_max': (self._selected_parent(probe) - self._parent_forward(probe)).abs().max(),
            'selected_vs_quadratic_count_error_max': (selected - self._quadratic_counts(probe, self.state_mode)).abs().max(),
            'transition_lift_norm_error_max': (small.square().sum(-1) - lifted.square().sum(-1)).abs().max(),
            'edge_conservation_error_max': (selected[:, 0].sum((1, 2)) - (support[2].float()-1).clamp_min(0)).abs().max(),
            'transition_residual_rms': self.transition_head(lifted).square().mean().sqrt(),
            'gap_residual_rms': self.gap_head(self._gap_lift(gap)).square().mean().sqrt(),
            'oracle_impressions': float(probe['static'].shape[0]),
        }
        reverse, padded, empty = dict(probe), dict(probe), dict(probe)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[key] = probe[key].flip(-1)
            empty[key] = probe[key][:, :0]
            fill = probe[key].new_zeros((probe[key].shape[0], 2))
            if key != 'history_mask':
                fill.fill_(1)
            padded[key] = torch.cat([probe[key], fill], -1)
        for label, altered in [('reverse', reverse), ('padding', padded)]:
            new, _ = self._selected_counts(altered, self._transition_support(altered), self.state_mode)
            result[label + '_count_error_max'] = (selected - new).abs().max()
        for label, altered in [('zero_length', empty), ('empty', dict(probe)), ('missing', dict(probe)), ('tied', dict(probe))]:
            if label != 'zero_length':
                key = {'empty':'history_mask', 'missing':'history_brand', 'tied':'history_age_seconds'}[label]
                altered[key] = torch.zeros_like(probe[key])
            new, _ = self._selected_counts(altered, self._transition_support(altered), self.state_mode)
            result[label + '_count_abs_max'] = new.abs().max()
        changed = dict(probe)
        changed['static'] = torch.zeros_like(probe['static'])
        altered, _ = self._selected_counts(changed, support, self.state_mode)
        result['target_change_transition_count_max'] = (selected - altered).abs().max()
        if self.state_mode == 'full':
            result['same_checkpoint_full_logit_error_max'] = (self.forward(probe) - _FrozenModel.forward(self, probe)).abs().max()
        return {k:float(v) for k,v in result.items()}


CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'moment_grouping': 'match', 'moment_pairing': 'independent', 'field_mode': 'contrast', 'field_grouping': 'rank', 'field_scope': 'global', 'field_layout': 'compact', 'global_direction': 'norm_collapsed', 'identity_mode': 'joint', 'context_mode': 'joint', 'repeat_mode': 'joint', 'overlap_mode': 'observed', 'transition_order': 'chronological', 'transition_pairing': 'product', 'gap_mode': 'observed', 'gap_pairing': 'product', 'gap_resolution': 'global', 'gap_channels': 'all', 'product_scope': 'pooled', 'gap_association': 'observed', 'gap_execution': 'direct', 'gap_parameterization': 'lifted', 'transition_state_mode': 'full'}


_SelectedModel = Model


class Model(_SelectedModel):
    """Share joint counts; replace recurrence-context resolution prospectively.

    Full is an execution control. Action/rotated change three statistical
    readouts, retaining age information in all other inherited branches.
    No parameter is appended, removed, or reinitialized. All batch-dependent
    work is local to forward; no cross-batch or label cache exists.
    """
    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        self.recurrence_context_mode = self.config.get('recurrence_context_mode', 'action')
        if self.recurrence_context_mode not in ('full', 'action', 'rotated'):
            raise ValueError('Unknown recurrence_context_mode')
        if self.state_mode != 'full':
            raise ValueError('Hold chronological transition states fixed at full')

    @staticmethod
    def _context_partition(batch, mode):
        action = (batch['history_type'] - 1).clamp(0, 3)
        if mode == 'action':
            return action, 4
        age = batch['history_age_seconds']
        age_bin = ((age >= 3600.0).long() + (age >= 86400.0).long() +
                   (age >= 604800.0).long())
        if mode == 'rotated':
            return (action + age_bin).remainder(4), 4
        return 4 * action + age_bin, 16

    @staticmethod
    def _fine_to_coarse(mode, device):
        fine = torch.arange(16, device=device)
        action = fine.div(4, rounding_mode='floor')
        if mode == 'rotated':
            return (action + fine.remainder(4)).remainder(4)
        return action

    def _repeat_lift(self, features, mode):
        if mode == 'full':
            return features.flatten(1)
        # [B,group,coarse_context,3] -> original [B,group,16,3].
        # Each coarse coordinate has four copies scaled by 1/sqrt(4).
        mapping = self._fine_to_coarse(mode, features.device)
        return (features.index_select(2, mapping) / 2.0).flatten(1)

    def _overlap_lift(self, gram, mode):
        if mode == 'full':
            return gram[:, :, self.overlap_rows, self.overlap_cols].flatten(1)
        # Only cross-coarse-context pairs are retained. Each of six unordered
        # coarse pairs occurs in 4*4=16 fine upper-triangle slots. Within-state
        # fine pairs are zero, so diagonal collision is NOT added as new input.
        mapping = self._fine_to_coarse(mode, gram.device)
        row, col = mapping[self.overlap_rows], mapping[self.overlap_cols]
        return (gram[:, :, row, col] * (row != col).float() / 4.0).flatten(1)

    @staticmethod
    def _entropy_collision(sum_xlogx, sum_square, mass):
        denom = mass.clamp_min(1)
        entropy = torch.where(mass > 0, (denom.log() - sum_xlogx / denom).clamp_min(0),
                              torch.zeros_like(mass))
        return entropy, sum_square / denom.square()

    def _shared_recurrence(self, batch, semantic, mode):
        """Return identity/context/repeat/overlap features and compact probes.

        The only new table is [B,L,K] with K=16 (full) or 4 (replacement).
        Pair/group keys and counts use the original complete-pair support.
        All statistics are empirical with-replacement summaries; pair bucket
        identity is never described as item identity. No quadratic event table
        is built during forward.
        """
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        membership = semantic * eligible[:, None, :].float()
        n = membership.sum(-1)
        batch_size, length = c.shape
        context, contexts = self._context_partition(batch, mode)
        # Two unchanged marginal sorts. Joint counts from the third sort feed
        # every remaining summary instead of repeating the same counting work.
        hc, cc = self._sorted_distribution(c, group, eligible, n)
        hb, cb = self._sorted_distribution(b, group, eligible, n)
        counts = n.new_zeros((batch_size, length, contexts))
        run_group = torch.zeros_like(c)
        run_mass = n.new_zeros((batch_size, length))
        if length:
            identity = c * self.brand_vocab + b
            packed = 4 * identity + group
            key = torch.where(eligible, packed, torch.full_like(packed, -1))
            ordered, order = key.sort(dim=-1)
            valid = ordered >= 0
            start = valid.clone()
            start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
            run = (start.long().cumsum(-1) - 1).clamp_min(0)
            run_mass = run_mass.scatter_add(1, run, valid.float())
            run_group = run_group.scatter_add(
                1, run, torch.where(start, ordered.remainder(4), torch.zeros_like(ordered)))
            sorted_context = context.gather(1, order)
            counts = n.new_zeros((batch_size, length * contexts)).scatter_add(
                1, contexts * run + sorted_context, valid.float()).reshape(
                    batch_size, length, contexts)
        joint_square = torch.zeros_like(n).scatter_add(1, run_group, run_mass.square())
        joint_xlogx = torch.zeros_like(n).scatter_add(
            1, run_group, run_mass * run_mass.clamp_min(1).log())
        hx, cx = self._entropy_collision(joint_xlogx, joint_square, n)
        scale = math.log(129.0)
        identity_features = torch.cat([
            torch.stack([torch.log1p(n) / scale, hc / scale, hb / scale, cc, cb], -1),
            torch.stack([hx / scale, cx], -1)
        ], -1).flatten(1)
        context_mass = n.new_zeros((batch_size, 4 * contexts)).scatter_add(
            1, contexts * group + context, eligible.float()).reshape(
                batch_size, 4, contexts)
        hy, cy = self._entropy_collision(
            (context_mass * context_mass.clamp_min(1).log()).sum(-1),
            context_mass.square().sum(-1), n)
        # Counts^2 and counts*log(counts) are shared by joint and conditional
        # summaries. Integer count conservation itself is independently probed.
        squared, xlogx = counts.square(), counts * counts.clamp_min(1).log()
        hxy, cxy = self._entropy_collision(
            torch.zeros_like(n).scatter_add(1, run_group, xlogx.sum(-1)),
            torch.zeros_like(n).scatter_add(1, run_group, squared.sum(-1)), n)
        context_features = torch.cat([
            torch.stack([torch.log1p(n) / scale, hx / scale, hy / scale, cx, cy], -1),
            torch.stack([hxy / scale, cxy], -1)
        ], -1).flatten(1)
        cell = contexts * run_group[:, :, None] + torch.arange(
            contexts, device=c.device)[None, None, :]
        conditional_square = n.new_zeros((batch_size, 4 * contexts)).scatter_add(
            1, cell.flatten(1), squared.flatten(1)).reshape(batch_size, 4, contexts)
        conditional_xlogx = n.new_zeros((batch_size, 4 * contexts)).scatter_add(
            1, cell.flatten(1), xlogx.flatten(1)).reshape(batch_size, 4, contexts)
        entropy, collision = self._entropy_collision(
            conditional_xlogx, conditional_square, context_mass)
        compact_repeat = torch.stack([
            torch.log1p(context_mass) / scale, entropy / scale, collision], -1)
        raw = n.new_zeros((batch_size, 4, contexts, contexts))
        if length:
            grams = []
            for g in range(4):
                selected = counts * (run_group == g).unsqueeze(-1).float()
                grams.append(torch.bmm(counts.transpose(1, 2), selected))
            raw = torch.stack(grams, dim=1)
        denominator = context_mass.unsqueeze(-1) * context_mass.unsqueeze(-2)
        gram = raw / denominator.clamp_min(1)
        features = (identity_features, context_features,
                    self._repeat_lift(compact_repeat, mode), self._overlap_lift(gram, mode))
        # Compact summaries for label-free host diagnostics, not cached state.
        return features, (n, context_mass, counts.sum((1, 2)), compact_repeat, gram)

    def _parent_from_recurrence(self, main_features, blocks):
        identity, context, repeat, overlap = blocks
        parent = self.output(main_features).squeeze(-1)
        return (parent + self.identity_head(identity).squeeze(-1) +
                self.context_head(context).squeeze(-1) +
                self.repeat_head(repeat).squeeze(-1) +
                self.overlap_head(overlap).squeeze(-1))

    def _selected_parent(self, batch):
        main, _, _, _, _, semantic, _ = self._shared_features(batch)
        blocks, _ = self._shared_recurrence(batch, semantic, self.recurrence_context_mode)
        return self._parent_from_recurrence(main, blocks)

    def _dense_recurrence_oracle(self, batch, mode):
        """Independent O(B L^2) event equality oracle; <=8 diagnostics rows.

        Direct per-event multiplicities independently verify joint/conditional
        entropy and collision. This code is never called by forward.
        """
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        semantic = self._membership(batch)[1]
        group = semantic.argmax(dim=1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        context, contexts = self._context_partition(batch, mode)
        cell = contexts * group + context
        batch_size = c.shape[0]
        context_mass = n.new_zeros((batch_size, 4 * contexts)).scatter_add(
            1, cell, eligible.float()).reshape(batch_size, 4, contexts)
        same_x = ((c[:, :, None] == c[:, None, :]) &
                  (b[:, :, None] == b[:, None, :]) &
                  (group[:, :, None] == group[:, None, :]) &
                  eligible[:, :, None] & eligible[:, None, :])
        same_xy = same_x & (context[:, :, None] == context[:, None, :])
        mx, mxy = same_x.sum(-1).float(), same_xy.sum(-1).float()
        weights = eligible.float()
        def group_moments(multiplicity):
            return (torch.zeros_like(n).scatter_add(1, group, multiplicity.clamp_min(1).log() * weights),
                    torch.zeros_like(n).scatter_add(1, group, multiplicity * weights))
        hx, cx = self._entropy_collision(*group_moments(mx), n)
        hxy, cxy = self._entropy_collision(*group_moments(mxy), n)
        hy, cy = self._entropy_collision(
            (context_mass * context_mass.clamp_min(1).log()).sum(-1),
            context_mass.square().sum(-1), n)
        hc, cc = self._dense_distribution(c, group, eligible, n)
        hb, cb = self._dense_distribution(b, group, eligible, n)
        scale = math.log(129.0)
        identity = torch.stack([torch.log1p(n)/scale, hc/scale, hb/scale, cc, cb, hx/scale, cx], -1).flatten(1)
        joint_context = torch.stack([torch.log1p(n)/scale, hx/scale, hy/scale, cx, cy, hxy/scale, cxy], -1).flatten(1)
        conditional_xlogx = n.new_zeros((batch_size, 4 * contexts)).scatter_add(
            1, cell, mxy.clamp_min(1).log() * weights).reshape(batch_size, 4, contexts)
        conditional_square = n.new_zeros((batch_size, 4 * contexts)).scatter_add(
            1, cell, mxy * weights).reshape(batch_size, 4, contexts)
        entropy, collision = self._entropy_collision(conditional_xlogx, conditional_square, context_mass)
        compact_repeat = torch.stack([torch.log1p(context_mass)/scale, entropy/scale, collision], -1)
        grams = []
        for g in range(4):
            membership = torch.stack([
                eligible & (group == g) & (context == k) for k in range(contexts)], 1).float()
            grams.append(torch.bmm(torch.bmm(membership, same_x.float()), membership.transpose(1, 2)))
        raw = torch.stack(grams, 1)
        gram = raw / (context_mass.unsqueeze(-1) * context_mass.unsqueeze(-2)).clamp_min(1)
        return (identity, joint_context, self._repeat_lift(compact_repeat, mode), self._overlap_lift(gram, mode))

    def diagnostics(self, batch):
        """Candidate-supplied probes; host validation remains the only scorer."""
        probe = {key: value[:min(8, value.shape[0])] for key, value in batch.items()}
        main, _, _, _, _, semantic, _ = self._shared_features(probe)
        mode = self.recurrence_context_mode
        blocks, aux = self._shared_recurrence(probe, semantic, mode)
        full, _ = self._shared_recurrence(probe, semantic, 'full')
        oracle = self._dense_recurrence_oracle(probe, mode)
        old_ids = self._identity_tables(probe, semantic)
        old_ctx = self._context_tables(probe, old_ids)
        legacy = (self._identity_features(old_ids), self._context_features(old_ctx),
                  _SelectedModel._selected_repeat(self, probe, old_ids, old_ctx),
                  _SelectedModel._selected_overlap(self, probe, old_ids, old_ctx))
        n, context_mass, counted, compact_repeat, gram = aux
        contexts = 16 if mode == 'full' else 4
        rows, cols = torch.triu_indices(contexts, contexts, offset=1, device=gram.device)
        compact_overlap = gram[:, :, rows, cols].flatten(1)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'recurrence_context_states': float(contexts),
            'source_sort_calls_per_parent': 3.0,
            'repeat_intrinsic_coordinates': float(4 * contexts * 3),
            'overlap_intrinsic_coordinates': float(4 * contexts * (contexts - 1) // 2),
            'oracle_impressions': float(probe['static'].shape[0]),
            'full_parent_logit_error_max': (self._parent_from_recurrence(main, full) -
                                           _SelectedModel._selected_parent(self, probe)).abs().max(),
            'count_conservation_error_max': (counted - n.sum(-1)).abs().max(),
            'context_margin_conservation_error_max': (context_mass.sum(-1) - n).abs().max(),
            'repeat_lift_norm_error_max': (blocks[2].square().sum(-1) - compact_repeat.square().flatten(1).sum(-1)).abs().max(),
            'overlap_lift_norm_error_max': (blocks[3].square().sum(-1) - compact_overlap.square().sum(-1)).abs().max(),
            'context_residual_rms': self.context_head(blocks[1]).square().mean().sqrt(),
            'repeat_residual_rms': self.repeat_head(blocks[2]).square().mean().sqrt(),
            'overlap_residual_rms': self.overlap_head(blocks[3]).square().mean().sqrt(),
        }
        for i, label in enumerate(('identity', 'context', 'repeat', 'overlap')):
            result['full_vs_selected_' + label + '_error_max'] = (full[i] - legacy[i]).abs().max()
            result['dense_oracle_' + label + '_error_max'] = (blocks[i] - oracle[i]).abs().max()
        # Reordering and masked nonzero padding cannot change these summaries.
        history_keys = ('history_category', 'history_brand', 'history_type',
                        'history_age_seconds', 'history_mask')
        reverse, padded, empty = dict(probe), dict(probe), dict(probe)
        for key in history_keys:
            reverse[key] = probe[key].flip(-1)
            empty[key] = probe[key][:, :0]
            fill = probe[key].new_zeros((probe[key].shape[0], 2))
            if key != 'history_mask':
                fill.fill_(1)
            padded[key] = torch.cat([probe[key], fill], -1)
        for label, altered in [('reverse', reverse), ('padding', padded)]:
            changed, _ = self._shared_recurrence(altered, self._membership(altered)[1], mode)
            result[label + '_statistic_error_max'] = max((x-y).abs().max() for x,y in zip(blocks, changed))
        for label, altered in [('zero_length', empty), ('empty', dict(probe)), ('missing', dict(probe))]:
            if label != 'zero_length':
                key = 'history_mask' if label == 'empty' else 'history_brand'
                altered[key] = torch.zeros_like(probe[key])
            changed, _ = self._shared_recurrence(altered, self._membership(altered)[1], mode)
            result[label + '_statistic_abs_max'] = max(x.abs().max() for x in changed)
        changed = dict(probe)
        changed['history_age_seconds'] = torch.zeros_like(probe['history_age_seconds'])
        altered, _ = self._shared_recurrence(changed, semantic, mode)
        result['age_change_statistic_delta_max'] = max((x-y).abs().max() for x,y in zip(blocks, altered))
        zero_repeat = torch.zeros_like(blocks[2])
        zero_overlap = torch.zeros_like(blocks[3])
        result['zero_repeat_residual_abs_max'] = self.repeat_head(zero_repeat).abs().max()
        result['zero_overlap_residual_abs_max'] = self.overlap_head(zero_overlap).abs().max()
        # These measurements are submitted for source review, not asserted as
        # independently verified scientific outcomes or timing measurements.
        return {key: float(value) for key, value in result.items()}


CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'moment_grouping': 'match', 'moment_pairing': 'independent', 'field_mode': 'contrast', 'field_grouping': 'rank', 'field_scope': 'global', 'field_layout': 'compact', 'global_direction': 'norm_collapsed', 'identity_mode': 'joint', 'context_mode': 'joint', 'repeat_mode': 'joint', 'overlap_mode': 'observed', 'transition_order': 'chronological', 'transition_pairing': 'product', 'gap_mode': 'observed', 'gap_pairing': 'product', 'gap_resolution': 'global', 'gap_channels': 'all', 'product_scope': 'pooled', 'gap_association': 'observed', 'gap_execution': 'direct', 'gap_parameterization': 'lifted', 'transition_state_mode': 'full', 'recurrence_context_mode': 'full'}
