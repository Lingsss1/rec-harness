"""Prospective target-availability score-fusion factorial.

Inherit the hash-verified r0005-query_blind incumbent: same parameter
constructors/order, context queries, history values, and 24 explicit features.
Replace only the shared relevance score for missing-target impressions.
Availability (ID > 0) binds to either its matching or the crossed field scorer.
Denominator is either active field count (at least one) or fixed two. All-known
rows explicitly use the original .5*(category_score+brand_score). All-missing
rows have zero scores and uniform history attention; empty histories pool zero.

The four standalone modules have identical code except literal CONFIG. This
is a parameter-preserving missingness hypothesis, not a novelty/reproduction
claim. Normalized local attention is retained, unlike original DIN sec.4.3.
QueryFormer-inspired blind context is inherited, not a new component here.
No I/O, extra data, labels, optimizer changes or new random draws. Diagnostics
are descriptive first-512 fixed-checkpoint measurements, not causal importance
or missing-stratum predictive metrics. See prospective proposal.json.
"""
import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'statistic_target': 'match', 'statistic_scale': 'count', 'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0, 'pair_statistic': 'support', 'field_pairing': 'aligned', 'field_routing': 'shared', 'context_gain': 0.5, 'query_context': 'blind', 'fusion_binding': 'available', 'fusion_scale': 'active'}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.field_pairing = self.config['field_pairing']
        self.field_routing = self.config['field_routing']
        self.query_context = self.config['query_context']
        self.fusion_binding = self.config['fusion_binding']
        self.fusion_scale = self.config['fusion_scale']
        if self.fusion_binding not in ('available', 'crossed'):
            raise ValueError('Unknown fusion_binding')
        if self.fusion_scale not in ('active', 'fixed'):
            raise ValueError('Unknown fusion_scale')
        if self.query_context != 'blind':
            raise ValueError('The incumbent blind query context is frozen')
        if self.query_context not in ('target', 'uniform', 'blind', 'adaptive'):
            raise ValueError('Unknown query_context')
        self.context_gain = float(self.config['context_gain'])
        if self.context_gain != 0.5:
            raise ValueError('The prospective residual gain is frozen at 0.5')
        if self.field_pairing != 'aligned' or self.field_routing != 'shared':
            raise ValueError('The aligned/shared parent retrieval is frozen')
        if self.field_pairing not in ('aligned', 'crossed'):
            raise ValueError('Unknown field_pairing')
        if self.field_routing not in ('shared', 'separate'):
            raise ValueError('Unknown field_routing')
        if self.config['pair_statistic'] != 'support':
            raise ValueError('The inherited support branch is frozen')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.temporal_statistic = self.config['temporal_statistic']
        self.pair_statistic = self.config['pair_statistic']
        if self.pair_statistic not in ('joint', 'factorized', 'duplicate', 'support'):
            raise ValueError('Unknown pair_statistic')
        if self.config['kind'] != 'din' or self.temporal_statistic != 'aligned':
            raise ValueError('The inherited aligned DIN branch is frozen')
        if self.temporal_statistic not in ('aligned', 'factorized', 'activity', 'duplicate'):
            raise ValueError('Unknown temporal_statistic')
        if (self.config['statistic_target'] != 'match' or
                self.config['statistic_scale'] != 'count'):
            raise ValueError('The inherited count branch is frozen')
        self.half_life = float(self.config['half_life_seconds'])
        if not math.isfinite(self.half_life) or self.half_life <= 0:
            raise ValueError('half_life_seconds must be finite and positive')
        dim = self.config['embedding_dim']
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        # Preserve all 24 parent features and their output coordinates.
        # No new nn.Modules or random draws: all modes inherit initialization.
        width = len(meta['fields']) * dim + dim + 24
        layers = []
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _statistics(self, batch):
        mask = batch['history_mask']
        types = batch['history_type']
        action_ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == action_ids) & mask.unsqueeze(-1)).float()
        counts = actions.sum(dim=1)
        # The host supplies nonnegative causal ages. Mask padding before exp;
        # no event selection, history extension, learned decay or label access.
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        weighted_actions = actions * decay.unsqueeze(-1)
        matches, supports, decayed_matches, decayed_supports = [], [], [], []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            history_ids = batch[column]
            target_ids = batch['static'][:, index:index + 1]
            valid = mask & (history_ids > 0)
            equal = valid & (target_ids > 0) & (history_ids == target_ids)
            valid_float = valid.unsqueeze(-1).float()
            equal_float = equal.unsqueeze(-1).float()
            supports.append((actions * valid_float).sum(dim=1))
            matches.append((actions * equal_float).sum(dim=1))
            decayed_supports.append((weighted_actions * valid_float).sum(dim=1))
            decayed_matches.append((weighted_actions * equal_float).sum(dim=1))
        # Each statistic is [B,2,4]: category/brand then pv/cart/fav/buy.
        return (counts, torch.stack(matches, dim=1), torch.stack(supports, dim=1),
                torch.stack(decayed_matches, dim=1),
                torch.stack(decayed_supports, dim=1))

    def _evidence_from_statistics(self, statistics):
        _, matches, supports, decayed_matches, decayed_supports = statistics
        factorized = matches * decayed_supports / supports.clamp_min(1.0)
        if self.temporal_statistic == 'aligned':
            extra = decayed_matches
        elif self.temporal_statistic == 'factorized':
            extra = factorized
        elif self.temporal_statistic == 'activity':
            extra = decayed_supports
        else:
            extra = matches
        # Same fixed monotone count transform on both branches and all modes.
        return torch.cat([
            torch.log1p(matches).flatten(start_dim=1),
            torch.log1p(extra).flatten(start_dim=1),
        ], dim=-1) / math.log(129.0)

    def _pair_statistics(self, batch):
        """Counts and decay masses within the SAME pair-valid event population.

        Pair-valid depends only on supplied event-field availability. Target
        equality is separately masked for each field, so ID zero never matches.
        The decayed independence construction uses decay-weighted marginals;
        it is not a uniform age/field permutation or a new event sequence.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        action_ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == action_ids) & mask.unsqueeze(-1)).float()
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        history_category = batch['history_category']
        history_brand = batch['history_brand']
        target_category = batch['static'][:, self.category:self.category + 1]
        target_brand = batch['static'][:, self.brand:self.brand + 1]
        valid = mask & (history_category > 0) & (history_brand > 0)
        category_equal = valid & (target_category > 0) & (history_category == target_category)
        brand_equal = valid & (target_brand > 0) & (history_brand == target_brand)
        joint_equal = category_equal & brand_equal
        valid_actions = actions * valid.unsqueeze(-1).float()
        weighted_actions = valid_actions * decay.unsqueeze(-1)
        category_match = category_equal.unsqueeze(-1).float()
        brand_match = brand_equal.unsqueeze(-1).float()
        joint_match = joint_equal.unsqueeze(-1).float()
        support = valid_actions.sum(dim=1)
        weighted_support = weighted_actions.sum(dim=1)
        category_count = (valid_actions * category_match).sum(dim=1)
        brand_count = (valid_actions * brand_match).sum(dim=1)
        category_decay = (weighted_actions * category_match).sum(dim=1)
        brand_decay = (weighted_actions * brand_match).sum(dim=1)
        joint_count = (valid_actions * joint_match).sum(dim=1)
        joint_decay = (weighted_actions * joint_match).sum(dim=1)
        independent_count = category_count * (brand_count / support.clamp_min(1.0))
        # Divide before multiply to avoid underflow in Wc*Wb. When U=0,
        # both marginals are zero. A denominator of one gives exact zero
        # without evaluating a division by zero or altering positive masses.
        safe_mass = torch.where(weighted_support > 0, weighted_support,
                                torch.ones_like(weighted_support))
        independent_decay = category_decay * (brand_decay / safe_mass)
        return {
            'support': support, 'weighted_support': weighted_support,
            'category_count': category_count, 'brand_count': brand_count,
            'category_decay': category_decay, 'brand_decay': brand_decay,
            'joint_count': joint_count, 'joint_decay': joint_decay,
            'independent_count': independent_count,
            'independent_decay': independent_decay,
        }

    def _pair_evidence(self, statistics, pairs):
        # Every mode receives all already computed quantities. This selector
        # changes only which eight deterministic scalars enter the same MLP.
        if self.pair_statistic == 'joint':
            count, mass = pairs['joint_count'], pairs['joint_decay']
        elif self.pair_statistic == 'factorized':
            count, mass = pairs['independent_count'], pairs['independent_decay']
        elif self.pair_statistic == 'support':
            count, mass = pairs['support'], pairs['weighted_support']
        else:
            # Original parent category marginals, including events with missing
            # brand. These duplicate existing features and contain no new info.
            count, mass = statistics[1][:, 0, :], statistics[3][:, 0, :]
        return torch.cat([torch.log1p(count), torch.log1p(mass)], dim=-1) / math.log(129.0)

    def _masked_weights(self, scores, mask):
        weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def _score(self, query, value):
        query = query[:, None, :].expand_as(value)
        return self.attention(torch.cat([
            query, value, query - value, query * value
        ], dim=-1)).squeeze(-1)

    def _context_queries(self, batch, fields, query_mode=None):
        """Attend only to available static fields other than category/brand.

        The shared scorer is deliberately reused across roles. All context
        alternatives are evaluated in all modes, but unused gradients do not
        enter the loss. Values and keys are the same supplied embeddings.
        """
        values = torch.stack(fields, dim=1)
        positions = torch.arange(len(fields), device=batch['static'].device)
        mask = ((batch['static'] > 0) &
                (positions != self.category).unsqueeze(0) &
                (positions != self.brand).unsqueeze(0))
        uniform = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        zero_anchor = torch.zeros_like(fields[self.category])
        blind = self._masked_weights(self._score(zero_anchor, values), mask)
        adaptive_category = self._masked_weights(
            self._score(fields[self.category], values), mask)
        adaptive_brand = self._masked_weights(
            self._score(fields[self.brand], values), mask)
        uniform_context = (uniform.unsqueeze(-1) * values).sum(1)
        blind_context = (blind.unsqueeze(-1) * values).sum(1)
        category_context = (adaptive_category.unsqueeze(-1) * values).sum(1)
        brand_context = (adaptive_brand.unsqueeze(-1) * values).sum(1)
        mode = self.query_context if query_mode is None else query_mode
        if mode == 'adaptive':
            selected_category, selected_brand = category_context, brand_context
            category_weights, brand_weights = adaptive_category, adaptive_brand
        elif mode == 'blind':
            selected_category, selected_brand = blind_context, blind_context
            category_weights, brand_weights = blind, blind
        elif mode == 'uniform':
            selected_category, selected_brand = uniform_context, uniform_context
            category_weights, brand_weights = uniform, uniform
        elif mode == 'target':
            selected_category = torch.zeros_like(fields[self.category])
            selected_brand = torch.zeros_like(fields[self.brand])
            category_weights = torch.zeros_like(uniform)
            brand_weights = torch.zeros_like(uniform)
        else:
            raise ValueError('Unknown diagnostic query_mode')
        # Bypass even zero additions in the exact historical-parent control.
        category_query, brand_query = fields[self.category], fields[self.brand]
        if mode != 'target':
            category_query = category_query + self.context_gain * selected_category
            brand_query = brand_query + self.context_gain * selected_brand
        return {
            'category_query': category_query, 'brand_query': brand_query,
            'mask': mask, 'uniform': uniform, 'blind': blind,
            'adaptive_category': adaptive_category, 'adaptive_brand': adaptive_brand,
            'selected_category_weights': category_weights,
            'selected_brand_weights': brand_weights,
        }

    def _fused_scores(self, batch, category_score, brand_score,
                      binding=None, scale=None):
        """Gate scores, never history events or values.

        Crossed binding has identical active counts and scale, but attaches each
        availability indicator to the other scorer. Both-known rows exactly
        select the original parent expression. Zero is a missing-ID marker, not
        an observed category/brand identity. No learned temperature is added.
        """
        category_available = (batch['static'][:, self.category:self.category + 1] > 0)
        brand_available = (batch['static'][:, self.brand:self.brand + 1] > 0)
        binding = self.fusion_binding if binding is None else binding
        scale = self.fusion_scale if scale is None else scale
        if binding == 'available':
            category_gate, brand_gate = category_available, brand_available
        elif binding == 'crossed':
            category_gate, brand_gate = brand_available, category_available
        else:
            raise ValueError('Unknown fusion binding')
        category_part = torch.where(category_gate, category_score, torch.zeros_like(category_score))
        brand_part = torch.where(brand_gate, brand_score, torch.zeros_like(brand_score))
        active = category_available.float() + brand_available.float()
        if scale == 'active':
            fused = (category_part + brand_part) / active.clamp_min(1.0)
        elif scale == 'fixed':
            fused = 0.5 * (category_part + brand_part)
        else:
            raise ValueError('Unknown fusion scale')
        parent = 0.5 * (category_score + brand_score)
        return torch.where(category_available & brand_available, parent, fused)

    def _field_views(self, batch, fields, query_mode=None):
        mask = batch['history_mask']
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        context = self.action(batch['history_type']) + self.recency(ages)
        category_value = self.embeddings[self.category](batch['history_category']) + 0.5 * context
        brand_value = self.embeddings[self.brand](batch['history_brand']) + 0.5 * context
        context = self._context_queries(batch, fields, query_mode=query_mode)
        category_query = context['category_query']
        brand_query = context['brand_query']
        category_score = self._score(category_query, category_value)
        brand_score = self._score(brand_query, brand_value)
        category_weights = self._masked_weights(category_score, mask)
        brand_weights = self._masked_weights(brand_score, mask)
        shared_score = self._fused_scores(batch, category_score, brand_score)
        shared_weights = self._masked_weights(shared_score, mask)
        shared_pool = (shared_weights.unsqueeze(-1) * (category_value + brand_value)).sum(1)
        separate_pool = ((category_weights.unsqueeze(-1) * category_value).sum(1) +
                         (brand_weights.unsqueeze(-1) * brand_value).sum(1))
        selected = separate_pool if self.field_routing == 'separate' else shared_pool
        return {
            'context': context,
            'selected': selected, 'shared_pool': shared_pool, 'separate_pool': separate_pool,
            'category_weights': category_weights, 'brand_weights': brand_weights,
            'shared_weights': shared_weights,
            'category_score': category_score, 'brand_score': brand_score,
            'shared_score': shared_score, 'history_value': category_value + brand_value,
        }

    def forward(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._field_views(batch, fields)
        statistics = self._statistics(batch)
        pairs = self._pair_statistics(batch)
        evidence = self._evidence_from_statistics(statistics)
        pair_evidence = self._pair_evidence(statistics, pairs)
        features = torch.cat(fields + [views['selected'], evidence, pair_evidence], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """48 scalars max; candidate-supplied, no labels, no population claim.

        Counterfactuals reuse weights and embeddings at this checkpoint. They
        measure activation only, not the predictive effect of retraining.
        Conditional means are zero when count=0, so counts must be read first.
        """
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._field_views(batch, fields)
        mask = batch['history_mask']
        lengths = mask.sum(-1).float()
        present = (lengths > 0).float()
        category_available = batch['static'][:, self.category] > 0
        brand_available = batch['static'][:, self.brand] > 0
        both_known = category_available & brand_available
        single_known = category_available ^ brand_available
        both_missing = ~(category_available | brand_available)
        parent_score = 0.5 * (views['category_score'] + views['brand_score'])
        parent_weights = self._masked_weights(parent_score, mask)
        parent_pool = (parent_weights.unsqueeze(-1) * views['history_value']).sum(1)
        weights = views['shared_weights']
        pool_distance = (views['selected'] - parent_pool).norm(dim=-1)
        weight_distance = (weights - parent_weights).abs().sum(-1)
        statistics = self._statistics(batch)
        evidence = self._evidence_from_statistics(statistics)
        pair_evidence = self._pair_evidence(statistics, self._pair_statistics(batch))
        actual_logits = self.output(torch.cat(
            fields + [views['selected'], evidence, pair_evidence], dim=-1)).squeeze(-1)
        parent_logits = self.output(torch.cat(
            fields + [parent_pool, evidence, pair_evidence], dim=-1)).squeeze(-1)
        logit_distance = (actual_logits - parent_logits).abs()

        def conditional_mean(value, group):
            group = group.float()
            return (value * group).sum() / group.sum().clamp_min(1.0)

        result = {
            'diagnostic_sample_count': lengths.new_tensor(lengths.shape[0]),
            'supplied_history_length_mean': lengths.mean(),
            'supplied_empty_history_count': (lengths == 0).float().sum(),
            'category_target_missing_count': (~category_available).float().sum(),
            'brand_target_missing_count': (~brand_available).float().sum(),
            'weight_sum_error_max': (weights.sum(-1) - present).abs().max(),
            'padded_weight_max': (weights * (~mask).float()).abs().max(),
            'empty_selected_pool_norm_max': (views['selected'].norm(dim=-1) * (1.0 - present)).max(),
            'parent_pool_distance_mean': pool_distance.mean(),
            'parent_weight_l1_distance_mean': weight_distance.mean(),
            'parent_absolute_logit_change_mean': logit_distance.mean(),
            'both_known_parent_score_error_max': ((views['shared_score'] - parent_score).abs() * both_known[:, None].float()).max(),
            'both_missing_shared_score_abs_max': (views['shared_score'].abs() * both_missing[:, None].float()).max(),
        }
        for name, group in (('both_known', both_known), ('single_known', single_known),
                            ('both_missing', both_missing)):
            result[name + '_count'] = group.float().sum()
            result[name + '_nonempty_count'] = (group.float() * present).sum()
            result[name + '_parent_pool_distance_mean'] = conditional_mean(pool_distance, group)
            result[name + '_parent_weight_distance_mean'] = conditional_mean(weight_distance, group)
            result[name + '_parent_logit_change_mean'] = conditional_mean(logit_distance, group)
        for name, available in (('category', category_available), ('brand', brand_available)):
            scores = views[name + '_score']
            mean = (scores * mask.float()).sum(-1) / lengths.clamp_min(1.0)
            variance = ((scores - mean[:, None]).square() * mask.float()).sum(-1) / lengths.clamp_min(1.0)
            group = (~available).float() * present
            result[name + '_missing_nonempty_count'] = group.sum()
            result[name + '_missing_score_variance_mean'] = conditional_mean(variance, group)
        # Complete factorial pathways at fixed weights, not retrained results.
        pools = {}
        for binding in ('available', 'crossed'):
            for scale in ('active', 'fixed'):
                scores = self._fused_scores(batch, views['category_score'],
                                            views['brand_score'], binding, scale)
                local_weights = self._masked_weights(scores, mask)
                pool = (local_weights.unsqueeze(-1) * views['history_value']).sum(1)
                pools[(binding, scale)] = pool
                result[binding + '_' + scale + '_single_known_parent_pool_distance_mean'] = conditional_mean(
                    (pool - parent_pool).norm(dim=-1), single_known)
        for scale in ('active', 'fixed'):
            result[scale + '_binding_pool_distance_single_known_mean'] = conditional_mean(
                (pools[('available', scale)] - pools[('crossed', scale)]).norm(dim=-1), single_known)
        for binding in ('available', 'crossed'):
            result[binding + '_scale_pool_distance_single_known_mean'] = conditional_mean(
                (pools[(binding, 'active')] - pools[(binding, 'fixed')]).norm(dim=-1), single_known)
        return result
