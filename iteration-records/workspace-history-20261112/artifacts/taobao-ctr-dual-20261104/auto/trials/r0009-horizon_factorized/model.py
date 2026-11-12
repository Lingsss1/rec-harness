"""Prospective learned temporal-shape versus amplitude replacement study.

Parent: r0005-query_blind, source 9cc179ba81d736d49723b152aff730a6c43f35b31179cc964b60ff575ae04023.
Preserve static embeddings, context queries, shared retrieval, all count features,
the eight fixed pair-support features, and classifier dimensions/initialization.
Change only the eight decayed target-match statistics. Eight zero-initialized
parameters (field x action) either change half-life or multiply fixed-day mass.
An aligned/factorized replacement crosses that matched-capacity comparison.
Both transformations are computed in every mode; backward costs may differ.

f=exp(log(4)*tanh(theta)); horizon h=86400*f seconds, amplitude a=f.
Horizon kernel exp(-log(2)*age/h); amplitude kernel a*exp(-log(2)*age/86400).
Horizon preserves unit zero-age weight; amplitude is a scaling control. Both
start exactly at the parent kernel. They do not have identical function classes.
The factorized replacement is match_count * weighted_valid_support / support.
Only the match branch changes; pair support retains the fixed one-day kernel.
No new random draws, custom loss, optimizer, events, fields, labels or I/O.

Nearest inspected primary source: DIN arXiv:1706.06978v4 section 4.3, archived
lines 260-283. Its local relevance activation and intensity motivation are
inherited, while its unnormalized pooling is not reproduced here. DMR author
README at commit 2c6e76804281954a3c969010a31712cd5054aca9 verifies the absence
of history item IDs only; its full paper was unavailable in the registry.
This is a local temporal-kernel experiment, with no verified novelty claim.
"""
import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'statistic_target': 'match', 'statistic_scale': 'count', 'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0, 'pair_statistic': 'support', 'field_pairing': 'aligned', 'field_routing': 'shared', 'context_gain': 0.5, 'query_context': 'blind', 'kernel_mode': 'horizon', 'kernel_alignment': 'factorized'}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.field_pairing = self.config['field_pairing']
        self.field_routing = self.config['field_routing']
        self.query_context = self.config['query_context']
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

        # Appended after parent constructors: preserve all parent RNG draws.
        self.kernel_mode = self.config['kernel_mode']
        self.kernel_alignment = self.config['kernel_alignment']
        if self.kernel_mode not in ('horizon', 'amplitude'):
            raise ValueError('Unknown kernel_mode')
        if self.kernel_alignment not in ('aligned', 'factorized'):
            raise ValueError('Unknown kernel_alignment')
        if self.half_life != 86400.0:
            raise ValueError('The center half-life is frozen at one day')
        self.kernel_theta = nn.Parameter(torch.zeros(2, 4))

    def _kernel_weights(self, ages, field_index, kernel_mode=None):
        # A bounded positive factor; no in-place updates or detached gradients.
        factor = torch.exp(math.log(4.0) * torch.tanh(self.kernel_theta[field_index]))
        day_kernel = torch.exp(-math.log(2.0) * ages.unsqueeze(-1) / self.half_life)
        horizon_kernel = torch.exp(
            -math.log(2.0) * ages.unsqueeze(-1) / (self.half_life * factor))
        amplitude_kernel = day_kernel * factor
        mode = self.kernel_mode if kernel_mode is None else kernel_mode
        if mode == 'horizon':
            return horizon_kernel
        if mode == 'amplitude':
            return amplitude_kernel
        if mode == 'fixed':
            return day_kernel.expand(-1, -1, 4)
        raise ValueError('Unknown diagnostic kernel_mode')

    def _statistics(self, batch, kernel_mode=None):
        mask = batch['history_mask']
        types = batch['history_type']
        action_ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == action_ids) & mask.unsqueeze(-1)).float()
        counts = actions.sum(dim=1)
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        matches, supports, decayed_matches, decayed_supports = [], [], [], []
        for field_index, (column, index) in enumerate((
                ('history_category', self.category), ('history_brand', self.brand))):
            history_ids = batch[column]
            target_ids = batch['static'][:, index:index + 1]
            valid = mask & (history_ids > 0)
            equal = valid & (target_ids > 0) & (history_ids == target_ids)
            valid_float = valid.unsqueeze(-1).float()
            equal_float = equal.unsqueeze(-1).float()
            weighted_actions = actions * self._kernel_weights(ages, field_index, kernel_mode)
            supports.append((actions * valid_float).sum(dim=1))
            matches.append((actions * equal_float).sum(dim=1))
            decayed_supports.append((weighted_actions * valid_float).sum(dim=1))
            decayed_matches.append((weighted_actions * equal_float).sum(dim=1))
        return (counts, torch.stack(matches, dim=1), torch.stack(supports, dim=1),
                torch.stack(decayed_matches, dim=1),
                torch.stack(decayed_supports, dim=1))

    def _evidence_from_statistics(self, statistics, alignment=None):
        _, matches, supports, decayed_matches, decayed_supports = statistics
        mode = self.kernel_alignment if alignment is None else alignment
        if mode == 'aligned':
            extra = decayed_matches
        elif mode == 'factorized':
            extra = matches * decayed_supports / supports.clamp_min(1.0)
        else:
            raise ValueError('Unknown diagnostic alignment')
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
        shared_weights = self._masked_weights(0.5 * (category_score + brand_score), mask)
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
        """45 descriptive scalars; first host-selected 512 impressions only.

        Counterfactuals at one checkpoint demonstrate activation, not value.
        Horizon parameters are predictive summaries, not causal lifetimes.
        """
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._field_views(batch, fields)
        statistics = self._statistics(batch)
        fixed_statistics = self._statistics(batch, kernel_mode='fixed')
        evidence = self._evidence_from_statistics(statistics)
        fixed_evidence = self._evidence_from_statistics(fixed_statistics, alignment='aligned')
        aligned_evidence = self._evidence_from_statistics(statistics, alignment='aligned')
        factorized_evidence = self._evidence_from_statistics(statistics, alignment='factorized')
        pair_evidence = self._pair_evidence(statistics, self._pair_statistics(batch))
        selected_logits = self.output(torch.cat(
            fields + [views['selected'], evidence, pair_evidence], dim=-1)).squeeze(-1)
        fixed_logits = self.output(torch.cat(
            fields + [views['selected'], fixed_evidence, pair_evidence], dim=-1)).squeeze(-1)
        mask = batch['history_mask']
        lengths = mask.sum(-1).float()
        present = (lengths > 0).float()
        factor = torch.exp(math.log(4.0) * torch.tanh(self.kernel_theta))
        result = {
            'diagnostic_sample_count': lengths.new_tensor(float(lengths.numel())),
            'empty_history_count': (lengths == 0).float().sum(),
            'supplied_history_length_mean': lengths.mean(),
            'kernel_factor_min': factor.min(),
            'kernel_factor_max': factor.max(),
            'kernel_log_factor_abs_mean': factor.log().abs().mean(),
            'kernel_saturated_fraction': (torch.tanh(self.kernel_theta).abs() > 0.95).float().mean(),
            'evidence_fixed_distance_mean': (evidence - fixed_evidence).abs().mean(),
            'aligned_factorized_distance_mean': (aligned_evidence - factorized_evidence).abs().mean(),
            'fixed_counterfactual_absolute_logit_change_mean': (selected_logits - fixed_logits).abs().mean(),
            'empty_absolute_logit_change_max': ((selected_logits - fixed_logits).abs() * (1.0 - present)).max(),
            'empty_selected_pool_norm_max': (views['selected'].norm(dim=-1) * (1.0 - present)).max(),
            'parent_shared_weight_sum_error_max': (views['shared_weights'].sum(-1) - present).abs().max(),
        }
        for f, field_name in enumerate(('category', 'brand')):
            for a, action_name in enumerate(('pv', 'cart', 'fav', 'buy')):
                prefix = field_name + '_' + action_name
                result[prefix + '_kernel_factor'] = factor[f, a]
                result[prefix + '_match_count_mean'] = statistics[1][:, f, a].mean()
                result[prefix + '_support_count_mean'] = statistics[2][:, f, a].mean()
                result[prefix + '_selected_mass_mean'] = statistics[3][:, f, a].mean()
        return result
