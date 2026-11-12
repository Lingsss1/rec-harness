"""Prospective r0020: relative-age magnitude versus ordinal chronology.

The hash-verified r0019-latest_activity model is inherited. Only the final
8 deterministic inputs are replaced; no parameter, initialization, optimizer,
loss, support, causal history or main-rank coordinate changes. The factorial
crosses ordinal/ratio encoding with oriented/folded auxiliary evidence.
Folded controls erase direction only from the auxiliary, not the whole model.
Every mode computes every table. Host alone trains and evaluates.
No source-system reproduction, priority or measured efficacy is claimed.
"""
import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'statistic_target': 'match', 'statistic_scale': 'count', 'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0, 'pair_statistic': 'support', 'field_pairing': 'aligned', 'field_routing': 'shared', 'context_gain': 0.5, 'query_context': 'blind', 'extreme_binding': 'match', 'extreme_reduction': 'latest', 'competition_binding': 'nonmatch', 'competition_reduction': 'latest', 'order_binding': 'crossed', 'order_statistic': 'rank', 'order_gate': 'required', 'order_value': 'rank', 'rank_encoding': 'raw', 'rank_auxiliary': 'duplicate', 'anchor_scope': 'same', 'anchor_value': 'rank', 'common_anchor': 'same', 'support_extension': 'extend', 'frontier_reduction': 'latest', 'frontier_population': 'activity', 'margin_encoding': 'ratio', 'margin_orientation': 'folded'}


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

        self.extreme_binding = self.config['extreme_binding']
        self.extreme_reduction = self.config['extreme_reduction']
        if self.extreme_binding not in ('match', 'activity'):
            raise ValueError('Unknown extreme_binding')
        if self.extreme_reduction not in ('latest', 'mean'):
            raise ValueError('Unknown extreme_reduction')
        if self.query_context != 'blind' or self.half_life != 86400.0:
            raise ValueError('The blind parent query and one-day decay are frozen')
        # Retain all parent initialization draws, including downstream layers.
        # Extra features are appended after the old last feature coordinate.
        first = self.output[0]
        zeros = first.weight.new_zeros(first.out_features, 8)
        first.weight = nn.Parameter(torch.cat([first.weight, zeros], dim=1))
        first.in_features += 8

        # The inherited eight extrema remain frozen in every factorial cell.
        if self.extreme_binding != 'match' or self.extreme_reduction != 'latest':
            raise ValueError('The r0010 match/latest incumbent is frozen')
        self.competition_binding = self.config['competition_binding']
        self.competition_reduction = self.config['competition_reduction']
        if self.competition_binding not in ('nonmatch', 'activity'):
            raise ValueError('Unknown competition_binding')
        if self.competition_reduction not in ('latest', 'mean'):
            raise ValueError('Unknown competition_reduction')
        # Parent classifier is already fully constructed. No RNG consumption.
        # Retain its original coordinates and append only the new eight columns.
        first = self.output[0]
        zeros = first.weight.new_zeros(first.out_features, 8)
        first.weight = nn.Parameter(torch.cat([first.weight, zeros], dim=1))
        first.in_features += 8

        if self.competition_binding != 'nonmatch' or self.competition_reduction != 'latest':
            raise ValueError('The full r0011-nonmatch_latest parent is frozen')
        self.order_binding = self.config['order_binding']
        self.order_statistic = self.config['order_statistic']
        if self.order_binding not in ('aligned', 'crossed'):
            raise ValueError('Unknown order_binding')
        if self.order_statistic not in ('rank', 'decay'):
            raise ValueError('Unknown order_statistic')
        # Parent parameter shapes and RNG draws above remain unchanged.
        first = self.output[0]
        zeros = first.weight.new_zeros(first.out_features, 8)
        first.weight = nn.Parameter(torch.cat([first.weight, zeros], dim=1))
        first.in_features += 8

        # Pure configuration selectors: preserve every ancestor RNG draw,
        # parameter shape and initialization, including final eight zeros.
        self.order_gate = self.config['order_gate']
        self.order_value = self.config['order_value']
        if self.order_gate not in ('common', 'required'):
            raise ValueError('Unknown order_gate')
        if self.order_value not in ('rank', 'presence'):
            raise ValueError('Unknown order_value')
        if self.order_binding != 'crossed' or self.order_statistic != 'rank':
            raise ValueError('The crossed/rank ancestor is frozen')

        # Pure selectors and deterministic zero columns; no random draws.
        if self.order_gate != 'required' or self.order_value != 'rank':
            raise ValueError('The full required/rank parent is frozen')
        self.rank_encoding = self.config['rank_encoding']
        self.rank_auxiliary = self.config['rank_auxiliary']
        if self.rank_encoding not in ('raw', 'centered'):
            raise ValueError('Unknown rank_encoding')
        if self.rank_auxiliary not in ('support', 'duplicate'):
            raise ValueError('Unknown rank_auxiliary')
        first = self.output[0]
        zeros = first.weight.new_zeros(first.out_features, 8)
        first.weight = nn.Parameter(torch.cat([first.weight, zeros], dim=1))
        first.in_features += 8

        # Pure scope/value selectors: parameter shapes and RNG are untouched.
        if self.rank_encoding != 'raw' or self.rank_auxiliary != 'duplicate':
            raise ValueError('The r0015 raw/duplicate source parent is frozen')
        self.anchor_scope = self.config['anchor_scope']
        self.anchor_value = self.config['anchor_value']
        if self.anchor_scope not in ('same', 'pooled'):
            raise ValueError('Unknown anchor_scope')
        if self.anchor_value not in ('rank', 'presence'):
            raise ValueError('Unknown anchor_value')

        # Pure selectors: no parameters, buffers, modules or RNG draws added.
        if (self.anchor_scope != 'same' or self.anchor_value != 'rank' or
                self.rank_encoding != 'raw' or self.rank_auxiliary != 'duplicate'):
            raise ValueError('The raw same-action rank parent is frozen')
        self.common_anchor = self.config['common_anchor']
        self.support_extension = self.config['support_extension']
        if self.common_anchor not in ('same', 'pooled'):
            raise ValueError('Unknown common_anchor')
        if self.support_extension not in ('closed', 'extend'):
            raise ValueError('Unknown support_extension')

        # Deterministic selectors only: retain every parent parameter/RNG draw.
        if self.common_anchor != 'same' or self.support_extension != 'extend':
            raise ValueError('The r0017 fallback anchor/support policy is frozen')
        self.frontier_reduction = self.config['frontier_reduction']
        self.frontier_population = self.config['frontier_population']
        if self.frontier_reduction not in ('latest', 'mean'):
            raise ValueError('Unknown frontier_reduction')
        if self.frontier_population not in ('nonmatch', 'activity'):
            raise ValueError('Unknown frontier_population')

        if self.frontier_reduction != 'latest' or self.frontier_population != 'activity':
            raise ValueError('The r0019 latest/activity source parent is frozen')
        self.margin_encoding = self.config['margin_encoding']
        self.margin_orientation = self.config['margin_orientation']
        if self.margin_encoding not in ('ordinal', 'ratio'):
            raise ValueError('Unknown margin_encoding')
        if self.margin_orientation not in ('oriented', 'folded'):
            raise ValueError('Unknown margin_orientation')

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

    def _extreme_statistics(self, batch, statistics):
        """Four [B,2,4] tables; maxima need no chronological array ordering.

        A zero sentinel handles L=0 as well as absent action/field populations.
        Ages underflowing to zero decay carry zero mass, without invalid divides.
        All fields/actions preserve the parent's valid event populations.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == ids) & mask.unsqueeze(-1)).float()
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        weighted = actions * decay.unsqueeze(-1)
        sentinel = weighted.new_zeros(weighted.shape[0], 1, 4)
        match_max, activity_max = [], []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            history = batch[column]
            target = batch['static'][:, index:index + 1]
            valid = mask & (history > 0)
            equal = valid & (target > 0) & (history == target)
            match_mass = weighted * equal.unsqueeze(-1).float()
            activity_mass = weighted * valid.unsqueeze(-1).float()
            match_max.append(torch.cat([match_mass, sentinel], dim=1).amax(dim=1))
            activity_max.append(torch.cat([activity_mass, sentinel], dim=1).amax(dim=1))
        _, count, support, mass, support_mass = statistics
        present = (count > 0).float()
        return {
            'match_latest': torch.stack(match_max, dim=1),
            'activity_latest': torch.stack(activity_max, dim=1) * present,
            'match_mean': mass / count.clamp_min(1.0),
            'activity_mean': support_mass / support.clamp_min(1.0) * present,
        }

    def _selected_extreme(self, summaries):
        return summaries[self.extreme_binding + '_' + self.extreme_reduction]

    def _base_features(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._field_views(batch, fields)
        statistics = self._statistics(batch)
        pairs = self._pair_statistics(batch)
        evidence = self._evidence_from_statistics(statistics)
        pair_evidence = self._pair_evidence(statistics, pairs)
        base = torch.cat(fields + [views['selected'], evidence, pair_evidence], dim=-1)
        return base, views, statistics

    def _competition_statistics(self, batch):
        """All four summaries and common support gate, each [B,2,4].

        Known positive target IDs and event field IDs are required. Exact
        inequality selects a complementary event population, without assigning
        a negative label or changing the original parent retrieval/masking.
        Direct population sums avoid subtracting nearly equal decay masses.
        A zero sentinel permits a zero-length history dimension.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == ids) & mask.unsqueeze(-1)).float()
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        sentinel = actions.new_zeros(actions.shape[0], 1, 4)
        nonmatch_latest, activity_latest = [], []
        nonmatch_mean, activity_mean, gates = [], [], []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            history = batch[column]
            target = batch['static'][:, index:index + 1]
            valid = mask & (history > 0) & (target > 0)
            nonmatch = valid & (history != target)
            valid_actions = actions * valid.unsqueeze(-1).float()
            nonmatch_actions = actions * nonmatch.unsqueeze(-1).float()
            support = valid_actions.sum(dim=1)
            count = nonmatch_actions.sum(dim=1)
            present = (count > 0).float()
            activity_mass = valid_actions * decay.unsqueeze(-1)
            nonmatch_mass = nonmatch_actions * decay.unsqueeze(-1)
            activity_latest.append(
                torch.cat([activity_mass, sentinel], dim=1).amax(dim=1) * present)
            nonmatch_latest.append(
                torch.cat([nonmatch_mass, sentinel], dim=1).amax(dim=1) * present)
            activity_mean.append(
                activity_mass.sum(dim=1) / support.clamp_min(1.0) * present)
            nonmatch_mean.append(
                nonmatch_mass.sum(dim=1) / count.clamp_min(1.0) * present)
            gates.append(present)
        return {
            'nonmatch_latest': torch.stack(nonmatch_latest, dim=1),
            'activity_latest': torch.stack(activity_latest, dim=1),
            'nonmatch_mean': torch.stack(nonmatch_mean, dim=1),
            'activity_mean': torch.stack(activity_mean, dim=1),
            'present': torch.stack(gates, dim=1),
        }

    def _selected_competition(self, summaries, binding=None, reduction=None):
        binding = self.competition_binding if binding is None else binding
        reduction = self.competition_reduction if reduction is None else reduction
        return summaries[binding + '_' + reduction]

    def _incumbent_features(self, batch):
        base, views, statistics = self._base_features(batch)
        extremes = self._extreme_statistics(batch, statistics)
        incumbent = torch.cat([
            base, self._selected_extreme(extremes).flatten(start_dim=1)
        ], dim=-1)
        return incumbent, views

    def _parent_features(self, batch):
        incumbent, views = self._incumbent_features(batch)
        competition = self._competition_statistics(batch)
        return torch.cat([
            incumbent, self._selected_competition(competition).flatten(start_dim=1)
        ], dim=-1), views

    def _order_statistics(self, batch):
        """Compute [B,2,4] summaries without relying on array ordering.

        For each field/action, take the minimum MATCH age as its anchor.
        Rank = mean(1[nonmatch_age < anchor] + .5*1[equal]).
        Decay proxy = .5 + .5*(mean(nonmatch_decay) - anchor_decay).
        Swapping category/brand anchors changes binding, not populations.
        Original summaries use the ancestor common gate: BOTH fields have a
        known-target match AND a nonmatch for the given action. Required-rank
        variants only need nonmatches in the receiving field and a match in
        the chosen anchor field. Both gates are computed without labels.
        Raw ages preserve order even when one-day decay numerically underflows.
        Absent minima are replaced by zero before further arithmetic.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        ids = torch.arange(1, 5, device=types.device)
        action_mask = (types.unsqueeze(-1) == ids) & mask.unsqueeze(-1)
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        expanded_ages = ages.unsqueeze(-1).expand(-1, -1, 4)
        sentinel = ages.new_full((ages.shape[0], 1, 4), torch.finfo(ages.dtype).max)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        anchors, match_presence, competitors, counts, means = [], [], [], [], []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            target = batch['static'][:, index:index + 1]
            history = batch[column]
            valid = mask & (history > 0) & (target > 0)
            matches = action_mask & (valid & (history == target)).unsqueeze(-1)
            nonmatches = action_mask & (valid & (history != target)).unsqueeze(-1)
            present = matches.any(dim=1)
            minimum = torch.cat([
                expanded_ages.masked_fill(~matches, torch.finfo(ages.dtype).max),
                sentinel
            ], dim=1).amin(dim=1)
            anchors.append(torch.where(present, minimum, torch.zeros_like(minimum)))
            match_presence.append(present)
            nonmatch_weights = nonmatches.float()
            count = nonmatch_weights.sum(dim=1)
            competitors.append(nonmatch_weights)
            counts.append(count)
            means.append((nonmatch_weights * decay.unsqueeze(-1)).sum(dim=1) /
                         count.clamp_min(1.0))
        anchor_table = torch.stack(anchors, dim=1)
        local_present = torch.stack(match_presence, dim=1) & (torch.stack(counts, dim=1) > 0)
        # Same gate in every field and factorial cell. Known-target status is
        # inherited from each population's validity test, including crossed.
        gate = local_present.all(dim=1, keepdim=True).expand(-1, 2, -1).float()
        result = {'present': gate, 'local_present': local_present.float(),
                  'anchors': anchor_table}
        for binding in ('aligned', 'crossed'):
            ranks, proxies = [], []
            for field in range(2):
                anchor_field = field if binding == 'aligned' else 1 - field
                anchor = anchors[anchor_field]
                newer = (ages.unsqueeze(-1) < anchor.unsqueeze(1)).float()
                equal = (ages.unsqueeze(-1) == anchor.unsqueeze(1)).float()
                rank = ((newer + 0.5 * equal) * competitors[field]).sum(dim=1)
                rank = rank / counts[field].clamp_min(1.0)
                anchor_decay = torch.exp(-math.log(2.0) * anchor / self.half_life)
                proxy = 0.5 + 0.5 * (means[field] - anchor_decay)
                ranks.append(rank)
                proxies.append(proxy)
            result[binding + '_rank'] = torch.stack(ranks, dim=1) * gate
            result[binding + '_decay'] = torch.stack(proxies, dim=1) * gate
            # Only the populations used in this conditional rank are needed.
            # For crossed binding they are local nonmatches and OTHER-field
            # matches. A missing receiving-field target already zeros its
            # nonmatch count; a missing anchor-field target zeros its presence.
            required_gate = torch.stack([
                match_presence[0 if binding == 'aligned' else 1] & (counts[0] > 0),
                match_presence[1 if binding == 'aligned' else 0] & (counts[1] > 0),
            ], dim=1).float()
            result[binding + '_required_present'] = required_gate
            result[binding + '_required_rank'] = torch.stack(ranks, dim=1) * required_gate
        return result

    def _selected_order(self, summaries, gate=None, value=None):
        gate = self.order_gate if gate is None else gate
        value = self.order_value if value is None else value
        if gate not in ('common', 'required') or value not in ('rank', 'presence'):
            raise ValueError('Unknown order replacement')
        support = (summaries['present'] if gate == 'common' else
                   summaries['crossed_required_present'])
        if value == 'presence':
            # Fixed uninformative midrank; support is the ONLY changing input.
            return 0.5 * support
        return (summaries['crossed_rank'] if gate == 'common' else
                summaries['crossed_required_rank'])

    def _rank_encodings(self, summaries):
        """All modes compute identical summaries, without modifying the parent."""
        gate = summaries['crossed_required_present']
        raw = summaries['crossed_required_rank']
        centered = raw - 0.5 * gate
        return {
            'gate': gate, 'raw': raw, 'centered': centered,
            'support': 0.5 * gate,
            'raw_duplicate': 0.5 * raw,
            'centered_duplicate': 0.5 * centered,
        }

    def _rank_inputs(self, encodings, encoding=None, auxiliary=None):
        encoding = self.rank_encoding if encoding is None else encoding
        auxiliary = self.rank_auxiliary if auxiliary is None else auxiliary
        if encoding not in ('raw', 'centered'):
            raise ValueError('Unknown diagnostic rank encoding')
        if auxiliary not in ('support', 'duplicate'):
            raise ValueError('Unknown diagnostic rank auxiliary')
        value = encodings[encoding]
        extra = (encodings['support'] if auxiliary == 'support' else
                 encodings[encoding + '_duplicate'])
        return torch.cat([value.flatten(start_dim=1),
                          extra.flatten(start_dim=1)], dim=-1)

    def _anchor_scope_statistics(self, batch, order):
        """All tensors [B,2,4] except anchor tables [B,2].

        Same-action results are the exact inherited tensors. Pooled anchors
        use valid known-type events whose field equals a positive target.
        Receiving field nonmatches still belong to their own action. Unknown
        targets never create matches or nonmatches. No event ordering assumed.
        A finite sentinel makes zero-length histories well-defined; absent
        anchor ages become zero before comparisons. Gates zero their results.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        ids = torch.arange(1, 5, device=types.device)
        action_mask = (types.unsqueeze(-1) == ids) & mask.unsqueeze(-1)
        known_action = action_mask.any(dim=-1)
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        sentinel = ages.new_full((ages.shape[0], 1), torch.finfo(ages.dtype).max)
        anchors, anchor_present, competitors, counts = [], [], [], []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            target = batch['static'][:, index:index + 1]
            history = batch[column]
            valid = mask & (history > 0) & (target > 0)
            matches = valid & (history == target) & known_action
            nonmatches = action_mask & (valid & (history != target)).unsqueeze(-1)
            present = matches.any(dim=1)
            minimum = torch.cat([
                ages.masked_fill(~matches, torch.finfo(ages.dtype).max), sentinel
            ], dim=1).amin(dim=1)
            anchors.append(torch.where(present, minimum, torch.zeros_like(minimum)))
            anchor_present.append(present)
            weights = nonmatches.float()
            competitors.append(weights)
            counts.append(weights.sum(dim=1))
        ranks, gates = [], []
        for field in range(2):
            # Fixed crossed-field binding, only action scope changes.
            anchor_field = 1 - field
            anchor = anchors[anchor_field][:, None, None]
            newer = (ages.unsqueeze(-1) < anchor).float()
            equal = (ages.unsqueeze(-1) == anchor).float()
            rank = ((newer + 0.5 * equal) * competitors[field]).sum(dim=1)
            rank = rank / counts[field].clamp_min(1.0)
            gate = ((counts[field] > 0) &
                    anchor_present[anchor_field][:, None]).float()
            ranks.append(rank * gate)
            gates.append(gate)
        return {
            'same_rank': order['crossed_required_rank'],
            'same_presence': order['crossed_required_present'],
            'pooled_rank': torch.stack(ranks, dim=1),
            'pooled_presence': torch.stack(gates, dim=1),
        }

    def _routing_tables(self, scopes):
        """Same support S is a subset of pooled support P, by construction.

        Four auxiliary values, before the fixed .5 multiplier:
          same_closed = r
          same_extend = r on S, pooled rank on P\S, else zero
          pooled_closed = pooled rank on S, else zero
          pooled_extend = pooled rank on P, else zero.
        The main eight rank coordinates stay r in every factorial cell.
        All tables are computed in every variant, using no new raw inputs.
        """
        same = scopes['same_rank']
        pooled = scopes['pooled_rank']
        gate = scopes['same_presence']
        return {
            'same_closed': same,
            'same_extend': torch.where(gate.bool(), same, pooled),
            'pooled_closed': pooled * gate,
            'pooled_extend': pooled,
        }

    def _routing_inputs(self, scopes, tables, common=None, extension=None):
        common = self.common_anchor if common is None else common
        extension = self.support_extension if extension is None else extension
        if common not in ('same', 'pooled') or extension not in ('closed', 'extend'):
            raise ValueError('Unknown diagnostic routing selector')
        return torch.cat([
            scopes['same_rank'].flatten(start_dim=1),
            (0.5 * tables[common + '_' + extension]).flatten(start_dim=1),
        ], dim=-1)

    def _frontier_statistics(self, batch, order, scopes, tables):
        """All ordinal summaries [B,2,4] on identical inherited support.

        A receiving coordinate keeps its crossed same-action match anchor if
        available, else the crossed pooled-action match anchor. Nonmatches
        and all valid activity retain the receiving action. Raw-age minima
        need no ordering and preserve ties even when decay underflows.
        Latest rank = 1[min_age < anchor] + .5*1[min_age == anchor].
        Mean rank uses that comparison averaged over the event population.
        A finite sentinel and zero absent minima handle padding and L=0.
        Every cell computes every summary; only the classifier input differs.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        ids = torch.arange(1, 5, device=types.device)
        action_mask = (types.unsqueeze(-1) == ids) & mask.unsqueeze(-1)
        known_action = action_mask.any(dim=-1)
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        expanded_ages = ages.unsqueeze(-1).expand(-1, -1, 4)
        sentinel = ages.new_full((ages.shape[0], 1, 4), torch.finfo(ages.dtype).max)
        pooled_sentinel = ages.new_full((ages.shape[0], 1), torch.finfo(ages.dtype).max)
        gate = scopes['pooled_presence']
        values = {key: [] for key in ('latest_nonmatch', 'latest_activity',
                                      'mean_nonmatch', 'mean_activity')}
        count_tables = {'nonmatch': [], 'activity': []}
        for field, (column, index) in enumerate((('history_category', self.category),
                                                ('history_brand', self.brand))):
            history = batch[column]
            target = batch['static'][:, index:index + 1]
            valid = mask & (history > 0) & (target > 0)
            nonmatch = action_mask & (valid & (history != target)).unsqueeze(-1)
            activity = action_mask & valid.unsqueeze(-1)
            other_column = 'history_brand' if field == 0 else 'history_category'
            other_index = self.brand if field == 0 else self.category
            other_history = batch[other_column]
            other_target = batch['static'][:, other_index:other_index + 1]
            pooled_matches = (mask & known_action & (other_history > 0) &
                              (other_target > 0) & (other_history == other_target))
            pooled_minimum = torch.cat([
                ages.masked_fill(~pooled_matches, torch.finfo(ages.dtype).max),
                pooled_sentinel
            ], dim=1).amin(dim=1)
            pooled_anchor = torch.where(pooled_matches.any(dim=1), pooled_minimum,
                                        torch.zeros_like(pooled_minimum))
            anchor = torch.where(scopes['same_presence'][:, field, :].bool(),
                                 order['anchors'][:, 1 - field, :],
                                 pooled_anchor[:, None])
            comparison = ((expanded_ages < anchor.unsqueeze(1)).float() +
                          0.5 * (expanded_ages == anchor.unsqueeze(1)).float())
            for population, population_mask in (('nonmatch', nonmatch),
                                                 ('activity', activity)):
                weights = population_mask.float()
                count = weights.sum(dim=1)
                count_tables[population].append(count)
                mean = (weights * comparison).sum(dim=1) / count.clamp_min(1.0)
                minimum = torch.cat([
                    expanded_ages.masked_fill(~population_mask, torch.finfo(ages.dtype).max),
                    sentinel
                ], dim=1).amin(dim=1)
                latest_age = torch.where(count > 0, minimum, torch.zeros_like(minimum))
                latest = ((latest_age < anchor).float() +
                          0.5 * (latest_age == anchor).float())
                values['mean_' + population].append(mean * gate[:, field, :])
                values['latest_' + population].append(latest * gate[:, field, :])
        result = {key: torch.stack(parts, dim=1) for key, parts in values.items()}
        result['recomputed_mean_nonmatch'] = result['mean_nonmatch']
        # Exact historical arithmetic for the parent-reproduction cell.
        result['mean_nonmatch'] = tables['same_extend']
        result['gate'] = gate
        result['nonmatch_count'] = torch.stack(count_tables['nonmatch'], dim=1)
        result['activity_count'] = torch.stack(count_tables['activity'], dim=1)
        return result

    def _frontier_inputs(self, scopes, frontier, reduction=None, population=None):
        reduction = self.frontier_reduction if reduction is None else reduction
        population = self.frontier_population if population is None else population
        if reduction not in ('latest', 'mean') or population not in ('nonmatch', 'activity'):
            raise ValueError('Unknown frontier replacement')
        return torch.cat([
            scopes['same_rank'].flatten(start_dim=1),
            (0.5 * frontier[reduction + '_' + population]).flatten(start_dim=1),
        ], dim=-1)

    def _margin_statistics(self, batch, order, scopes, frontier):
        """[B,2,4] relative-age tables, using the exact parent populations.

        a = crossed match anchor age, c = latest receiving activity age.
        Same-action anchor is retained where present; otherwise pooled-action
        match anchor is used. Parent nonmatch/anchor gate is UNCHANGED.
        Ratio = (1+a)/(2+a+c) in seconds, evaluated after max scaling to
        avoid overflow. This is a deterministic covariate, not a probability
        estimate. The one-second offset is fixed, never calibrated.
        Ordinal uses the EXACT parent tensor. Folding maps r to .5+|r-.5|
        on support. Ties retain .5; no event obtains an outcome label.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        ids = torch.arange(1, 5, device=types.device)
        action_mask = (types.unsqueeze(-1) == ids) & mask.unsqueeze(-1)
        known_action = action_mask.any(dim=-1)
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        expanded = ages.unsqueeze(-1).expand(-1, -1, 4)
        sentinel = ages.new_full((ages.shape[0], 1, 4), torch.finfo(ages.dtype).max)
        pooled_sentinel = ages.new_full((ages.shape[0], 1), torch.finfo(ages.dtype).max)
        gate = frontier['gate']
        ratios, ordinals, anchor_ages, activity_ages = [], [], [], []
        for field, (column, index) in enumerate((('history_category', self.category),
                                                ('history_brand', self.brand))):
            history = batch[column]
            target = batch['static'][:, index:index + 1]
            activity = action_mask & (mask & (history > 0) & (target > 0)).unsqueeze(-1)
            minimum = torch.cat([
                expanded.masked_fill(~activity, torch.finfo(ages.dtype).max), sentinel
            ], dim=1).amin(dim=1)
            latest = torch.where(activity.any(dim=1), minimum, torch.zeros_like(minimum))
            other_column = 'history_brand' if field == 0 else 'history_category'
            other_index = self.brand if field == 0 else self.category
            other_history = batch[other_column]
            other_target = batch['static'][:, other_index:other_index + 1]
            matches = (mask & known_action & (other_history > 0) &
                       (other_target > 0) & (other_history == other_target))
            pooled_minimum = torch.cat([
                ages.masked_fill(~matches, torch.finfo(ages.dtype).max), pooled_sentinel
            ], dim=1).amin(dim=1)
            pooled = torch.where(matches.any(dim=1), pooled_minimum,
                                 torch.zeros_like(pooled_minimum))
            anchor = torch.where(scopes['same_presence'][:, field, :].bool(),
                                 order['anchors'][:, 1 - field, :], pooled[:, None])
            supported = gate[:, field, :].bool()
            anchor = torch.where(supported, anchor, torch.zeros_like(anchor))
            latest = torch.where(supported, latest, torch.zeros_like(latest))
            scale = torch.maximum(torch.maximum(anchor, latest), torch.ones_like(anchor))
            a, c, offset = anchor / scale, latest / scale, 1.0 / scale
            ratio = (a + offset) / (a + c + 2.0 * offset)
            ratios.append(ratio * gate[:, field, :])
            ordinals.append(((latest < anchor).float() +
                             0.5 * (latest == anchor).float()) * gate[:, field, :])
            anchor_ages.append(anchor)
            activity_ages.append(latest)
        ratio = torch.stack(ratios, dim=1)
        ordinal = frontier['latest_activity']
        return {
            'ordinal_oriented': ordinal,
            'ordinal_folded': torch.maximum(ordinal, gate - ordinal),
            'ratio_oriented': ratio,
            'ratio_folded': torch.maximum(ratio, gate - ratio),
            'recomputed_ordinal': torch.stack(ordinals, dim=1),
            'gate': gate,
            'anchor_age': torch.stack(anchor_ages, dim=1),
            'activity_age': torch.stack(activity_ages, dim=1),
        }

    def _margin_inputs(self, scopes, margins, encoding=None, orientation=None):
        encoding = self.margin_encoding if encoding is None else encoding
        orientation = self.margin_orientation if orientation is None else orientation
        if encoding not in ('ordinal', 'ratio') or orientation not in ('oriented', 'folded'):
            raise ValueError('Unknown margin replacement')
        return torch.cat([
            scopes['same_rank'].flatten(start_dim=1),
            (0.5 * margins[encoding + '_' + orientation]).flatten(start_dim=1),
        ], dim=-1)

    def forward(self, batch):
        parent, _ = self._parent_features(batch)
        order = self._order_statistics(batch)
        scopes = self._anchor_scope_statistics(batch, order)
        tables = self._routing_tables(scopes)
        frontier = self._frontier_statistics(batch, order, scopes, tables)
        margins = self._margin_statistics(batch, order, scopes, frontier)
        return self.output(torch.cat([
            parent, self._margin_inputs(scopes, margins)
        ], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        """Candidate-supplied first512 checkpoint sensitivities, no labels.

        These are implementation probes, not retrained ablations or population
        mechanism evidence. Conditional averages use support denominator >=1.
        """
        parent, views = self._parent_features(batch)
        order = self._order_statistics(batch)
        scopes = self._anchor_scope_statistics(batch, order)
        tables = self._routing_tables(scopes)
        frontier = self._frontier_statistics(batch, order, scopes, tables)
        margins = self._margin_statistics(batch, order, scopes, frontier)
        inputs = self._margin_inputs(scopes, margins)
        actual = self.output(torch.cat([parent, inputs], dim=-1)).squeeze(-1)
        other_encoding = 'ordinal' if self.margin_encoding == 'ratio' else 'ratio'
        other_orientation = 'folded' if self.margin_orientation == 'oriented' else 'oriented'
        replaced_encoding = self.output(torch.cat([
            parent, self._margin_inputs(scopes, margins, encoding=other_encoding)
        ], dim=-1)).squeeze(-1)
        replaced_orientation = self.output(torch.cat([
            parent, self._margin_inputs(scopes, margins, orientation=other_orientation)
        ], dim=-1)).squeeze(-1)
        parent_inputs = self._frontier_inputs(scopes, frontier)
        parent_logits = self.output(torch.cat([parent, parent_inputs], dim=-1)).squeeze(-1)
        zero_inputs = torch.cat([scopes['same_rank'].flatten(start_dim=1),
                                 torch.zeros_like(scopes['same_rank']).flatten(start_dim=1)], dim=-1)
        zero_logits = self.output(torch.cat([parent, zero_inputs], dim=-1)).squeeze(-1)
        selected = margins[self.margin_encoding + '_' + self.margin_orientation]
        gate = margins['gate']
        ratio, ordinal = margins['ratio_oriented'], margins['ordinal_oriented']
        anchor, latest = margins['anchor_age'], margins['activity_age']
        ties = (anchor == latest).float() * gate
        gap = (ratio - ordinal).abs()
        empty = (~batch['history_mask'].any(dim=1)).float()
        missing = ((batch['static'][:, self.category] == 0) |
                   (batch['static'][:, self.brand] == 0)).float()
        result = {
            'diagnostic_sample_count': selected.new_tensor(selected.shape[0]),
            'empty_history_count': empty.sum(),
            'missing_target_count': missing.sum(),
            'same_support_fraction': scopes['same_presence'].mean(),
            'selected_support_fraction': gate.mean(),
            'fallback_support_fraction': (gate - scopes['same_presence']).mean(),
            'support_subset_violation_max': (scopes['same_presence'] - gate).clamp_min(0.0).max(),
            'ordinal_recompute_error_max': (ordinal - margins['recomputed_ordinal']).abs().max(),
            'parent_input_identity_error_max': (self._margin_inputs(scopes, margins, 'ordinal', 'oriented') - parent_inputs).abs().max(),
            'main_rank_identity_error_max': (inputs[:, :8] - parent_inputs[:, :8]).abs().max(),
            'selected_bounds_violation_max': torch.maximum((-selected).clamp_min(0.0).max(), (selected - 1.0).clamp_min(0.0).max()),
            'outside_support_abs_max': (selected.abs() * (1.0 - gate)).max(),
            'empty_inputs_abs_max': (inputs.abs() * empty[:, None]).max(),
            'missing_inputs_abs_max': (inputs.abs() * missing[:, None]).max(),
            'tie_midpoint_error_max': ((ratio - 0.5).abs() * ties).max(),
            'ratio_direction_violation_max': ((0.5 - ratio) * (anchor > latest).float() * gate + (ratio - 0.5) * (anchor < latest).float() * gate).clamp_min(0.0).max(),
            'folding_identity_error_max': (margins['ratio_folded'] - (0.5 * gate + (ratio - 0.5 * gate).abs())).abs().max(),
            'ratio_ordinal_gap_given_support': (gap * gate).sum() / gate.sum().clamp_min(1.0),
            'tie_fraction_given_support': ties.sum() / gate.sum().clamp_min(1.0),
            'ratio_changed_given_support': ((gap > 1e-6).float() * gate).sum() / gate.sum().clamp_min(1.0),
            'selected_aux_abs_mean': (0.5 * selected).abs().mean(),
            'encoding_replacement_logit_distance_mean': (actual - replaced_encoding).abs().mean(),
            'orientation_replacement_logit_distance_mean': (actual - replaced_orientation).abs().mean(),
            'parent_replacement_logit_distance_mean': (actual - parent_logits).abs().mean(),
            'zero_aux_logit_distance_mean': (actual - zero_logits).abs().mean(),
            'aux_input_weight_norm': self.output[0].weight[:, -8:].norm(),
            'parent_weight_sum_error_max': (views['shared_weights'].sum(-1) - (1.0 - empty)).abs().max(),
            'empty_parent_pool_norm_max': (views['selected'].norm(dim=-1) * empty).max(),
        }
        for f, field in enumerate(('category', 'brand')):
            for a, action in enumerate(('pv', 'cart', 'fav', 'buy')):
                prefix = field + '_' + action + '_'
                local_gate = gate[:, f, a]
                denom = local_gate.sum().clamp_min(1.0)
                result[prefix + 'support_fraction'] = local_gate.mean()
                result[prefix + 'ratio_ordinal_gap'] = (gap[:, f, a] * local_gate).sum() / denom
                result[prefix + 'tie_fraction'] = ties[:, f, a].sum() / denom
        return result
