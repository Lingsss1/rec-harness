"""Required support versus order variability: prospective replacement factorial.

Exact parent: r0012-crossed_rank, source SHA256
2a5369ebf80f473b7b722bf817544a6818c0aa3a30fc480f9650925de36780ab.
Preserve all embeddings, normalized shared retrieval, preceding 40 scalar
coordinates, constructor RNG draws, classifier shape and training recipe.
Replace only the final eight crossed rank inputs. Required support for field f
uses nonmatches in f and a match anchor in the OTHER field; the older common
gate additionally requires the two unused populations. Presence controls use
0.5 times the chosen gate. All summaries are computed in all new variants.
Historical common/rank is the fourth factorial cell; do not count it as a new
independent result. No new parameters or tensor initialization are introduced.

Category/brand histories do not identify items. No causal-transition, negative
preference, priority, full-DIN or full-DMR reproduction claim is made. DIN's
primary section 4.3 uses unnormalized activation; normalized ancestor retrieval
is retained here. DMR access was limited to its author README.

Host-only scratch execution and validation follow proposal.json. Diagnostics
are descriptive replacements on the first 512 impressions, not retrained
ablation effects or outcome estimates. No local training is authorized.
"""
import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'statistic_target': 'match', 'statistic_scale': 'count', 'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0, 'pair_statistic': 'support', 'field_pairing': 'aligned', 'field_routing': 'shared', 'context_gain': 0.5, 'query_context': 'blind', 'extreme_binding': 'match', 'extreme_reduction': 'latest', 'competition_binding': 'nonmatch', 'competition_reduction': 'latest', 'order_binding': 'crossed', 'order_statistic': 'rank', 'order_gate': 'required', 'order_value': 'presence'}


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

    def forward(self, batch):
        parent, _ = self._parent_features(batch)
        summaries = self._order_statistics(batch)
        extra = self._selected_order(summaries).flatten(start_dim=1)
        return self.output(torch.cat([parent, extra], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        """56 descriptive scalars; no labels or population AUC are used.

        Both gate/value replacements are made at one frozen checkpoint and
        share all preceding features. They measure coordinate use, not the
        effects of retraining any control. Gate zeros can coincide with genuine
        zero ranks, so nonzero-feature frequency is NOT support frequency.
        """
        parent, views = self._parent_features(batch)
        summaries = self._order_statistics(batch)
        selected = self._selected_order(summaries)
        common_gate = summaries['present']
        required_gate = summaries['crossed_required_present']
        selected_gate = common_gate if self.order_gate == 'common' else required_gate
        common_rank = summaries['crossed_rank']
        required_rank = summaries['crossed_required_rank']
        new_gate = required_gate - common_gate
        other_gate = 'required' if self.order_gate == 'common' else 'common'
        other_value = 'presence' if self.order_value == 'rank' else 'rank'
        gate_control = self._selected_order(summaries, gate=other_gate)
        value_control = self._selected_order(summaries, value=other_value)
        empty = (~batch['history_mask'].any(dim=1)).float()
        any_missing_target = ((batch['static'][:, self.category] == 0) |
                              (batch['static'][:, self.brand] == 0)).float()
        actual = self.output(torch.cat([parent, selected.flatten(start_dim=1)], dim=-1)).squeeze(-1)
        zero = self.output(torch.cat([parent, torch.zeros_like(selected).flatten(start_dim=1)], dim=-1)).squeeze(-1)
        gated = self.output(torch.cat([parent, gate_control.flatten(start_dim=1)], dim=-1)).squeeze(-1)
        valued = self.output(torch.cat([parent, value_control.flatten(start_dim=1)], dim=-1)).squeeze(-1)
        result = {
            'diagnostic_sample_count': selected.new_tensor(selected.shape[0]),
            'empty_history_count': empty.sum(),
            'common_support_fraction': common_gate.mean(),
            'required_support_fraction': required_gate.mean(),
            'new_support_fraction': new_gate.mean(),
            'selected_support_fraction': selected_gate.mean(),
            'support_subset_violation_max': (common_gate - required_gate).clamp_min(0.0).max(),
            'common_rank_agreement_error_max': ((required_rank - common_rank).abs() * common_gate).max(),
            'required_rank_outside_support_abs_max': (required_rank.abs() * (1.0 - required_gate)).max(),
            'selected_outside_support_abs_max': (selected.abs() * (1.0 - selected_gate)).max(),
            'empty_selected_abs_max': (selected.abs() * empty[:, None, None]).max(),
            'missing_required_target_selected_abs_max': (selected.abs() * any_missing_target[:, None, None]).max(),
            'selected_min': selected.min(),
            'selected_max': selected.max(),
            'rank_bounds_violation_max': torch.stack([
                (-required_rank).clamp_min(0.0).max(),
                (required_rank - 1.0).clamp_min(0.0).max(),
            ]).max(),
            'gate_feature_distance_mean': (selected - gate_control).abs().mean(),
            'value_feature_distance_mean': (selected - value_control).abs().mean(),
            'extra_zero_logit_distance_mean': (actual - zero).abs().mean(),
            'gate_logit_distance_mean': (actual - gated).abs().mean(),
            'value_logit_distance_mean': (actual - valued).abs().mean(),
            'empty_extra_zero_logit_distance_max': ((actual - zero).abs() * empty).max(),
            'extra_input_weight_norm': self.output[0].weight[:, -8:].norm(),
            'parent_weight_sum_error_max': (views['shared_weights'].sum(-1) - (1.0 - empty)).abs().max(),
            'empty_parent_pool_norm_max': (views['selected'].norm(dim=-1) * empty).max(),
        }
        for field_index, field in enumerate(('category', 'brand')):
            for action_index, action in enumerate(('pv', 'cart', 'fav', 'buy')):
                prefix = field + '_' + action + '_'
                result[prefix + 'common_support'] = common_gate[:, field_index, action_index].mean()
                result[prefix + 'required_support'] = required_gate[:, field_index, action_index].mean()
                result[prefix + 'new_rank_mean'] = (required_rank[:, field_index, action_index] * new_gate[:, field_index, action_index]).sum() / new_gate[:, field_index, action_index].sum().clamp_min(1.0)
                result[prefix + 'selected_mean'] = selected[:, field_index, action_index].mean()
        return result
