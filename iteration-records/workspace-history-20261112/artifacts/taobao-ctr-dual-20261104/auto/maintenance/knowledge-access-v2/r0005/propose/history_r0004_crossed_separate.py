"""Field-correspondence by attention-routing factorial on r0002-support_pair.

Parent source SHA256 f11d409d2dfe3653c1fc1c788784f512f2dbc7ab13dbbace4c01f387cf0bada7.
All 24 incumbent count/decay/support statistics, embeddings, scorer parameter
shapes, output MLP and initialization order are inherited. Replace one scorer
call on early-added fields with two calls to the SAME scorer on field values:
    vc = category_embedding + (action_embedding + age_embedding)/2
    vb = brand_embedding    + (action_embedding + age_embedding)/2
Aligned queries are (target_category, target_brand); crossed swaps these queries
while keeping keys/values and coordinate order fixed. Shared routing applies
softmax((sc+sb)/2) to vc+vb. Separate routing applies softmax(sc) to vc and
softmax(sb) to vb, then sums. Both pooling alternatives are computed in all modes.
The pooled width stays D: no appended summaries, extra parameters or optimizer.
The same supplied event mask is used in every head, including missing fields;
zero field IDs embed as zero, but their action/time context remains available.
Empty histories pool to exact zero. No added events, labels, I/O or supervision.

Nearest primary work: DIN arXiv:1706.06978v4 sec.4.3; DIF-SR arXiv:2204.11046v1
secs.4.2.3-4.3. DIF-SR fuses attribute self-attention scores before one softmax
and uses item-ID values plus auxiliary attribute predictors. This local target
cross-attention uses one shared nonlinear scorer and separate field-value routes
with a crossed-query factorial control. Neither reproduction nor novelty claimed.
Parent versus all four cells also changes early fusion to shared-score late
fusion, so that comparison alone cannot establish the proposed mechanism.
"""
import math
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'statistic_target': 'match', 'statistic_scale': 'count',
    'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0,
    'pair_statistic': 'support',
    'field_pairing': 'crossed', 'field_routing': 'separate',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.field_pairing = self.config['field_pairing']
        self.field_routing = self.config['field_routing']
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
        # Preserve sixteen parent statistics and append eight replacement inputs.
        # All four modes have exactly the same parameters and initialization.
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

    def _field_views(self, batch, fields):
        mask = batch['history_mask']
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        context = self.action(batch['history_type']) + self.recency(ages)
        category_value = self.embeddings[self.category](batch['history_category']) + 0.5 * context
        brand_value = self.embeddings[self.brand](batch['history_brand']) + 0.5 * context
        category_query, brand_query = fields[self.category], fields[self.brand]
        if self.field_pairing == 'crossed':
            category_query, brand_query = brand_query, category_query
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
        """29 finite descriptive scalars on the host-selected first 512 rows.

        Includes counterfactual pooling for the current scores, not a retrained
        ablation. Weight or norm differences are not causal importance evidence.
        """
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._field_views(batch, fields)
        mask = batch['history_mask']
        lengths = mask.sum(-1).float()
        present = (lengths > 0).float()
        uniform = mask.float() / lengths[:, None].clamp_min(1)
        result = {
            'supplied_history_length_mean': lengths.mean(),
            'supplied_empty_history_fraction': (lengths == 0).float().mean(),
            'field_weight_l1_distance_mean':
                (views['category_weights'] - views['brand_weights']).abs().sum(-1).mean(),
            'shared_pool_norm_mean': views['shared_pool'].norm(dim=-1).mean(),
            'separate_pool_norm_mean': views['separate_pool'].norm(dim=-1).mean(),
            'selected_pool_norm_mean': views['selected'].norm(dim=-1).mean(),
            'separate_shared_pool_distance_mean':
                (views['separate_pool'] - views['shared_pool']).norm(dim=-1).mean(),
            'empty_selected_pool_norm_max':
                (views['selected'].norm(dim=-1) * (1.0 - present)).max(),
        }
        for name, index, column in (
                ('category', self.category, 'history_category'),
                ('brand', self.brand, 'history_brand')):
            weights = views[name + '_weights']
            score = views[name + '_score']
            score_mean = (score * mask.float()).sum(-1) / lengths.clamp_min(1)
            score_variance = (((score - score_mean[:, None]) ** 2) *
                              mask.float()).sum(-1) / lengths.clamp_min(1)
            target = batch['static'][:, index:index + 1]
            valid = mask & (batch[column] > 0)
            equal = valid & (target > 0) & (batch[column] == target)
            match_mass = (weights * equal.float()).sum(-1)
            effective = present / weights.square().sum(-1).clamp_min(1e-9)
            result[name + '_target_missing_fraction'] = (target == 0).float().mean()
            result[name + '_valid_event_fraction_mean'] = (
                valid.float().sum(-1) / lengths.clamp_min(1)).mean()
            result[name + '_match_present_fraction'] = equal.any(-1).float().mean()
            result[name + '_weight_sum_error_max'] = (weights.sum(-1) - present).abs().max()
            result[name + '_padded_weight_max'] = (weights * (~mask).float()).max()
            result[name + '_effective_events_when_present'] = (
                effective.sum() / present.sum().clamp_min(1))
            result[name + '_score_variance_mean'] = score_variance.mean()
            result[name + '_weight_uniform_l1_distance_mean'] = (
                weights - uniform).abs().sum(-1).mean()
            result[name + '_match_attention_mass_mean'] = match_mass.mean()
            result[name + '_match_uniform_mass_mean'] = (uniform * equal.float()).sum(-1).mean()
        result['shared_weight_sum_error_max'] = (views['shared_weights'].sum(-1) - present).abs().max()
        return result
