"""Same-event category/brand match statistics with matched replacement controls.

Parent: r0001-aligned_time, SHA256
16a631418c8fc964684a6bf9853b36ee3a9159b34431f1c97d4db3d204f9ab26.
Preserves its sixteen field/action count and one-day decay features, normalized
DIN-style attention and training recipe. Appends eight fixed statistics:
  joint:      J, K       same-event category AND brand match count/decay mass
  factorized: Cc*Cb/S, Wc*Wb/U   pair-valid marginal-independence construction
  duplicate:  Ccat, Wcat          redundant original category statistics
  support:    S, U                target-independent pair-valid action support
S=0 or U=0 yields zero for the corresponding independence statistic.
All modes compute every quantity and use identical parameter shapes/RNG calls.
Only encoded equality is observed; conjunction is not known item identity.
Related primary work inspected: DIN arXiv:1706.06978v4 sec. 4.3;
DGIN arXiv:2311.10764v1 secs. 3.3.1-2; TIN arXiv:2308.08487v4 sec. 3.1;
EDIN/DINMP arXiv:2104.06312v1 secs. 3.1.2-3 and 3.2.
This is a local adaptation and prospective controlled test, not a reproduction
or a verified novelty claim. No additional history, labels or I/O are used.
"""
import math
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'statistic_target': 'match', 'statistic_scale': 'count',
    'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0,
    'pair_statistic': 'joint',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
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

    def forward(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']) +
                   self.action(batch['history_type']) + self.recency(ages))
        mask = batch['history_mask']
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        scores = self.attention(torch.cat([
            query, history, query - history, query * history
        ], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        pooled = (weights.unsqueeze(-1) * history).sum(1)
        statistics = self._statistics(batch)
        pairs = self._pair_statistics(batch)
        evidence = self._evidence_from_statistics(statistics)
        pair_evidence = self._pair_evidence(statistics, pairs)
        features = torch.cat(fields + [pooled, evidence, pair_evidence], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """55 descriptive finite scalars on the host-selected first 512 rows.

        These quantify supplied-event opportunity and statistic disagreement,
        not population estimates, labels, performance or semantic identity.
        """
        statistics = self._statistics(batch)
        pairs = self._pair_statistics(batch)
        counts = statistics[0]
        evidence = torch.cat([self._evidence_from_statistics(statistics),
                              self._pair_evidence(statistics, pairs)], dim=-1)
        count_gap = pairs['joint_count'] - pairs['independent_count']
        decay_gap = pairs['joint_decay'] - pairs['independent_decay']
        target_valid = ((batch['static'][:, self.category] > 0) &
                        (batch['static'][:, self.brand] > 0))
        result = {
            'supplied_history_length_mean': counts.sum(-1).mean(),
            'supplied_empty_history_fraction': (counts.sum(-1) == 0).float().mean(),
            'pair_target_present_fraction': target_valid.float().mean(),
            'pair_support_mean': pairs['support'].mean(),
            'joint_count_mean': pairs['joint_count'].mean(),
            'factorized_count_mean': pairs['independent_count'].mean(),
            'joint_decay_mean': pairs['joint_decay'].mean(),
            'factorized_decay_mean': pairs['independent_decay'].mean(),
            'pair_count_gap_absolute_mean': count_gap.abs().mean(),
            'pair_decay_gap_absolute_mean': decay_gap.abs().mean(),
            'pair_count_gap_nonzero_fraction': (count_gap.abs() > 1e-6).float().mean(),
            'pair_decay_gap_nonzero_fraction': (decay_gap.abs() > 1e-6).float().mean(),
            'pair_support_empty_fraction': (pairs['support'].sum(-1) == 0).float().mean(),
            'evidence_feature_mean': evidence.mean(),
            'evidence_feature_max': evidence.max(),
        }
        for a, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            for key in ('support', 'weighted_support', 'category_count', 'brand_count',
                        'joint_count', 'joint_decay', 'independent_count', 'independent_decay'):
                result[name + '_pair_' + key + '_mean'] = pairs[key][:, a].mean()
            result[name + '_joint_match_present_fraction'] = (pairs['joint_count'][:, a] > 0).float().mean()
            result[name + '_pair_count_gap_nonzero_fraction'] = (count_gap[:, a].abs() > 1e-6).float().mean()
        return result
