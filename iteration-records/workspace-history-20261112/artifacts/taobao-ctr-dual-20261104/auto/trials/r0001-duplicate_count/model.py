"""Action/field match counts plus a controlled temporal statistic.

Parent: r0000-match_count. Inherits its normalized DIN-style attention,
embeddings, count features and optimizer. Related primary sources read:
DIN arXiv:1706.06978v4 sec. 4.3; DGIN arXiv:2311.10764v1 secs. 3.3.1-2;
TIN arXiv:2308.08487v4 sec. 3.1. This is an adaptation, not a reproduction
or a verified novelty claim. Only encoded category/brand equality is known.

For each field/action, C is match count, S is valid-field support count,
W is exponentially decayed match count and U is decayed support count.
The additional eight statistics are chosen by temporal_statistic:
  aligned: W (preserves match/age association)
  factorized: C * U / max(S, 1) (exchangeable-age expectation)
  activity: U (target-independent valid-field activity recency)
  duplicate: C (same inputs as parent; matched extra parameter shapes)
Every mode computes all statistics and preserves C as the first eight inputs.
The fixed decay half-life is one day; there is no validation tuning or new
history. The underlying attention already has age embeddings in every mode.
"""
import math
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'statistic_target': 'match', 'statistic_scale': 'count',
    'temporal_statistic': 'duplicate', 'half_life_seconds': 86400.0,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.temporal_statistic = self.config['temporal_statistic']
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
        # Eight inherited counts and eight replacement temporal statistics.
        # All four modes have exactly the same parameters and initialization.
        width = len(meta['fields']) * dim + dim + 16
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
        evidence = self._evidence_from_statistics(self._statistics(batch))
        features = torch.cat(fields + [pooled, evidence], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        # At most 59 finite scalar descriptions of the host's first 512
        # validation impressions. No labels or model-selection metrics.
        statistics = self._statistics(batch)
        counts, matches, supports, decayed_matches, decayed_supports = statistics
        factorized = matches * decayed_supports / supports.clamp_min(1.0)
        evidence = self._evidence_from_statistics(statistics)
        alignment = decayed_matches - factorized
        result = {
            'supplied_history_length_mean': counts.sum(-1).mean(),
            'supplied_empty_history_fraction': (counts.sum(-1) == 0).float().mean(),
            'evidence_feature_mean': evidence.mean(),
            'evidence_feature_max': evidence.max(),
            'alignment_absolute_mean': alignment.abs().mean(),
            'alignment_nonzero_fraction': (alignment.abs() > 1e-6).float().mean(),
            'mean_valid_field_decay': (decayed_supports.sum() /
                                       supports.sum().clamp_min(1.0)),
        }
        for a, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            result[name + '_supplied_count_mean'] = counts[:, a].mean()
            for f, field in enumerate(('category', 'brand')):
                key = field + '_' + name
                result[key + '_encoded_match_count_mean'] = matches[:, f, a].mean()
                result[key + '_field_support_count_mean'] = supports[:, f, a].mean()
                result[key + '_match_present_fraction'] = (matches[:, f, a] > 0).float().mean()
                result[key + '_decayed_match_mean'] = decayed_matches[:, f, a].mean()
                result[key + '_factorized_match_mean'] = factorized[:, f, a].mean()
                result[key + '_decayed_support_mean'] = decayed_supports[:, f, a].mean()
        return result
