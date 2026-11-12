"""DIN reference plus explicit action/field evidence statistics.

Prospective adaptation of DIN intensity (1706.06978v4, sec. 4.3) and
DGIN heterogeneous counts (2311.10764v1, sec. 3.3.1). Not a reproduction
or a priority claim. Equality is on supplied encoded category/brand IDs,
not item identity. Only the supplied causal history prefix is inspected.

The 2x2 controls preserve every trainable tensor and the optimizer recipe:
  statistic_target: match | support
  statistic_scale: count | rate
"""
import math
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'statistic_target': 'match', 'statistic_scale': 'count',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.statistic_target = self.config['statistic_target']
        self.statistic_scale = self.config['statistic_scale']
        if self.statistic_target not in ('match', 'support'):
            raise ValueError('Unknown statistic_target')
        if self.statistic_scale not in ('count', 'rate'):
            raise ValueError('Unknown statistic_scale')
        dim = self.config['embedding_dim']
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        # Eight appended scalars add exactly 8*128=1024 parameters at the
        # default width. The four new variants have identical tensor shapes.
        width = len(meta['fields']) * dim + dim + 8
        layers = []
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _statistics(self, batch):
        mask = batch['history_mask']
        types = batch['history_type']
        # [B,L,4]; type 0 never contributes even if mask is malformed.
        action_ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == action_ids) & mask.unsqueeze(-1)).float()
        counts = actions.sum(dim=1)
        supports = []
        matches = []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            history_ids = batch[column]
            target_ids = batch['static'][:, index:index + 1]
            valid = mask & (history_ids > 0)
            equal = valid & (target_ids > 0) & (history_ids == target_ids)
            supports.append((actions * valid.unsqueeze(-1).float()).sum(dim=1))
            matches.append((actions * equal.unsqueeze(-1).float()).sum(dim=1))
        # [B,2,4], category first, then brand; within each field pv/cart/fav/buy.
        return counts, torch.stack(matches, dim=1), torch.stack(supports, dim=1)

    def _evidence(self, batch):
        counts, matches, supports = self._statistics(batch)
        values = matches if self.statistic_target == 'match' else supports
        if self.statistic_scale == 'rate':
            # Per-action proportions, expressed as pseudo-counts at a fixed
            # exposure of 128. Both modes share the same [0,1] log transform.
            # This is not extra history; 128 is the frozen supplied limit.
            values = 128.0 * values / counts.unsqueeze(1).clamp_min(1.0)
        return (torch.log1p(values) / math.log(129.0)).flatten(start_dim=1)

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
        features = torch.cat(fields + [pooled, self._evidence(batch)], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        # Descriptive first-512 checkpoint measurements only; no labels,
        # validation-wide metrics, mutable state, or independent assessment.
        counts, matches, supports = self._statistics(batch)
        features = self._evidence(batch)
        result = {
            'supplied_history_length_mean': counts.sum(-1).mean(),
            'supplied_empty_history_fraction': (counts.sum(-1) == 0).float().mean(),
            'evidence_feature_mean': features.mean(),
            'evidence_feature_max': features.max(),
        }
        for a, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            result[name + '_supplied_count_mean'] = counts[:, a].mean()
            for f, field in enumerate(('category', 'brand')):
                key = field + '_' + name
                result[key + '_encoded_match_count_mean'] = matches[:, f, a].mean()
                result[key + '_field_support_count_mean'] = supports[:, f, a].mean()
                result[key + '_encoded_match_present_fraction'] = (
                    matches[:, f, a] > 0).float().mean()
        return result
