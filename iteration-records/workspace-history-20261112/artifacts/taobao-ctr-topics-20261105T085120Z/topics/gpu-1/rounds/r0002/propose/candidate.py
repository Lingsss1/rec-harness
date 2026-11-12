"""Target-match multiset statistics alongside the frozen DIN-style summary.

Factorial controls replace joint same-event matches with C*B/N and replace
bounded log counts with proportions. All four settings allocate identical
parameter tensors. Equality refers to supplied bucket IDs, not raw identity.
No claim of a new global attention mechanism or of source reproduction.
"""
import math
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'joint_mode': 'observed', 'statistic_mode': 'count',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.joint_mode = self.config['joint_mode']
        self.statistic_mode = self.config['statistic_mode']
        if self.joint_mode not in ('observed', 'independent'):
            raise ValueError('Unknown joint_mode')
        if self.statistic_mode not in ('count', 'rate'):
            raise ValueError('Unknown statistic_mode')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Preserve the reference's module creation order and embedding sizes.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim + 4
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

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

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']) +
                   self.action(batch['history_type']) + self.recency(ages))
        mask = batch['history_mask']
        if history.shape[1] == 0:
            summary = static.new_zeros((static.shape[0], history.shape[-1]))
            weights = static.new_zeros(mask.shape)
        else:
            query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
            scores = self.attention(torch.cat(
                [query, history, query - history, query * history], dim=-1
            )).squeeze(-1)
            weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
            summary = (weights.unsqueeze(-1) * history).sum(1)
        counts = self._match_counts(batch)
        statistics = self._statistics(counts)
        return static, summary, weights, counts, statistics

    def forward(self, batch):
        static, summary, _, _, statistics = self._represent(batch)
        features = torch.cat([static, summary, statistics], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        # Exploratory first-512 validation diagnostics, not population evidence.
        _, summary, weights, counts, statistics = self._represent(batch)
        n, c, b, joint, independent = counts
        denom = n.clamp_min(1)
        result = {
            'empty_history_fraction': (n == 0).float().mean(),
            'visible_count_mean': n.mean(),
            'visible_count_at_limit_fraction': (n == 128).float().mean(),
            'category_match_count_mean': c.mean(),
            'brand_match_count_mean': b.mean(),
            'joint_match_count_mean': joint.mean(),
            'independent_joint_count_mean': independent.mean(),
            'category_match_present_fraction': (c > 0).float().mean(),
            'brand_match_present_fraction': (b > 0).float().mean(),
            'joint_match_present_fraction': (joint > 0).float().mean(),
            'joint_independent_abs_gap_mean': (joint - independent).abs().mean(),
            'joint_independent_rate_abs_gap_mean': ((joint - independent) / denom).abs().mean(),
            'attention_mass_mean': weights.sum(-1).mean(),
            'summary_rms': summary.square().mean().sqrt(),
            'target_category_missing_fraction': (batch['static'][:, self.category] == 0).float().mean(),
            'target_brand_missing_fraction': (batch['static'][:, self.brand] == 0).float().mean(),
        }
        for i, name in enumerate(('activity', 'category', 'brand', 'joint')):
            result['selected_' + name + '_mean'] = statistics[:, i].mean()
        return {key: float(value) for key, value in result.items()}
