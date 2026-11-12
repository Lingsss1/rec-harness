"""Separate direct target matches from learned history weighting.

Mean pooling and content-only weighting replace target-conditioned attention,
each crossed with match counts versus activity alone. The existing parent and
attention_activity trial supply the target-conditioned cells. All allocated
parameter tensors and their creation order are inherited. Mean mode skips the
scorer but retains its dormant parameters; content mode supplies a zero query
to the unchanged scorer, reducing effective input rank. Active capacity is not
matched. These are local mechanism contrasts, not globally novel pooling.

Bucket equality is not raw identity. No extra history, labels or optimizer.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'summary_mode': 'mean', 'relevance_mode': 'activity'}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.joint_mode = self.config['joint_mode']
        self.statistic_mode = self.config['statistic_mode']
        self.summary_mode = self.config['summary_mode']
        self.relevance_mode = self.config['relevance_mode']
        if self.summary_mode not in ('mean', 'content', 'attention', 'none'):
            raise ValueError('Unknown summary_mode')
        if self.relevance_mode not in ('match', 'activity'):
            raise ValueError('Unknown relevance_mode')
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

    def _full_statistics(self, counts):
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

    def _statistics(self, counts):
        statistics = self._full_statistics(counts)
        if self.relevance_mode == 'activity':
            statistics = torch.cat([
                statistics[:, :1], torch.zeros_like(statistics[:, 1:])
            ], dim=-1)
        return statistics

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        mask = batch['history_mask']
        if self.summary_mode == 'none' or mask.shape[1] == 0:
            summary = static.new_zeros((static.shape[0], self.embeddings[self.category].embedding_dim))
            weights = static.new_zeros(mask.shape)
        else:
            ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            history = (self.embeddings[self.category](batch['history_category']) +
                       self.embeddings[self.brand](batch['history_brand']) +
                       self.action(batch['history_type']) + self.recency(ages))
            if self.summary_mode == 'mean':
                # Target dependence now enters via counts and the final MLP.
                # The value embeddings remain trainable; only weights are fixed.
                weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
            else:
                if self.summary_mode == 'content':
                    # Same tensors and scorer input width as the parent. Query
                    # and product columns are dormant; history/-history remain.
                    query = torch.zeros_like(history)
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
        effective_events = (weights.sum(-1) > 0).float() / weights.square().sum(-1).clamp_min(1e-9)
        # A label-free counterfactual source diagnostic, never an extra score:
        # rotate valid target IDs within their own fields, holding history fixed.
        rotated = dict(batch)
        rotated_static = batch['static'].clone()
        for index in (self.category, self.brand):
            rotated_static[:, index] = batch['static'][:, index].roll(1, 0)
        rotated['static'] = rotated_static
        _, rotated_summary, rotated_weights, _, _ = self._represent(rotated)
        weight_change = (weights - rotated_weights).abs().sum(-1)
        dormant = 0
        if self.summary_mode == 'mean':
            dormant = sum(p.numel() for p in self.attention.parameters())
        elif self.summary_mode == 'none':
            dormant = sum(p.numel() for module in (self.action, self.recency, self.attention)
                          for p in module.parameters())
        result = {
            'trainable_history_values_enabled': float(self.summary_mode != 'none'),
            'learned_weighting_enabled': float(self.summary_mode in ('content', 'attention')),
            'target_in_weighting_enabled': float(self.summary_mode == 'attention'),
            'target_match_readout_enabled': float(self.relevance_mode == 'match'),
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'fully_dormant_history_module_parameters': float(dormant),
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
            'effective_weighted_events_mean': effective_events.mean(),
            'effective_event_fraction_mean': (effective_events / denom).mean(),
            'weight_entropy_mean': -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean(),
            'rotated_target_summary_rms_delta': (summary - rotated_summary).square().mean().sqrt(),
            'rotated_target_weight_l1_mean': weight_change.mean(),
            'target_category_missing_fraction': (batch['static'][:, self.category] == 0).float().mean(),
            'target_brand_missing_fraction': (batch['static'][:, self.brand] == 0).float().mean(),
        }
        for i, name in enumerate(('activity', 'category', 'brand', 'joint')):
            result['selected_' + name + '_mean'] = statistics[:, i].mean()
        return {key: float(value) for key, value in result.items()}
