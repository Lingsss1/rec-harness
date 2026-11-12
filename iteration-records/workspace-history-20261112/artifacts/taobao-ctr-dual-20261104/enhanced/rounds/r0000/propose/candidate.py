"""Field-context residual query experiment; matched replacements use config.mode.

Inherits the public baseline's encodings, softmax attention and classifier.
Only the route/input of a 16 -> 64 -> 16 residual adapter changes.
No optimizer, loss, supervision, history budget or data access changes.
"""
import math
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64],
    'lr': 0.001, 'adapter_hidden': 64, 'mode': 'context_query',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.mode = self.config['mode']
        if self.mode not in ('context_query', 'target_query', 'context_readout'):
            raise ValueError('Unknown adapter replacement')
        if self.config['kind'] != 'din':
            raise ValueError('This study holds DIN history attention fixed')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.context_indices = tuple(i for i in range(len(meta['fields']))
                                     if i not in (self.category, self.brand))
        dim = self.config['embedding_dim']
        # Keep the parent's construction order so its seed initialization is
        # preserved. The extra adapter is created after every inherited layer.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)
        self.adapter = nn.Sequential(
            nn.Linear(dim, self.config['adapter_hidden']), nn.ReLU(),
            nn.Linear(self.config['adapter_hidden'], dim),
        )
        # All replacements start at the exact parent function, independent of
        # context scale. The first adapter layer begins learning after the last.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def _encode(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        features = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (
            self.embeddings[self.category](batch['history_category'])
            + self.embeddings[self.brand](batch['history_brand'])
            + self.action(batch['history_type']) + self.recency(ages)
        )
        mask = batch['history_mask']
        target = fields[self.category] + fields[self.brand]
        if self.context_indices:
            context = torch.stack([fields[i] for i in self.context_indices], dim=1).sum(1)
            context = context / math.sqrt(len(self.context_indices))
        else:
            context = torch.zeros_like(target)
        # Input dimensions, adapter capacity and initialization are identical.
        adapter_input = target / math.sqrt(2.0) if self.mode == 'target_query' else context
        delta = self.adapter(adapter_input)
        delta = delta * mask.any(-1, keepdim=True).to(delta.dtype)
        query = target if self.mode == 'context_readout' else target + delta
        return features, history, mask, target, delta, query

    def _weights(self, query, history, mask):
        query = query[:, None, :].expand_as(history)
        scores = self.attention(torch.cat(
            [query, history, query - history, query * history], dim=-1
        )).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def forward(self, batch):
        features, history, mask, target, delta, query = self._encode(batch)
        weights = self._weights(query, history, mask)
        pooled = (weights.unsqueeze(-1) * history).sum(1)
        if self.mode == 'context_readout':
            pooled = pooled + delta
        return self.output(torch.cat([features, pooled], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        # Descriptive only: host calls on its fixed first 512 validation rows.
        # No labels, state mutation, random perturbations or persistent cache.
        features, history, mask, target, delta, query = self._encode(batch)
        weights = self._weights(query, history, mask)
        base_weights = self._weights(target, history, mask)
        length = mask.sum(-1)
        nonempty = length > 0
        category = batch['static'][:, self.category]
        brand = batch['static'][:, self.brand]
        category_match = ((batch['history_category'] == category[:, None])
                          & (category[:, None] != 0) & mask)
        brand_match = ((batch['history_brand'] == brand[:, None])
                       & (brand[:, None] != 0) & mask)
        overlap = category_match.any(-1)
        absent = nonempty & ~overlap
        present = nonempty & overlap

        def mean_where(values, select):
            select = select.to(values.dtype)
            return (values * select).sum() / select.sum().clamp_min(1)

        tv = 0.5 * (weights - base_weights).abs().sum(-1)
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(-1)
        pooled = (weights.unsqueeze(-1) * history).sum(1)
        if self.mode == 'context_readout':
            pooled = pooled + delta
        actual_logits = self.output(torch.cat([features, pooled], dim=-1)).squeeze(-1)
        base_pool = (base_weights.unsqueeze(-1) * history).sum(1)
        disabled_logits = self.output(torch.cat([features, base_pool], dim=-1)).squeeze(-1)
        change = (actual_logits - disabled_logits).abs()
        stats = {
            'diagnostic_rows': length.new_tensor(length.numel()).float(),
            'nonempty_rows': nonempty.sum().float(),
            'category_no_overlap_nonempty_rows': absent.sum().float(),
            'category_overlap_rows': present.sum().float(),
            'mean_history_length': length.float().mean(),
            'nonpadding_category_fraction': (category != 0).float().mean(),
            'nonpadding_brand_fraction': (brand != 0).float().mean(),
            'brand_overlap_given_no_category_overlap': mean_where(brand_match.any(-1).float(), absent),
            'delta_norm_nonempty': mean_where(delta.norm(dim=-1), nonempty),
            'target_norm_nonempty': mean_where(target.norm(dim=-1), nonempty),
            'entropy_nonempty': mean_where(entropy, nonempty),
            'attention_tv_no_category_overlap': mean_where(tv, absent),
            'attention_tv_category_overlap': mean_where(tv, present),
            'logit_change_no_category_overlap': mean_where(change, absent),
            'logit_change_category_overlap': mean_where(change, present),
            'category_match_weight_mass': mean_where((weights * category_match.float()).sum(-1), nonempty),
            'brand_match_weight_mass': mean_where((weights * brand_match.float()).sum(-1), nonempty),
        }
        for action in range(1, 5):
            action_mask = (batch['history_type'] == action) & mask
            stats['action_%d_event_fraction' % action] = (
                action_mask.sum().float() / mask.sum().clamp_min(1)
            )
            stats['action_%d_weight_mass' % action] = mean_where(
                (weights * action_mask.float()).sum(-1), nonempty
            )
        return {name: float(value.item()) for name, value in stats.items()}
