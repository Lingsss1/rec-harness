"""Four-slot history study with shared event scorer and fixed-width head.

A 2x2 factorial changes only partition identity and attention normalization.
Behavior slots: pv/cart/fav/buy. Control slots: valid-event ordinal modulo four.
All events, field embeddings, action/age values, scorer and head are shared.
This is a local adaptation, not a novel attention architecture or exact DIN.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16,
    'hidden': [128, 64],
    'lr': 0.001,
    'grouping': 'behavior',
    'normalization': 'within_group',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.grouping = self.config['grouping']
        normalization = self.config['normalization']
        if self.grouping not in ('behavior', 'interleaved'):
            raise ValueError('Unknown grouping')
        if normalization not in ('within_group', 'global'):
            raise ValueError('Unknown normalization')
        self.local_weight = float(normalization == 'within_group')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + 4 * dim
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)
        self.register_buffer('slot_ids', torch.arange(4).view(1, 4, 1))

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        mask = batch['history_mask']
        # Retain the reference's value encoding and supplied recency bins.
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (self.embeddings[self.category](batch['history_category'])
                   + self.embeddings[self.brand](batch['history_brand'])
                   + self.action(batch['history_type']) + self.recency(ages))
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        scores = self.attention(torch.cat(
            [query, history, query - history, query * history], dim=-1
        )).squeeze(-1)
        # Padding never increments the valid-event rank. Both partitions use
        # each valid event once and never request a different history prefix.
        ordinal = (mask.long().cumsum(-1) - 1).remainder(4)
        behavior = batch['history_type'] - 1
        group_id = behavior if self.grouping == 'behavior' else ordinal
        group_mask = (group_id[:, None, :] == self.slot_ids) & mask[:, None, :]
        if history.shape[1] == 0:
            slots = history.new_zeros((static.shape[0], 4, history.shape[-1]))
            empty_weights = history.new_zeros(group_mask.shape)
            return static, slots, empty_weights, empty_weights, group_mask
        global_weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
        global_weights = global_weights / global_weights.sum(-1, keepdim=True).clamp_min(1e-9)
        global_group_weights = global_weights[:, None, :] * group_mask.float()
        local_weights = scores[:, None, :].expand_as(group_mask).masked_fill(
            ~group_mask, -1e4
        ).softmax(-1) * group_mask.float()
        local_weights = local_weights / local_weights.sum(-1, keepdim=True).clamp_min(1e-9)
        # Compute both paths in every variant to keep the operator structure
        # and autograd graph matched. The coefficient is fixed, not learned.
        weights = (self.local_weight * local_weights
                   + (1.0 - self.local_weight) * global_group_weights)
        slots = torch.bmm(weights, history)
        return static, slots, weights, global_group_weights, group_mask

    def forward(self, batch):
        static, slots, _, _, _ = self._represent(batch)
        return self.output(torch.cat([static, slots.flatten(1)], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        # Host calls under no_grad on its fixed first-512 validation sample.
        # These exploratory diagnostics do not estimate population utility.
        _, slots, weights, global_weights, group_mask = self._represent(batch)
        result = {
            'empty_history_fraction': (~batch['history_mask'].any(-1)).float().mean(),
            'visible_event_count_mean': batch['history_mask'].float().sum(-1).mean(),
            'summary_rms': slots.square().mean().sqrt(),
            'total_attention_mass_mean': weights.sum((1, 2)).mean(),
        }
        for i in range(4):
            name = 'slot_' + str(i + 1)
            result[name + '_count_mean'] = group_mask[:, i, :].float().sum(-1).mean()
            result[name + '_present_fraction'] = group_mask[:, i, :].any(-1).float().mean()
            result[name + '_global_mass_mean'] = global_weights[:, i, :].sum(-1).mean()
            result[name + '_selected_mass_mean'] = weights[:, i, :].sum(-1).mean()
            # True-action counts remain interpretable in the position controls.
            action_mask = (batch['history_type'] == i + 1) & batch['history_mask']
            result['action_' + str(i + 1) + '_count_mean'] = action_mask.float().sum(-1).mean()
        return {key: float(value) for key, value in result.items()}
