"""Field-preserving values under the unchanged target-aware event scorer.

Three variants share all parameter tensors and the 4D history head:
  fields: concatenate category, brand, action, and recency weighted means;
  collapsed: project those four slots onto their common mean;
  collapsed_rms: the same projection, rescaled to retain the source norm.
All variants use each supplied event, one global masked softmax, and the same
sum-of-fields key and target category-plus-brand query. No field-wise scorer,
new raw information, optimizer, loss, or learned projection is introduced.
This is an attribution study, not a new attention or reproduction claim.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16,
    'hidden': [128, 64],
    'lr': 0.001,
    'value_mode': 'fields',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.value_mode = self.config['value_mode']
        if self.value_mode not in ('fields', 'collapsed', 'collapsed_rms'):
            raise ValueError('Unknown value_mode')
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

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        components = [
            self.embeddings[self.category](batch['history_category']),
            self.embeddings[self.brand](batch['history_brand']),
            self.action(batch['history_type']),
            self.recency(ages),
        ]
        # Use the reference's left-associated sum and identical scorer inputs.
        history = components[0] + components[1] + components[2] + components[3]
        mask = batch['history_mask']
        if history.shape[1] == 0:
            raw_slots = history.new_zeros((static.shape[0], 4, history.shape[-1]))
            weights = history.new_zeros(mask.shape)
        else:
            query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
            scores = self.attention(torch.cat(
                [query, history, query - history, query * history], dim=-1
            )).squeeze(-1)
            weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
            raw_slots = torch.stack([
                (weights.unsqueeze(-1) * component).sum(1)
                for component in components
            ], dim=1)
        collapsed = raw_slots.mean(dim=1, keepdim=True).expand_as(raw_slots)
        if self.value_mode == 'fields':
            slots = raw_slots
        elif self.value_mode == 'collapsed':
            slots = collapsed
        else:
            # Clamp BEFORE sqrt: all-empty histories must have finite gradients.
            # This deterministic control retains one extra scalar (source norm),
            # but cannot reconstruct the discarded field contrast directions.
            source_energy = raw_slots.square().sum((1, 2), keepdim=True)
            projected_energy = collapsed.square().sum((1, 2), keepdim=True)
            scale = (source_energy.clamp_min(1e-12)
                     / projected_energy.clamp_min(1e-12)).sqrt()
            slots = collapsed * scale
        return static, raw_slots, slots, weights

    def forward(self, batch):
        static, _, slots, _ = self._represent(batch)
        return self.output(torch.cat([static, slots.flatten(1)], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        # Exploratory only: host uses its fixed first 512 validation impressions.
        _, raw_slots, slots, weights = self._represent(batch)
        collapsed = raw_slots.mean(dim=1, keepdim=True).expand_as(raw_slots)
        source_energy = raw_slots.square().sum((1, 2))
        projected_energy = collapsed.square().sum((1, 2))
        selected_energy = slots.square().sum((1, 2))
        result = {
            'empty_history_fraction': (~batch['history_mask'].any(-1)).float().mean(),
            'visible_event_count_mean': batch['history_mask'].float().sum(-1).mean(),
            'attention_mass_mean': weights.sum(-1).mean(),
            'attention_entropy_mean': -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean(),
            'raw_slot_rms': raw_slots.square().mean().sqrt(),
            'selected_slot_rms': slots.square().mean().sqrt(),
            'field_contrast_energy_fraction_mean': (
                (raw_slots - collapsed).square().sum((1, 2))
                / source_energy.clamp_min(1e-12)
            ).mean(),
            'selected_to_raw_energy_ratio_mean': (
                selected_energy / source_energy.clamp_min(1e-12)
            ).mean(),
            'norm_control_degenerate_fraction': (
                (projected_energy < 1e-12) & (source_energy >= 1e-12)
            ).float().mean(),
            'selected_norm_relative_error_mean': (
                (selected_energy - source_energy).abs()
                / source_energy.clamp_min(1e-12)
            ).mean(),
        }
        for i, name in enumerate(('category', 'brand', 'action', 'recency')):
            result[name + '_summary_rms'] = raw_slots[:, i].square().mean().sqrt()
        return {key: float(value) for key, value in result.items()}
