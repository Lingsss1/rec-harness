"""Prospective action-group normalization study; no architectural novelty claim.

Inherits the supplied DIN-style reference, which differs from original DIN.
All factorial cells share parameter shapes; only routing/normalization differ.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'routing': 'displaced', 'normalization': 'local',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.routing = self.config['routing']
        self.normalization = self.config['normalization']
        if self.routing not in ('action', 'displaced'):
            raise ValueError('routing must be action or displaced')
        if self.normalization not in ('local', 'global'):
            raise ValueError('normalization must be local or global')
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
        # Preserve the original global summary in addition to four group views.
        width = len(meta['fields']) * dim + 5 * dim
        layers = []
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    @staticmethod
    def _masked_softmax(scores, mask):
        # Absent groups have exactly zero weight, including all-empty rows.
        scores = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    @staticmethod
    def _displaced_types(types, mask):
        # Rotate labels by half the number of VALID events; exact group counts
        # are preserved per row, including with noncontiguous padding.
        length = types.shape[1]
        positions = torch.arange(length, device=types.device)[None, :].expand_as(types)
        valid_positions = positions.masked_fill(~mask, length).sort(-1).indices
        counts = mask.sum(-1, keepdim=True)
        ranks = (mask.long().cumsum(-1) - 1).clamp_min(0)
        source_ranks = (ranks + counts // 2) % counts.clamp_min(1)
        source_positions = valid_positions.gather(1, source_ranks)
        return types.gather(1, source_positions)

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (self.embeddings[self.category](batch['history_category'])
                   + self.embeddings[self.brand](batch['history_brand'])
                   + self.action(batch['history_type']) + self.recency(ages))
        mask = batch['history_mask']
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        scores = self.attention(torch.cat(
            [query, history, query - history, query * history], dim=-1
        )).squeeze(-1)
        global_weights = self._masked_softmax(scores, mask)
        displaced = self._displaced_types(batch['history_type'], mask)
        routed = batch['history_type'] if self.routing == 'action' else displaced
        group_ids = torch.arange(1, 5, device=mask.device)[None, :, None]
        group_mask = (routed[:, None, :] == group_ids) & mask[:, None, :]
        # Both weights are computed in every cell to match arithmetic work.
        local_weights = self._masked_softmax(scores[:, None, :].expand_as(group_mask), group_mask)
        grouped_global = global_weights[:, None, :] * group_mask.float()
        grouped_weights = local_weights if self.normalization == 'local' else grouped_global
        original = torch.bmm(global_weights[:, None, :].to(history.dtype), history).squeeze(1)
        grouped = torch.bmm(grouped_weights.to(history.dtype), history).flatten(1)
        features = torch.cat([*fields, original, grouped], dim=-1)
        return features, global_weights, grouped_weights, group_mask, displaced

    def forward(self, batch):
        features, _, _, _, _ = self._represent(batch)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        # First 512 validation rows only: descriptive measurements, not proof.
        _, global_weights, grouped_weights, group_mask, displaced = self._represent(batch)
        mask = batch['history_mask']
        types = batch['history_type']
        counts = mask.sum(-1)
        valid = mask.float().sum().clamp_min(1)
        out = {
            'history_count_mean': counts.float().mean(),
            'empty_history_fraction': (counts == 0).float().mean(),
            'displaced_label_change_fraction': ((displaced != types) & mask).float().sum() / valid,
            'global_weight_sum_mean': global_weights.sum(-1).mean(),
        }
        for i, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            action_mask = (types == i + 1) & mask
            present = action_mask.any(-1).float()
            denominator = present.sum().clamp_min(1)
            mass = (global_weights * action_mask.float()).sum(-1)
            out[name + '_event_count_mean'] = action_mask.sum(-1).float().mean()
            out[name + '_present_fraction'] = present.mean()
            out[name + '_global_mass_when_present'] = (mass * present).sum() / denominator
            out[name + '_global_mass_below_005_when_present'] = ((mass < 0.05).float() * present).sum() / denominator
            out[name + '_group_weight_sum_mean'] = grouped_weights[:, i, :].sum(-1).mean()
            out[name + '_routed_present_fraction'] = group_mask[:, i, :].any(-1).float().mean()
        return out
