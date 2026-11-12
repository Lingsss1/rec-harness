"""Field/action history readout with a norm-matched directional fusion study.

Inherited: shared field embeddings, logarithmic age buckets, DIN-style relation
features, and the reference MLP. This is an adaptation, not a DIN reproduction.
The four standalone variants implement a target-query x directional-fusion
factorial. Token construction is deliberately identical across all four.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': False, 'directional_fusion': False,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        dim = self.config['embedding_dim']
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.target_query = self.config['target_query']
        self.directional_fusion = self.config['directional_fusion']
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * dim, 32), nn.PReLU(), nn.Linear(32, dim)
        )
        layers = []
        width = len(meta['fields']) * dim + 2 * dim
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _encode(self, batch):
        fields = [e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)]
        # Four means per semantic field; missing behavior groups remain zero.
        types = torch.arange(1, 5, device=batch['static'].device)
        membership = ((batch['history_type'][:, None, :] == types[None, :, None])
                      & batch['history_mask'][:, None, :])
        counts = membership.sum(-1)
        pool = membership.float() / counts.clamp_min(1).unsqueeze(-1)
        ages = (torch.log2(batch['history_age_seconds'].clamp_min(0) + 1).long()
                + 1).clamp(0, 32)
        cat = torch.bmm(pool, self.embeddings[self.category](batch['history_category']))
        brand = torch.bmm(pool, self.embeddings[self.brand](batch['history_brand']))
        age = torch.bmm(pool, self.recency(ages))
        present = (counts > 0)[:, None, :].expand(-1, 2, -1)
        tokens = (torch.stack([cat, brand], dim=1) + age[:, None, :, :]
                  + self.action(types)[None, None, :, :])
        tokens = tokens * present.unsqueeze(-1).float()
        target = torch.stack([fields[self.category], fields[self.brand]], dim=1)
        context = (tokens.sum(2)
                   / present.sum(2, keepdim=True).clamp_min(1).float())
        return fields, tokens, present, target, context

    def _read(self, tokens, present, query):
        query = query.unsqueeze(2).expand_as(tokens)
        relation = torch.cat([query, tokens, query - tokens, query * tokens], dim=-1)
        scores = self.attention(relation).squeeze(-1).masked_fill(~present, -1e4)
        weights = scores.softmax(-1) * present.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        # Every output of the d-dimensional gate contributes in both controls.
        gates = 0.5 + self.gate(relation).sigmoid()
        amplitude = gates.mean(-1, keepdim=True)
        scalar = (weights.unsqueeze(-1) * tokens * amplitude).sum(2)
        raw_directional = (weights.unsqueeze(-1) * tokens * gates).sum(2)
        # Match the final per-field history norm, not just individual token norms.
        scalar_norm = scalar.norm(dim=-1, keepdim=True)
        raw_norm = raw_directional.norm(dim=-1, keepdim=True)
        directional = raw_directional * (scalar_norm / raw_norm.clamp_min(1e-6))
        return scalar, directional, weights, gates

    def forward(self, batch):
        fields, tokens, present, target, context = self._encode(batch)
        query = target if self.target_query else context
        scalar, directional, _, _ = self._read(tokens, present, query)
        summary = directional if self.directional_fusion else scalar
        features = torch.cat([torch.cat(fields, dim=-1), summary.flatten(1)], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        _, tokens, present, target, context = self._encode(batch)
        query = target if self.target_query else context
        scalar, directional, weights, gates = self._read(tokens, present, query)
        _, _, context_weights, _ = self._read(tokens, present, context)
        scalar_norm = scalar.norm(dim=-1)
        directional_norm = directional.norm(dim=-1)
        cosine = ((scalar * directional).sum(-1)
                  / (scalar_norm * directional_norm).clamp_min(1e-9))
        active = present.any(-1).float()
        count = active.sum().clamp_min(1)
        return {
            'nonempty_fraction': float(active.mean()),
            'present_behavior_groups': float(present.float().sum(-1).mean()),
            'scalar_summary_norm': float(scalar_norm.mean()),
            'directional_summary_norm': float(directional_norm.mean()),
            'norm_matching_absolute_error': float((scalar_norm - directional_norm).abs().max()),
            'direction_scalar_cosine_nonempty': float((cosine * active).sum() / count),
            'attention_entropy': float((-(weights * weights.clamp_min(1e-9).log()).sum(-1)
                                        * active).sum() / count),
            'gate_channel_std': float(gates.std(-1, unbiased=False).mean()),
            'attention_l1_vs_context': float((weights - context_weights).abs().sum(-1).mean()),
            'uses_target_query': float(self.target_query),
            'uses_directional_fusion': float(self.directional_fusion),
        }
