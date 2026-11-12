"""Controlled field-query to event-value routing.

Inherited: local DIN-style scorer and norm-matched field retrieval from
 g2-r0001-target_separate. New intervention: arithmetic opinion pooling reads
 the full event rather than restricting each query to its own field value.
 Target-independent and crossed-value replacements provide attribution.
 These are local adaptations of existing attention, not a priority claim.
 No optimizer, loss, information boundary or training recipe changes.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'value_route': 'aligned',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.value_route = self.config['value_route']
        if self.value_route not in ('aligned', 'crossed', 'union'):
            raise ValueError('value_route must be aligned, crossed or union')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Keep ancestor parameter construction order and shapes for paired seeds.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _encode(self, batch):
        fields = [e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)]
        mask = batch['history_mask']
        ages = (torch.log2(batch['history_age_seconds'].clamp_min(0) + 1).long()
                + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        common = self.action(batch['history_type']) + self.recency(ages)
        uniform = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        context = torch.stack([
            (uniform.unsqueeze(-1) * category).sum(1),
            (uniform.unsqueeze(-1) * brand).sum(1),
        ], dim=1)
        target = torch.stack([fields[self.category], fields[self.brand]], dim=1)
        pooled = (uniform.unsqueeze(-1) * (category + brand + common)).sum(1)
        return fields, category, brand, common, mask, pooled, target, context

    def _score(self, query, keys):
        query = query[:, None, :].expand_as(keys)
        relation = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        return self.attention(relation).squeeze(-1)

    @staticmethod
    def _weights(scores, mask):
        weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def _read(self, category, brand, common, mask, query):
        category_score = self._score(query[:, 0], category + common)
        brand_score = self._score(query[:, 1], brand + common)
        wc = self._weights(category_score, mask)
        wb = self._weights(brand_score, mask)
        wj = self._weights(0.5 * (category_score + brand_score), mask)
        wu = 0.5 * (wc + wb)
        # Aligned expression is unchanged from the incumbent, including order.
        aligned = (
            wc.unsqueeze(-1) * category
            + wb.unsqueeze(-1) * brand
            + (0.5 * (wc + wb)).unsqueeze(-1) * common
        ).sum(1)
        crossed = (
            wb.unsqueeze(-1) * category
            + wc.unsqueeze(-1) * brand
            + (0.5 * (wc + wb)).unsqueeze(-1) * common
        ).sum(1)
        # This equals (aligned + crossed)/2 before norm matching. Neither
        # fields nor events are shuffled: each full value stays at its event.
        union = (wu.unsqueeze(-1) * (category + brand + common)).sum(1)
        geometric = (wj.unsqueeze(-1) * (category + brand + common)).sum(1)
        return {'aligned': aligned, 'crossed': crossed, 'union': union,
                'geometric': geometric}, wc, wb, wu, wj

    @staticmethod
    def _match_norm(raw, pooled):
        raw_norm = raw.norm(dim=-1, keepdim=True)
        reference_norm = pooled.norm(dim=-1, keepdim=True)
        matched = raw * (reference_norm / raw_norm.clamp_min(1e-6))
        # Empty and degenerate summaries use the reference; both branches finite.
        return torch.where(raw_norm > 1e-6, matched, pooled)

    def forward(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        reads, _, _, _, _ = self._read(category, brand, common, mask, query)
        summary = self._match_norm(reads[self.value_route], pooled)
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        reads, wc, wb, wu, wj = self._read(category, brand, common, mask, query)
        _, cc, cb, _, _ = self._read(category, brand, common, mask, context)
        selected_raw = reads[self.value_route]
        selected = self._match_norm(selected_raw, pooled)
        active = mask.any(-1).float()
        count = active.sum().clamp_min(1)
        reference_norm = pooled.norm(dim=-1)
        selected_norm = selected.norm(dim=-1)

        def active_mean(values):
            return float((values * active).sum() / count)

        def cosine(left, right):
            return ((left * right).sum(-1)
                    / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-9))

        def entropy(weight):
            return -(weight * weight.clamp_min(1e-9).log()).sum(-1)

        target_category = batch['static'][:, self.category, None]
        target_brand = batch['static'][:, self.brand, None]
        cm = (batch['history_category'] == target_category) & (target_category != 0) & mask
        bm = (batch['history_brand'] == target_brand) & (target_brand != 0) & mask
        return {
            'nonempty_fraction': float(active.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'reference_norm_mean': float(reference_norm.mean()),
            'selected_norm_mean': float(selected_norm.mean()),
            'norm_absolute_error_max': float((selected_norm - reference_norm).abs().max()),
            'norm_relative_error_max': float(((selected_norm - reference_norm).abs()
                                             / reference_norm.clamp_min(1e-6)).max()),
            'norm_fallback_fraction': float((selected_raw.norm(dim=-1) <= 1e-6).float().mean()),
            'union_decomposition_error_max': float((reads['union'] - 0.5 *
                                                    (reads['aligned'] + reads['crossed'])).abs().max()),
            'union_aligned_cosine_nonempty': active_mean(cosine(reads['union'], reads['aligned'])),
            'union_geometric_cosine_nonempty': active_mean(cosine(reads['union'], reads['geometric'])),
            'aligned_crossed_cosine_nonempty': active_mean(cosine(reads['aligned'], reads['crossed'])),
            'field_attention_l1_nonempty': active_mean((wc - wb).abs().sum(-1)),
            'union_geometric_attention_l1_nonempty': active_mean((wu - wj).abs().sum(-1)),
            'attention_l1_vs_context_nonempty': active_mean(0.5 * ((wc - cc).abs().sum(-1)
                                                                  + (wb - cb).abs().sum(-1))),
            'union_attention_entropy_nonempty': active_mean(entropy(wu)),
            'geometric_attention_entropy_nonempty': active_mean(entropy(wj)),
            # Hash-bucket equality is a proxy; collisions are not semantic matches.
            'category_bucket_match_mass_nonempty': active_mean((wc * cm.float()).sum(-1)),
            'brand_bucket_match_mass_nonempty': active_mean((wb * bm.float()).sum(-1)),
            'union_category_bucket_match_mass_nonempty': active_mean((wu * cm.float()).sum(-1)),
            'union_brand_bucket_match_mass_nonempty': active_mean((wu * bm.float()).sum(-1)),
            'uses_target_query': float(self.target_query),
            'uses_union_values': float(self.value_route == 'union'),
            'uses_crossed_values': float(self.value_route == 'crossed'),
        }
