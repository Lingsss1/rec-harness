"""Field-identity preservation x target-query factorial.

Concatenation is established prior art, not an architecture novelty claim.
Compare two field contributions with a duplicated additive summary. All four
variants match parameter shapes, initialization order and total history norm.
Duplicating the sum matches allocated capacity, not input rank or optimization.
No optimizer, loss, information-boundary or training-recipe changes.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': False, 'late_fusion': True,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.late_fusion = self.config['late_fusion']
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        self.dim = dim
        # Inherit encoder/scorer shapes and initialization order.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        # Same width across all four variants; +dim versus the incumbent.
        width = len(meta['fields']) * dim + 2 * dim
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
        category_weight = self._weights(category_score, mask)
        brand_weight = self._weights(brand_score, mask)
        # left + right equals the ancestor's summary up to summation rounding.
        left = (category_weight.unsqueeze(-1) * (category + 0.5 * common)).sum(1)
        right = (brand_weight.unsqueeze(-1) * (brand + 0.5 * common)).sum(1)
        return left, right, category_weight, brand_weight

    @staticmethod
    def _duplicate(vector):
        # Isometric embedding into twice the width; both blocks remain active.
        return torch.cat([vector, vector], dim=-1) * (2.0 ** -0.5)

    def _fuse(self, left, right, pooled):
        separated = torch.cat([left, right], dim=-1)
        collapsed = self._duplicate(left + right)
        reference = self._duplicate(pooled)
        raw = separated if self.late_fusion else collapsed
        return self._match_norm(raw, reference), raw, reference

    @staticmethod
    def _match_norm(raw, pooled):
        raw_norm = raw.norm(dim=-1, keepdim=True)
        reference_norm = pooled.norm(dim=-1, keepdim=True)
        matched = raw * (reference_norm / raw_norm.clamp_min(1e-6))
        # Degenerate vectors use the reference itself, including empty histories.
        # Both branches are finite before where; no divisions by zero occur.
        return torch.where(raw_norm > 1e-6, matched, pooled)

    def forward(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        left, right, _, _ = self._read(category, brand, common, mask, query)
        summary, _, _ = self._fuse(left, right, pooled)
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        left, right, wc, wb = self._read(category, brand, common, mask, query)
        _, _, cc, cb = self._read(category, brand, common, mask, context)
        selected, selected_raw, reference = self._fuse(left, right, pooled)
        ancestor = (wc.unsqueeze(-1) * category + wb.unsqueeze(-1) * brand
                    + (0.5 * (wc + wb)).unsqueeze(-1) * common).sum(1)
        collapsed = self._match_norm(self._duplicate(left + right), reference)
        ancestor_matched = self._match_norm(ancestor, pooled)
        active = mask.any(-1).float()
        active_count = active.sum().clamp_min(1)
        reference_norm = pooled.norm(dim=-1)
        nl, nr = left.norm(dim=-1), right.norm(dim=-1)
        entropy = -0.5 * ((wc * wc.clamp_min(1e-9).log()).sum(-1)
                          + (wb * wb.clamp_min(1e-9).log()).sum(-1))
        category_match = ((batch['history_category'] == batch['static'][:, self.category, None])
                          & (batch['history_category'] != 0) & mask)
        brand_match = ((batch['history_brand'] == batch['static'][:, self.brand, None])
                       & (batch['history_brand'] != 0) & mask)
        static = torch.cat(fields, dim=-1)
        logits = self.output(torch.cat([static, selected], dim=-1)).squeeze(-1)
        swapped = torch.cat([selected[:, self.dim:], selected[:, :self.dim]], dim=-1)
        swapped_logits = self.output(torch.cat([static, swapped], dim=-1)).squeeze(-1)

        def active_mean(value):
            return float((value * active).sum() / active_count)

        return {
            'nonempty_fraction': float(active.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'reference_norm_mean': float(reference_norm.mean()),
            'selected_norm_mean': float(selected.norm(dim=-1).mean()),
            'norm_absolute_error_max': float((selected.norm(dim=-1) - reference_norm).abs().max()),
            'norm_relative_error_max': float(((selected.norm(dim=-1) - reference_norm).abs()
                                              / reference_norm.clamp_min(1e-6)).max()),
            'norm_fallback_fraction': float((selected_raw.norm(dim=-1) <= 1e-6).float().mean()),
            'ancestor_decomposition_error_max': float((left + right - ancestor).abs().max()),
            'collapsed_ancestor_error_max': float((collapsed - self._duplicate(ancestor_matched)).abs().max()),
            'duplicate_isometry_error_max': float((reference.norm(dim=-1) - reference_norm).abs().max()),
            'left_right_cosine_nonempty': active_mean((left * right).sum(-1) / (nl * nr).clamp_min(1e-9)),
            'sum_to_branch_norm_ratio_nonempty': active_mean((left + right).norm(dim=-1) / (nl + nr).clamp_min(1e-9)),
            'left_energy_fraction_nonempty': active_mean(nl.square() / (nl.square() + nr.square()).clamp_min(1e-9)),
            'branch_difference_norm_nonempty': active_mean((left - right).norm(dim=-1)),
            'field_attention_l1_nonempty': float(((wc - wb).abs().sum(-1) * active).sum()
                                                / active_count),
            'attention_l1_vs_context_nonempty': float((0.5 * ((wc - cc).abs().sum(-1)
                                                               + (wb - cb).abs().sum(-1))
                                                       * active).sum() / active_count),
            'field_attention_entropy_nonempty': float((entropy * active).sum() / active_count),
            'padding_attention_mass_max': float((0.5 * (wc + wb) * (~mask).float()).sum(-1).max()),
            'category_bucket_match_mass_nonempty': active_mean((wc * category_match.float()).sum(-1)),
            'brand_bucket_match_mass_nonempty': active_mean((wb * brand_match.float()).sum(-1)),
            # Frozen-head sensitivity, not a causal performance estimate.
            'branch_swap_logit_abs_change_nonempty': active_mean((logits - swapped_logits).abs()),
            'uses_target_query': float(self.target_query),
            'uses_late_fusion': float(self.late_fusion),
            'history_output_width': float(2 * self.dim),
        }
