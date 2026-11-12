"""Semantic retrieval keys versus metadata-mixed keys: a query-by-key factorial.

Inherited: DIN-style relation scorer, independent field softmaxes, aligned values,
and target-independent pooled-norm matching from g2-r0001-target_separate.
Intervention: omit action/recency from relevance keys only. Both metadata
embeddings remain in every event value and in the reference norm. This is an
ablation hypothesis, not a new-attention priority claim. No optimizer/loss change.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'separate_reads': True, 'semantic_keys': False,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.separate_reads = self.config['separate_reads']
        self.semantic_keys = self.config['semantic_keys']
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Same construction order, parameter shapes and output width as baseline.
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

    def _read(self, category, brand, common, mask, query, semantic_keys=None):
        use_semantic = self.semantic_keys if semantic_keys is None else semantic_keys
        # The values below are unchanged: metadata is excluded from keys only.
        category_key = category if use_semantic else category + common
        brand_key = brand if use_semantic else brand + common
        category_score = self._score(query[:, 0], category_key)
        brand_score = self._score(query[:, 1], brand_key)
        category_weight = self._weights(category_score, mask)
        brand_weight = self._weights(brand_score, mask)
        # Consensus is a geometric, rather than arithmetic, opinion pool.
        joint_weight = self._weights(0.5 * (category_score + brand_score), mask)
        separate = (
            category_weight.unsqueeze(-1) * category
            + brand_weight.unsqueeze(-1) * brand
            + (0.5 * (category_weight + brand_weight)).unsqueeze(-1) * common
        ).sum(1)
        joint = (joint_weight.unsqueeze(-1) * (category + brand + common)).sum(1)
        return separate, joint, category_weight, brand_weight, joint_weight

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
        separate, joint, _, _, _ = self._read(category, brand, common, mask, query)
        summary = self._match_norm(separate if self.separate_reads else joint, pooled)
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        separate, joint, wc, wb, wj = self._read(category, brand, common, mask, query)
        _, _, cc, cb, _ = self._read(category, brand, common, mask, context)
        semantic, _, sc, sb, _ = self._read(category, brand, common, mask, query, True)
        mixed, _, mc, mb, _ = self._read(category, brand, common, mask, query, False)
        _, _, zc, zb, _ = self._read(category, brand, torch.zeros_like(common), mask, query)
        selected_raw = separate if self.separate_reads else joint
        selected = self._match_norm(selected_raw, pooled)
        semantic_matched = self._match_norm(semantic, pooled)
        mixed_matched = self._match_norm(mixed, pooled)
        active = mask.any(-1)
        active_float = active.float()
        active_count = active_float.sum().clamp_min(1)
        def mean_active(value):
            return float((value * active_float).sum() / active_count)
        weights = 0.5 * (wc + wb)
        ref_norm = pooled.norm(dim=-1)
        norm_error = (selected.norm(dim=-1) - ref_norm).abs()
        cosine = ((semantic_matched * mixed_matched).sum(-1)
                  / (semantic_matched.norm(dim=-1) * mixed_matched.norm(dim=-1)).clamp_min(1e-9))
        entropy = -0.5 * ((wc * wc.clamp_min(1e-9).log()).sum(-1)
                          + (wb * wb.clamp_min(1e-9).log()).sum(-1))
        category_match = (mask & (batch['history_category'] > 0)
                          & (batch['static'][:, self.category, None] > 0)
                          & (batch['history_category'] == batch['static'][:, self.category, None]))
        brand_match = (mask & (batch['history_brand'] > 0)
                       & (batch['static'][:, self.brand, None] > 0)
                       & (batch['history_brand'] == batch['static'][:, self.brand, None]))
        decomposed = ((wc.unsqueeze(-1) * (category + 0.5 * common)).sum(1)
                      + (wb.unsqueeze(-1) * (brand + 0.5 * common)).sum(1))
        diagnostics = {
            'nonempty_fraction': float(active_float.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'reference_norm_mean': float(ref_norm.mean()),
            'selected_norm_mean': float(selected.norm(dim=-1).mean()),
            'norm_absolute_error_max': float(norm_error.max()),
            'norm_relative_error_max': float((norm_error / ref_norm.clamp_min(1e-6)).max()),
            'norm_fallback_fraction': float((selected_raw.norm(dim=-1) <= 1e-6).float().mean()),
            'empty_history_summary_norm_max': float((selected.norm(dim=-1) * (~active).float()).max()),
            'padding_attention_mass_max': float((weights * (~mask).float()).sum(-1).max()),
            'attention_sum_error_max': float((weights.sum(-1) - active_float).abs().max()),
            'value_decomposition_error_max': float((separate - decomposed).abs().max()),
            'semantic_mixed_summary_cosine_nonempty': mean_active(cosine),
            'semantic_mixed_attention_l1_nonempty': mean_active(0.5 * ((sc - mc).abs().sum(-1)
                                                                                    + (sb - mb).abs().sum(-1))),
            'metadata_zeroed_attention_l1_nonempty': mean_active(0.5 * ((wc - zc).abs().sum(-1)
                                                                                   + (wb - zb).abs().sum(-1))),
            'field_attention_l1_nonempty': mean_active((wc - wb).abs().sum(-1)),
            'attention_l1_vs_context_nonempty': mean_active(0.5 * ((wc - cc).abs().sum(-1)
                                                                               + (wb - cb).abs().sum(-1))),
            'field_attention_entropy_nonempty': mean_active(entropy),
            'joint_attention_entropy_nonempty': mean_active(-(wj * wj.clamp_min(1e-9).log()).sum(-1)),
            'category_bucket_match_mass_nonempty': mean_active((wc * category_match.float()).sum(-1)),
            'brand_bucket_match_mass_nonempty': mean_active((wb * brand_match.float()).sum(-1)),
            'selected_mean_log2_age_nonempty': mean_active((weights * torch.log2(
                batch['history_age_seconds'].clamp_min(0) + 1)).sum(-1)),
            'uniform_mean_log2_age_nonempty': mean_active((mask.float() * torch.log2(
                batch['history_age_seconds'].clamp_min(0) + 1)).sum(-1)
                / mask.sum(-1).clamp_min(1)),
            'common_to_semantic_event_norm_ratio': float((common.norm(dim=-1) * mask.float()).sum()
                / ((0.5 * (category.norm(dim=-1) + brand.norm(dim=-1)) * mask.float()).sum().clamp_min(1e-6))),
            'uses_target_query': float(self.target_query),
            'uses_semantic_keys': float(self.semantic_keys),
            'uses_separate_reads': float(self.separate_reads),
        }
        for action_id in (1, 2, 3, 4):
            action_mask = (mask & (batch['history_type'] == action_id)).float()
            diagnostics['action_%d_attention_mass_nonempty' % action_id] = mean_active(
                (weights * action_mask).sum(-1))
            diagnostics['action_%d_uniform_share_nonempty' % action_id] = mean_active(
                action_mask.sum(-1) / mask.sum(-1).clamp_min(1))
        return diagnostics
