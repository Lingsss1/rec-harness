"""Residual query coupling, with matched self-refinement controls.

The other field may rotate each retrieval query without changing its norm.
The shared projection is appended after inherited modules and starts at zero.
Separate field reads, metadata-mixed keys, aligned values, pooled-summary norm,
classifier width, optimizer and loss are inherited. No novelty is asserted.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': False, 'separate_reads': True,
    'cross_query': True,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.separate_reads = self.config['separate_reads']
        self.cross_query = self.config['cross_query']
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
        # Append after all inherited random initialization. Same construction
        # and trainable capacity in every cell; only projection input differs.
        self.query_residual = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.query_residual.weight)

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

    def _refine(self, base, cross=None):
        cross = self.cross_query if cross is None else cross
        source = base.flip(1) if cross else base
        raw = base + self.query_residual(source)
        # Own-query norm is the reference, including zero/missing own fields.
        # Thus an absent own query stays zero even when its counterpart exists.
        return self._match_norm(raw, base)

    @staticmethod
    def _weights(scores, mask):
        weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def _read(self, category, brand, common, mask, query):
        category_score = self._score(query[:, 0], category + common)
        brand_score = self._score(query[:, 1], brand + common)
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
        query = self._refine(target if self.target_query else context)
        separate, joint, _, _, _ = self._read(category, brand, common, mask, query)
        summary = self._match_norm(separate if self.separate_reads else joint, pooled)
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        base = target if self.target_query else context
        query = self._refine(base)
        separate, joint, wc, wb, wj = self._read(category, brand, common, mask, query)
        _, _, cc, cb, _ = self._read(category, brand, common, mask, self._refine(context))
        off_separate, off_joint, oc, ob, _ = self._read(category, brand, common, mask, base)
        alt_query = self._refine(base, cross=not self.cross_query)
        _, _, ac, ab, _ = self._read(category, brand, common, mask, alt_query)
        separate_matched = self._match_norm(separate, pooled)
        joint_matched = self._match_norm(joint, pooled)
        selected_raw = separate if self.separate_reads else joint
        selected = separate_matched if self.separate_reads else joint_matched
        active = mask.any(-1).float()
        active_count = active.sum().clamp_min(1)
        reference_norm = pooled.norm(dim=-1)
        separate_norm = separate_matched.norm(dim=-1)
        joint_norm = joint_matched.norm(dim=-1)
        cosine = ((separate_matched * joint_matched).sum(-1)
                  / (separate_norm * joint_norm).clamp_min(1e-9))
        entropy = -0.5 * ((wc * wc.clamp_min(1e-9).log()).sum(-1)
                          + (wb * wb.clamp_min(1e-9).log()).sum(-1))
        off_summary = self._match_norm(off_separate if self.separate_reads else off_joint, pooled)
        static = torch.cat(fields, dim=-1)
        logits = self.output(torch.cat([static, selected], dim=-1)).squeeze(-1)
        off_logits = self.output(torch.cat([static, off_summary], dim=-1)).squeeze(-1)
        query_norm = query.norm(dim=-1)
        base_norm = base.norm(dim=-1)
        query_cosine = ((query * base).sum(-1)
                        / (query_norm * base_norm).clamp_min(1e-9))
        query_active = (base_norm > 1e-6).float()
        query_count = query_active.sum().clamp_min(1)
        # Same pooled-norm construction means magnitude cannot explain the
        # within-checkpoint projection-off contrast. This is a sensitivity
        # diagnostic, not a separately trained mechanism attribution control.
        return {
            'nonempty_fraction': float(active.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'reference_norm_mean': float(reference_norm.mean()),
            'selected_norm_mean': float(selected.norm(dim=-1).mean()),
            'norm_absolute_error_max': float((selected.norm(dim=-1) - reference_norm).abs().max()),
            'norm_relative_error_max': float(((selected.norm(dim=-1) - reference_norm).abs()
                                              / reference_norm.clamp_min(1e-6)).max()),
            'norm_fallback_fraction': float((selected_raw.norm(dim=-1) <= 1e-6).float().mean()),
            'separate_joint_cosine_nonempty': float((cosine * active).sum() / active_count),
            'field_attention_l1_nonempty': float(((wc - wb).abs().sum(-1) * active).sum()
                                                / active_count),
            'attention_l1_vs_context_nonempty': float((0.5 * ((wc - cc).abs().sum(-1)
                                                               + (wb - cb).abs().sum(-1))
                                                       * active).sum() / active_count),
            'field_attention_entropy_nonempty': float((entropy * active).sum() / active_count),
            'joint_attention_entropy_nonempty': float((-(wj * wj.clamp_min(1e-9).log()).sum(-1)
                                                       * active).sum() / active_count),
            'uses_target_query': float(self.target_query),
            'uses_separate_reads': float(self.separate_reads),
            'uses_cross_query': float(self.cross_query),
            'residual_weight_frobenius': float(self.query_residual.weight.norm()),
            'query_norm_absolute_error_max': float((query_norm - base_norm).abs().max()),
            'query_norm_relative_error_max': float(((query_norm - base_norm).abs()
                                                    / base_norm.clamp_min(1e-6)).max()),
            'query_cosine_to_base_present': float((query_cosine * query_active).sum() / query_count),
            'query_relative_residual_present': float(((query - base).norm(dim=-1)
                                                       / base_norm.clamp_min(1e-6)
                                                       * query_active).sum() / query_count),
            'attention_l1_projection_off_nonempty': float((0.5 * ((wc - oc).abs().sum(-1)
                                                                   + (wb - ob).abs().sum(-1))
                                                           * active).sum() / active_count),
            'attention_l1_other_route_nonempty': float((0.5 * ((wc - ac).abs().sum(-1)
                                                               + (wb - ab).abs().sum(-1))
                                                       * active).sum() / active_count),
            'projection_off_logit_abs_change_nonempty': float(((logits - off_logits).abs()
                                                               * active).sum() / active_count),
            'padding_attention_mass_max': float(((wc + wb) * (~mask).float()).abs().max()),
            'attention_sum_error_max': float(torch.stack([
                (wc.sum(-1) - active).abs(), (wb.sum(-1) - active).abs()
            ]).max()),
            'empty_history_summary_norm_max': float((selected.norm(dim=-1) * (1 - active)).max()),
        }
