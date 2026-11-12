"""Event-level field retrieval with a target-query x readout factorial.

DIN-style relation scoring is inherited, not a novelty claim. Separate field
softmaxes are compared with a shared consensus softmax using the same scores,
values, parameters and output width. All summaries are normalized to the norm
of a target-independent masked-mean history. No optimizer or loss changes.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'separate_reads': True,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.separate_reads = self.config['separate_reads']
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
        query = target if self.target_query else context
        separate, joint, _, _, _ = self._read(category, brand, common, mask, query)
        summary = self._match_norm(separate if self.separate_reads else joint, pooled)
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        separate, joint, wc, wb, wj = self._read(category, brand, common, mask, query)
        _, _, cc, cb, _ = self._read(category, brand, common, mask, context)
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
        }
