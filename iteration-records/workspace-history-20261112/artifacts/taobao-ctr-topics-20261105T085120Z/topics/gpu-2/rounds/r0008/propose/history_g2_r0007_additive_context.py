"""Norm-controlled explicit field interactions after inherited target retrieval.

Prospective factorial: product/additive residual input x target/context source.
All cells start at the incumbent function and preserve its output width, initial
parameters, attention, loss, and optimizer. No performance or novelty claim.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'separate_reads': True,
    'value_target': False, 'product_values': False,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.separate_reads = self.config['separate_reads']
        self.value_target = self.config['value_target']
        self.product_values = self.config['product_values']
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
        # Construct after all inherited parameters; zeros consume no RNG draws.
        # Every factorial cell has the same trainable 16 x 16 residual map.
        self.value_projection = nn.Parameter(torch.zeros(dim, dim))

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

    def _value_driver(self, hc, hb, query, product=None):
        """Normalize operator inputs to the same retrieved semantic norm.

        The queries only affect values here. Attention always uses the inherited
        target queries in all four registered cells. Products are coordinatewise
        explicit second-order interactions, not an exact UniDot implementation.
        """
        product = self.product_values if product is None else product
        if product:
            raw = query[:, 0] * hc + query[:, 1] * hb
        else:
            raw = query[:, 0] + hc + query[:, 1] + hb
        raw_norm = raw.norm(dim=-1, keepdim=True)
        reference_norm = torch.cat([hc, hb], dim=-1).norm(dim=-1, keepdim=True)
        scaled = raw * (reference_norm / raw_norm.clamp_min(1e-6))
        # No fabricated value direction when the product is degenerate.
        return torch.where(raw_norm > 1e-6, scaled, torch.zeros_like(raw))

    def _state(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        retrieval_query = target if self.target_query else context
        separate, joint, wc, wb, wj = self._read(
            category, brand, common, mask, retrieval_query)
        hc = (wc.unsqueeze(-1) * category).sum(1)
        hb = (wb.unsqueeze(-1) * brand).sum(1)
        base = separate if self.separate_reads else joint
        return fields, mask, pooled, target, context, base, hc, hb, wc, wb, common

    def _summary(self, base, hc, hb, pooled, query, product=None):
        driver = self._value_driver(hc, hb, query, product)
        residual = driver @ self.value_projection.t()
        summary = self._match_norm(base + residual, pooled)
        return summary, driver, residual

    def _logits(self, fields, summary):
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def forward(self, batch):
        fields, _, pooled, target, context, base, hc, hb, _, _, _ = self._state(batch)
        query = target if self.value_target else context
        summary, _, _ = self._summary(base, hc, hb, pooled, query)
        return self._logits(fields, summary)

    def diagnostics(self, batch):
        fields, mask, pooled, target, context, base, hc, hb, wc, wb, common = self._state(batch)
        query = target if self.value_target else context
        other = context if self.value_target else target
        selected, driver, residual = self._summary(base, hc, hb, pooled, query)
        alternate_source, other_driver, _ = self._summary(base, hc, hb, pooled, other)
        alternate_operator, _, _ = self._summary(
            base, hc, hb, pooled, query, not self.product_values)
        off = self._match_norm(base, pooled)
        selected_logits = self._logits(fields, selected)
        active = mask.any(-1).float()
        count = active.sum().clamp_min(1)
        empty = 1 - active
        reference_norm = pooled.norm(dim=-1)
        semantic_norm = torch.cat([hc, hb], dim=-1).norm(dim=-1)
        driver_norm = driver.norm(dim=-1)
        valid_driver = (driver_norm > 1e-6).float()
        cosine = (driver * other_driver).sum(-1) / (
            driver_norm * other_driver.norm(dim=-1)).clamp_min(1e-9)
        common_read = (0.5 * (wc + wb)).unsqueeze(-1) * common
        decomposition = hc + hb + common_read.sum(1)
        def mean_active(value):
            return float((value * active).sum() / count)
        return {
            'nonempty_fraction': float(active.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'uses_target_retrieval': float(self.target_query),
            'uses_target_values': float(self.value_target),
            'uses_product_values': float(self.product_values),
            'projection_frobenius': float(self.value_projection.norm()),
            'reference_norm_mean': float(reference_norm.mean()),
            'selected_norm_mean': float(selected.norm(dim=-1).mean()),
            'norm_absolute_error_max': float((selected.norm(dim=-1) - reference_norm).abs().max()),
            'norm_relative_error_max': float(((selected.norm(dim=-1) - reference_norm).abs()
                                             / reference_norm.clamp_min(1e-6)).max()),
            'driver_norm_error_max_valid': float(((driver_norm - semantic_norm).abs()
                                                  * valid_driver).max()),
            'degenerate_driver_fraction': float((1 - valid_driver).mean()),
            'residual_relative_norm_nonempty': mean_active(
                residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-6)),
            'residual_off_logit_abs_change_nonempty': mean_active(
                (selected_logits - self._logits(fields, off)).abs()),
            'value_source_switch_logit_abs_change_nonempty': mean_active(
                (selected_logits - self._logits(fields, alternate_source)).abs()),
            'value_operator_switch_logit_abs_change_nonempty': mean_active(
                (selected_logits - self._logits(fields, alternate_operator)).abs()),
            'driver_source_cosine_nonempty': mean_active(cosine),
            'parent_decomposition_error_max': float((base - decomposition).abs().max()),
            'padding_attention_mass_max': float(torch.stack([
                (wc * (~mask).float()).sum(-1), (wb * (~mask).float()).sum(-1)]).abs().max()),
            'attention_sum_error_max': float(torch.stack([
                wc.sum(-1) - active, wb.sum(-1) - active]).abs().max()),
            'empty_history_summary_norm_max': float((selected.norm(dim=-1) * empty).max()),
            'empty_history_driver_norm_max': float((driver_norm * empty).max()),
            'field_attention_l1_nonempty': mean_active((wc - wb).abs().sum(-1)),
            'missing_target_category_fraction': float((batch['static'][:, self.category] == 0).float().mean()),
            'missing_target_brand_fraction': float((batch['static'][:, self.brand] == 0).float().mean()),
        }
