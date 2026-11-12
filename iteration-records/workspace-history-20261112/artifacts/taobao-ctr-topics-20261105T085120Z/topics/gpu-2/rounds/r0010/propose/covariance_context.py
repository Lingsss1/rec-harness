"""Attended within-field covariance readout with matched diagonal controls.

The added moment driver is q * (C @ q), or its inherited diagonal restriction
q.square() * diag(C), summed across category and brand. Covariance is contracted
through centered events without constructing [B,D,D] tensors during forward.
All four cells share parameters, initialization, attention and normalization.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'separate_reads': True,
    'value_target': True, 'product_values': True,
    'moment_variance': True, 'moment_target': False,
    'covariance_values': True,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.separate_reads = self.config['separate_reads']
        self.value_target = self.config['value_target']
        self.product_values = self.config['product_values']
        self.moment_variance = self.config['moment_variance']
        self.moment_target = self.config['moment_target']
        self.covariance_values = self.config['covariance_values']
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
        # Added after every incumbent parameter, without consuming RNG draws.
        self.moment_projection = nn.Parameter(torch.zeros(dim, dim))

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
        separate, joint, wc, wb, _ = self._read(
            category, brand, common, mask, retrieval_query)
        hc = (wc.unsqueeze(-1) * category).sum(1)
        hb = (wb.unsqueeze(-1) * brand).sum(1)
        # Center explicitly to avoid subtracting two nearly equal second moments.
        # Same allowed event prefix and field-specific attention as the first moment.
        dc = category - hc[:, None, :]
        db = brand - hb[:, None, :]
        vc = (wc.unsqueeze(-1) * dc.square()).sum(1)
        vb = (wb.unsqueeze(-1) * db.square()).sum(1)
        base = separate if self.separate_reads else joint
        return fields, mask, pooled, target, context, base, hc, hb, vc, vb, wc, wb, common, dc, db

    @staticmethod
    def _normalize_driver(raw, hc, hb):
        norm = raw.norm(dim=-1, keepdim=True)
        reference = torch.cat([hc, hb], dim=-1).norm(dim=-1, keepdim=True)
        scaled = raw * (reference / norm.clamp_min(1e-6))
        return torch.where(norm > 1e-6, scaled, torch.zeros_like(raw))

    @staticmethod
    def _covariance_raw(dc, db, wc, wb, query):
        # C q = sum_l w_l delta_l (delta_l dot q), with C centered separately
        # under each field's existing attention. No category/brand cross terms.
        qc, qb = query[:, 0], query[:, 1]
        c_dot = (dc * qc[:, None, :]).sum(-1)
        b_dot = (db * qb[:, None, :]).sum(-1)
        c_read = (dc * (wc * c_dot).unsqueeze(-1)).sum(1)
        b_read = (db * (wb * b_dot).unsqueeze(-1)).sum(1)
        return qc * c_read + qb * b_read

    def _moment_driver(self, hc, hb, vc, vb, query, dc, db, wc, wb,
                       covariance=None):
        covariance = self.covariance_values if covariance is None else covariance
        if covariance:
            raw = self._covariance_raw(dc, db, wc, wb, query)
        else:
            # Exact inherited diagonal-variance operation and order.
            raw = query[:, 0].square() * vc + query[:, 1].square() * vb
        return self._normalize_driver(raw, hc, hb)

    def _summary(self, base, hc, hb, vc, vb, pooled, target, context,
                 dc, db, wc, wb, covariance=None, moment_target=None,
                 branch_on=True):
        old_query = target if self.value_target else context
        old_driver = self._value_driver(hc, hb, old_query)
        old_residual = old_driver @ self.value_projection.t()
        moment_target = self.moment_target if moment_target is None else moment_target
        query = target if moment_target else context
        driver = self._moment_driver(hc, hb, vc, vb, query, dc, db, wc, wb,
                                     covariance)
        residual = driver @ self.moment_projection.t()
        raw_summary = base + old_residual
        if branch_on:
            raw_summary = raw_summary + residual
        return self._match_norm(raw_summary, pooled), driver, residual, old_residual

    def _logits(self, fields, summary):
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def forward(self, batch):
        (fields, _, pooled, target, context, base, hc, hb, vc, vb,
         wc, wb, _, dc, db) = self._state(batch)
        summary, _, _, _ = self._summary(
            base, hc, hb, vc, vb, pooled, target, context, dc, db, wc, wb)
        return self._logits(fields, summary)

    def diagnostics(self, batch):
        (fields, mask, pooled, target, context, base, hc, hb, vc, vb,
         wc, wb, common, dc, db) = self._state(batch)
        args = (base, hc, hb, vc, vb, pooled, target, context, dc, db, wc, wb)
        selected, driver, residual, old_residual = self._summary(*args)
        off, _, _, _ = self._summary(*args, branch_on=False)
        alternate, alternate_driver, _, _ = self._summary(
            *args, covariance=not self.covariance_values)
        alternate_source, _, _, _ = self._summary(
            *args, moment_target=not self.moment_target)
        query = target if self.moment_target else context
        full_raw = self._covariance_raw(dc, db, wc, wb, query)
        diag_raw = query[:, 0].square() * vc + query[:, 1].square() * vb
        # Explicit small covariance matrices only in the host's diagnostic call.
        # This verifies the event contraction; it adds no model parameters.
        cc = torch.einsum('bli,bl,blj->bij', dc, wc, dc)
        cb = torch.einsum('bli,bl,blj->bij', db, wb, db)
        explicit = (query[:, 0] * torch.einsum('bij,bj->bi', cc, query[:, 0])
                    + query[:, 1] * torch.einsum('bij,bj->bi', cb, query[:, 1]))
        logits = self._logits(fields, selected)
        active = mask.any(-1).float()
        empty = 1 - active
        singleton = (mask.sum(-1) == 1).float()
        short = ((mask.sum(-1) > 0) & (mask.sum(-1) <= 16)).float()
        long = (mask.sum(-1) > 16).float()
        semantic_norm = torch.cat([hc, hb], -1).norm(dim=-1)
        driver_norm = driver.norm(dim=-1)
        valid = (driver_norm > 1e-6).float()
        reference_norm = pooled.norm(dim=-1)
        change = (logits - self._logits(fields, off)).abs()
        switch_change = (logits - self._logits(fields, alternate)).abs()
        def group_mean(value, group):
            return float((value * group).sum() / group.sum().clamp_min(1))
        def mean_active(value):
            return group_mean(value, active)
        cosine = (driver * alternate_driver).sum(-1) / (
            driver_norm * alternate_driver.norm(dim=-1)).clamp_min(1e-9)
        decomposition = hc + hb + ((0.5 * (wc + wb)).unsqueeze(-1) * common).sum(1)
        return {
            'nonempty_fraction': float(active.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'uses_full_covariance': float(self.covariance_values),
            'uses_target_moment_operand': float(self.moment_target),
            'incumbent_projection_frobenius': float(self.value_projection.norm()),
            'moment_projection_frobenius': float(self.moment_projection.norm()),
            'full_contraction_error_max': float((full_raw - explicit).abs().max()),
            'full_contraction_relative_error_max': float(
                ((full_raw - explicit).norm(dim=-1) / explicit.norm(dim=-1).clamp_min(1e-6)).max()),
            'covariance_diagonal_error_max': float(torch.maximum(
                (cc.diagonal(dim1=-2, dim2=-1) - vc).abs().max(),
                (cb.diagonal(dim1=-2, dim2=-1) - vb).abs().max())),
            'full_quadratic_form_min': float(full_raw.sum(-1).min()),
            'full_raw_negative_coordinate_fraction': mean_active((full_raw < 0).float().mean(-1)),
            'offdiagonal_raw_norm_nonempty': mean_active((full_raw - diag_raw).norm(dim=-1)),
            'full_raw_norm_nonempty': mean_active(full_raw.norm(dim=-1)),
            'diagonal_raw_norm_nonempty': mean_active(diag_raw.norm(dim=-1)),
            'statistic_driver_cosine_nonempty': mean_active(cosine),
            'singleton_count': float(singleton.sum()),
            'singleton_full_raw_norm_max': float((full_raw.norm(dim=-1) * singleton).max()),
            'empty_full_raw_norm_max': float((full_raw.norm(dim=-1) * empty).max()),
            'reference_norm_mean': float(reference_norm.mean()),
            'selected_norm_mean': float(selected.norm(dim=-1).mean()),
            'norm_absolute_error_max': float((selected.norm(dim=-1) - reference_norm).abs().max()),
            'norm_relative_error_max': float(((selected.norm(dim=-1) - reference_norm).abs()
                                             / reference_norm.clamp_min(1e-6)).max()),
            'moment_driver_norm_error_max_valid': float(((driver_norm - semantic_norm).abs() * valid).max()),
            'degenerate_moment_driver_fraction': float((1 - valid).mean()),
            'moment_residual_relative_norm_nonempty': mean_active(
                residual.norm(dim=-1) / (base + old_residual).norm(dim=-1).clamp_min(1e-6)),
            'moment_off_logit_abs_change_nonempty': mean_active(change),
            'moment_off_logit_abs_change_short': group_mean(change, short),
            'moment_off_logit_abs_change_long': group_mean(change, long),
            'statistic_switch_logit_abs_change_nonempty': mean_active(switch_change),
            'statistic_switch_logit_abs_change_short': group_mean(switch_change, short),
            'statistic_switch_logit_abs_change_long': group_mean(switch_change, long),
            'moment_source_switch_logit_abs_change_nonempty': mean_active(
                (logits - self._logits(fields, alternate_source)).abs()),
            'short_diagnostic_count': float(short.sum()),
            'long_diagnostic_count': float(long.sum()),
            'parent_decomposition_error_max': float((base - decomposition).abs().max()),
            'padding_attention_mass_max': float(torch.stack([
                (wc * (~mask).float()).sum(-1), (wb * (~mask).float()).sum(-1)]).abs().max()),
            'attention_sum_error_max': float(torch.stack([
                wc.sum(-1) - active, wb.sum(-1) - active]).abs().max()),
            'empty_history_summary_norm_max': float((selected.norm(dim=-1) * empty).max()),
            'empty_history_moment_driver_norm_max': float((driver_norm * empty).max()),
            'missing_target_category_fraction': float((batch['static'][:, self.category] == 0).float().mean()),
            'missing_target_brand_fraction': float((batch['static'][:, self.brand] == 0).float().mean()),
        }
