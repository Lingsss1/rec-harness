"""Same-event semantic interaction versus factorized marginal interaction.

Inherits the diagonal-variance incumbent. Adds one zero-initialized D x D map
fed by q_category * q_brand * E_u[category * brand], or the capacity-matched
replacement q_category * q_brand * E_u[category] * E_u[brand]. The common
probability measure u is the arithmetic average of the two inherited reads.
Only the new operator and its operand source vary in the 2 x 2 design.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'separate_reads': True,
    'value_target': True, 'product_values': True,
    'moment_variance': True, 'moment_target': True,
    'event_pair': False, 'pair_target': True,
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
        self.event_pair = self.config['event_pair']
        self.pair_target = self.config['pair_target']
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
        # Appended after inherited parameters: initial inherited RNG is preserved.
        self.pair_projection = nn.Parameter(torch.zeros(dim, dim))

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
        vc = (wc.unsqueeze(-1) * (category - hc[:, None, :]).square()).sum(1)
        vb = (wb.unsqueeze(-1) * (brand - hb[:, None, :]).square()).sum(1)
        base = separate if self.separate_reads else joint
        return fields, mask, pooled, target, context, base, hc, hb, vc, vb, wc, wb, common, category, brand

    @staticmethod
    def _normalize_driver(raw, hc, hb):
        norm = raw.norm(dim=-1, keepdim=True)
        reference = torch.cat([hc, hb], dim=-1).norm(dim=-1, keepdim=True)
        scaled = raw * (reference / norm.clamp_min(1e-6))
        return torch.where(norm > 1e-6, scaled, torch.zeros_like(raw))

    def _moment_driver(self, hc, hb, vc, vb, query, variance=None):
        variance = self.moment_variance if variance is None else variance
        # Var(q * x) = q^2 * Var(x) for fixed q. The replacement control
        # uses (q * E[x])^2 and cannot recover event dispersion from that mean.
        sc, sb = (vc, vb) if variance else (hc.square(), hb.square())
        raw = query[:, 0].square() * sc + query[:, 1].square() * sb
        return self._normalize_driver(raw, hc, hb)

    def _components(self, batch):
        (fields, mask, pooled, target, context, base, hc, hb, vc, vb,
         wc, wb, common, category, brand) = self._state(batch)
        # The same probability measure defines both marginal means. The control
        # equals E_{i,j~u independently}[category_i * brand_j]; it can be computed
        # exactly without sampling, permutations, or a quadratic event matrix.
        u = 0.5 * (wc + wb)
        mc = (u.unsqueeze(-1) * category).sum(1)
        mb = (u.unsqueeze(-1) * brand).sum(1)
        paired = (u.unsqueeze(-1) * category * brand).sum(1)
        independent = mc * mb
        return dict(fields=fields, mask=mask, pooled=pooled, target=target,
                    context=context, base=base, hc=hc, hb=hb, vc=vc, vb=vb,
                    wc=wc, wb=wb, common=common, u=u, mc=mc, mb=mb,
                    category=category, brand=brand, paired=paired,
                    independent=independent)

    def _summary(self, s, event_pair=None, pair_target=None, branch_on=True):
        # Parent arithmetic order and both parent operators remain intact.
        old_query = s['target'] if self.value_target else s['context']
        old_driver = self._value_driver(s['hc'], s['hb'], old_query)
        old_residual = old_driver @ self.value_projection.t()
        moment_query = s['target'] if self.moment_target else s['context']
        moment_driver = self._moment_driver(
            s['hc'], s['hb'], s['vc'], s['vb'], moment_query)
        moment_residual = moment_driver @ self.moment_projection.t()
        event_pair = self.event_pair if event_pair is None else event_pair
        pair_target = self.pair_target if pair_target is None else pair_target
        query = s['target'] if pair_target else s['context']
        statistic = s['paired'] if event_pair else s['independent']
        raw_driver = query[:, 0] * query[:, 1] * statistic
        driver = self._normalize_driver(raw_driver, s['hc'], s['hb'])
        residual = driver @ self.pair_projection.t()
        parent_raw = s['base'] + old_residual
        parent_raw = parent_raw + moment_residual
        raw = parent_raw + residual if branch_on else parent_raw
        return self._match_norm(raw, s['pooled']), driver, residual, parent_raw

    def _logits(self, fields, summary):
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def forward(self, batch):
        s = self._components(batch)
        summary, _, _, _ = self._summary(s)
        return self._logits(s['fields'], summary)

    def diagnostics(self, batch):
        s = self._components(batch)
        selected, driver, residual, parent_raw = self._summary(s)
        off, _, _, _ = self._summary(s, branch_on=False)
        alternate, alternate_driver, _, _ = self._summary(
            s, event_pair=not self.event_pair)
        alternate_source, _, _, _ = self._summary(
            s, pair_target=not self.pair_target)
        logits = self._logits(s['fields'], selected)
        change = (logits - self._logits(s['fields'], off)).abs()
        switch_change = (logits - self._logits(s['fields'], alternate)).abs()
        # E[c*b] - E[c]*E[b] = E[(c-E[c])*(b-E[b])] under the SAME u.
        # This is cross-field diagonal covariance, not within-field full Cq.
        covariance = (s['u'].unsqueeze(-1)
                      * (s['category'] - s['mc'][:, None, :])
                      * (s['brand'] - s['mb'][:, None, :])).sum(1)
        identity_error = s['paired'] - s['independent'] - covariance
        active = s['mask'].any(-1).float()
        empty = 1 - active
        singleton = (s['mask'].sum(-1) == 1).float()
        short = ((s['mask'].sum(-1) > 0) & (s['mask'].sum(-1) <= 16)).float()
        long = (s['mask'].sum(-1) > 16).float()
        missing = (batch['static'][:, self.brand] == 0).float()
        def group_mean(value, group):
            return float((value * group).sum() / group.sum().clamp_min(1))
        def mean_active(value):
            return group_mean(value, active)
        driver_norm = driver.norm(dim=-1)
        semantic_norm = torch.cat([s['hc'], s['hb']], -1).norm(dim=-1)
        valid = (driver_norm > 1e-6).float()
        ref_norm = s['pooled'].norm(dim=-1)
        cosine = (driver * alternate_driver).sum(-1) / (
            driver_norm * alternate_driver.norm(dim=-1)).clamp_min(1e-9)
        decomposition = (s['hc'] + s['hb']
                         + (s['u'].unsqueeze(-1) * s['common']).sum(1))
        # Diagnostic parent function uses the same fitted embeddings and weights;
        # it is NOT a separately retrained ablation or a population effect.
        parent_summary = self._match_norm(parent_raw, s['pooled'])
        return {
            'uses_same_event_pairs': float(self.event_pair),
            'uses_target_pair_operand': float(self.pair_target),
            'nonempty_fraction': float(active.mean()),
            'supplied_event_count_mean': float(s['mask'].float().sum(-1).mean()),
            'pair_projection_frobenius': float(self.pair_projection.norm()),
            'value_projection_frobenius': float(self.value_projection.norm()),
            'moment_projection_frobenius': float(self.moment_projection.norm()),
            'cross_covariance_identity_error_max': float(identity_error.abs().max()),
            'cross_covariance_identity_relative_error_max': float(
                (identity_error.norm(dim=-1) / (
                    s['paired'].norm(dim=-1) + s['independent'].norm(dim=-1)
                    + covariance.norm(dim=-1)).clamp_min(1e-6)).max()),
            'cross_covariance_norm_nonempty': mean_active(covariance.norm(dim=-1)),
            'paired_statistic_norm_nonempty': mean_active(s['paired'].norm(dim=-1)),
            'independent_statistic_norm_nonempty': mean_active(s['independent'].norm(dim=-1)),
            'statistic_driver_cosine_nonempty': mean_active(cosine),
            'singleton_count': float(singleton.sum()),
            'singleton_pair_difference_max': float(
                ((s['paired'] - s['independent']).norm(dim=-1) * singleton).max()),
            'empty_pair_statistic_norm_max': float((s['paired'].norm(dim=-1) * empty).max()),
            'reference_norm_mean': float(ref_norm.mean()),
            'selected_norm_mean': float(selected.norm(dim=-1).mean()),
            'norm_absolute_error_max': float((selected.norm(dim=-1) - ref_norm).abs().max()),
            'norm_relative_error_max': float(((selected.norm(dim=-1) - ref_norm).abs()
                                             / ref_norm.clamp_min(1e-6)).max()),
            'pair_driver_norm_error_max_valid': float(((driver_norm - semantic_norm).abs() * valid).max()),
            'degenerate_pair_driver_fraction': float((1 - valid).mean()),
            'pair_residual_relative_norm_nonempty': mean_active(
                residual.norm(dim=-1) / parent_raw.norm(dim=-1).clamp_min(1e-6)),
            'branch_off_parent_summary_error_max': float((off - parent_summary).abs().max()),
            'pair_off_logit_abs_change_nonempty': mean_active(change),
            'pair_off_logit_abs_change_short': group_mean(change, short),
            'pair_off_logit_abs_change_long': group_mean(change, long),
            'statistic_switch_logit_abs_change_nonempty': mean_active(switch_change),
            'statistic_switch_logit_abs_change_short': group_mean(switch_change, short),
            'statistic_switch_logit_abs_change_long': group_mean(switch_change, long),
            'pair_source_switch_logit_abs_change_nonempty': mean_active(
                (logits - self._logits(s['fields'], alternate_source)).abs()),
            'missing_brand_pair_off_logit_abs_change': group_mean(change, missing),
            'missing_brand_diagnostic_count': float(missing.sum()),
            'short_diagnostic_count': float(short.sum()),
            'long_diagnostic_count': float(long.sum()),
            'parent_decomposition_error_max': float((s['base'] - decomposition).abs().max()),
            'padding_attention_mass_max': float(torch.stack([
                (s['wc'] * (~s['mask']).float()).sum(-1),
                (s['wb'] * (~s['mask']).float()).sum(-1)]).abs().max()),
            'attention_sum_error_max': float(torch.stack([
                s['wc'].sum(-1) - active, s['wb'].sum(-1) - active]).abs().max()),
            'empty_history_summary_norm_max': float((selected.norm(dim=-1) * empty).max()),
            'empty_history_pair_driver_norm_max': float((driver_norm * empty).max()),
            'missing_target_category_fraction': float((batch['static'][:, self.category] == 0).float().mean()),
            'missing_target_brand_fraction': float(missing.mean()),
        }
