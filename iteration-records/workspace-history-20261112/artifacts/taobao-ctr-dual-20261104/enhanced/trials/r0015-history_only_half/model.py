"""Prospective target/history product-gradient routing factorial.

Two mixed cells complete the existing full/full and scale-half/scale-half
controls (r0013-full_audit, r0013-scale_half_audit). Keep the aligned unit
product forward, learned constructor, native autograd, optimizer, host loss,
seed and all other paths. Donor identities are the inherited detached stable
inverse-norm rank half-rotation within eligible events in an impression.
No new donor family, trainable parameter, training hook or RNG is introduced.

Mixed cells detach the opposite product operand, normalize each role with
its assigned backward rule, then add its zero-valued surrogate correction to
the detached common forward. Thus each operand receives exactly one VJP;
the two roles still accumulate into the SAME embedding tables. Other query,
static, history-value and attention paths remain fully trainable. These are
surrogate updates, not gradients of a new scalar population objective.

Host diagnostics are first-512 validation input/forward checks and tiny
synthetic float64 role VJPs, not supervised training or optimizer evidence.
Cross-seed robustness and literature priority remain unverified.
"""
import torch
from torch import nn
CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'product_query_gradient': 'full', 'product_history_gradient': 'scale_half_tangent'}

class Model(nn.Module):

    def __init__(self, meta, config=None):
        super().__init__()
        config = {**CONFIG, **(config or {})}
        self.config = config
        self.kind = config['kind']
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = config['embedding_dim']
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1))
        layers = []
        width = len(meta['fields']) * dim + (dim if self.kind != 'static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _encode(self, batch):
        fields = [embedding(batch['static'][:, i]) for (i, embedding) in enumerate(self.embeddings)]
        features = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        history = category + brand
        action = self.action(batch['history_type'])
        history = history + action
        recency = self.recency(ages)
        history = history + recency
        return (fields, features, history, (category, brand, action, recency))

    @staticmethod
    def _weights(scores, mask):
        scores = scores.masked_fill(~mask, -10000.0)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-09)

    def _logits(self, features, summary):
        features = torch.cat([features, summary], dim=-1)
        return self.output(features).squeeze(-1)

    @staticmethod
    def _rotation_donors(inverse, valid_regular, gradient):
        """Integer donor identities; detached keys, stable position tie break.

        No index/age neighborhood is imposed by scale ordering. Circular
        adjacency has one extreme wrap; scalar similarity is a manipulation
        check, not guaranteed for arbitrary short or heavy-tailed groups.
        """
        positions = torch.arange(inverse.shape[-1], device=inverse.device, dtype=torch.long)[None, :].expand_as(inverse)
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            keys = inverse.detach().masked_fill(~valid_regular, float('inf'))
        elif gradient in ('history_adjacent_tangent', 'history_reassigned_tangent'):
            keys = positions.to(inverse.dtype).masked_fill(~valid_regular, float('inf'))
        else:
            raise ValueError('rotation mode required')
        order = keys.argsort(dim=-1, stable=True)
        rank = torch.zeros_like(order).scatter(-1, order, positions)
        count = valid_regular.sum(-1, keepdim=True)
        shift = 1 if gradient.endswith('adjacent_tangent') else count // 2
        donor_rank = (rank + shift).remainder(count.clamp_min(1))
        donor = order.gather(-1, donor_rank)
        return torch.where(valid_regular, donor, positions)

    @staticmethod
    def _event_scales(inverse, valid_regular, gradient, action_type=None):
        """Detached, within-impression scales; no labels or other rows used."""
        if gradient in ('action_mean_tangent', 'action_reassigned_tangent', 'action_adjacent_tangent'):
            if action_type is None:
                raise ValueError('behavior-stratified scales require history_type')
            base_mode = {'action_mean_tangent': 'history_mean_tangent', 'action_reassigned_tangent': 'history_reassigned_tangent', 'action_adjacent_tangent': 'history_adjacent_tangent'}[gradient]
            result = inverse
            for action in range(1, 5):
                selected = valid_regular & (action_type == action)
                assigned = Model._event_scales(inverse, selected, base_mode)
                result = torch.where(selected, assigned, result)
            return result
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            donor = Model._rotation_donors(inverse, valid_regular, gradient)
            return inverse.gather(-1, donor)
        if gradient == 'full':
            return inverse
        if gradient == 'constant_tangent':
            return torch.ones_like(inverse) * 0.2
        count = valid_regular.sum(-1, keepdim=True)
        values = torch.where(valid_regular, inverse, torch.zeros_like(inverse))
        if gradient == 'history_mean_tangent':
            mean = values.sum(-1, keepdim=True) / count.clamp_min(1)
            return torch.where(valid_regular, mean, inverse)
        if gradient in ('history_reassigned_tangent', 'history_adjacent_tangent'):
            rank = valid_regular.long().cumsum(-1) - 1
            packed = torch.zeros_like(inverse).scatter_add(-1, rank.clamp_min(0), values)
            shift = 1 if gradient == 'history_adjacent_tangent' else count // 2
            donor = (rank + shift).remainder(count.clamp_min(1))
            rotated = packed.gather(-1, donor)
            return torch.where(valid_regular, rotated, inverse)
        raise ValueError('unknown normalization_gradient')

    @staticmethod
    def _normalize_product(product, gradient='full', mask=None, action_type=None):
        squared = product.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-06).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        regular = squared >= 1e-06
        valid_regular = regular.squeeze(-1)
        if mask is not None:
            valid_regular = valid_regular & mask
        inverse = denominator.squeeze(-1).reciprocal().detach()
        target = Model._event_scales(inverse, valid_regular, gradient, action_type)
        coefficient = torch.where(regular, denominator * target.unsqueeze(-1), torch.ones_like(denominator)).detach()
        surrogate = unit * coefficient
        return unit.detach() + (surrogate - surrogate.detach())

    @staticmethod
    def _route_product(qc, qb, category, brand, mask, query_mode, history_mode):
        allowed = ('full', 'scale_half_tangent')
        if query_mode not in allowed or history_mode not in allowed:
            raise ValueError('Only the precommitted own/scale-half role rules are supported')
        product = qc * category + qb * brand
        if query_mode == history_mode:
            return Model._normalize_product(product, query_mode, mask)
        if not torch.is_grad_enabled():
            return Model._normalize_product(product, 'full', mask)
        query_raw = qc * category.detach() + qb * brand.detach()
        history_raw = qc.detach() * category + qb.detach() * brand
        query_unit = Model._normalize_product(query_raw, query_mode, mask)
        history_unit = Model._normalize_product(history_raw, history_mode, mask)
        base = Model._normalize_product(product, 'full', mask).detach()
        return base + (query_unit - query_unit.detach()) + (history_unit - history_unit.detach())

    def _product(self, fields, components, pairing, normalization, mask=None, action_type=None):
        if pairing != 'aligned' or normalization != 'unit':
            raise ValueError('This study fixes aligned unit products')
        (category, brand, _, _) = components
        return self._route_product(fields[self.category][:, None, :], fields[self.brand][:, None, :], category, brand, mask, self.config['product_query_gradient'], self.config['product_history_gradient'])

    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def forward(self, batch):
        (fields, features, history, components) = self._encode(batch)
        product = self._product(fields, components, self.config['product_pairing'], self.config['product_normalization'], batch['history_mask'], batch['history_type'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        return self._logits(features, summary)

    @staticmethod
    def _vjp_audit():
        """Tiny CPU float64 autograd check, called by host, never an update.

        Candidate-authored product-channel check, not an independent audit
        of training execution, optimizer state or whole-model gradients.
        """
        modes = ('full', 'constant_tangent', 'history_mean_tangent', 'history_reassigned_tangent', 'action_mean_tangent', 'action_reassigned_tangent', 'history_adjacent_tangent', 'action_adjacent_tangent', 'scale_adjacent_tangent', 'scale_half_tangent')
        data = [[[1.0, 2.0, 0.0, 0.0]] * 6, [[1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, -1.0, 3.0, 0.0], [0.0002, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]], [[0.001, 0.0, 0.0, 0.0], [0.0002, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0], [3.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [3.0, 2.0, 1.0, 0.0], [4.0, 1.0, 0.0, 2.0], [1.0, -1.0, 2.0, 0.0], [2.0, 1.0, 0.0, 0.0]]]
        data.extend([[[1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [3.0, -1.0, 1.0, 0.0], [4.0, 1.0, 0.0, 2.0], [5.0, 0.0, 1.0, 0.0], [6.0, -1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0], [4.0, 1.0, 0.0, 2.0], [5.0, -1.0, 1.0, 0.0], [6.0, 1.0, 0.0, 1.0]]])
        mask_rows = [[False] * 6, [True, False, False, False, False, False], [True, False, True, True, True, False], [True] * 6, [True, True, True, False, True, True]]
        action_rows = [[1, 2, 3, 4, 1, 2], [2, 1, 1, 1, 1, 1], [1, 2, 1, 3, 4, 2], [1, 1, 2, 2, 3, 4], [1, 2, 1, 2, 1, 2]]
        mask_rows.extend([[True] * 6, [True] * 6])
        action_rows.extend([[1] * 6, [0, 5, 2, 2, 2, 2]])
        actions = torch.tensor(action_rows, dtype=torch.long, device='cpu')
        mask = torch.tensor(mask_rows, dtype=torch.bool, device='cpu')
        base = torch.tensor(data, dtype=torch.float64, device='cpu')
        r2 = base.square().sum(-1)
        radius = r2.clamp_min(1e-06).sqrt()
        inverse = radius.reciprocal()
        unit = base / radius.unsqueeze(-1)
        regular = r2 >= 1e-06
        eligible = regular & mask
        probe1 = torch.arange(1, base.numel() + 1, dtype=base.dtype, device='cpu').reshape_as(base) / base.numel()
        probe2 = base + torch.tensor([0.2, -0.3, 0.5, -0.7], dtype=base.dtype, device='cpu')
        result = {'audit_synthetic_boundary_events': float((r2 == 1e-06).sum()), 'audit_synthetic_floored_valid_events': float(((r2 < 1e-06) & mask).sum())}
        forward_error = 0.0
        for mode in modes:
            scales = inverse.clone()
            if mode == 'constant_tangent':
                scales.fill_(0.2)
            elif mode in ('history_mean_tangent', 'history_reassigned_tangent', 'action_mean_tangent', 'action_reassigned_tangent', 'history_adjacent_tangent', 'action_adjacent_tangent', 'scale_adjacent_tangent', 'scale_half_tangent'):
                groups = range(1, 5) if mode.startswith('action_') else (None,)
                for row in range(len(data)):
                    for group in groups:
                        ids = [j for j in range(6) if bool(eligible[row, j]) and (group is None or action_rows[row][j] == group)]
                        if mode.startswith('scale_'):
                            ids = sorted(ids, key=lambda j: float(inverse[row, j]))
                        for (position, j) in enumerate(ids):
                            if mode.endswith('mean_tangent'):
                                scales[row, j] = inverse[row, ids].mean()
                            else:
                                shift = 1 if mode.endswith('adjacent_tangent') else len(ids) // 2
                                donor = ids[(position + shift) % len(ids)]
                                scales[row, j] = inverse[row, donor]
            max_error = 0.0
            with torch.enable_grad():
                for probe in (probe1, probe2):
                    x = base.clone().requires_grad_(True)
                    out = Model._normalize_product(x, mode, mask, actions)
                    (actual,) = torch.autograd.grad(out, x, grad_outputs=probe)
                    tangent = probe - unit * (unit * probe).sum(-1, keepdim=True)
                    expected = torch.where(regular.unsqueeze(-1), tangent * scales.unsqueeze(-1), probe * inverse.unsqueeze(-1))
                    error = ((actual - expected).abs() / (1 + expected.abs())).max()
                    max_error = max(max_error, float(error.detach()))
                    forward_error = max(forward_error, float((out.detach() - unit).abs().max()))
            result['audit_vjp_scaled_error_' + mode] = max_error
        result['audit_synthetic_forward_abs_max'] = forward_error
        result['audit_vjp_probe_count_per_mode'] = 2.0
        return result

    @staticmethod
    def _route_vjp_audit():
        """Independent scalar-donor lists and analytic four-operand VJP checks.

        Float64 CPU data, no RNG. Includes empty, singleton, holes, ties,
        below-floor, zero, and ordinary events. Checks both mixed cells and
        exact inherited endpoints. Gradcheck is inappropriate for the intended
        nonconservative surrogate; check its declared VJP instead.
        """
        base = torch.tensor([[[1.0, 2.0, 0.0, 0.0]] * 6, [[1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, -1.0, 3.0, 0.0], [0.0002, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]], [[0.001, 0.0, 0.0, 0.0], [0.0002, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0], [3.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [3.0, 2.0, 1.0, 0.0], [4.0, 1.0, 0.0, 2.0], [1.0, -1.0, 2.0, 0.0], [2.0, 1.0, 0.0, 0.0]]], dtype=torch.float64, device='cpu')
        mask = torch.tensor([[False] * 6, [True, False, False, False, False, False], [True, False, True, True, True, False], [True] * 6, [True, True, True, False, True, True]], device='cpu')
        qc0 = torch.ones(base.shape[0], 1, base.shape[-1], dtype=base.dtype, device='cpu')
        qb0 = torch.ones_like(qc0)
        (hc0, hb0) = (base * 0.75, base * 0.25)
        raw = qc0 * hc0 + qb0 * hb0
        r2 = raw.square().sum(-1)
        denominator = r2.clamp_min(1e-06).sqrt()
        own = denominator.reciprocal()
        unit = raw / denominator.unsqueeze(-1)
        eligible = mask & (r2 >= 1e-06)
        donated = own.clone()
        for row in range(raw.shape[0]):
            ids = [j for j in range(raw.shape[1]) if bool(eligible[row, j])]
            ids.sort(key=lambda j: float(own[row, j]))
            for (rank, j) in enumerate(ids):
                donated[row, j] = own[row, ids[(rank + len(ids) // 2) % len(ids)]]
        probe = torch.arange(1, raw.numel() + 1, dtype=raw.dtype, device='cpu').reshape_as(raw) / raw.numel()
        probes = (probe, raw + raw.new_tensor([0.2, -0.3, 0.5, -0.7]))
        result = {}
        forward_error = 0.0
        cells = [('own_own', 'full', 'full'), ('half_own', 'scale_half_tangent', 'full'), ('own_half', 'full', 'scale_half_tangent'), ('half_half', 'scale_half_tangent', 'scale_half_tangent')]
        with torch.enable_grad():
            for (name, qmode, hmode) in cells:
                error = 0.0
                for v in probes:
                    operands = tuple((x.clone().requires_grad_(True) for x in (qc0, qb0, hc0, hb0)))
                    out = Model._route_product(*operands, mask, qmode, hmode)
                    actual = torch.autograd.grad(out, operands, grad_outputs=v)
                    tangent = v - unit * (unit * v).sum(-1, keepdim=True)

                    def product_vjp(mode):
                        scales = own if mode == 'full' else donated
                        return torch.where((r2 >= 1e-06).unsqueeze(-1), tangent * scales.unsqueeze(-1), v * own.unsqueeze(-1))
                    (gq, gh) = (product_vjp(qmode), product_vjp(hmode))
                    expected = ((gq * hc0).sum(1, keepdim=True), (gq * hb0).sum(1, keepdim=True), gh * qc0, gh * qb0)
                    for (a, e) in zip(actual, expected):
                        error = max(error, float(((a - e).abs() / (1 + e.abs())).max().detach()))
                    forward_error = max(forward_error, float((out.detach() - unit).abs().max()))
                result['role_vjp_scaled_error_' + name] = error
        result['role_synthetic_forward_abs_max'] = forward_error
        result['role_synthetic_probes_per_cell'] = float(len(probes))
        result['role_synthetic_eligible_events'] = float(eligible.sum())
        if max((result['role_vjp_scaled_error_' + c[0]] for c in cells)) > 1e-09 or forward_error != 0:
            raise RuntimeError('Role-routing synthetic contract failed; suspend attribution')
        return result

    def diagnostics(self, batch):
        """62 scalar keys: first-512 forward/scales, first-32 VJPs, synthetic.

        Candidate-authored ordered diagnostics, not training or population
        measurements. All counterfactual rules share each probe checkpoint.
        """
        (fields, features, history, components) = self._encode(batch)
        mask = batch['history_mask']
        (category, brand, _, _) = components
        (qc, qb) = (fields[self.category][:, None, :], fields[self.brand][:, None, :])
        raw = qc * category + qb * brand
        r2 = raw.square().sum(-1)
        own = r2.clamp_min(1e-06).sqrt().reciprocal().detach()
        eligible = mask & (r2 >= 1e-06)
        donor = self._event_scales(own, eligible, 'scale_half_tangent')
        full = self._normalize_product(raw, 'full', mask)
        common_logits = None
        forward_error = raw.new_zeros(())
        product_error = raw.new_zeros(())
        for (qmode, hmode) in [('full', 'full'), ('scale_half_tangent', 'full'), ('full', 'scale_half_tangent'), ('scale_half_tangent', 'scale_half_tangent')]:
            product = self._route_product(qc, qb, category, brand, mask, qmode, hmode)
            weights = self._weights(self._scores_from_product(fields, history, product), mask)
            logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
            product_error = torch.maximum(product_error, (product - full).abs().max())
            if common_logits is None:
                common_logits = logits
            forward_error = torch.maximum(forward_error, (logits - common_logits).abs().max())
        active = self._product(fields, components, 'aligned', 'unit', mask)
        weights = self._weights(self._scores_from_product(fields, history, active), mask)
        summary = (weights.unsqueeze(-1) * history).sum(1)

        def sums(x):
            return torch.where(eligible, x, torch.zeros_like(x)).sum(-1)
        result = {'validation_samples': float(mask.shape[0]), 'validation_valid_events': float(mask.sum()), 'validation_eligible_events': float(eligible.sum()), 'validation_empty_rows': float((~mask.any(-1)).sum()), 'validation_floored_valid_events': float((mask & (r2 < 1e-06)).sum()), 'validation_boundary_valid_events': float((mask & (r2 == 1e-06)).sum()), 'validation_common_four_cell_product_abs_max': float(product_error), 'validation_common_four_cell_logit_abs_max': float(forward_error), 'validation_forward_helper_abs_max': float((self.forward(batch) - self._logits(features, summary)).abs().max()), 'validation_masked_padding_mass_max': float((weights * (~mask).float()).sum(-1).max()), 'validation_weight_sum_error_max': float((weights.sum(-1) - mask.any(-1).float()).abs().max()), 'validation_empty_summary_l2_max': float((summary.norm(dim=-1) * ~mask.any(-1)).max()), 'validation_donor_sum_relative_error_max': float(((sums(donor) - sums(own)).abs() / (1 + sums(own).abs())).max()), 'validation_donor_square_sum_relative_error_max': float(((sums(donor.square()) - sums(own.square())).abs() / (1 + sums(own.square()).abs())).max())}
        result.update(self._vjp_audit())
        result.update(self._route_vjp_audit())
        return result
