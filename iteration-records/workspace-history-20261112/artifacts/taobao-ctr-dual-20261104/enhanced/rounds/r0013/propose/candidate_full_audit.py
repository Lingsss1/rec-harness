"""Diagnostic descendant of r0012-scale_half, with unchanged training methods.

A fixed first-32 validation logit-mean probe compares six inherited backward
rules at one checkpoint, including total category/brand embedding gradients.
This is label-free local prediction sensitivity, not training-loss gradients,
Adam state, updates, causal mediation, population evidence or a new model.
One scratch seed-42 replay is necessary because no checkpoint is exported.
All graph construction and VJPs are confined to diagnostics; no parameter,
buffer, optimizer, RNG, loss, forward path or gradient accumulation is added.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'full'}


class Model(nn.Module):
    def __init__(self,meta,config=None):
        super().__init__()
        config={**CONFIG,**(config or {})}
        self.config=config
        self.kind=config['kind']
        self.category=meta['category_field_index'];self.brand=meta['brand_field_index']
        dim=config['embedding_dim']
        self.embeddings=nn.ModuleList([nn.Embedding(size,dim,padding_idx=0) for size in meta['buckets']])
        self.action=nn.Embedding(5,dim,padding_idx=0)
        self.recency=nn.Embedding(33,dim,padding_idx=0)
        self.attention=nn.Sequential(nn.Linear(dim*4,64),nn.PReLU(),nn.Linear(64,1))
        layers=[]
        width=len(meta['fields'])*dim+(dim if self.kind!='static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width,hidden),nn.ReLU()]);width=hidden
        layers.append(nn.Linear(width,1))
        self.output=nn.Sequential(*layers)


    def _encode(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        features = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        history = category + brand
        action = self.action(batch['history_type'])
        history = history + action
        recency = self.recency(ages)
        # Keep the parent's operation order as well as its association.
        history = history + recency
        return fields, features, history, (category, brand, action, recency)


    @staticmethod
    def _weights(scores, mask):
        # Identical frozen softmax arithmetic, including all-masked rows.
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)


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
        positions = torch.arange(inverse.shape[-1], device=inverse.device,
                                 dtype=torch.long)[None, :].expand_as(inverse)
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
        if gradient in ('action_mean_tangent', 'action_reassigned_tangent',
                        'action_adjacent_tangent'):
            if action_type is None:
                raise ValueError('behavior-stratified scales require history_type')
            base_mode = {'action_mean_tangent': 'history_mean_tangent',
                         'action_reassigned_tangent': 'history_reassigned_tangent',
                         'action_adjacent_tangent': 'history_adjacent_tangent'}[gradient]
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
            # Pack valid regular events in supplied temporal order. A rotation
            # by floor(n/2) or one has no fixed index for n>=2; n=0/1 is unchanged.
            # Padding and floored events contribute zero and are never donors.
            rank = valid_regular.long().cumsum(-1) - 1
            packed = torch.zeros_like(inverse).scatter_add(
                -1, rank.clamp_min(0), values)
            shift = 1 if gradient == 'history_adjacent_tangent' else count // 2
            donor = (rank + shift).remainder(count.clamp_min(1))
            rotated = packed.gather(-1, donor)
            return torch.where(valid_regular, rotated, inverse)
        raise ValueError('unknown normalization_gradient')

    @staticmethod
    def _normalize_product(product, gradient='full', mask=None, action_type=None):
        squared = product.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-6).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        regular = squared >= 1e-6
        valid_regular = regular.squeeze(-1)
        if mask is not None:
            valid_regular = valid_regular & mask
        inverse = denominator.squeeze(-1).reciprocal().detach()
        target = Model._event_scales(inverse, valid_regular, gradient, action_type)
        # On valid regular events J=target*(I-u*u^T). On floored
        # events retain J=1000*I. Row replacements retain full J on
        # masked events. All cross-event scale derivatives are stopped.
        coefficient = torch.where(regular, denominator * target.unsqueeze(-1),
                                  torch.ones_like(denominator)).detach()
        surrogate = unit * coefficient
        return unit.detach() + (surrogate - surrogate.detach())


    def _product(self, fields, components, pairing, normalization, mask=None, action_type=None):
        category, brand, _, _ = components
        qc = fields[self.category][:, None, :]
        qb = fields[self.brand][:, None, :]
        if pairing == 'aligned':
            product = qc * category + qb * brand
        elif pairing == 'crossed':
            product = qc * brand + qb * category
        else:
            raise ValueError('product_pairing must be aligned or crossed')
        if normalization == 'unit':
            return self._normalize_product(
                product, self.config.get('normalization_gradient', 'full'), mask, action_type)
        if normalization != 'raw':
            raise ValueError('product_normalization must be raw or unit')
        return product

    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def forward(self, batch):
        fields, features, history, components = self._encode(batch)
        product = self._product(fields, components,
                                self.config['product_pairing'],
                                self.config['product_normalization'], batch['history_mask'],
                                batch['history_type'])
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
        modes = ('full', 'constant_tangent', 'history_mean_tangent',
                 'history_reassigned_tangent', 'action_mean_tangent',
                 'action_reassigned_tangent', 'history_adjacent_tangent',
                 'action_adjacent_tangent', 'scale_adjacent_tangent',
                 'scale_half_tangent')
        # Empty, singleton, holed two-event, boundary/floor, and longer rows.
        data = [
            [[1., 2., 0., 0.]] * 6,
            [[1., 0., 0., 0.], [2., 1., 0., 0.], [0., 0., 0., 0.],
             [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
            [[1., 2., 0., 0.], [0., 0., 0., 0.], [2., -1., 3., 0.],
             [0.0002, 0., 0., 0.], [0., 0., 0., 0.], [4., 1., 0., 0.]],
            [[0.001, 0., 0., 0.], [0.0002, 0., 0., 0.], [0., 0., 0., 0.],
             [1., 2., 3., 0.], [3., 0., 1., 0.], [2., 0., 0., 0.]],
            [[1., 1., 0., 0.], [2., -1., 0., 0.], [3., 2., 1., 0.],
             [4., 1., 0., 2.], [1., -1., 2., 0.], [2., 1., 0., 0.]],
        ]
        data.extend([
            [[1., 0., 0., 0.], [2., 1., 0., 0.], [3., -1., 1., 0.],
             [4., 1., 0., 2.], [5., 0., 1., 0.], [6., -1., 0., 1.]],
            [[1., 1., 0., 0.], [2., -1., 0., 0.], [3., 0., 1., 0.],
             [4., 1., 0., 2.], [5., -1., 1., 0.], [6., 1., 0., 1.]],
        ])
        mask_rows = [[False] * 6, [True, False, False, False, False, False],
                     [True, False, True, True, True, False], [True] * 6,
                     [True, True, True, False, True, True]]
        action_rows = [[1, 2, 3, 4, 1, 2], [2, 1, 1, 1, 1, 1],
                       [1, 2, 1, 3, 4, 2], [1, 1, 2, 2, 3, 4],
                       [1, 2, 1, 2, 1, 2]]
        mask_rows.extend([[True] * 6, [True] * 6])
        action_rows.extend([[1] * 6, [0, 5, 2, 2, 2, 2]])
        actions = torch.tensor(action_rows, dtype=torch.long, device='cpu')
        mask = torch.tensor(mask_rows, dtype=torch.bool, device='cpu')
        base = torch.tensor(data, dtype=torch.float64, device='cpu')
        r2 = base.square().sum(-1)
        radius = r2.clamp_min(1e-6).sqrt()
        inverse = radius.reciprocal()
        unit = base / radius.unsqueeze(-1)
        regular = r2 >= 1e-6
        eligible = regular & mask
        probe1 = torch.arange(1, base.numel() + 1, dtype=base.dtype, device='cpu').reshape_as(base) / base.numel()
        probe2 = base + torch.tensor([0.2, -0.3, 0.5, -0.7], dtype=base.dtype, device='cpu')
        result = {
            'audit_synthetic_boundary_events': float((r2 == 1e-6).sum()),
            'audit_synthetic_floored_valid_events': float(((r2 < 1e-6) & mask).sum()),
        }
        forward_error = 0.0
        # Reference selection uses explicit small Python index lists, separate
        # from the vectorized packing implementation used in the real forward.
        for mode in modes:
            scales = inverse.clone()
            if mode == 'constant_tangent':
                scales.fill_(0.2)
            elif mode in ('history_mean_tangent', 'history_reassigned_tangent',
                          'action_mean_tangent', 'action_reassigned_tangent',
                          'history_adjacent_tangent', 'action_adjacent_tangent',
                          'scale_adjacent_tangent', 'scale_half_tangent'):
                # Independent reference: Python lists enumerate each donor set.
                groups = range(1, 5) if mode.startswith('action_') else (None,)
                for row in range(len(data)):
                    for group in groups:
                        ids = [j for j in range(6) if bool(eligible[row, j]) and
                               (group is None or action_rows[row][j] == group)]
                        if mode.startswith('scale_'):
                            # Python's stable list sort is independent of the
                            # vectorized argsort/scatter donor implementation.
                            ids = sorted(ids, key=lambda j: float(inverse[row, j]))
                        for position, j in enumerate(ids):
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
                    actual, = torch.autograd.grad(out, x, grad_outputs=probe)
                    tangent = probe - unit * (unit * probe).sum(-1, keepdim=True)
                    expected = torch.where(regular.unsqueeze(-1),
                                           tangent * scales.unsqueeze(-1),
                                           probe * inverse.unsqueeze(-1))
                    error = ((actual - expected).abs() / (1 + expected.abs())).max()
                    max_error = max(max_error, float(error.detach()))
                    forward_error = max(forward_error, float((out.detach() - unit).abs().max()))
            result['audit_vjp_scaled_error_' + mode] = max_error
        result['audit_synthetic_forward_abs_max'] = forward_error
        result['audit_vjp_probe_count_per_mode'] = 2.0
        return result

    def _common_state_probe(self, batch):
        """Six common-state VJPs; no labels, updates or writes to .grad.

        Probe: mean prediction logit on the first min(32,B) supplied rows.
        Shared gradients include ALL paths to the two embedding tables;
        product gradients measure only the product channel. Ratios use a
        1e-30 norm floor, with reference norms reported for interpretation.
        This checkpoint probe cannot reconstruct accumulated Adam moments.
        """
        sample = {key: batch[key][:32] for key in ('static', 'history_category', 'history_brand',
                'history_type', 'history_age_seconds', 'history_mask')}
        modes = (('full', 'full'), ('fixed', 'constant_tangent'),
                 ('history_adjacent', 'history_adjacent_tangent'),
                 ('history_half', 'history_reassigned_tangent'),
                 ('scale_adjacent', 'scale_adjacent_tangent'),
                 ('scale_half', 'scale_half_tangent'))
        # Avoid duplicate autograd inputs even in a degenerate field mapping.
        params = tuple(dict.fromkeys((self.embeddings[self.category].weight,
                                      self.embeddings[self.brand].weight)))

        def dot(left, right):
            return sum((a * b).sum(dtype=torch.float64)
                       for a, b in zip(left, right))

        result = {'probe_samples': float(sample['static'].shape[0])}
        max_forward_error = 0.0
        with torch.enable_grad():
            fields, features, history, components = self._encode(sample)
            mask, actions = sample['history_mask'], sample['history_type']
            raw = self._product(fields, components, 'aligned', 'raw', mask, actions)
            for index, (name, mode) in enumerate(modes):
                product = self._normalize_product(raw, mode, mask, actions)
                scores = self._scores_from_product(fields, history, product)
                weights = self._weights(scores, mask)
                logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
                grads = torch.autograd.grad(logits.mean(), (raw,) + params,
                                             retain_graph=index < len(modes) - 1,
                                             create_graph=False)
                grads = tuple(g.detach() for g in grads)
                groups = ((grads[0],), grads[1:])
                if index == 0:
                    reference_logits = logits.detach()
                    references = groups
                    ref_norms = tuple(dot(g, g).clamp_min(0).sqrt() for g in groups)
                    result['probe_full_product_gradient_norm'] = float(ref_norms[0])
                    result['probe_full_embedding_gradient_norm'] = float(ref_norms[1])
                else:
                    max_forward_error = max(max_forward_error,
                        float((logits.detach() - reference_logits).abs().max()))
                    for label, group, ref, ref_norm in zip(
                            ('product', 'embedding'), groups, references, ref_norms):
                        norm = dot(group, group).clamp_min(0).sqrt()
                        difference = tuple(g - r for g, r in zip(group, ref))
                        diff_norm = dot(difference, difference).clamp_min(0).sqrt()
                        lead = 'probe_' + name + '_' + label
                        result[lead + '_norm_ratio'] = float(norm / ref_norm.clamp_min(1e-30))
                        result[lead + '_relative_difference'] = float(diff_norm / ref_norm.clamp_min(1e-30))
                        result[lead + '_cosine'] = float(dot(group, ref) /
                            (norm * ref_norm).clamp_min(1e-30))
        result['probe_common_forward_abs_max'] = max_forward_error
        return result

    def diagnostics(self, batch):
        """62 scalar keys: first-512 forward/scales, first-32 VJPs, synthetic.

        Candidate-authored ordered diagnostics, not training or population
        measurements. All counterfactual rules share each probe checkpoint.
        """
        fields, features, history, components = self._encode(batch)
        mask, actions = batch['history_mask'], batch['history_type']
        raw = self._product(fields, components, 'aligned', 'raw', mask, actions)
        r2 = raw.square().sum(-1)
        own = r2.clamp_min(1e-6).sqrt().reciprocal().detach()
        eligible = mask & (r2 >= 1e-6)
        active = self._normalize_product(raw, self.config['normalization_gradient'], mask, actions)
        full = self._normalize_product(raw, 'full', mask, actions)
        weights = self._weights(self._scores_from_product(fields, history, active), mask)
        logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
        fw = self._weights(self._scores_from_product(fields, history, full), mask)
        full_logits = self._logits(features, (fw.unsqueeze(-1) * history).sum(1))

        def sums(value):
            return torch.where(eligible, value, torch.zeros_like(value)).sum(-1)

        def mean(value):
            return sums(value).sum() / eligible.sum().clamp_min(1)

        group_mean = sums(own) / eligible.sum(-1).clamp_min(1)
        centered = own - group_mean.unsqueeze(-1)
        variance = mean(centered.square())
        result = {
            'diagnostic_samples': own.new_tensor(mask.shape[0]),
            'valid_events': mask.sum(),
            'eligible_events': eligible.sum(),
            'empty_samples': (~mask.any(-1)).sum(),
            'floored_valid_events': (mask & (r2 < 1e-6)).sum(),
            'boundary_valid_events': (mask & (r2 == 1e-6)).sum(),
            'within_history_own_scale_std': variance.sqrt(),
            'same_state_forward_logit_abs_max': (full_logits - logits).abs().max(),
        }
        sum_error, square_error = 0.0, 0.0
        for name, rule in (('history_adjacent', 'history_adjacent_tangent'),
                           ('history_half', 'history_reassigned_tangent'),
                           ('scale_adjacent', 'scale_adjacent_tangent'),
                           ('scale_half', 'scale_half_tangent')):
            assigned = self._event_scales(own, eligible, rule, actions)
            sum_error = max(sum_error, float(((sums(assigned) - sums(own)).abs() /
                                             (1 + sums(own).abs())).max()))
            square_error = max(square_error, float(((sums(assigned.square()) - sums(own.square())).abs() /
                                                    (1 + sums(own.square()).abs())).max()))
            result[name + '_own_centered_correlation'] = mean(
                centered * (assigned - group_mean.unsqueeze(-1))) / variance.clamp_min(1e-12)
        result = {name: float(value.detach().item()) for name, value in result.items()}
        result['rotation_scale_sum_relative_error_max'] = sum_error
        result['rotation_squared_scale_sum_relative_error_max'] = square_error
        result.update(self._common_state_probe(batch))
        result.update(self._vjp_audit())
        return result
