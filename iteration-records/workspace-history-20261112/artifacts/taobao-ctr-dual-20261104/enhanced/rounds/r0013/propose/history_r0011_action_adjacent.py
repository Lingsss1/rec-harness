"""Distance controls for within-impression backward-scale reassignment.

Inherited forward: p=qc*hc+qb*hb; u=p/sqrt(max(sum(p*p),1e-6)).
The new history_adjacent_tangent/action_adjacent_tangent modes rotate valid
regular scales 1/r by ONE eligible position, within the history or same action.
Existing reassigned modes rotate by floor(n/2). Both rotations preserve each
selected group's scale multiset and have no fixed index for n>=2. They coincide
for n<=3; wraparound is not temporally local. Adjacent supplied positions need
not be adjacent in elapsed time and same-action gaps can be large.
Only the detached product-channel backward scale changes: J=s_donor*(I-u*u^T)
above/on the floor; below the floor all retain 1000*I. Masked/floored positions
are not donors; empty/singleton groups and unknown action codes retain own
scales. No cross-row input, parameter, state, RNG, optimizer or loss is added.

This contrasts reassignment rules, not semantic relevance or general necessity.
DIN v4 section 4.3 is the attention precedent, but this field-adapted softmax
reference differs from published DIN. External normalization/gradient priority
is unverified; no novelty or seed-robustness claim is made. Diagnostics are
candidate-authored first-512 descriptions and small synthetic VJP checks,
not representative training gradients or independent audit evidence.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'action_adjacent_tangent'}


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
                 'action_adjacent_tangent')
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
                          'history_adjacent_tangent', 'action_adjacent_tangent'):
                # Independent reference: Python lists enumerate each donor set.
                groups = range(1, 5) if mode.startswith('action_') else (None,)
                for row in range(len(data)):
                    for group in groups:
                        ids = [j for j in range(6) if bool(eligible[row, j]) and
                               (group is None or action_rows[row][j] == group)]
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

    def diagnostics(self, batch):
        """Ordered first-512 descriptions; no labels or population claims."""
        fields, features, history, components = self._encode(batch)
        mask, actions = batch['history_mask'], batch['history_type']
        raw = self._product(fields, components, 'aligned', 'raw', mask, actions)
        r2 = raw.square().sum(-1)
        own = r2.clamp_min(1e-6).sqrt().reciprocal().detach()
        eligible = mask & (r2 >= 1e-6)
        mode = self.config['normalization_gradient']
        active_scale = self._event_scales(own, eligible, mode, actions)
        action_grouping = mode.startswith('action_')
        half_mode = 'action_reassigned_tangent' if action_grouping else 'history_reassigned_tangent'
        mean_mode = 'action_mean_tangent' if action_grouping else 'history_mean_tangent'
        half_scale = self._event_scales(own, eligible, half_mode, actions)
        group_mean = self._event_scales(own, eligible, mean_mode, actions)
        active = self._normalize_product(raw, mode, mask, actions)
        full = self._normalize_product(raw, 'full', mask, actions)
        weights = self._weights(self._scores_from_product(fields, history, active), mask)
        summary = (weights.unsqueeze(-1) * history).sum(1)
        logits = self._logits(features, summary)
        full_weights = self._weights(self._scores_from_product(fields, history, full), mask)
        full_logits = self._logits(features, (full_weights.unsqueeze(-1) * history).sum(1))

        def mean(value, selected):
            return torch.where(selected, value, torch.zeros_like(value)).sum() / selected.sum().clamp_min(1)

        centered = own - group_mean
        variance = mean(centered.square(), eligible)
        result = {
            'diagnostic_samples': mask.new_tensor(mask.shape[0], dtype=torch.long),
            'valid_events': mask.sum(),
            'empty_samples': (~mask.any(-1)).sum(),
            'eligible_events': eligible.sum(),
            'floored_valid_events': (mask & (r2 < 1e-6)).sum(),
            'boundary_valid_events': (mask & (r2 == 1e-6)).sum(),
            'action_grouping': own.new_tensor(float(action_grouping)),
            'own_scale_mean': mean(own, eligible),
            'active_scale_mean': mean(active_scale, eligible),
            'within_group_own_scale_std': variance.sqrt(),
            'active_own_centered_correlation': mean(centered * (active_scale - group_mean), eligible) / variance.clamp_min(1e-12),
            'half_own_centered_correlation': mean(centered * (half_scale - group_mean), eligible) / variance.clamp_min(1e-12),
            'active_own_scale_abs_difference_mean': mean((active_scale - own).abs(), eligible),
            'half_own_scale_abs_difference_mean': mean((half_scale - own).abs(), eligible),
            'active_half_scale_abs_difference_mean': mean((active_scale - half_scale).abs(), eligible),
            'active_half_scale_changed_fraction': mean((active_scale != half_scale).float(), eligible),
            'same_state_full_active_product_abs_max': (full - active).abs().max(),
            'same_state_full_active_logit_abs_max': (full_logits - logits).abs().max(),
            'forward_helper_logit_abs_max': (self.forward(batch) - logits).abs().max(),
            'masked_padding_mass_max': (weights * (~mask).float()).sum(-1).max(),
            'weight_sum_error_max': (weights.sum(-1) - mask.any(-1).float()).abs().max(),
            'empty_summary_l2_max': (summary.square().sum(-1).sqrt() * (~mask.any(-1))).max(),
        }
        # Actual donor position and age displacement, including wraparound.
        positions = torch.arange(own.shape[-1], device=own.device, dtype=own.dtype)[None, :].expand_as(own)
        for name, values in (('position', positions), ('age_seconds', batch['history_age_seconds'])):
            active_donor = self._event_scales(values, eligible, mode, actions)
            half_donor = self._event_scales(values, eligible, half_mode, actions)
            result['active_donor_' + name + '_abs_gap_mean'] = mean((active_donor - values).abs(), eligible)
            result['half_donor_' + name + '_abs_gap_mean'] = mean((half_donor - values).abs(), eligible)
        donor_actions = self._event_scales(actions.float(), eligible, mode, actions)
        result['active_donor_action_mismatch_fraction'] = mean((donor_actions != actions).float(), eligible)
        errors = []
        singleton = own.sum() * 0
        small = own.sum() * 0
        large = own.sum() * 0
        groups = range(1, 5) if action_grouping else (None,)
        for group in groups:
            selected = eligible if group is None else eligible & (actions == group)
            n = selected.sum(-1)
            singleton = singleton + (n == 1).sum()
            small = small + torch.where((n >= 2) & (n <= 3), n, torch.zeros_like(n)).sum()
            large = large + torch.where(n >= 4, n, torch.zeros_like(n)).sum()
            def sums(value):
                return torch.where(selected, value, torch.zeros_like(value)).sum(-1)
            own_sum, own_square_sum = sums(own), sums(own.square())
            errors.append(torch.stack([
                ((sums(active_scale) - own_sum).abs() / (1 + own_sum.abs())).max(),
                ((sums(active_scale.square()) - own_square_sum).abs() / (1 + own_square_sum.abs())).max(),
            ]))
        conservation = torch.stack(errors).max(0).values
        result['active_group_scale_sum_relative_error_max'] = conservation[0]
        result['active_group_squared_scale_sum_relative_error_max'] = conservation[1]
        result['singleton_group_events'] = singleton
        result['group_size_2_3_events'] = small
        result['group_size_ge4_events'] = large
        for action in range(1, 5):
            result['action_' + str(action) + '_events'] = (eligible & (actions == action)).sum()
        result = {name: float(value.detach().item()) for name, value in result.items()}
        result.update(self._vjp_audit())
        return result
