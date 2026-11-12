"""Inference-only elimination of backward donor/surrogate work.

Common standalone implementation for a fixed flag-off/flag-on comparison.
Parent: r0030-aligned_half, source SHA256
4b4c46af768722616e064d6793fd540d3e9adc5810294f2f7f62619f0d79c7ca.
The guard uses autograd state, never model.training: eval-time derivatives
retain the inherited half-rank surrogate. All parameter creation, training
operations, mean BCE and host recipe remain inherited. No optimizer or seed
changes. Inference shortcut is an implementation optimization, not a new
prediction or gradient mechanism. Host-only diagnostics are prospective;
Torch is unavailable locally. Both modules differ only in literal CONFIG.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'none', 'inference_shortcut': True}


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
        if self.category == self.brand:
            raise ValueError('This registered study requires distinct category/brand tables')
        if config['kind'] != 'din' or config['product_pairing'] not in ('aligned', 'crossed') or config['product_normalization'] != 'unit':
            raise ValueError('This study fixes target-aware unit products and declared pairing')
        if config['normalization_gradient'] not in ('full', 'scale_half_tangent'):
            raise ValueError('Only the fixed own/half normalization rules are admitted')
        if config['row_intervention'] not in ('none', 'total_magnitude'):
            raise ValueError('Only the registered complete-gradient factorial is admitted')
        self._path_cache = None
        self._cached_logits = None
        self._training_loss_calls = 0
        self.register_buffer('_gate_rows', torch.full((3, 8), -1., dtype=torch.float64))
        self.register_buffer('_calls_through_checkpoint', torch.zeros((), dtype=torch.float64))

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
    def _normalize_product(product, gradient='full', mask=None, action_type=None,
                           inference_shortcut=False):
        squared = product.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-6).sqrt()
        unit = product / denominator
        # The half rule changes only derivatives. With autograd disabled,
        # its cancelling surrogate contributes no value; skip donor ranking.
        # Do not gate on self.training: callers may need gradients in eval.
        if gradient == 'full' or (inference_shortcut and not torch.is_grad_enabled()):
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


    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)


    def _product(self, fields, components, mask):
        category, brand, _, _ = components
        qc, qb = fields[self.category], fields[self.brand]
        if self.training and torch.is_grad_enabled():
            # These four clones are exclusive product-path endpoints. Other
            # static, attention-query, history and value paths use originals.
            qc, qb, category, brand = qc.clone(), qb.clone(), category.clone(), brand.clone()
            self._path_cache = (qc, qb, category, brand)
        if self.config['product_pairing'] == 'aligned':
            product = qc[:, None, :] * category + qb[:, None, :] * brand
        else:
            # Same term count and width; only history-field partners change.
            product = qc[:, None, :] * brand + qb[:, None, :] * category
        return self._normalize_product(
            product, self.config['normalization_gradient'], mask,
            inference_shortcut=self.config.get('inference_shortcut', False))


    def forward(self, batch):
        self._path_cache = None
        self._cached_logits = None
        fields, features, history, components = self._encode(batch)
        product = self._product(fields, components, batch['history_mask'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
        if self._path_cache is not None:
            self._cached_logits = logits
        return logits


    def training_loss(self, batch, logits, train_labels):
        # This module intentionally implements only the registered none arm.
        # Keep host mean BCE and release the parent's cloned graph references.
        if self.config['row_intervention'] != 'none':
            raise ValueError('Observer-free candidate requires row_intervention=none')
        labels = train_labels.to(dtype=logits.dtype).reshape_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='mean')
        if not self.training or not torch.is_grad_enabled():
            return loss
        if self._path_cache is None or self._cached_logits is not logits:
            raise RuntimeError('training_loss requires immediately preceding logits')
        self._path_cache = None
        self._cached_logits = None
        return loss

    @staticmethod
    def _synthetic_gradient_audit():
        """Bounded host-side VJP check, artificial float64 tensors, no RNG.

        Hand-specified donor indices independently enumerate a four-event
        rotation, singleton, empty history, stable ties and masked/floor cases.
        This checks product normalization only, not whole-model gradients.
        """
        values = [[[1., 0., 0.], [0., 2., 0.], [0., 0., 4.], [3., 0., 0.]],
                  [[0., 0., 0.], [0., 0.0001, 0.], [2., 0., 0.], [0., 5., 0.]],
                  [[1., 1., 0.], [2., 1., 0.], [1., 0., 0.], [3., 1., 0.]],
                  [[1., 0., 0.], [0., 1., 0.], [0., 0., 2.], [0., 0., 0.]]]
        raw = torch.tensor(values, dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0],
                             [0, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        donor = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3],
                              [0, 1, 2, 3], [1, 2, 0, 3]], dtype=torch.long)
        squared = raw.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-6).sqrt()
        unit = raw / denominator
        inverse = denominator.squeeze(-1).reciprocal()
        regular = squared >= 1e-6
        eligible = regular.squeeze(-1) & mask
        actual_donor = Model._rotation_donors(inverse, eligible, 'scale_half_tangent')
        probe = (torch.arange(raw.numel(), dtype=raw.dtype).reshape_as(raw) + 1) / 17.
        tangent = probe - unit * (unit * probe).sum(-1, keepdim=True)
        out = {'synthetic_donor_abs_error': float((actual_donor - donor).abs().max())}
        with torch.enable_grad():
            for mode in ('full', 'scale_half_tangent'):
                leaf = raw.clone().requires_grad_(True)
                normalized = Model._normalize_product(leaf, mode, mask)
                (actual,) = torch.autograd.grad((normalized * probe).sum(), leaf)
                scale = inverse if mode == 'full' else inverse.gather(-1, donor)
                expected = torch.where(regular, scale.unsqueeze(-1) * tangent, probe * 1000.)
                relative = (actual - expected).abs() / (1. + expected.abs())
                out['synthetic_vjp_relative_error_' + mode] = float(relative.max())
                out['synthetic_forward_abs_error_' + mode] = float((normalized.detach() - unit).abs().max())
        return out

    @staticmethod
    def _shortcut_audit():
        """Flag-on/off artificial probes, independent of model state or RNG.

        Includes tied ranks, empty/singleton groups, padding, below-floor,
        boundary and above-floor products. Gradient comparisons run under
        enable_grad even though the host calls diagnostics under no_grad.
        These are bounded candidate-owned checks, not population proofs.
        """
        raw = torch.tensor([
            [[1., 0., 0.], [0., 2., 0.], [0., 0., 4.], [3., 0., 0.]],
            [[0., 0., 0.], [0.0001, 0., 0.], [0.001, 0., 0.], [0.001001, 0., 0.]],
            [[1., 1., 0.], [2., 1., 0.], [1., 0., 0.], [3., 1., 0.]],
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 2.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0.0001, 0.], [2., 0., 0.], [0., 5., 0.]],
        ], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0],
                             [1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
        probe = (torch.arange(raw.numel(), dtype=raw.dtype).reshape_as(raw) + 1.) / 17.
        with torch.no_grad():
            old = Model._normalize_product(raw, 'scale_half_tangent', mask,
                                           inference_shortcut=False)
            new = Model._normalize_product(raw, 'scale_half_tangent', mask,
                                           inference_shortcut=True)
            squared = raw.square().sum(-1)
            out = {
                'shortcut_synthetic_no_grad_product_abs_error': float((old - new).abs().max()),
                'shortcut_synthetic_no_grad_finite_fraction': float(torch.isfinite(new).double().mean()),
                'shortcut_synthetic_floor_events': float(((squared < 1e-6) & mask).sum()),
                'shortcut_synthetic_boundary_events': float(((squared == 1e-6) & mask).sum()),
            }
        with torch.enable_grad():
            left = raw.clone().requires_grad_(True)
            right = raw.clone().requires_grad_(True)
            old = Model._normalize_product(left, 'scale_half_tangent', mask,
                                           inference_shortcut=False)
            new = Model._normalize_product(right, 'scale_half_tangent', mask,
                                           inference_shortcut=True)
            (g_old,) = torch.autograd.grad((old * probe).sum(), left)
            (g_new,) = torch.autograd.grad((new * probe).sum(), right)
            out['shortcut_synthetic_grad_product_abs_error'] = float((old.detach() - new.detach()).abs().max())
            out['shortcut_synthetic_grad_vjp_abs_error'] = float((g_old - g_new).abs().max())
        return out

    def diagnostics(self, batch):
        # First512 ordered validation impressions; no labels or training probes.
        fields, features, history, components = self._encode(batch)
        category, brand, _, _ = components
        qc, qb = fields[self.category], fields[self.brand]
        raw = (qc[:, None, :] * category + qb[:, None, :] * brand
               if self.config['product_pairing'] == 'aligned' else
               qc[:, None, :] * brand + qb[:, None, :] * category)
        mask = batch['history_mask']
        full = self._normalize_product(raw, 'full', mask)
        half = self._normalize_product(raw, 'scale_half_tangent', mask)
        product = half if self.config['normalization_gradient'] == 'scale_half_tangent' else full
        weights = self._weights(self._scores_from_product(fields, history, product), mask)
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        full_weights = self._weights(self._scores_from_product(fields, history, full), mask)
        full_logits = self._logits(features, (full_weights.unsqueeze(-1) * history).sum(1))
        actual = self.forward(batch)
        squared = raw.square().sum(-1)
        inverse = squared.clamp_min(1e-6).sqrt().reciprocal()
        eligible = (squared >= 1e-6) & mask
        assigned = self._event_scales(inverse.detach(), eligible, 'scale_half_tangent')
        count = eligible.sum(-1)
        own = torch.where(eligible, inverse, torch.zeros_like(inverse))
        target = torch.where(eligible, assigned, torch.zeros_like(assigned))
        has_history = mask.any(-1)
        events = eligible.sum().clamp_min(1)
        norm_error = (product.square().sum(-1).sqrt() - 1.).abs()
        out = {
            'audit_schema': 9.,
            'inference_shortcut_enabled': float(self.config.get('inference_shortcut', False)),
            'diagnostics_autograd_disabled': float(not torch.is_grad_enabled()),
            'training_observers_enabled': 0.,
            'row_intervention_enabled': 0.,
            'active_pairing_crossed': float(self.config['product_pairing'] == 'crossed'),
            'active_gradient_half': float(self.config['normalization_gradient'] == 'scale_half_tangent'),
            'validation_samples': float(batch['static'].shape[0]),
            'validation_valid_events': float(mask.sum()),
            'validation_empty_history_samples': float((~has_history).sum()),
            'validation_eligible_events': float(eligible.sum()),
            'validation_floor_events': float(((squared < 1e-6) & mask).sum()),
            'validation_boundary_events': float(((squared == 1e-6) & mask).sum()),
            'validation_eligible_singleton_rows': float((count == 1).sum()),
            'validation_eligible_multiple_rows': float((count > 1).sum()),
            'validation_forward_helper_abs_max': float((actual - reference).abs().max()),
            'validation_full_half_product_abs_max': float((half - full).abs().max()),
            'validation_full_active_logit_abs_max': float((actual - full_logits).abs().max()),
            'validation_logits_finite_fraction': float(torch.isfinite(actual).to(torch.float32).mean()),
            'validation_masked_mass_max': float((weights * (~mask).to(weights.dtype)).sum(-1).max()),
            'validation_weight_sum_error_max': float((weights.sum(-1) - has_history.to(weights.dtype)).abs().max()),
            'validation_empty_summary_norm_max': float((summary.square().sum(-1).sqrt() * (~has_history)).max()),
            'validation_unit_norm_error_max_regular': float(torch.where(eligible, norm_error, torch.zeros_like(norm_error)).max()),
            'validation_half_scale_abs_difference_mean': float((target - own).abs().sum() / events),
            'validation_half_scale_sum_relative_error_max': float(((target.sum(-1) - own.sum(-1)).abs() / own.sum(-1).clamp_min(1e-15)).max()),
            'validation_half_scale_squared_sum_relative_error_max': float(((target.square().sum(-1) - own.square().sum(-1)).abs() / own.square().sum(-1).clamp_min(1e-15)).max()),
        }
        out.update(self._synthetic_gradient_audit())
        out.update(self._shortcut_audit())
        return out
