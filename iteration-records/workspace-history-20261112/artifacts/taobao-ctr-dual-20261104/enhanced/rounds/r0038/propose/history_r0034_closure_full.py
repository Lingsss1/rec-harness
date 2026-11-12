"""Prospective source closure for aligned full versus rank-half normalization.

Both standalone modules have identical code except literal CONFIG. The parent
is r0033-global_half_control, source SHA256
9e8f81495ca127ea4cd59d750d6097bfcd9a3c61bd0b11d191b64c7b2522ee1e.
This pair admits the already implemented native-full derivative as a matched
control. It introduces no new modeling mechanism and makes no novelty claim.
Half means detached inverse-norm donor reassignment by floor(n/2) rank rotation
within each impression's valid, nonfloored events, not scalar multiplication
by 0.5. Forward values, parameter construction, scalar mean BCE and host Adam
remain inherited. Row interventions, sparse routing and training observers are
disabled. Inference uses the inherited no-grad shortcut in both arms.

Changes from the parent: this description; constructor guard admits full and
requires sparse_full_gradient=False; diagnostics identify the selected rule.
All other methods, buffers, parameter/RNG construction and operation order are
byte-preserved. Existing synthetic routing checks remain as bounded input-only
checks of the inherited normalization helpers, not an active routing study.
The host still uses seed42; neither module changes seed, data order or protocol.
No local Torch execution or training. Source/syntax checks are not VJP tests.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'full', 'row_intervention': 'none', 'inference_shortcut': True, 'surrogate_implementation': 'legacy', 'sparse_full_gradient': False}


class _IdentityScaledVJP(torch.autograd.Function):
    """Value identity; multiply the incoming VJP by a detached coefficient.

    Save only the scalar-per-event coefficient, not another full product.
    Returning an input alias is safe: downstream code makes no in-place edits.
    This deliberately delegates the entire normalization derivative to Torch,
    preserving the parent's operation graph before the scale multiplication.
    """
    @staticmethod
    def forward(ctx, unit, coefficient):
        ctx.save_for_backward(coefficient)
        return unit

    @staticmethod
    def backward(ctx, upstream):
        (coefficient,) = ctx.saved_tensors
        return upstream * coefficient, None


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
        if config.get('surrogate_implementation', 'legacy') not in ('legacy', 'identity_vjp'):
            raise ValueError('Only the registered legacy/identity VJP comparison is admitted')
        if config['row_intervention'] != 'none':
            raise ValueError('This study fixes row_intervention=none')
        if type(config.get('sparse_full_gradient', False)) is not bool:
            raise ValueError('sparse_full_gradient must be a literal boolean')
        if config['product_pairing'] != 'aligned' or config.get('surrogate_implementation', 'legacy') != 'legacy' or config.get('sparse_full_gradient', False):
            raise ValueError('Source closure fixes aligned products, legacy transport and no sparse routing')
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
                           inference_shortcut=False, surrogate_implementation='legacy',
                           sparse_full_gradient=False):
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
        if surrogate_implementation == 'identity_vjp':
            return _IdentityScaledVJP.apply(unit, coefficient)
        if surrogate_implementation != 'legacy':
            raise ValueError('Unknown surrogate implementation')
        surrogate = unit * coefficient
        half = unit.detach() + (surrogate - surrogate.detach())
        if sparse_full_gradient:
            if mask is None:
                raise ValueError('Supplied causal history mask is required for routing')
            # Count supplied events, including valid events below the norm floor.
            # The decision uses only non-differentiable per-impression inputs.
            short = (mask.sum(-1, keepdim=True) <= 16).unsqueeze(-1)
            return torch.where(short, unit, half)
        return half


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
            inference_shortcut=self.config.get('inference_shortcut', False),
            surrogate_implementation=self.config.get('surrogate_implementation', 'legacy'),
            sparse_full_gradient=self.config.get('sparse_full_gradient', False))


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
    def _routing_audit():
        """Prospective bounded checks; independently enumerate donors in Python.

        Artificial cases include lengths 0/1/2/16/17, non-prefix masks, ties,
        floors and boundary-adjacent products. Count supplied events, not just
        norm-eligible events. Local Torch execution is not claimed.
        """
        out = {}
        lengths = (0, 1, 2, 16, 17, 16, 17, 17)
        for dtype, tag in ((torch.float32, 'f32'), (torch.float64, 'f64')):
            raw = torch.tensor([
                [[0.1 * (j + 1), 0.25 + 0.1 * (j % 3), 0.05]
                 for j in range(17)] for _ in lengths], dtype=dtype)
            mask = torch.tensor([[j < length for j in range(17)]
                                 for length in lengths], dtype=torch.bool)
            mask[7] = torch.tensor([j % 2 == 0 for j in range(17)], dtype=torch.bool)
            for row in (5, 6):
                raw[row, 0] = torch.tensor([0., 0., 0.], dtype=dtype)
                raw[row, 1] = torch.tensor([0.0001, 0., 0.], dtype=dtype)
                raw[row, 2] = torch.tensor([0.001, 0., 0.], dtype=dtype)
                raw[row, 3] = torch.tensor([1., 0., 0.], dtype=dtype)
                raw[row, 4] = torch.tensor([1., 0., 0.], dtype=dtype)
                raw[row, 5] = torch.tensor([0.001001, 0., 0.], dtype=dtype)
            squared = raw.square().sum(-1, keepdim=True)
            denominator = squared.clamp_min(1e-6).sqrt()
            unit = raw / denominator
            inverse = denominator.squeeze(-1).reciprocal()
            regular = squared >= 1e-6
            eligible = regular.squeeze(-1) & mask
            short = mask.sum(-1) <= 16
            # Separate Python sorting/ranking oracle, not the production helper.
            donors = torch.arange(17, dtype=torch.long)[None, :].expand(8, -1).clone()
            for row in range(8):
                order = sorted([j for j in range(17) if bool(eligible[row, j])],
                               key=lambda j: (float(inverse[row, j]), j))
                for rank, event in enumerate(order):
                    donors[row, event] = order[(rank + len(order) // 2) % len(order)]
            assigned = inverse.gather(-1, donors)
            actual_donor = Model._rotation_donors(inverse, eligible, 'scale_half_tangent')
            probe = (torch.arange(raw.numel(), dtype=dtype).reshape_as(raw) + 1.) / 317.
            tangent = probe - unit * (unit * probe).sum(-1, keepdim=True)
            grads = {}
            value_error = raw.new_zeros(())
            with torch.enable_grad():
                for name, mode, fallback in (
                    ('full', 'full', False),
                    ('half', 'scale_half_tangent', False),
                    ('routed', 'scale_half_tangent', True),
                ):
                    leaf = raw.clone().requires_grad_(True)
                    value = Model._normalize_product(
                        leaf, mode, mask, inference_shortcut=True,
                        surrogate_implementation='legacy', sparse_full_gradient=fallback)
                    (actual,) = torch.autograd.grad((value * probe).sum(), leaf)
                    scale = inverse if name == 'full' else assigned
                    if fallback:
                        scale = torch.where(short[:, None], inverse, assigned)
                    expected = torch.where(regular, scale.unsqueeze(-1) * tangent, probe * 1000.)
                    error = ((actual - expected).abs() / (1. + expected.abs())).max()
                    out['routing_' + tag + '_' + name + '_vjp_scaled_error'] = float(error)
                    value_error = torch.maximum(value_error, (value.detach() - unit).abs().max())
                    grads[name] = actual.detach()
            with torch.no_grad():
                value = Model._normalize_product(
                    raw, 'scale_half_tangent', mask, inference_shortcut=True,
                    sparse_full_gradient=True)
            prefix = 'routing_' + tag + '_'
            out.update({
                prefix + 'donor_abs_error': float((actual_donor - donors).abs().max()),
                prefix + 'forward_abs_error': float(value_error),
                prefix + 'no_grad_forward_abs_error': float((value - unit).abs().max()),
                prefix + 'short_matches_full_vjp_scaled_error': float(
                    ((grads['routed'][short] - grads['full'][short]).abs() /
                     (1. + grads['full'][short].abs())).max()),
                prefix + 'long_matches_half_vjp_scaled_error': float(
                    ((grads['routed'][~short] - grads['half'][~short]).abs() /
                     (1. + grads['half'][~short].abs())).max()),
                prefix + 'short_changed_vjp_abs_max': float(
                    (grads['routed'][short] - grads['half'][short]).abs().max()),
                prefix + 'length_16_rows': float((mask.sum(-1) == 16).sum()),
                prefix + 'length_17_rows': float((mask.sum(-1) == 17).sum()),
                prefix + 'floor_events': float(((~regular.squeeze(-1)) & mask).sum()),
                prefix + 'boundary_events': float(((squared.squeeze(-1) == 1e-6) & mask).sum()),
            })
        return out

    def diagnostics(self, batch):
        """At most 49 finite scalars; inputs only, under host no_grad.

        No training probes or stored labels. Saved buffers are inherited,
        remain unused and have identical layout in the two modules.
        """
        fields, features, history, components = self._encode(batch)
        raw = (fields[self.category][:, None, :] * components[0] +
               fields[self.brand][:, None, :] * components[1])
        mask = batch['history_mask']
        full = self._normalize_product(raw, 'full', mask)
        weights = self._weights(self._scores_from_product(fields, history, full), mask)
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        actual = self.forward(batch)
        count = mask.sum(-1)
        short = count <= 16
        squared = raw.square().sum(-1)
        eligible = (squared >= 1e-6) & mask
        out = {
            'audit_schema': 12.,
            'normalization_half_enabled': float(self.config['normalization_gradient'] == 'scale_half_tangent'),
            'sparse_full_gradient_enabled': float(self.config['sparse_full_gradient']),
            'short_history_cutoff': 16.,
            'validation_samples': float(mask.shape[0]),
            'validation_short_rows': float(short.sum()),
            'validation_long_rows': float((~short).sum()),
            'validation_empty_rows': float((count == 0).sum()),
            'validation_short_regular_events': float((eligible & short[:, None]).sum()),
            'validation_long_regular_events': float((eligible & (~short[:, None])).sum()),
            'validation_floor_events': float(((squared < 1e-6) & mask).sum()),
            'validation_boundary_events': float(((squared == 1e-6) & mask).sum()),
            'validation_forward_abs_error': float((actual - reference).abs().max()),
            'validation_logits_finite_fraction': float(torch.isfinite(actual).float().mean()),
            'validation_masked_mass_max': float((weights * (~mask).to(weights.dtype)).sum(-1).max()),
            'validation_weight_sum_error_max': float((weights.sum(-1) - mask.any(-1).to(weights.dtype)).abs().max()),
            'validation_empty_summary_norm_max': float((summary.square().sum(-1).sqrt() * (~mask.any(-1))).max()),
            'training_observers_enabled': 0.,
            'row_intervention_enabled': 0.,
        }
        out.update(self._routing_audit())
        return out
