"""No-row backbone with unused loss observers removed.

Operational source bridge from r0027-frozen_none, not a new learning rule.
CONFIG, __init__, and all prediction/helper methods are preserved verbatim,
including endpoint clones and small unused audit buffers. The half-rotation
normalization surrogate remains part of the inherited forward/backward rule.

The parent none branch already returns mean BCE exactly; its extra retained
backward, hypothetical row targets and derivative gates have no intended
update contribution. Here training_loss returns the same BCE without that
observer package. No custom optimizer, surgery, RNG or host-setting edits.
This does not guarantee finite-precision trajectory replay; the host will
measure population validation scores and cost. Old gate buffers stay unused
and are not presented as successful training diagnostics. Validation checks
are candidate-owned, label-free, and restricted to supplied inputs.

Only syntax/source checks run locally; Torch execution and training remain
for the authorized host. See proposal.json for the frozen engineering guards,
historical control limitations, one-job budget and assessment reserve.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'none'}


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
        if config['kind'] != 'din' or config['product_pairing'] != 'aligned' or config['product_normalization'] != 'unit':
            raise ValueError('This study fixes target-aware aligned unit products')
        if config['normalization_gradient'] not in ('full', 'scale_half_tangent'):
            raise ValueError('Only the fixed own/half normalization rules are admitted')
        if config['row_intervention'] not in ('none', 'total_magnitude'):
            raise ValueError('Only the registered complete-gradient factorial is admitted')
        if config['normalization_gradient'] != 'scale_half_tangent':
            raise ValueError('This bounded study fixes the inherited half-scale rule')
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
        product = qc[:, None, :] * category + qb[:, None, :] * brand
        return self._normalize_product(product, self.config['normalization_gradient'], mask)


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

    def diagnostics(self, batch):
        # Host invokes on the first512 validation inputs under no_grad.
        # No training gates are run or claimed. No labels or external state.
        fields, features, history, components = self._encode(batch)
        raw = (fields[self.category][:, None, :] * components[0] +
               fields[self.brand][:, None, :] * components[1])
        product = self._normalize_product(raw, 'full', batch['history_mask'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        actual = self.forward(batch)
        has_history = batch['history_mask'].any(-1)
        return {
            'audit_schema': 7.,
            'training_observers_enabled': 0.,
            'magnitude_enabled': float(self.config['row_intervention'] != 'none'),
            'parent_gate_buffers_unused': 1.,
            'validation_samples': float(batch['static'].shape[0]),
            'validation_empty_history_samples': float((~has_history).sum()),
            'validation_forward_abs_max': float((actual - reference).abs().max()),
            'validation_logits_finite_fraction': float(torch.isfinite(actual).float().mean()),
            'validation_masked_mass_max': float(
                (weights * (~batch['history_mask']).to(weights.dtype)).sum(-1).max()),
            'validation_weight_sum_error_max': float(
                (weights.sum(-1) - has_history.to(weights.dtype)).abs().max()),
            'validation_empty_summary_norm_max': float(
                (summary.square().sum(-1).sqrt() * (~has_history)).max()),
        }
