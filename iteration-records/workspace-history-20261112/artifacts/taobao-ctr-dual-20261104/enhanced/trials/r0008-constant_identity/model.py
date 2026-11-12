"""Fixed-scale replacement of the aligned normalized product backward rule.

p = qc*hc + qb*hb; r = sqrt(max(sum(p**2), 1e-6)); u = p/r.
Every mode returns u at the same finite state. On squared norms >= 1e-6:
full gives J=(I-u*u^T)/r, constant_tangent gives J=0.2*(I-u*u^T),
and constant_identity gives J=0.2*I. Below the floor all retain J=1000*I.
The latter two are explicit surrogate gradients, not derivatives of u.
The fixed scalar 0.2 is prospectively chosen from earlier ordered diagnostic
means near 0.18-0.20; it is NOT calibrated on a representative training sample.
No sweep, training-time calibration, new state, optimizer or loss is introduced.
This is a local mechanism control, not a novelty or performance claim.
Inherits the DIN-style field/history adapter and forward-matched gradient
contrasts of Xu et al., NeurIPS 2019, sections 3.2-3.3 (not LayerNorm itself).
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'constant_identity'}


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
    def _normalize_product(product, gradient='full'):
        # Exactly the incumbent's forward arithmetic and squared-norm floor.
        denominator = product.square().sum(-1, keepdim=True).clamp_min(1e-6).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        regular = product.square().sum(-1, keepdim=True) >= 1e-6
        if gradient == 'constant_tangent':
            # Detached r cancels inverse-norm scaling only above the floor.
            scale = torch.where(regular, denominator * 0.2,
                                torch.ones_like(denominator)).detach()
            surrogate = unit * scale
        elif gradient == 'constant_identity':
            scale = torch.where(regular, torch.ones_like(denominator) * 0.2,
                                denominator.reciprocal()).detach()
            surrogate = product * scale
        else:
            raise ValueError('normalization_gradient must be full, constant_tangent or constant_identity')
        # Exact zero forward correction for finite tensors; no extra parameters.
        return unit.detach() + (surrogate - surrogate.detach())


    def _product(self, fields, components, pairing, normalization):
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
                product, self.config.get('normalization_gradient', 'full'))
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
                                self.config['product_normalization'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        return self._logits(features, summary)

    def diagnostics(self, batch):
        """Label-free first-512 measurements, not population or gradient evidence.

        The host calls under no_grad at its best checkpoint. Analytic probe
        values below do NOT test actual autograd or optimizer updates. Empty
        group means/minima/maxima are zero and require the reported counts.
        """
        fields, features, history, components = self._encode(batch)
        mask = batch['history_mask']
        mf = mask.float()
        lengths = mask.sum(-1)
        nonempty = lengths > 0
        raw = self._product(fields, components, 'aligned', 'raw')
        r2 = raw.square().sum(-1)
        r = r2.clamp_min(1e-6).sqrt()
        inverse = r.reciprocal()
        regular = mask & (r2 >= 1e-6)
        unit = self._normalize_product(raw, 'full')
        active = self._normalize_product(raw, self.config['normalization_gradient'])
        scores = self._scores_from_product(fields, history, active)
        weights = self._weights(scores, mask)
        summary = (weights.unsqueeze(-1) * history).sum(1)
        logits = self._logits(features, summary)
        full_weights = self._weights(self._scores_from_product(fields, history, unit), mask)
        full_logits = self._logits(features, (full_weights.unsqueeze(-1) * history).sum(1))

        def mean(value, selection):
            return (value * selection).sum() / selection.sum().clamp_min(1)

        def norm(value):
            return value.square().sum(-1).sqrt()

        def group_stats(name, selection, result):
            result[name + '_events'] = selection.sum()
            result[name + '_inverse_norm_mean'] = mean(inverse, selection)

        result = {
            'diagnostic_samples': lengths.new_tensor(mask.shape[0]),
            'valid_events': lengths.sum(),
            'nonempty_samples': nonempty.sum(),
            'empty_samples': (~nonempty).sum(),
            'regular_valid_events': regular.sum(),
            'floored_valid_events': (mask & (r2 < 1e-6)).sum(),
            'boundary_valid_events': (mask & (r2 == 1e-6)).sum(),
            'constant_scale': mf.new_tensor(0.2),
            'gradient_has_projection': mf.new_tensor(float(self.config['normalization_gradient'] != 'constant_identity')),
            'target_brand_zero_samples': (batch['static'][:, self.brand] == 0).sum(),
            'masked_padding_mass_max': (weights * (~mask).float()).sum(-1).max(),
            'weight_sum_error_max': (weights.sum(-1) - nonempty.float()).abs().max(),
            'empty_summary_l2_max': (norm(summary) * (~nonempty)).max(),
            'forward_helper_logit_abs_max': (self.forward(batch) - logits).abs().max(),
            'same_state_full_active_product_abs_max': (unit - active).abs().max(),
            'same_state_full_active_logit_abs_max': (full_logits - logits).abs().max(),
            'unit_norm_error_max_regular': ((norm(unit) - 1).abs() * regular).max(),
            'raw_norm_mean_valid': mean(r2.sqrt(), mask),
            'inverse_norm_mean_valid': mean(inverse, mask),
            'inverse_norm_std_valid': mean((inverse - mean(inverse, mask)).square(), mask).sqrt(),
            'inverse_norm_mean_regular': mean(inverse, regular),
            'inverse_norm_std_regular': mean((inverse - mean(inverse, regular)).square(), regular).sqrt(),
            'inverse_norm_max_valid': (inverse * mf).max(),
            'inverse_norm_abs_difference_from_constant_regular': mean((inverse - 0.2).abs(), regular),
            'inverse_norm_above_constant_fraction_regular': mean((inverse > 0.2).float(), regular),
        }
        for action in range(1, 5):
            group_stats('action_' + str(action), mask & (batch['history_type'] == action), result)
        target_brand_missing = (batch['static'][:, self.brand] == 0)[:, None]
        group_stats('target_brand_missing', mask & target_brand_missing, result)
        group_stats('target_brand_present', mask & ~target_brand_missing, result)
        group_stats('history_brand_missing', mask & (batch['history_brand'] == 0), result)
        group_stats('history_brand_present', mask & (batch['history_brand'] > 0), result)
        category_match = ((batch['static'][:, self.category, None] > 0) &
                          (batch['history_category'] > 0) &
                          (batch['history_category'] == batch['static'][:, self.category, None]))
        group_stats('category_match', mask & category_match, result)
        group_stats('category_no_match', mask & ~category_match, result)
        # Analytic Jacobian action on a fixed unit probe, not measured gradients.
        probe = torch.ones_like(raw) / (raw.shape[-1] ** 0.5)
        projected = probe - unit * (unit * probe).sum(-1, keepdim=True)
        full_probe = projected * inverse.unsqueeze(-1)
        constant_probe = projected * 0.2
        result['analytic_full_probe_norm_mean_regular'] = mean(norm(full_probe), regular)
        result['analytic_constant_tangent_probe_norm_mean_regular'] = mean(norm(constant_probe), regular)
        result['analytic_full_vs_constant_probe_difference_regular'] = mean(norm(full_probe - constant_probe), regular)
        return {name: float(value.detach().item()) for name, value in result.items()}

