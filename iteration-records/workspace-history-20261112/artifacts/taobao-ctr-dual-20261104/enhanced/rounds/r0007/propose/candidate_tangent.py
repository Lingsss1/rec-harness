"""Forward-matched decomposition of the aligned L2-product backward path.

For p=qc*hc+qb*hb, r=sqrt(max(sum(p**2),1e-6)), every mode
returns u=p/r at fixed weights. Above the fixed floor the Jacobians are:
  full: (I-u*u^T)/r; detached: I/r; tangent: I-u*u^T; identity: I.
Below the floor all modes retain I/1e-3. Full/detached are existing controls;
tangent/identity complete a radial-projection by inverse-norm-scaling factorial.
The latter two are explicit surrogate gradients, not derivatives of the
forward function. The subtraction is parenthesized to make an exact zero
forward correction for finite tensors. No learned parameters, data, loss,
optimizer, RNG consumption or selection rules are changed.
Inherits DIN-style attention and the DetachNorm diagnostic idea (Xu et al.,
NeurIPS 2019, sections 3.2-3.3); does not reproduce LayerNorm or claim novelty.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'tangent'}


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
        # Keep the inherited forward and fixed squared-norm floor unchanged.
        squared_norm = product.square().sum(-1, keepdim=True)
        denominator = squared_norm.clamp_min(1e-6).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        if gradient == 'detached':
            return product / denominator.detach()
        regular = squared_norm >= 1e-6
        if gradient == 'tangent':
            # Freeze only the outer factor: r * d(p/r)/dp = I-u*u^T.
            # Below the floor retain the inherited finite derivative I/1e-3.
            scale = torch.where(regular, denominator,
                                torch.ones_like(denominator)).detach()
            surrogate = unit * scale
        elif gradient == 'identity':
            # No projection or inverse-norm scaling above the floor.
            scale = torch.where(regular, torch.ones_like(denominator),
                                denominator.reciprocal()).detach()
            surrogate = product * scale
        else:
            raise ValueError('normalization_gradient must be full, detached, tangent or identity')
        # Same-state forward is u; only the explicit product-block Jacobian changes.
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
        """At most 64 label-free scalars, restricted to first 512 host rows.

        All substitutions use one checkpoint; they measure activity only.
        Nonzero encoded-ID matches refer only to the supplied history window.
        Empty subgroup means are zero; inspect the corresponding counts.
        Unit-vector scale matching applies only above the fixed norm floor.
        """
        fields, features, history, components = self._encode(batch)
        mask = batch['history_mask']
        mf = mask.float()
        lengths = mask.sum(-1)
        nonempty = lengths > 0
        modes = [('aligned_raw', 'aligned', 'raw'),
                 ('crossed_raw', 'crossed', 'raw'),
                 ('aligned_unit', 'aligned', 'unit'),
                 ('crossed_unit', 'crossed', 'unit')]
        products = {}
        scores = {}
        weights = {}
        summaries = {}
        logits = {}
        for name, pairing, normalization in modes:
            products[name] = self._product(fields, components, pairing, normalization)
            scores[name] = self._scores_from_product(fields, history, products[name])
            weights[name] = self._weights(scores[name], mask)
            summaries[name] = (weights[name].unsqueeze(-1) * history).sum(1)
            logits[name] = self._logits(features, summaries[name])
        active_name = self.config['product_pairing'] + '_' + self.config['product_normalization']
        active = weights[active_name]

        def norm(value):
            return value.square().sum(-1).sqrt()

        def selected_mean(value, selection):
            return (value * selection).sum() / selection.sum().clamp_min(1)

        target_category = batch['static'][:, self.category, None]
        target_brand = batch['static'][:, self.brand, None]
        cm = (mask & (target_category > 0) & (batch['history_category'] > 0) &
              (batch['history_category'] == target_category))
        bm = (mask & (target_brand > 0) & (batch['history_brand'] > 0) &
              (batch['history_brand'] == target_brand))
        cp = cm.any(-1)
        bp = bm.any(-1)
        dr2 = products['aligned_raw'].square().sum(-1)
        xr2 = products['crossed_raw'].square().sum(-1)
        joint_regular = mask & (dr2 >= 1e-6) & (xr2 >= 1e-6)
        result = {
            'diagnostic_samples': lengths.new_tensor(mask.shape[0]),
            'visible_valid_events': lengths.sum(),
            'empty_samples': (~nonempty).sum(),
            'nonempty_samples': nonempty.sum(),
            'visible_category_match_samples': cp.sum(),
            'visible_brand_match_samples': bp.sum(),
            'visible_category_only_samples': (cp & ~bp).sum(),
            'visible_brand_only_samples': (bp & ~cp).sum(),
            'visible_both_match_samples': (cp & bp).sum(),
            'visible_neither_nonempty_samples': (nonempty & ~cp & ~bp).sum(),
            'target_category_zero_samples': (target_category.squeeze(-1) == 0).sum(),
            'target_brand_zero_samples': (target_brand.squeeze(-1) == 0).sum(),
            'visible_category_zero_events': (mask & (batch['history_category'] == 0)).sum(),
            'visible_brand_zero_events': (mask & (batch['history_brand'] == 0)).sum(),
            'aligned_floor_events': (mask & (dr2 < 1e-6)).sum(),
            'crossed_floor_events': (mask & (xr2 < 1e-6)).sum(),
            'joint_nonfloored_events': joint_regular.sum(),
            'active_pairing_is_aligned': mf.new_tensor(float(self.config['product_pairing'] == 'aligned')),
            'active_normalization_is_unit': mf.new_tensor(float(self.config['product_normalization'] == 'unit')),
            'masked_padding_mass_max': (active * (~mask).float()).sum(-1).max(),
            'weight_sum_error_max': (active.sum(-1) - nonempty.float()).abs().max(),
            'empty_summary_l2_max': (norm(summaries[active_name]) * (~nonempty)).max(),
            'forward_helper_logit_abs_max': (self.forward(batch) - logits[active_name]).abs().max(),
            'unit_pair_norm_difference_max_valid': (
                (norm(products['aligned_unit']) - norm(products['crossed_unit'])).abs() * mf).max(),
            'unit_pair_cosine_mean_joint_nonfloored': selected_mean(
                (products['aligned_unit'] * products['crossed_unit']).sum(-1), joint_regular),
        }
        baseline_scores = scores['aligned_raw']
        baseline_weights = weights['aligned_raw']
        for name, _, _ in modes:
            product_norm = norm(products[name])
            delta = scores[name] - baseline_scores
            center = (delta * mf).sum(-1) / lengths.clamp_min(1)
            result[name + '_product_l2_mean_valid_event'] = selected_mean(product_norm, mask)
            result[name + '_centered_score_delta_abs_mean_valid_event'] = selected_mean(
                (delta - center[:, None]).abs(), mask)
            result[name + '_weight_l1_from_aligned_raw_mean_nonempty'] = selected_mean(
                (weights[name] - baseline_weights).abs().sum(-1), nonempty)
            result[name + '_summary_l2_from_aligned_raw_mean_nonempty'] = selected_mean(
                norm(summaries[name] - summaries['aligned_raw']), nonempty)
            result[name + '_logit_abs_from_aligned_raw_mean_nonempty'] = selected_mean(
                (logits[name] - logits['aligned_raw']).abs(), nonempty)
            result[name + '_category_match_mass_mean_present'] = selected_mean(
                (weights[name] * cm.float()).sum(-1), cp)
            result[name + '_brand_match_mass_mean_present'] = selected_mean(
                (weights[name] * bm.float()).sum(-1), bp)
            if name.endswith('_unit'):
                raw_name = name.replace('_unit', '_raw')
                regular = mask & (products[raw_name].square().sum(-1) >= 1e-6)
                result[name + '_norm_error_max_nonfloored'] = (
                    (product_norm - 1).abs() * regular).max()
        # These no-grad checks verify forward arithmetic only, not backward paths.
        mode = self.config.get('normalization_gradient', 'full')
        raw_product = products[self.config['product_pairing'] + '_raw']
        full_product = self._normalize_product(raw_product, 'full')
        active_product = self._normalize_product(raw_product, mode)
        full_scores = self._scores_from_product(fields, history, full_product)
        full_weights = self._weights(full_scores, mask)
        full_summary = (full_weights.unsqueeze(-1) * history).sum(1)
        full_logits = self._logits(features, full_summary)
        squared_norm = raw_product.square().sum(-1)
        result['gradient_is_tangent_only'] = mf.new_tensor(float(mode == 'tangent'))
        result['gradient_is_identity'] = mf.new_tensor(float(mode == 'identity'))
        result['same_state_full_surrogate_product_abs_max'] = (
            full_product - active_product).abs().max()
        result['same_state_full_surrogate_logit_abs_max'] = (
            full_logits - logits[active_name]).abs().max()
        result['active_inverse_norm_mean_valid_event'] = selected_mean(
            squared_norm.clamp_min(1e-6).sqrt().reciprocal(), mask)
        result['active_regular_valid_events'] = (mask & (squared_norm >= 1e-6)).sum()
        result['active_floored_valid_events'] = (mask & (squared_norm < 1e-6)).sum()
        return {name: float(value.detach().item()) for name, value in result.items()}
