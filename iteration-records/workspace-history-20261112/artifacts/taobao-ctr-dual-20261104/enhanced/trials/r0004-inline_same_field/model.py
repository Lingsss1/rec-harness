"""Inline realization of the existing same-field product intervention.

This is an implementation bridge, not a new scientific mechanism. It retains
baseline construction, masks, sum association, attention normalization and
classifier. The two configuration flags document the fixed joint restriction;
this module is registered only for kind=din with both flags true. Training uses
this inline forward; the inherited helper path is called only in diagnostics.
"""
import torch
from torch import nn

CONFIG={'kind':'din','embedding_dim':16,'hidden':[128,64],'lr':0.001,
        'remove_cross_field':True,'remove_context_product':True}


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

    def forward(self,batch):
        fields=[embedding(batch['static'][:,i]) for i,embedding in enumerate(self.embeddings)]
        features=torch.cat(fields,dim=-1)
        if self.kind!='static':
            ages=(torch.log2(batch['history_age_seconds']+1).long()+1).clamp(0,32)
            # Bind the two existing lookups; no additional embedding call.
            history_category = self.embeddings[self.category](batch['history_category'])
            history_brand = self.embeddings[self.brand](batch['history_brand'])
            history=(history_category+history_brand+
                     self.action(batch['history_type'])+self.recency(ages))
            mask=batch['history_mask']
            if self.kind=='pooled':
                weights=mask.float()/mask.sum(-1,keepdim=True).clamp_min(1)
            else:
                query=(fields[self.category]+fields[self.brand])[:,None,:].expand_as(history)
                # Change only the explicit fourth scorer block to same-field terms.
                product=(fields[self.category][:,None,:]*history_category+
                         fields[self.brand][:,None,:]*history_brand)
                scores=self.attention(torch.cat([query,history,query-history,product],dim=-1)).squeeze(-1)
                scores=scores.masked_fill(~mask,-1e4)
                weights=scores.softmax(-1)*mask.float()
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
            features=torch.cat([features,(weights.unsqueeze(-1)*history).sum(1)],dim=-1)
        return self.output(features).squeeze(-1)

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


    def _product(self, query, fields, history, components, remove_cross, remove_context):
        category, brand, action, recency = components
        if not remove_cross:
            if not remove_context:
                # Exact ordinary-reference operation, including association.
                return query * history
            return query * (category + brand)
        qc = fields[self.category][:, None, :]
        qb = fields[self.brand][:, None, :]
        diagonal = qc * category + qb * brand
        if remove_context:
            return diagonal
        return diagonal + query * (action + recency)


    def _scores(self, fields, history, components, remove_cross, remove_context):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        product = self._product(query, fields, history, components, remove_cross, remove_context)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)


    @staticmethod
    def _weights(scores, mask):
        # Identical frozen softmax arithmetic, including all-masked rows.
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)


    def _logits(self, features, summary):
        features = torch.cat([features, summary], dim=-1)
        return self.output(features).squeeze(-1)


    def diagnostics(self, batch):
        """58 label-free scalars on the host's ordered first 512 rows.

        All four substitutions use this one frozen checkpoint. Weight/logit
        changes demonstrate intervention activity, not relevance or benefit.
        Matches below mean nonzero encoded-ID equality in the visible window;
        they do not establish raw-ID equality or match the host slice scope.
        Empty subgroup means return 0 and must be read with support counts.
        """
        fields, features, history, components = self._encode(batch)
        category, brand, action, recency = components
        mask = batch['history_mask']
        mf = mask.float()
        lengths = mask.sum(-1)
        nonempty = lengths > 0
        empty = ~nonempty
        qc = fields[self.category][:, None, :]
        qb = fields[self.brand][:, None, :]
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        modes = [('ordinary', False, False), ('cross_only', True, False),
                 ('context_only', False, True), ('same_field', True, True)]
        scores = {}
        weights = {}
        summaries = {}
        logits = {}
        for name, cross, context in modes:
            scores[name] = self._scores(fields, history, components, cross, context)
            weights[name] = self._weights(scores[name], mask)
            summaries[name] = (weights[name].unsqueeze(-1) * history).sum(1)
            logits[name] = self._logits(features, summaries[name])
        active_name = {(False, False): 'ordinary', (True, False): 'cross_only',
                       (False, True): 'context_only', (True, True): 'same_field'}[
                           (self.config['remove_cross_field'],
                            self.config['remove_context_product'])]
        active = weights[active_name]
        target_category = batch['static'][:, self.category, None]
        target_brand = batch['static'][:, self.brand, None]
        cm = (mask & (target_category > 0) & (batch['history_category'] > 0) &
              (batch['history_category'] == target_category))
        bm = (mask & (target_brand > 0) & (batch['history_brand'] > 0) &
              (batch['history_brand'] == target_brand))
        cp = cm.any(-1)
        bp = bm.any(-1)
        groups = [('category_only', cp & ~bp), ('brand_only', bp & ~cp),
                  ('neither', nonempty & ~cp & ~bp)]

        def selected_mean(values, selection):
            return (values * selection).sum() / selection.sum().clamp_min(1)

        def norm(values):
            return values.square().sum(-1).sqrt()

        def event_norm_mean(values):
            return (norm(values) * mf).sum() / mf.sum().clamp_min(1)

        diagonal = qc * category + qb * brand
        cross = qc * brand + qb * category
        context = query * (action + recency)
        ordinary_product = query * history
        decomposition_error = norm(ordinary_product - (diagonal + cross + context))
        decomposition_relative = decomposition_error / norm(ordinary_product).clamp_min(1)
        result = {
            'diagnostic_samples': lengths.new_tensor(mask.shape[0]),
            'visible_valid_events': lengths.sum(),
            'empty_samples': empty.sum(),
            'nonempty_samples': nonempty.sum(),
            'target_category_zero_samples': (target_category.squeeze(-1) == 0).sum(),
            'target_brand_zero_samples': (target_brand.squeeze(-1) == 0).sum(),
            'visible_category_match_samples': cp.sum(),
            'visible_brand_match_samples': bp.sum(),
            'visible_both_match_samples': (cp & bp).sum(),
            'visible_category_only_samples': groups[0][1].sum(),
            'visible_brand_only_samples': groups[1][1].sum(),
            'visible_neither_nonempty_samples': groups[2][1].sum(),
            'active_remove_cross_field': mf.new_tensor(float(self.config['remove_cross_field'])),
            'active_remove_context_product': mf.new_tensor(float(self.config['remove_context_product'])),
            'masked_padding_mass_max': (active * (~mask).float()).sum(-1).max(),
            'weight_sum_error_max': (active.sum(-1) - nonempty.float()).abs().max(),
            'empty_summary_l2_max': (norm(summaries[active_name]) * empty).max(),
            'product_decomposition_relative_l2_max': (decomposition_relative * mf).max(),
            'diagonal_product_l2_mean_valid_event': event_norm_mean(diagonal),
            'cross_product_l2_mean_valid_event': event_norm_mean(cross),
            'context_product_l2_mean_valid_event': event_norm_mean(context),
            'ordinary_product_l2_mean_valid_event': event_norm_mean(ordinary_product),
            'visible_category_zero_events': (mask & (batch['history_category'] == 0)).sum(),
            'visible_brand_zero_events': (mask & (batch['history_brand'] == 0)).sum(),
        }
        for name, _, _ in modes:
            weight_l1 = (weights[name] - weights['ordinary']).abs().sum(-1)
            delta = scores[name] - scores['ordinary']
            # Remove row-wise logit shifts that softmax cannot identify.
            center = (delta * mf).sum(-1) / lengths.clamp_min(1)
            centered_abs = (delta - center[:, None]).abs() * mf
            result[name + '_weight_l1_from_ordinary_mean_nonempty'] = selected_mean(weight_l1, nonempty)
            result[name + '_score_delta_centered_abs_mean_valid_event'] = centered_abs.sum() / mf.sum().clamp_min(1)
            result[name + '_summary_l2_from_ordinary_mean_nonempty'] = selected_mean(
                norm(summaries[name] - summaries['ordinary']), nonempty)
            result[name + '_logit_abs_from_ordinary_mean_nonempty'] = selected_mean(
                (logits[name] - logits['ordinary']).abs(), nonempty)
            result[name + '_category_match_mass_mean_present'] = selected_mean(
                (weights[name] * cm.float()).sum(-1), cp)
            result[name + '_brand_match_mass_mean_present'] = selected_mean(
                (weights[name] * bm.float()).sum(-1), bp)
            if name != 'ordinary':
                for group_name, selection in groups:
                    result[name + '_weight_l1_from_ordinary_mean_' + group_name] = selected_mean(
                        weight_l1, selection)
        result['inline_helper_logit_abs_max'] = (self.forward(batch) - logits['same_field']).abs().max()
        return {name: float(value.detach().item()) for name, value in result.items()}

