"""Fixed half-strength within-type count correction, prepared for host execution.

Learned modules and initialization order match the verified round-one sources.
For valid event i, weights are proportional to exp(score_i) / count(type_i)^alpha.
Only alpha=0.5 is submitted. Alpha=0/1 retain exact ancestor arithmetic paths
for label-free same-checkpoint diagnostic controls. No new inputs or parameters.
This is an attribution follow-up, with no novelty or performance claim.
"""
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'kind': 'din', 'count_alpha': 0.5}


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
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']) +
                   self.action(batch['history_type']) + self.recency(ages))
        return fields, history

    @staticmethod
    def _type_counts(batch):
        mask = batch['history_mask']
        counts = torch.zeros((mask.shape[0], 5), device=mask.device, dtype=torch.float32)
        return counts.scatter_add(1, batch['history_type'], mask.float())

    def _scores(self, fields, history):
        if self.kind == 'pooled':
            return torch.zeros(history.shape[:2], device=history.device, dtype=history.dtype)
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, query * history], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def _weights(self, scores, batch, alpha):
        mask = batch['history_mask']
        if self.kind == 'pooled':
            if alpha == 0.0:
                # Exact historical masked-mean arithmetic, without renormalizing.
                return mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
            counts = self._type_counts(batch)
            event_counts = counts.gather(1, batch['history_type']).clamp_min(1)
            if alpha == 1.0:
                # Exact full-correction endpoint; no extra power operation.
                weights = mask.float() / event_counts
            else:
                weights = mask.float() / event_counts.pow(alpha)
        else:
            if alpha != 0.0:
                counts = self._type_counts(batch)
                log_counts = counts.gather(1, batch['history_type']).clamp_min(1).log()
                if alpha == 1.0:
                    scores = scores - log_counts
                else:
                    scores = scores - alpha * log_counts
            scores = scores.masked_fill(~mask, -1e4)
            weights = scores.softmax(-1) * mask.float()
        # The supplied fixed-length padded batch has zero mass on empty rows.
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def forward(self, batch):
        if self.kind == 'static':
            fields = [embedding(batch['static'][:, i])
                      for i, embedding in enumerate(self.embeddings)]
            return self.output(torch.cat(fields, dim=-1)).squeeze(-1)
        fields, history = self._encode(batch)
        scores = self._scores(fields, history)
        weights = self._weights(scores, batch, self.config['count_alpha'])
        features = torch.cat([torch.cat(fields, dim=-1),
                              (weights.unsqueeze(-1) * history).sum(1)], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        # Descriptive only: host supplies the first 512 validation impressions.
        # No labels, outcome comparisons, training updates, or persistent state.
        fields, history = self._encode(batch)
        scores = self._scores(fields, history)
        ordinary = self._weights(scores, batch, 0.0)
        half = self._weights(scores, batch, 0.5)
        corrected = self._weights(scores, batch, 1.0)
        active = self._weights(scores, batch, self.config['count_alpha'])
        mask = batch['history_mask']
        counts = self._type_counts(batch)
        lengths = mask.sum(-1)
        nonempty = lengths > 0
        present_types = (counts[:, 1:] > 0).sum(-1)
        mixed = present_types > 1
        pv_share = counts[:, 1] / lengths.clamp_min(1)
        target_category = batch['static'][:, self.category, None]
        target_brand = batch['static'][:, self.brand, None]
        category_match = (mask & (target_category > 0) &
                          (batch['history_category'] > 0) &
                          (batch['history_category'] == target_category)).any(-1)
        brand_match = (mask & (target_brand > 0) &
                       (batch['history_brand'] > 0) &
                       (batch['history_brand'] == target_brand)).any(-1)
        absent = nonempty & ~category_match
        result = {
            'diagnostic_samples': lengths.new_tensor(mask.shape[0]),
            'empty_samples': (~nonempty).sum(),
            'single_type_samples': (present_types == 1).sum(),
            'mixed_type_samples': mixed.sum(),
            'pv_share_ge_90pct_mixed_samples': (mixed & (pv_share >= 0.9)).sum(),
            'visible_length_gt64_samples': (lengths > 64).sum(),
            'visible_category_absent_nonempty_samples': absent.sum(),
            'visible_category_absent_mixed_samples': (absent & mixed).sum(),
            'visible_category_absent_brand_present_samples': (absent & brand_match).sum(),
            'visible_category_absent_pv90_mixed_samples': (absent & mixed & (pv_share >= 0.9)).sum(),
            'target_category_zero_samples': (target_category.squeeze(-1) == 0).sum(),
            'target_brand_zero_samples': (target_brand.squeeze(-1) == 0).sum(),
            'masked_padding_mass_max': (active * (~mask).float()).sum(-1).max(),
            'weight_sum_error_max': (active.sum(-1) - nonempty.float()).abs().max(),
        }
        result['diagnostic_active_alpha'] = lengths.new_tensor(
            self.config['count_alpha'], dtype=torch.float32)
        result['known_target_category_absent_nonempty_samples'] = (
            absent & (target_category.squeeze(-1) > 0)).sum()
        result['known_target_brand_absent_nonempty_samples'] = (
            nonempty & ~brand_match & (target_brand.squeeze(-1) > 0)).sum()
        result['visible_category_zero_events'] = (
            mask & (batch['history_category'] == 0)).sum()
        result['visible_brand_zero_events'] = (
            mask & (batch['history_brand'] == 0)).sum()
        conditional_error = scores.new_zeros(())
        conditional_groups = lengths.new_zeros(())
        for action, name in enumerate(('pv', 'cart', 'fav', 'buy'), start=1):
            belongs = (batch['history_type'] == action) & mask
            present = counts[:, action] > 0
            ordinary_mass = (ordinary * belongs.float()).sum(-1)
            corrected_mass = (corrected * belongs.float()).sum(-1)
            half_mass = (half * belongs.float()).sum(-1)
            denom = present.sum().clamp_min(1)
            result[name + '_visible_events'] = counts[:, action].sum()
            result[name + '_present_samples'] = present.sum()
            result[name + '_ordinary_mass_mean_present'] = (ordinary_mass * present).sum() / denom
            result[name + '_corrected_mass_mean_present'] = (corrected_mass * present).sum() / denom
            result[name + '_half_mass_mean_present'] = (half_mass * present).sum() / denom
            per_row_score = ((scores * belongs.float()).sum(-1) /
                             counts[:, action].clamp_min(1))
            result[name + '_score_mean_present'] = (per_row_score * present).sum() / denom
            # Conditional weights should be unchanged within each type. Exclude
            # numerically vanishing type masses and disclose eligible groups.
            eligible = ((ordinary_mass > 1e-6) & (half_mass > 1e-6) &
                        (corrected_mass > 1e-6))
            conditional_ordinary = ordinary / ordinary_mass[:, None].clamp_min(1e-9)
            conditional_half = half / half_mass[:, None].clamp_min(1e-9)
            conditional_full = corrected / corrected_mass[:, None].clamp_min(1e-9)
            error = torch.maximum((conditional_ordinary - conditional_half).abs(),
                                  (conditional_ordinary - conditional_full).abs())
            row_error = (error * belongs.float()).max(-1).values
            conditional_error = torch.maximum(conditional_error,
                                               (row_error * eligible).max())
            conditional_groups = conditional_groups + eligible.sum()
        ordinary_summary = (ordinary.unsqueeze(-1) * history).sum(1)
        corrected_summary = (corrected.unsqueeze(-1) * history).sum(1)
        half_summary = (half.unsqueeze(-1) * history).sum(1)
        result['conditional_type_weight_change_max'] = conditional_error
        result['conditional_type_eligible_groups'] = conditional_groups
        result['empty_summary_l2_max'] = (
            active.unsqueeze(-1).mul(history).sum(1).square().sum(-1).sqrt() *
            (~nonempty)).max()
        single_row_change = torch.maximum((ordinary - half).abs(),
                                          (ordinary - corrected).abs()).max(-1).values
        result['single_type_weight_change_max'] = (
            single_row_change * (present_types == 1)).max()
        half_distance = (ordinary_summary - half_summary).square().sum(-1).sqrt()
        half_full_distance = (half_summary - corrected_summary).square().sum(-1).sqrt()
        distance = (ordinary_summary - corrected_summary).square().sum(-1).sqrt()
        for name, selection in [('nonempty', nonempty), ('mixed', mixed),
                                ('category_absent_mixed', absent & mixed),
                                ('single_type', present_types == 1)]:
            result['summary_l2_change_mean_' + name] = (
                (distance * selection).sum() / selection.sum().clamp_min(1))
            result['half_summary_l2_change_mean_' + name] = (
                (half_distance * selection).sum() / selection.sum().clamp_min(1))
            result['half_full_summary_l2_change_mean_' + name] = (
                (half_full_distance * selection).sum() / selection.sum().clamp_min(1))
        # Return plain finite scalar values, not tensors or per-impression arrays.
        return {name: float(value.detach().item()) for name, value in result.items()}
