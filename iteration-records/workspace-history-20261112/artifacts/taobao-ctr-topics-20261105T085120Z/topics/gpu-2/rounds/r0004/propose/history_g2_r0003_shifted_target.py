"""Count-prior test for field-aligned target retrieval, without extra parameters.

Inherit g2-r0001-target_separate's parameters and readout. Only change the
softmax base measure: w_i is proportional to exp(score_i)/n_action(i).
The shifted control preserves correction histograms; real event values,
action embeddings, masks and order never move. No reproduction/priority claim.
"""
import torch
from torch import nn

CONFIG = {
    'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_query': True, 'count_prior': 'shifted',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.count_prior = self.config['count_prior']
        if self.count_prior not in ('action', 'shifted', 'none'):
            raise ValueError('count_prior must be action, shifted or none')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Preserve ancestor parameter construction order for paired seeds.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _encode(self, batch):
        fields = [e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)]
        mask = batch['history_mask']
        ages = (torch.log2(batch['history_age_seconds'].clamp_min(0) + 1).long()
                + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        common = self.action(batch['history_type']) + self.recency(ages)
        uniform = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        context = torch.stack([
            (uniform.unsqueeze(-1) * category).sum(1),
            (uniform.unsqueeze(-1) * brand).sum(1),
        ], dim=1)
        target = torch.stack([fields[self.category], fields[self.brand]], dim=1)
        pooled = (uniform.unsqueeze(-1) * (category + brand + common)).sum(1)
        return fields, category, brand, common, mask, pooled, target, context

    @staticmethod
    def _priors(types, mask):
        """Use validity ranks to handle arbitrary padding, including empty rows.

        Packing/rotation affects only the correction labels, never real event
        embeddings. All variants calculate both priors for matched bookkeeping.
        """
        valid = mask.long()
        counts = torch.zeros((types.shape[0], 5), device=types.device,
                             dtype=torch.float32)
        counts.scatter_add_(1, types, mask.float())
        rank = (valid.cumsum(-1) - 1).clamp_min(0)
        packed = torch.zeros_like(types)
        packed.scatter_add_(1, rank, types * valid)
        length = valid.sum(-1, keepdim=True)
        shifted_rank = (rank + length // 2) % length.clamp_min(1)
        shifted = packed.gather(1, shifted_rank) * valid
        ap = counts.gather(1, types).clamp_min(1).log() * mask.float()
        sp = counts.gather(1, shifted).clamp_min(1).log() * mask.float()
        return ap, sp, counts, shifted

    def _score(self, query, keys):
        query = query[:, None, :].expand_as(keys)
        relation = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        return self.attention(relation).squeeze(-1)

    @staticmethod
    def _weights(scores, mask):
        weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def _chosen_prior(self, ap, sp):
        if self.count_prior == 'action':
            return ap
        if self.count_prior == 'shifted':
            return sp
        return None

    def _read(self, category, brand, common, mask, query, prior):
        sc = self._score(query[:, 0], category + common)
        sb = self._score(query[:, 1], brand + common)
        # Replay keeps ancestor arithmetic, with no subtraction.
        wc = self._weights(sc if prior is None else sc - prior, mask)
        wb = self._weights(sb if prior is None else sb - prior, mask)
        aligned = (
            wc.unsqueeze(-1) * category
            + wb.unsqueeze(-1) * brand
            + (0.5 * (wc + wb)).unsqueeze(-1) * common
        ).sum(1)
        return aligned, wc, wb, sc, sb

    @staticmethod
    def _match_norm(raw, pooled):
        raw_norm = raw.norm(dim=-1, keepdim=True)
        reference_norm = pooled.norm(dim=-1, keepdim=True)
        matched = raw * (reference_norm / raw_norm.clamp_min(1e-6))
        return torch.where(raw_norm > 1e-6, matched, pooled)

    def forward(self, batch):
        fields, category, brand, common, mask, pooled, target, context = self._encode(batch)
        ap, sp, _, _ = self._priors(batch['history_type'], mask)
        prior = self._chosen_prior(ap, sp)
        query = target if self.target_query else context
        raw, _, _, _, _ = self._read(category, brand, common, mask, query, prior)
        summary = self._match_norm(raw, pooled)
        return self.output(torch.cat([torch.cat(fields, dim=-1), summary], dim=-1)).squeeze(-1)

    def diagnostics(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        types = batch['history_type']
        ap, sp, counts, shifted = self._priors(types, mask)
        prior = self._chosen_prior(ap, sp)
        query = target if self.target_query else context
        raw, wc, wb, sc, sb = self._read(category, brand, common, mask, query, prior)
        _, cc, cb, _, _ = self._read(category, brand, common, mask, context, prior)
        gc, gb = self._weights(sc, mask), self._weights(sb, mask)
        selected = self._match_norm(raw, pooled)
        active = mask.any(-1).float()
        present = (counts[:, 1:] > 0).float()
        groups = present.sum(-1)
        mixed = (groups > 1).float()
        single = (groups == 1).float()

        def mean(values, condition=active):
            return float((values * condition).sum() / condition.sum().clamp_min(1))

        def entropy(w):
            return -(w * w.clamp_min(1e-9).log()).sum(-1)

        reference_norm = pooled.norm(dim=-1)
        norm_error = (selected.norm(dim=-1) - reference_norm).abs()
        zero_weight = self._weights(-ap, mask)
        zero_mass = torch.zeros_like(counts)
        zero_mass.scatter_add_(1, types, zero_weight)
        ideal_mass = present / groups[:, None].clamp_min(1)
        shifted_counts = torch.zeros_like(counts)
        shifted_counts.scatter_add_(1, shifted, mask.float())
        attention_weight = 0.5 * (wc + wb)
        global_weight = 0.5 * (gc + gb)
        result = {
            'nonempty_fraction': float(active.mean()),
            'mixed_action_fraction': float(mixed.mean()),
            'supplied_event_count_mean': float(mask.float().sum(-1).mean()),
            'present_action_count_nonempty': mean(groups),
            'largest_action_count_share_nonempty': mean(counts[:, 1:].max(-1).values
                                                       / counts.sum(-1).clamp_min(1)),
            'reference_norm_mean': float(reference_norm.mean()),
            'norm_absolute_error_max': float(norm_error.max()),
            'norm_relative_error_max': float((norm_error / reference_norm.clamp_min(1e-6)).max()),
            'norm_fallback_fraction': float((raw.norm(dim=-1) <= 1e-6).float().mean()),
            'action_equal_score_mass_error_max': float((zero_mass[:, 1:] - ideal_mass).abs().max()),
            'shifted_count_histogram_error_max': float((shifted_counts - counts).abs().max()),
            'prior_histogram_sum_error_max': float((ap.sum(-1) - sp.sum(-1)).abs().max()),
            'shifted_prior_difference_per_event_mixed': mean((ap - sp).abs().sum(-1)
                                                              / mask.sum(-1).clamp_min(1), mixed),
            'shifted_label_changed_fraction_mixed': mean(((types != shifted) & mask).float().sum(-1)
                                                         / mask.sum(-1).clamp_min(1), mixed),
            'attention_l1_vs_global_nonempty': mean(0.5 * ((wc - gc).abs().sum(-1)
                                                         + (wb - gb).abs().sum(-1))),
            'attention_l1_vs_global_single_action': mean(0.5 * ((wc - gc).abs().sum(-1)
                                                             + (wb - gb).abs().sum(-1)), single),
            'attention_l1_vs_context_nonempty': mean(0.5 * ((wc - cc).abs().sum(-1)
                                                          + (wb - cb).abs().sum(-1))),
            'field_attention_l1_nonempty': mean((wc - wb).abs().sum(-1)),
            'selected_attention_entropy_nonempty': mean(0.5 * (entropy(wc) + entropy(wb))),
            'global_attention_entropy_nonempty': mean(0.5 * (entropy(gc) + entropy(gb))),
            'padding_attention_mass_max': float((attention_weight * (~mask).float()).sum(-1).max()),
            'uses_target_query': float(self.target_query),
            'uses_action_prior': float(self.count_prior == 'action'),
            'uses_shifted_prior': float(self.count_prior == 'shifted'),
        }
        for action_id in range(1, 5):
            member = ((types == action_id) & mask).float()
            result['action_%d_count_nonempty' % action_id] = mean(counts[:, action_id])
            result['action_%d_selected_mass_nonempty' % action_id] = mean((attention_weight * member).sum(-1))
            result['action_%d_global_mass_nonempty' % action_id] = mean((global_weight * member).sum(-1))
        # Bucket equality is a proxy; collisions are not semantic matches.
        tc = batch['static'][:, self.category, None]
        tb = batch['static'][:, self.brand, None]
        cm = (batch['history_category'] == tc) & (tc != 0) & mask
        bm = (batch['history_brand'] == tb) & (tb != 0) & mask
        result['category_bucket_match_mass_nonempty'] = mean((wc * cm.float()).sum(-1))
        result['brand_bucket_match_mass_nonempty'] = mean((wb * bm.float()).sum(-1))
        return result
