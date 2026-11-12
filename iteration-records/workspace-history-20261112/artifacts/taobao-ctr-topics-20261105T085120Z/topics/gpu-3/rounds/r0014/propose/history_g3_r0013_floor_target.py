"""Within-match-group evidence floor, crossed with query source.

The complete paired_match representation, separate field scorers, equal group
mass, norm matching and classifier are inherited from balanced_target. The only
new retrieval operation mixes learned weights with uniform within-group weights:
    w = (1 - floor) * w_learned + floor * w_uniform_group.
At floor=0.5 every valid event has at least 1/(2*K*n_group) mass. The scorer stays
active in every cell; no new trainable parameters, loss or optimizer are added.
This is a prospective regularization contrast, with no novelty or utility claim.
Context queries are target independent; match partitions still use the target.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': True, 'separate_reads': True, 'group_balanced': True, 'learned_weights': True, 'uniform_floor': 0.5}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.uniform_floor = float(self.config['uniform_floor'])
        if self.uniform_floor not in (0.0, 0.5):
            raise ValueError('This study predeclares only floors 0 and 0.5')
        if not (self.config['group_balanced'] and self.config['learned_weights']):
            raise ValueError('This study fixes balanced learned retrieval')
        self.target_query = self.config['target_query']
        self.group_balanced = self.config['group_balanced']
        self.learned_weights = self.config['learned_weights']
        self.separate_reads = self.config['separate_reads']
        if not self.separate_reads:
            raise ValueError('This study fixes separate field reads')
        self.joint_mode = self.config['joint_mode']
        self.statistic_mode = self.config['statistic_mode']
        self.grouping_mode = self.config['grouping_mode']
        if self.joint_mode not in ('observed', 'independent'):
            raise ValueError('Unknown joint_mode')
        if self.statistic_mode not in ('count', 'rate'):
            raise ValueError('Unknown statistic_mode')
        if self.grouping_mode not in ('match', 'rank', 'occupied_global'):
            raise ValueError('Unknown grouping_mode')
        # Only the appended retrieval normalization, query and weighting change.
        # Original learned means and observed count tables remain fixed.
        if self.grouping_mode != 'match':
            raise ValueError('This study fixes learned pooling to match')
        self.histogram_grouping = self.config['histogram_grouping']
        self.histogram_pairing = self.config['histogram_pairing']
        if self.histogram_grouping not in ('match', 'rank'):
            raise ValueError('Unknown histogram_grouping')
        if self.histogram_pairing not in ('observed', 'independent'):
            raise ValueError('Unknown histogram_pairing')
        if (self.joint_mode, self.statistic_mode, self.histogram_grouping,
                self.histogram_pairing) != ('observed', 'count', 'match', 'observed'):
            raise ValueError('This study fixes the complete paired_match representation')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        # Preserve ancestor allocation order and activate its existing scorer.
        # All four variants allocate identical shapes and compute both weightings.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + 4 * dim + 4 + 64 + dim
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _match_counts(self, batch):
            mask = batch['history_mask']
            target_c = batch['static'][:, self.category:self.category + 1]
            target_b = batch['static'][:, self.brand:self.brand + 1]
            hc = batch['history_category']
            hb = batch['history_brand']
            # Zero represents padding/missing; missing-to-missing is never a match.
            cm = mask & (target_c > 0) & (hc > 0) & (hc == target_c)
            bm = mask & (target_b > 0) & (hb > 0) & (hb == target_b)
            n = mask.float().sum(-1)
            c = cm.float().sum(-1)
            b = bm.float().sum(-1)
            joint = (cm & bm).float().sum(-1)
            independent = c * b / n.clamp_min(1)
            return n, c, b, joint, independent

    def _statistics(self, counts):
            n, c, b, joint, independent = counts
            selected = joint if self.joint_mode == 'observed' else independent
            if self.statistic_mode == 'count':
                # Fixed scale from frozen history_limit=128, never from validation.
                # For fractional independence estimates the same log1p applies.
                return torch.log1p(torch.stack([n, c, b, selected], -1)) / math.log(129.0)
            denom = n.clamp_min(1)
            return torch.stack([
                (n > 0).float(), c / denom, b / denom, selected / denom
            ], -1)

    def _membership(self, batch):
        mask = batch['history_mask']
        target_c = batch['static'][:, self.category:self.category + 1]
        target_b = batch['static'][:, self.brand:self.brand + 1]
        hc, hb = batch['history_category'], batch['history_brand']
        cm = mask & (target_c > 0) & (hc > 0) & (hc == target_c)
        bm = mask & (target_b > 0) & (hb > 0) & (hb == target_b)
        semantic = torch.stack([
            mask & ~cm & ~bm, cm & ~bm, bm & ~cm, cm & bm
        ], dim=1).float()  # [B,4,L]
        sizes = semantic.sum(-1)
        if self.grouping_mode == 'rank':
            # Assign in supplied input order, counting only valid positions.
            # Prefix thresholds reproduce true cardinalities even with gaps.
            # This is a positional replacement, not a randomized control.
            rank = mask.long().cumsum(-1) - 1
            upper = sizes.long().cumsum(-1)
            lower = upper - sizes.long()
            membership = ((rank[:, None, :] >= lower[:, :, None]) &
                          (rank[:, None, :] < upper[:, :, None]) &
                          mask[:, None, :]).float()
        else:
            membership = semantic
        return membership, semantic, sizes

    def _represent(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        membership, semantic, sizes = self._membership(batch)
        mask = batch['history_mask']
        if mask.shape[1] == 0:
            pools = static.new_zeros((static.shape[0], 4, self.config['embedding_dim']))
            global_mean = pools[:, 0, :]
        else:
            ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
            history = (self.embeddings[self.category](batch['history_category']) +
                       self.embeddings[self.brand](batch['history_brand']) +
                       self.action(batch['history_type']) + self.recency(ages))
            # [B,4,L] @ [B,L,d] avoids an expanded [B,4,L,d] temporary.
            # Empty groups divide by one and have an exactly zero numerator.
            pools = torch.bmm(membership, history) / sizes.clamp_min(1).unsqueeze(-1)
            n = mask.float().sum(-1, keepdim=True).clamp_min(1)
            global_mean = (history * mask.unsqueeze(-1)).sum(1) / n
            if self.grouping_mode == 'occupied_global':
                pools = global_mean[:, None, :] * (sizes > 0).unsqueeze(-1)
        counts = self._match_counts(batch)
        statistics = self._statistics(counts)
        return static, pools, counts, statistics, membership, semantic, sizes, global_mean

    def _histogram(self, batch, semantic, sizes):
        """Return raw [B,group,action,age] tables and the selected readout.

        No learned values enter this calculation. Every variant computes BOTH
        tables, so the pairing switch changes information rather than a neural
        event-scoring path. The log transform is applied after replacement;
        only RAW margins, not sums of log-transformed cells, are preserved.
        """
        mask = batch['history_mask']
        if self.histogram_grouping == 'rank':
            rank = mask.long().cumsum(-1) - 1
            upper = sizes.long().cumsum(-1)
            lower = upper - sizes.long()
            membership = ((rank[:, None, :] >= lower[:, :, None]) &
                          (rank[:, None, :] < upper[:, :, None]) &
                          mask[:, None, :]).float()
        else:
            membership = semantic
        # Contract: valid events have type 1..4; masked positions have no mass.
        action_id = (batch['history_type'] - 1).clamp(0, 3)
        age = batch['history_age_seconds']
        age_id = ((age >= 3600.0).long() + (age >= 86400.0).long() +
                  (age >= 604800.0).long())
        cell = (4 * action_id + age_id)[:, None, :].expand(-1, 4, -1)
        observed = membership.new_zeros((mask.shape[0], 4, 16))
        observed = observed.scatter_add(2, cell, membership)
        observed = observed.reshape(mask.shape[0], 4, 4, 4)
        action_margin = observed.sum(-1)
        age_margin = observed.sum(-2)
        expected = (action_margin.unsqueeze(-1) * age_margin.unsqueeze(-2) /
                    sizes.clamp_min(1)[:, :, None, None])
        selected = observed if self.histogram_pairing == 'observed' else expected
        features = torch.log1p(selected.flatten(1)) / math.log(129.0)
        return features, observed, expected, membership

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

    def _score(self, query, keys):
        query = query[:, None, :].expand_as(keys)
        relation = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        return self.attention(relation).squeeze(-1)

    @staticmethod
    def _weights(scores, mask):
        weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    @staticmethod
    def _group_weights(scores, membership):
        """Each occupied match group receives 1/K mass; empty groups receive 0.

        A separate stable softmax is evaluated inside each group. This is a
        change to event-weight allocation, not the original group features.
        """
        valid = membership.bool()
        expanded = scores[:, None, :].expand_as(membership)
        # A finite minimum keeps an all-masked group finite before zeroing it.
        weights = expanded.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        occupied = valid.any(-1).to(weights.dtype)
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        return (weights * prior.unsqueeze(-1)).sum(1)

    @staticmethod
    def _uniform_group_weights(membership):
        sizes = membership.sum(-1, keepdim=True)
        occupied = (sizes.squeeze(-1) > 0).to(membership.dtype)
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        return (membership / sizes.clamp_min(1) * prior.unsqueeze(-1)).sum(1)

    def _read(self, category, brand, common, mask, query, membership):
        if mask.shape[1] == 0:
            zero = category.sum(1)
            weights = mask.float()
            return zero, weights, weights, weights, weights, weights, weights
        # Zero scaling is the uniform control. It preserves the scorer's
        # allocation and graph while giving its parameters exactly zero gradient.
        # Every cell computes global and grouped weights, including zero-scaled
        # backward paths. Measured host cost, not graph similarity, is decisive.
        scale = float(self.learned_weights)
        category_score = self._score(query[:, 0], category + common) * scale
        brand_score = self._score(query[:, 1], brand + common) * scale
        global_c = self._weights(category_score, mask)
        global_b = self._weights(brand_score, mask)
        balanced_c = self._group_weights(category_score, membership)
        balanced_b = self._group_weights(brand_score, membership)
        alpha = float(self.group_balanced)
        wc = alpha * balanced_c + (1.0 - alpha) * global_c
        wb = alpha * balanced_b + (1.0 - alpha) * global_b
        # Both cells execute the same arithmetic and allocate the same scorer.
        # The nonzero learned coefficient leaves its gradient path active.
        uniform_group = self._uniform_group_weights(membership)
        wc = (1.0 - self.uniform_floor) * wc + self.uniform_floor * uniform_group
        wb = (1.0 - self.uniform_floor) * wb + self.uniform_floor * uniform_group
        raw = (wc.unsqueeze(-1) * category + wb.unsqueeze(-1) * brand
               + (0.5 * (wc + wb)).unsqueeze(-1) * common).sum(1)
        return raw, wc, wb, global_c, global_b, balanced_c, balanced_b

    @staticmethod
    def _match_norm(raw, pooled):
        raw_norm = raw.norm(dim=-1, keepdim=True)
        reference_norm = pooled.norm(dim=-1, keepdim=True)
        matched = raw * (reference_norm / raw_norm.clamp_min(1e-6))
        # Degenerate vectors use the reference itself, including empty histories.
        # Both branches are finite before where; no divisions by zero occur.
        return torch.where(raw_norm > 1e-6, matched, pooled)


    def _retrieval(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        membership = self._membership(batch)[1]
        query = target if self.target_query else context
        raw, wc, wb, gc, gb, bc, bb = self._read(
            category, brand, common, mask, query, membership)
        return {'summary': self._match_norm(raw, pooled), 'pooled': pooled,
                'raw': raw, 'wc': wc, 'wb': wb, 'global_c': gc, 'global_b': gb,
                'balanced_c': bc, 'balanced_b': bb, 'membership': membership,
                'mask': mask, 'target': target, 'context': context}

    def _features(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        original = torch.cat([static, pools.flatten(1), statistics, histogram], dim=-1)
        summary = self._retrieval(batch)['summary']
        return torch.cat([original, summary], dim=-1), original, summary

    def forward(self, batch):
        return self.output(self._features(batch)[0]).squeeze(-1)

    def diagnostics(self, batch):
        # Candidate-supplied first-512 measurements, not causal mechanism proof.
        def maximum(x):
            return x.abs().max() if x.numel() else 0.0

        def entropy(w):
            return -(w * w.clamp_min(1e-9).log()).sum(-1)

        static, pools, counts, stats, _, semantic, sizes, global_mean = self._represent(batch)
        histogram, observed, expected, _ = self._histogram(batch, semantic, sizes)
        features, original, summary = self._features(batch)
        r = self._retrieval(batch)
        mask = r['mask']
        n, c, b, both, _ = counts
        active = mask.any(-1).float()
        denom = active.sum().clamp_min(1)
        occupied = (sizes > 0).float()
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        u = self._uniform_group_weights(semantic)
        wc, wb = r['wc'], r['wb']
        bc, bb = r['balanced_c'], r['balanced_b']
        fc, fb = 0.5 * (bc + u), 0.5 * (bb + u)
        selected_entropy = 0.5 * (entropy(wc) + entropy(wb))
        free_entropy = 0.5 * (entropy(bc) + entropy(bb))
        floor_entropy = 0.5 * (entropy(fc) + entropy(fb))
        _, category, brand, common, _, _, target, context = self._encode(batch)
        fixed_floor_raw = (fc.unsqueeze(-1) * category +
                           fb.unsqueeze(-1) * brand +
                           (0.5 * (fc + fb)).unsqueeze(-1) * common).sum(1)
        fixed_free_raw = (bc.unsqueeze(-1) * category +
                          bb.unsqueeze(-1) * brand +
                          (0.5 * (bc + bb)).unsqueeze(-1) * common).sum(1)
        floor_summary = self._match_norm(fixed_floor_raw, r['pooled'])
        free_summary = self._match_norm(fixed_free_raw, r['pooled'])
        target_raw = self._read(category, brand, common, mask, target, semantic)[0]
        context_raw = self._read(category, brand, common, mask, context, semantic)[0]
        target_summary = self._match_norm(target_raw, r['pooled'])
        context_summary = self._match_norm(context_raw, r['pooled'])
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(mask, batch[key], torch.zeros_like(batch[key]))
        reversed_batch = dict(batch)
        zero_length = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reversed_batch[key] = batch[key].flip(1)
            zero_length[key] = batch[key][:, :0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        empty_features, empty_original, empty_summary = self._features(empty)
        zero_features, zero_original, zero_summary = self._features(zero_length)
        padded_features = self._features(padded)[0]
        reversed_features = self._features(reversed_batch)[0]
        selected_group_mass = torch.stack([(semantic * wc[:, None, :]).sum(-1),
                                          (semantic * wb[:, None, :]).sum(-1)])
        expected_c = (1.0-self.uniform_floor)*bc+self.uniform_floor*u
        expected_b = (1.0-self.uniform_floor)*bb+self.uniform_floor*u
        # An exact same-score contraction identity. Norm matching is nonlinear,
        # so no corresponding contraction is asserted for summaries or logits.
        contraction = torch.stack([wc-u-(1.0-self.uniform_floor)*(bc-u),
                                   wb-u-(1.0-self.uniform_floor)*(bb-u)])
        floor_features = torch.cat([original, floor_summary], -1)
        free_features = torch.cat([original, free_summary], -1)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'scorer_allocated_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'scorer_active_coefficient': 1.0-self.uniform_floor,
            'uses_target_query': float(self.target_query),
            'uniform_floor': self.uniform_floor,
            'original_feature_width': float(original.shape[-1]),
            'appended_feature_width': float(summary.shape[-1]),
            'nonempty_fraction': active.mean(),
            'visible_event_count_mean': n.mean(),
            'occupied_groups_mean': occupied.sum(-1).mean(),
            'partition_coverage_error_max': maximum(semantic.sum(1)-mask.float()),
            'group_count_identity_error_max': maximum(sizes-torch.stack([n-c-b+both,c-both,b-both,both],-1)),
            'global_reconstruction_error_max': maximum((pools*sizes.unsqueeze(-1)).sum(1)/n.clamp_min(1).unsqueeze(-1)-global_mean),
            'hist_group_size_error_max': maximum(observed.sum((-1,-2))-sizes),
            'hist_action_margin_error_max': maximum(observed.sum(-1)-expected.sum(-1)),
            'hist_age_margin_error_max': maximum(observed.sum(-2)-expected.sum(-2)),
            'selected_weight_mass_error_max': maximum(torch.stack([wc.sum(-1)-active,wb.sum(-1)-active])),
            'selected_group_mass_error_max': maximum(selected_group_mass-prior.unsqueeze(0)),
            'masked_weight_abs_max': maximum(torch.stack([wc,wb],1)*(~mask)[:,None,:].float()),
            'floor_lower_bound_violation_max': maximum(torch.stack([(self.uniform_floor*u-wc).clamp_min(0),(self.uniform_floor*u-wb).clamp_min(0)])),
            'selected_mixture_identity_error_max': maximum(torch.stack([wc-expected_c,wb-expected_b])),
            'fixed_score_contraction_error_max': maximum(contraction),
            'fixed_score_floor_entropy_violation_max': maximum((free_entropy-floor_entropy).clamp_min(0)),
            'selected_entropy_nonempty': (selected_entropy*active).sum()/denom,
            'free_entropy_nonempty_same_scores': (free_entropy*active).sum()/denom,
            'floor_entropy_nonempty_same_scores': (floor_entropy*active).sum()/denom,
            'uniform_entropy_nonempty': (entropy(u)*active).sum()/denom,
            'selected_vs_uniform_l1_nonempty': (0.5*((wc-u).abs().sum(-1)+(wb-u).abs().sum(-1))*active).sum()/denom,
            'free_vs_uniform_l1_nonempty_same_scores': (0.5*((bc-u).abs().sum(-1)+(bb-u).abs().sum(-1))*active).sum()/denom,
            'field_attention_l1_nonempty': ((wc-wb).abs().sum(-1)*active).sum()/denom,
            'floor_free_summary_rms_fixed_scores': (floor_summary-free_summary).square().mean().sqrt(),
            'floor_free_logit_rms_fixed_scores': (self.output(floor_features)-self.output(free_features)).square().mean().sqrt(),
            'query_replacement_summary_rms_fixed_groups': (target_summary-context_summary).square().mean().sqrt(),
            'norm_absolute_error_max': maximum(summary.norm(dim=-1)-r['pooled'].norm(dim=-1)),
            'norm_fallback_fraction': (r['raw'].norm(dim=-1)<=1e-6).float().mean(),
            'appended_feature_rms': summary.square().mean().sqrt(),
            'singleton_summary_error_max': maximum((summary-r['pooled'])*(n==1).unsqueeze(-1)),
            'empty_summary_abs_max': maximum(empty_summary),
            'empty_history_feature_abs_max': maximum(empty_original[:,static.shape[-1]:]),
            'zero_length_summary_abs_max': maximum(zero_summary),
            'zero_length_history_feature_abs_max': maximum(zero_original[:,static.shape[-1]:]),
            'empty_logits_finite': float(torch.isfinite(self.output(empty_features)).all()),
            'zero_length_logits_finite': float(torch.isfinite(self.output(zero_features)).all()),
            'padding_feature_error_max': maximum(features-padded_features),
            'padding_logit_error_max': maximum(self.output(features)-self.output(padded_features)),
            'permutation_feature_error_max': maximum(features-reversed_features),
            'permutation_logit_error_max': maximum(self.output(features)-self.output(reversed_features)),
            'logits_finite': float(torch.isfinite(self.output(features)).all()),
            'classifier_retrieval_weight_rms': self.output[0].weight[:,original.shape[-1]:].square().mean().sqrt(),
        }
        return {key: float(value) for key, value in result.items()}
