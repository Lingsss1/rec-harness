"""Group-size-corrected event attention versus equal match-group mass.

Inherits the full paired_match plus separate-read hybrid. The only intervention
is w_i proportional to exp(s_i)/n_group(i), restoring relative group evidence.
A 2x2 query-content by mass-rule factorial shares all active parameter shapes.
No training/input/loss changes. No measured utility or priority claim.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': True, 'separate_reads': True, 'group_balanced': True, 'learned_weights': True, 'evidence_mass': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.evidence_mass = self.config['evidence_mass']
        self.group_balanced = self.config['group_balanced']
        self.learned_weights = self.config['learned_weights']
        self.separate_reads = self.config['separate_reads']
        if not self.group_balanced or not self.learned_weights:
            raise ValueError('This factorial fixes learned group-normalized retrieval')
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
        # Only appended retrieval group mass and query content change.
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
        # Preserve incumbent allocation order and all active parameter shapes.
        # All four variants allocate identical shapes and compute all allocations.
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

    def _density_weights(self, scores, membership):
        """Inverse-size event softmax; group mass is mean-exp score evidence.

        w_i = exp(s_i)/n_g / sum_h mean_{j in h} exp(s_j).
        Conditional weights inside each group remain its original softmax.
        Duplicating an entire group's empirical score distribution leaves this
        allocation unchanged. Duplicating a subset need not do so. All groups
        still use current-target match membership, even with context queries.
        """
        if scores.shape[1] == 0:
            return torch.zeros_like(scores)
        sizes = membership.sum(-1)
        event_sizes = (membership * sizes.unsqueeze(-1)).sum(1)
        valid = membership.bool().any(1)
        corrected = scores - event_sizes.clamp_min(1).log()
        # A fixed -1e4 padding sentinel would dominate more-negative valid
        # logits. Match the incumbent group softmax's finite-dtype masking.
        weights = corrected.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * valid.to(weights.dtype)
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def _read(self, category, brand, common, mask, query, membership):
        if mask.shape[1] == 0:
            zero = category.sum(1)
            weights = mask.float()
            return zero, weights, weights, weights, weights, weights, weights
        category_score = self._score(query[:, 0], category + common)
        brand_score = self._score(query[:, 1], brand + common)
        # Every cell executes all allocations. No added scorer or hidden width.
        global_c = self._weights(category_score, mask)
        global_b = self._weights(brand_score, mask)
        balanced_c = self._group_weights(category_score, membership)
        balanced_b = self._group_weights(brand_score, membership)
        density_c = self._density_weights(category_score, membership)
        density_b = self._density_weights(brand_score, membership)
        alpha = float(self.evidence_mass)
        wc = alpha * density_c + (1.0 - alpha) * balanced_c
        wb = alpha * density_b + (1.0 - alpha) * balanced_b
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
        """Label-free, candidate-supplied first-512 measurements, not proof.

        Offset/replication interventions fix keys, queries and memberships.
        They test allocation algebra, not end-to-end input invariance: actual
        event duplication would change original counts and context queries.
        """
        def maximum(x):
            return x.abs().max() if x.numel() else 0.0

        def rms(x):
            return x.square().mean().sqrt() if x.numel() else 0.0

        static, pools, counts, stats, _, semantic, sizes, global_mean = self._represent(batch)
        histogram, observed, expected, _ = self._histogram(batch, semantic, sizes)
        features, original, summary = self._features(batch)
        r = self._retrieval(batch)
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        query = target if self.target_query else context
        scores_c = self._score(query[:, 0], category + common)
        scores_b = self._score(query[:, 1], brand + common)
        density_c = self._density_weights(scores_c, semantic)
        density_b = self._density_weights(scores_b, semantic)
        bc, bb = r['balanced_c'], r['balanced_b']
        n = counts[0]
        active = (n > 0).float()
        denom = active.sum().clamp_min(1)
        occupied = (sizes > 0).float()
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        k = occupied.sum(-1)

        def masses(weights):
            return (semantic * weights[:, None, :]).sum(-1)

        def mean_exp_prior(scores):
            if scores.shape[1] == 0:
                return torch.zeros_like(sizes)
            masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            scaled = (masked - masked.max(-1, keepdim=True).values).exp() * mask.float()
            evidence = (semantic * scaled[:, None, :]).sum(-1) / sizes.clamp_min(1)
            return evidence / evidence.sum(-1, keepdim=True).clamp_min(1e-9)

        mc, mb = masses(r['wc']), masses(r['wb'])
        dc, db = masses(density_c), masses(density_b)
        # Cross multiplication avoids unstable division by tiny group masses.
        cond_c = semantic * (density_c[:, None, :] * prior.unsqueeze(-1)
                            - bc[:, None, :] * dc.unsqueeze(-1))
        cond_b = semantic * (density_b[:, None, :] * prior.unsqueeze(-1)
                            - bb[:, None, :] * db.unsqueeze(-1))
        equal_raw = (bc.unsqueeze(-1) * category + bb.unsqueeze(-1) * brand
                     + (0.5 * (bc + bb)).unsqueeze(-1) * common).sum(1)
        density_raw = (density_c.unsqueeze(-1) * category + density_b.unsqueeze(-1) * brand
                       + (0.5 * (density_c + density_b)).unsqueeze(-1) * common).sum(1)
        equal_summary = self._match_norm(equal_raw, pooled)
        density_summary = self._match_norm(density_raw, pooled)
        selected_expected = density_summary if self.evidence_mass else equal_summary
        target_raw = self._read(category, brand, common, mask, target, semantic)[0]
        context_raw = self._read(category, brand, common, mask, context, semantic)[0]
        # Fixed diagnostic offsets; adding a common constant cancels everywhere,
        # but different group offsets must survive only the evidence-mass rule.
        offsets = scores_c.new_tensor([0.0, 1.0, -1.0, 0.5])
        shifted = scores_c + (semantic * offsets[None, :, None]).sum(1)
        shifted_equal = self._group_weights(shifted, semantic) if mask.shape[1] else bc
        shifted_density = self._density_weights(shifted, semantic)
        offset_expected = dc * offsets.exp().unsqueeze(0)
        offset_expected = offset_expected / offset_expected.sum(-1, keepdim=True).clamp_min(1e-9)
        # Duplicate every event of group 0; exclude other copied positions.
        selector = semantic.new_tensor([1.0, 0.0, 0.0, 0.0])[None, :, None]
        duplicate_membership = torch.cat([semantic, semantic * selector], dim=-1)
        duplicate_scores = torch.cat([scores_c, scores_c], dim=-1)
        duplicate_weights = self._density_weights(duplicate_scores, duplicate_membership)
        length = mask.shape[1]
        recombined = duplicate_weights[:, :length] + duplicate_weights[:, length:]

        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(mask, batch[key], torch.zeros_like(batch[key]))
        padded_features = self._features(padded)[0]
        reverse = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[key] = batch[key].flip(1)
        reverse_features = self._features(reverse)[0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        empty_features, empty_original, empty_summary = self._features(empty)
        zero = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero[key] = batch[key][:, :0]
        zero_features, zero_original, zero_summary = self._features(zero)
        ref_norm = pooled.norm(dim=-1)
        selected_norm = summary.norm(dim=-1)
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'scorer_allocated_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'uses_target_query': float(self.target_query),
            'uses_evidence_group_mass': float(self.evidence_mass),
            'original_feature_width': float(original.shape[-1]),
            'appended_feature_width': float(summary.shape[-1]),
            'nonempty_fraction': active.mean(),
            'visible_event_count_mean': n.mean(),
            'occupied_groups_mean': k.mean(),
            'multi_group_fraction': (k > 1).float().mean(),
            'partition_coverage_error_max': maximum(semantic.sum(1) - mask.float()),
            'hist_group_size_error_max': maximum(observed.sum((-1, -2)) - sizes),
            'global_reconstruction_error_max': maximum((pools * sizes.unsqueeze(-1)).sum(1)
                                                       / n.clamp_min(1).unsqueeze(-1) - global_mean),
            'selected_weight_mass_error_max': maximum(torch.stack([r['wc'].sum(-1) - active,
                                                                  r['wb'].sum(-1) - active])),
            'balanced_group_mass_error_max': maximum(torch.stack([masses(bc) - prior, masses(bb) - prior])),
            'density_theory_mass_error_max': maximum(torch.stack([dc - mean_exp_prior(scores_c),
                                                                  db - mean_exp_prior(scores_b)])),
            'within_group_conditional_error_max': maximum(torch.stack([cond_c, cond_b])),
            'masked_weight_abs_max': maximum(torch.stack([r['wc'], r['wb']], 1)
                                              * (~mask)[:, None, :].float()),
            'selected_prior_l1_nonempty': ((0.5 * ((mc - prior).abs().sum(-1)
                                                                  + (mb - prior).abs().sum(-1))) * active).sum() / denom,
            'density_prior_l1_nonempty': ((0.5 * ((dc - prior).abs().sum(-1)
                                                                 + (db - prior).abs().sum(-1))) * active).sum() / denom,
            'field_attention_l1_nonempty': ((r['wc'] - r['wb']).abs().sum(-1) * active).sum() / denom,
            'allocation_summary_rms_fixed_scores': rms(density_summary - equal_summary),
            'allocation_logit_rms_fixed_weights': rms(self.output(torch.cat([original, density_summary], -1))
                                                      - self.output(torch.cat([original, equal_summary], -1))),
            'target_context_summary_rms_fixed_groups': rms(self._match_norm(target_raw, pooled)
                                                           - self._match_norm(context_raw, pooled)),
            'equal_group_offset_weight_error_max': maximum(shifted_equal - bc),
            'density_group_offset_mass_delta_l1': ((masses(shifted_density) - dc).abs().sum(-1)
                                                    * active).sum() / denom,
            'density_group_offset_theory_error_max': maximum(masses(shifted_density) - offset_expected),
            'group_replication_weight_error_max': maximum(recombined - density_c),
            'single_group_weight_error_max': maximum((density_c - bc) * (k == 1).unsqueeze(-1)),
            'selected_summary_identity_error_max': maximum(summary - selected_expected),
            'norm_error_max': maximum(selected_norm - ref_norm),
            'norm_fallback_fraction': (r['raw'].norm(dim=-1) <= 1e-6).float().mean(),
            'singleton_summary_error_max': maximum((summary - pooled) * (n == 1).unsqueeze(-1)),
            'empty_summary_abs_max': maximum(empty_summary),
            'empty_history_feature_abs_max': maximum(empty_original[:, static.shape[-1]:]),
            'zero_length_summary_abs_max': maximum(zero_summary),
            'zero_length_history_feature_abs_max': maximum(zero_original[:, static.shape[-1]:]),
            'empty_logits_finite': float(torch.isfinite(self.output(empty_features)).all()),
            'zero_length_logits_finite': float(torch.isfinite(self.output(zero_features)).all()),
            'padding_feature_error_max': maximum(features - padded_features),
            'padding_logit_error_max': maximum(self.output(features) - self.output(padded_features)),
            'permutation_feature_error_max': maximum(features - reverse_features),
            'permutation_logit_error_max': maximum(self.output(features) - self.output(reverse_features)),
            'logits_finite': float(torch.isfinite(self.output(features)).all()),
            'classifier_retrieval_weight_rms': rms(self.output[0].weight[:, original.shape[-1]:]),
        }
        for g, name in enumerate(('neither', 'category_only', 'brand_only', 'both')):
            result[name + '_selected_mass_mean'] = ((0.5 * (mc[:, g] + mb[:, g])) * active).sum() / denom
            result[name + '_density_mass_mean'] = ((0.5 * (dc[:, g] + db[:, g])) * active).sum() / denom
            result[name + '_equal_mass_mean'] = (prior[:, g] * active).sum() / denom
        return {key: float(value) for key, value in result.items()}
