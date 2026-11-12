"""Groupwise versus global-only norm matching, crossed with readout query.

All original paired_match features and active parameter allocation are retained.
Only the appended retrieval geometry changes: optionally constrain each group's
contribution norm before mixing, then keep the inherited global norm constraint.
Context queries use history only; semantic match groups still use the target.
No training/input/scorer changes, measured gain or verified novelty claim.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': False, 'separate_reads': True, 'group_balanced': True, 'learned_weights': True, 'local_group_norm': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.local_group_norm = self.config['local_group_norm']
        self.group_balanced = self.config['group_balanced']
        self.learned_weights = self.config['learned_weights']
        self.separate_reads = self.config['separate_reads']
        if not self.group_balanced or not self.learned_weights:
            raise ValueError('This study fixes balanced learned event weights')
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
        # Only pre-mix group normalization and the readout query change.
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
        # Preserve ancestor parameter shapes and allocation order exactly.
        # All cells compute both geometry paths; every parameter stays active.
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


    def _group_geometry(self, category, brand, common, membership, wc, wb):
        """Contributions include inherited 1/K attention mass, K occupied groups.

        Match each contribution to its own unweighted-mean contribution norm.
        No new learned parameters; raw event weights and membership stay fixed.
        Raw group mass remains 1/K, but this radial operation is NOT attention
        weight renormalization. Final global output norm is constrained as well.
        """
        sizes = membership.sum(-1)
        occupied = (sizes > 0).to(category.dtype)
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        if category.shape[1] == 0:
            group_raw = category.new_zeros((category.shape[0], 4, category.shape[-1]))
            reference = torch.zeros_like(group_raw)
        else:
            weighted = (wc.unsqueeze(-1) * category + wb.unsqueeze(-1) * brand
                        + (0.5 * (wc + wb)).unsqueeze(-1) * common)
            group_raw = torch.bmm(membership, weighted)
            means = (torch.bmm(membership, category + brand + common)
                     / sizes.clamp_min(1).unsqueeze(-1))
            reference = means * prior.unsqueeze(-1)
        group_matched = self._match_norm(group_raw, reference)
        return group_raw, reference, group_matched, prior

    def _retrieval(self, batch, use_target=None):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        membership = self._membership(batch)[1]
        use_target = self.target_query if use_target is None else use_target
        query = target if use_target else context
        raw, wc, wb, gc, gb, bc, bb = self._read(
            category, brand, common, mask, query, membership)
        group_raw, reference, group_matched, prior = self._group_geometry(
            category, brand, common, membership, wc, wb)
        local_raw = group_matched.sum(1)
        # Both finite branches remain in the graph in every factorial cell.
        # At alpha=0 the inherited raw reduction is selected exactly; added
        # branch gradients are zero, while all inherited parameters stay active.
        alpha = float(self.local_group_norm)
        selected_raw = alpha * local_raw + (1.0 - alpha) * raw
        return {'summary': self._match_norm(selected_raw, pooled), 'pooled': pooled,
                'raw': raw, 'local_raw': local_raw, 'selected_raw': selected_raw,
                'group_raw': group_raw, 'group_reference': reference,
                'group_matched': group_matched, 'prior': prior,
                'wc': wc, 'wb': wb, 'global_c': gc, 'global_b': gb,
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
        """Label-free candidate-supplied checks on host's first 512 impressions.

        These measurements check manipulation and numerical identities, and do
        not establish population utility or a causal explanation of AUC changes.
        """
        def maximum(x):
            return x.abs().max() if x.numel() else 0.0

        def masked_mean(x, valid):
            valid = valid.to(x.dtype)
            return (x * valid).sum() / valid.sum().clamp_min(1)

        static, pools, counts, stats, _, semantic, sizes, global_mean = self._represent(batch)
        _, observed, _, _ = self._histogram(batch, semantic, sizes)
        features, original, summary = self._features(batch)
        r = self._retrieval(batch)
        mask = r['mask']
        n = counts[0]
        active = mask.any(-1)
        occupied = sizes > 0
        k = occupied.sum(-1)
        multi = k > 1
        wc, wb = r['wc'], r['wb']
        mass_c = (semantic * wc[:, None, :]).sum(-1)
        mass_b = (semantic * wb[:, None, :]).sum(-1)
        group_norm = r['group_raw'].norm(dim=-1)
        reference_norm = r['group_reference'].norm(dim=-1)
        matched_norm = r['group_matched'].norm(dim=-1)
        eligible = occupied & (group_norm > 1e-6) & (reference_norm > 1e-6)
        ratios = group_norm / reference_norm.clamp_min(1e-6)
        ratio_mean = masked_mean(ratios, eligible)
        ratio_sd = masked_mean((ratios - ratio_mean).square(), eligible).sqrt()
        eligible_count = eligible.sum(-1).clamp_min(1)
        row_mean = (ratios * eligible).sum(-1) / eligible_count
        row_sd = (((ratios - row_mean[:, None]).square() * eligible).sum(-1)
                  / eligible_count).sqrt()
        global_summary = self._match_norm(r['raw'], r['pooled'])
        local_summary = self._match_norm(r['local_raw'], r['pooled'])
        global_features = torch.cat([original, global_summary], -1)
        local_features = torch.cat([original, local_summary], -1)
        target_summary = self._retrieval(batch, use_target=True)['summary']
        context_summary = self._retrieval(batch, use_target=False)['summary']
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(mask, batch[key], torch.zeros_like(batch[key]))
        padded_features = self._features(padded)[0]
        reverse = dict(batch)
        history_keys = ('history_category', 'history_brand', 'history_type',
                        'history_age_seconds', 'history_mask')
        for key in history_keys:
            reverse[key] = batch[key].flip(1)
        reversed_features = self._features(reverse)[0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        empty_features, empty_original, empty_summary = self._features(empty)
        zero_length = dict(batch)
        for key in history_keys:
            zero_length[key] = batch[key][:, :0]
        zero_features, zero_original, zero_summary = self._features(zero_length)
        safe_single = ((k == 1) & (r['raw'].norm(dim=-1) > 1e-6)
                       & (r['local_raw'].norm(dim=-1) > 1e-6)
                       & ((group_norm > 1e-6) | ~occupied).all(-1))
        expected_summary = local_summary if self.local_group_norm else global_summary
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'scorer_allocated_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'uses_target_query': float(self.target_query),
            'uses_pre_mix_group_norm': float(self.local_group_norm),
            'original_feature_width': float(original.shape[-1]),
            'appended_feature_width': float(summary.shape[-1]),
            'nonempty_fraction': active.float().mean(),
            'visible_event_count_mean': n.mean(),
            'occupied_groups_mean': k.float().mean(),
            'multi_group_fraction': multi.float().mean(),
            'partition_coverage_error_max': maximum(semantic.sum(1) - mask.float()),
            'hist_group_size_error_max': maximum(observed.sum((-1, -2)) - sizes),
            'global_reconstruction_error_max': maximum((pools*sizes.unsqueeze(-1)).sum(1)/n.clamp_min(1).unsqueeze(-1)-global_mean),
            'group_raw_reconstruction_error_max': maximum(r['group_raw'].sum(1)-r['raw']),
            'weight_mass_error_max': maximum(torch.stack([wc.sum(-1)-active.float(),wb.sum(-1)-active.float()])),
            'balanced_group_mass_error_max': maximum(torch.stack([mass_c-r['prior'],mass_b-r['prior']])),
            'masked_weight_abs_max': maximum(torch.stack([wc,wb],1)*(~mask)[:,None,:].float()),
            'field_attention_l1_nonempty': masked_mean((wc-wb).abs().sum(-1),active),
            'group_radial_eligible_fraction': eligible.float().sum()/occupied.float().sum().clamp_min(1),
            'group_radial_ratio_mean': ratio_mean,
            'group_radial_ratio_sd': ratio_sd,
            'within_row_group_ratio_sd_mean': masked_mean(row_sd,eligible.sum(-1)>1),
            'group_norm_error_max': maximum(matched_norm-reference_norm),
            'occupied_group_fallback_fraction': masked_mean((group_norm<=1e-6).float(),occupied),
            'empty_group_matched_abs_max': maximum(r['group_matched']*(~occupied).unsqueeze(-1)),
            'global_final_norm_error_max': maximum(global_summary.norm(dim=-1)-r['pooled'].norm(dim=-1)),
            'local_final_norm_error_max': maximum(local_summary.norm(dim=-1)-r['pooled'].norm(dim=-1)),
            'selected_final_norm_error_max': maximum(summary.norm(dim=-1)-r['pooled'].norm(dim=-1)),
            'local_global_summary_rms_multigroup': masked_mean((local_summary-global_summary).square().mean(-1),multi).sqrt(),
            'local_global_logit_delta_rms_fixed_weights': (self.output(local_features)-self.output(global_features)).square().mean().sqrt(),
            'target_context_summary_rms_fixed_groups': (target_summary-context_summary).square().mean().sqrt(),
            'selected_summary_identity_error_max': maximum(summary-expected_summary),
            'single_group_safe_fraction': safe_single.float().mean(),
            'single_group_summary_error_max': maximum((local_summary-global_summary)*safe_single.unsqueeze(-1)),
            'singleton_summary_error_max': maximum((summary-r['pooled'])*(n==1).unsqueeze(-1)),
            'global_mix_fallback_nonempty_fraction': masked_mean((r['raw'].norm(dim=-1)<=1e-6).float(),active),
            'local_mix_fallback_nonempty_fraction': masked_mean((r['local_raw'].norm(dim=-1)<=1e-6).float(),active),
            'empty_summary_abs_max': maximum(empty_summary),
            'empty_history_feature_abs_max': maximum(empty_original[:,static.shape[-1]:]),
            'zero_length_summary_abs_max': maximum(zero_summary),
            'zero_length_history_feature_abs_max': maximum(zero_original[:,static.shape[-1]:]),
            'zero_length_logits_finite': float(torch.isfinite(self.output(zero_features)).all()),
            'empty_logits_finite': float(torch.isfinite(self.output(empty_features)).all()),
            'padding_feature_error_max': maximum(features-padded_features),
            'padding_logit_error_max': maximum(self.output(features)-self.output(padded_features)),
            'permutation_feature_error_max': maximum(features-reversed_features),
            'permutation_logit_error_max': maximum(self.output(features)-self.output(reversed_features)),
            'logits_finite': float(torch.isfinite(self.output(features)).all()),
            'classifier_retrieval_weight_rms': self.output[0].weight[:,original.shape[-1]:].square().mean().sqrt(),
        }
        return {key: float(value) for key, value in result.items()}
