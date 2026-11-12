"""Signed within-group history deviations crossed with target/context queries.

All original group pools and histograms remain available. Both aggregation
paths execute in every cell; no extra learned parameters or input information.
Zero-safe norm matching is shared by candidate and raw controls.
No utility, causal mechanism, source reproduction or priority claim yet.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': True, 'separate_reads': True, 'group_balanced': True, 'learned_weights': True, 'center_readout': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
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


    @staticmethod
    def _value_sum(wc, wb, category, brand, common):
        return (wc.unsqueeze(-1) * category + wb.unsqueeze(-1) * brand
                + (0.5 * (wc + wb)).unsqueeze(-1) * common).sum(1)

    @staticmethod
    def _zero_norm(raw, pooled):
        # A zero deviation means no selective evidence. The same finite guard
        # is used by raw controls; compare it to the inherited fallback below.
        norm = raw.norm(dim=-1, keepdim=True)
        matched = raw * (pooled.norm(dim=-1, keepdim=True) / norm.clamp_min(1e-6))
        return torch.where(norm > 1e-6, matched, torch.zeros_like(raw))

    def _choose_summary(self, raw, centered, pooled):
        selected = centered if self.config['center_readout'] else raw
        return self._zero_norm(selected, pooled)

    def _retrieval(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        membership = self._membership(batch)[1]
        query = target if self.target_query else context
        raw, wc, wb, gc, gb, bc, bb = self._read(
            category, brand, common, mask, query, membership)
        # Group softmax at zero scores gives exactly the neutral allocation
        # under the same numerical reduction as the learned weights.
        neutral_w = (self._group_weights(torch.zeros_like(wc), membership)
                     if mask.shape[1] else wc)
        neutral = self._value_sum(neutral_w, neutral_w, category, brand, common)
        # Compute signed coefficients before reducing values. Gradients pass
        # through both terms; no stop-gradient or extra loss is introduced.
        dc, db = wc - neutral_w, wb - neutral_w
        centered = self._value_sum(dc, db, category, brand, common)
        return {'summary': self._choose_summary(raw, centered, pooled),
                'raw': raw, 'centered': centered, 'neutral': neutral,
                'neutral_w': neutral_w, 'dc': dc, 'db': db,
                'pooled': pooled, 'wc': wc, 'wb': wb,
                'global_c': gc, 'global_b': gb, 'balanced_c': bc, 'balanced_b': bb,
                'membership': membership, 'mask': mask,
                'target': target, 'context': context}

    def _features(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        original = torch.cat([static, pools.flatten(1), statistics, histogram], dim=-1)
        summary = self._retrieval(batch)['summary']
        return torch.cat([original, summary], dim=-1), original, summary

    def forward(self, batch):
        return self.output(self._features(batch)[0]).squeeze(-1)

    def diagnostics(self, batch):
        # Candidate-supplied, label-free first512 checks; not population evidence.
        def maximum(x):
            return x.abs().max() if x.numel() else 0.0

        def rms(x):
            return x.square().mean().sqrt() if x.numel() else 0.0

        _, pools, counts, _, _, semantic, sizes, _ = self._represent(batch)
        _, observed, _, _ = self._histogram(batch, semantic, sizes)
        features, original, summary = self._features(batch)
        r = self._retrieval(batch)
        mask, n = r['mask'], counts[0]
        active = mask.any(-1).float()
        denom = active.sum().clamp_min(1)
        occupied = (sizes > 0).float()
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        neutral_from_pools = (pools * prior.unsqueeze(-1)).sum(1)
        signed_c = (semantic * r['dc'][:, None, :]).sum(-1)
        signed_b = (semantic * r['db'][:, None, :]).sum(-1)
        raw_norm = r['raw'].norm(dim=-1)
        centered_norm = r['centered'].norm(dim=-1)
        selected_raw = r['centered'] if self.config['center_readout'] else r['raw']
        eligible = (selected_raw.norm(dim=-1) > 1e-6).float()
        ref_norm = r['pooled'].norm(dim=-1)
        raw_summary = self._zero_norm(r['raw'], r['pooled'])
        centered_summary = self._zero_norm(r['centered'], r['pooled'])
        ancestor_summary = self._match_norm(r['raw'], r['pooled'])
        _, category, brand, common, _, _, target, context = self._encode(batch)
        # Replace query only; keep target-dependent group membership fixed.
        tr = self._read(category, brand, common, mask, target, semantic)
        cr = self._read(category, brand, common, mask, context, semantic)
        td = self._value_sum(tr[1]-r['neutral_w'], tr[2]-r['neutral_w'], category, brand, common)
        cd = self._value_sum(cr[1]-r['neutral_w'], cr[2]-r['neutral_w'], category, brand, common)
        ts = self._choose_summary(tr[0], td, r['pooled'])
        cs = self._choose_summary(cr[0], cd, r['pooled'])
        # Every occupied singleton group has zero within-group contrast.
        singleton_groups = ((sizes <= 1).all(-1) & mask.any(-1)).float()
        # Fixed-score group offsets test the signed coefficients, not the
        # normalized summary (whose reference norm can change with offsets).
        group_id = torch.arange(1, 5, dtype=category.dtype, device=category.device)
        offset = (semantic * group_id[None, :, None]).sum(1).unsqueeze(-1)
        shifted = self._value_sum(r['dc'], r['db'], category+offset,
                                  brand+2*offset, common+3*offset)
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(mask, batch[key], torch.zeros_like(batch[key]))
        pf = self._features(padded)[0]
        reverse = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            reverse[key] = batch[key].flip(1)
        rf = self._features(reverse)[0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        ef, eo, es = self._features(empty)
        zero = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero[key] = batch[key][:, :0]
        zf, zo, zs = self._features(zero)
        static_width = len(self.embeddings) * self.config['embedding_dim']
        logits = self.output(features).squeeze(-1)
        raw_logits = self.output(torch.cat([original, raw_summary], -1)).squeeze(-1)
        centered_logits = self.output(torch.cat([original, centered_summary], -1)).squeeze(-1)
        ancestor_logits = self.output(torch.cat([original, ancestor_summary], -1)).squeeze(-1)
        return {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'scorer_allocated_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'uses_centered_readout': float(self.config['center_readout']),
            'uses_target_query': float(self.target_query),
            'original_feature_width': float(original.shape[-1]),
            'appended_feature_width': float(summary.shape[-1]),
            'nonempty_fraction': active.mean(),
            'visible_event_count_mean': n.mean(),
            'occupied_groups_mean': occupied.sum(-1).mean(),
            'partition_coverage_error_max': maximum(semantic.sum(1)-mask.float()),
            'hist_group_size_error_max': maximum(observed.sum((-1,-2))-sizes),
            'weight_mass_error_max': maximum(r['wc'].sum(-1)-active)+maximum(r['wb'].sum(-1)-active),
            'group_mass_error_max': maximum((semantic*r['wc'][:,None,:]).sum(-1)-prior)+maximum((semantic*r['wb'][:,None,:]).sum(-1)-prior),
            'signed_group_mass_error_max': maximum(signed_c)+maximum(signed_b),
            'masked_signed_weight_abs_max': maximum(r['dc']*(~mask).float())+maximum(r['db']*(~mask).float()),
            'neutral_pool_identity_error_max': maximum(r['neutral']-neutral_from_pools),
            'raw_subtraction_identity_error_max': maximum(r['centered']-(r['raw']-r['neutral'])),
            'neutral_raw_rms': rms(r['neutral']),
            'learned_raw_rms': rms(r['raw']),
            'centered_raw_rms': rms(r['centered']),
            'centered_raw_nonempty_nearzero_fraction': (((centered_norm <= 1e-6).float())*active).sum()/denom,
            'raw_nonempty_nearzero_fraction': (((raw_norm <= 1e-6).float())*active).sum()/denom,
            'centered_reference_norm_ratio_nonempty': ((centered_norm/ref_norm.clamp_min(1e-6))*active).sum()/denom,
            'selected_norm_error_eligible_max': maximum((summary.norm(dim=-1)-ref_norm)*eligible),
            'selected_ineligible_summary_abs_max': maximum(summary*(1-eligible).unsqueeze(-1)),
            'raw_control_ancestor_summary_error_max': maximum(raw_summary-ancestor_summary),
            'raw_control_ancestor_logit_error_max': maximum(raw_logits-ancestor_logits),
            'centered_raw_summary_rms_delta': rms(centered_summary-raw_summary),
            'centered_raw_logit_rms_delta_fixed_weights': rms(centered_logits-raw_logits),
            'query_replacement_summary_rms_fixed_groups': rms(ts-cs),
            'query_replacement_logit_rms_fixed_groups': rms(self.output(torch.cat([original,ts],-1))-self.output(torch.cat([original,cs],-1))),
            'all_singleton_group_fraction': singleton_groups.mean(),
            'singleton_group_centered_raw_error_max': maximum(r['centered']*singleton_groups.unsqueeze(-1)),
            'fixed_score_group_offset_centered_raw_error_max': maximum(shifted-r['centered']),
            'field_attention_l1_nonempty': ((r['wc']-r['wb']).abs().sum(-1)*active).sum()/denom,
            'selected_vs_neutral_l1_nonempty': (0.5*((r['dc']).abs().sum(-1)+(r['db']).abs().sum(-1))*active).sum()/denom,
            'empty_summary_abs_max': maximum(es),
            'empty_original_history_abs_max': maximum(eo[:,static_width:]),
            'zero_length_summary_abs_max': maximum(zs),
            'zero_length_original_history_abs_max': maximum(zo[:,static_width:]),
            'empty_logits_finite': torch.isfinite(self.output(ef)).float().mean(),
            'zero_length_logits_finite': torch.isfinite(self.output(zf)).float().mean(),
            'padding_feature_error_max': maximum(pf-features),
            'padding_logit_error_max': maximum(self.output(pf)-self.output(features)),
            'permutation_feature_error_max': maximum(rf-features),
            'permutation_logit_error_max': maximum(self.output(rf)-self.output(features)),
            'logits_finite': torch.isfinite(logits).float().mean(),
            'classifier_readout_weight_rms': rms(self.output[0].weight[:,-self.config['embedding_dim']:]),
        }
