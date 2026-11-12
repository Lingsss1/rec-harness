"""Exact encoded-key multiplicity correction inside occupied match groups.

The full paired_match feature path and parameter allocation are inherited.
Distinct measure subtracts log occurrence count before within-group attention.
Counts use (match group, field ID, action ID, inherited log-recency bucket).
Event controls execute identical preprocessing with correction multiplier zero.
Only fixed-query RAW retrieval has duplicate invariance; original count/pool
features, context queries and the reference norm remain frequency sensitive.
No performance, source replication or priority claim is made by this module.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': False, 'separate_reads': True, 'group_balanced': True, 'learned_weights': True, 'key_measure': 'event'}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.key_measure = self.config['key_measure']
        if self.key_measure not in ('distinct', 'event'):
            raise ValueError('Unknown key_measure')
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
        # Preserve exact ancestor learned allocation and initialization order.
        # Key counting adds no parameters; all cells execute the same counting.
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
    def _run_multiplicities(keys, mask):
        """Count exact integer keys per row, excluding masked positions.

        Sort and scatter use O(B L) storage, not a [B,L,L] equality tensor.
        The -1 sentinel cannot collide with a nonnegative constructed key.
        Padding multiplicity is one so log(count) is always finite.
        """
        if keys.shape[1] == 0:
            return torch.ones_like(keys, dtype=torch.float32)
        ordered, order = keys.masked_fill(~mask, -1).sort(dim=-1)
        valid = mask.gather(1, order)
        starts = torch.ones_like(ordered, dtype=torch.bool)
        starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        runs = starts.long().cumsum(-1) - 1
        totals = torch.zeros_like(ordered, dtype=torch.float32).scatter_add(
            1, runs, valid.float())
        sorted_counts = totals.gather(1, runs).clamp_min(1)
        return torch.ones_like(sorted_counts).scatter(1, order, sorted_counts)

    def _key_multiplicities(self, batch, membership):
        mask = batch['history_mask']
        group = membership.argmax(dim=1)
        ages = (torch.log2(batch['history_age_seconds'].clamp_min(0) + 1).long()
                + 1).clamp(0, 32)
        action = batch['history_type']
        # Mixed-radix integer encoding is injective over the contract's
        # nonnegative field IDs, action0..4, recency0..32 and group0..3.
        common = (action * 33 + ages) * 4 + group
        category_key = batch['history_category'] * (5 * 33 * 4) + common
        brand_key = batch['history_brand'] * (5 * 33 * 4) + common
        return (self._run_multiplicities(category_key, mask),
                self._run_multiplicities(brand_key, mask))

    def _read(self, category, brand, common, mask, query, membership,
              multiplicities, measure=None):
        if mask.shape[1] == 0:
            zero = category.sum(1)
            weights = mask.float()
            return zero, weights, weights, weights, weights, weights, weights
        selected = self.key_measure if measure is None else measure
        scale = float(self.learned_weights)
        correction = float(selected == 'distinct')
        category_score = (self._score(query[:, 0], category + common) * scale
                          - correction * multiplicities[0].log())
        brand_score = (self._score(query[:, 1], brand + common) * scale
                       - correction * multiplicities[1].log())
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


    def _retrieval(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        membership = self._membership(batch)[1]
        multiplicities = self._key_multiplicities(batch, membership)
        query = target if self.target_query else context
        raw, wc, wb, gc, gb, bc, bb = self._read(
            category, brand, common, mask, query, membership, multiplicities)
        return {'summary': self._match_norm(raw, pooled), 'pooled': pooled,
                'raw': raw, 'wc': wc, 'wb': wb, 'global_c': gc, 'global_b': gb,
                'balanced_c': bc, 'balanced_b': bb, 'membership': membership,
                'mask': mask, 'target': target, 'context': context,
                'multiplicities': multiplicities}

    def _features(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        original = torch.cat([static, pools.flatten(1), statistics, histogram], dim=-1)
        summary = self._retrieval(batch)['summary']
        return torch.cat([original, summary], dim=-1), original, summary

    def forward(self, batch):
        return self.output(self._features(batch)[0]).squeeze(-1)

    def diagnostics(self, batch):
        """Label-free first512 checks, supplied by the candidate implementation.

        Duplication compares raw readouts at a FIXED query. It does not assert
        full-logit invariance or causal usefulness. Missing fields stay visible
        as ID0 keys but cannot enter a positive semantic match.
        """
        def maximum(x):
            return x.abs().max() if x.numel() else 0.0

        static, pools, counts, stats, _, semantic, sizes, mean = self._represent(batch)
        histogram, observed, expected, _ = self._histogram(batch, semantic, sizes)
        features, original, summary = self._features(batch)
        r = self._retrieval(batch)
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        mc, mb = r['multiplicities']
        query = target if self.target_query else context
        event = self._read(category, brand, common, mask, query, semantic,
                           (mc, mb), 'event')
        distinct = self._read(category, brand, common, mask, query, semantic,
                              (mc, mb), 'distinct')
        tq = self._read(category, brand, common, mask, target, semantic, (mc, mb))[0]
        cq = self._read(category, brand, common, mask, context, semantic, (mc, mb))[0]
        n = counts[0]
        active = mask.any(-1).float()
        denom = active.sum().clamp_min(1)
        occupied = (sizes > 0).float()
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        mass_c = (semantic * r['wc'][:, None, :]).sum(-1)
        mass_b = (semantic * r['wb'][:, None, :]).sum(-1)
        unique_c = (mask.float() / mc).sum(-1)
        unique_b = (mask.float() / mb).sum(-1)
        all_unique = (((mc == 1) | ~mask).all(-1) &
                      ((mb == 1) | ~mask).all(-1))
        history_keys = ('history_category', 'history_brand', 'history_type',
                        'history_age_seconds', 'history_mask')
        reverse = dict(batch)
        for key in history_keys:
            reverse[key] = batch[key].flip(1)
        rev_features = self._features(reverse)[0]
        rev_counts = self._key_multiplicities(reverse, self._membership(reverse)[1])
        padded = dict(batch)
        for key in history_keys[:-1]:
            padded[key] = torch.where(mask, batch[key], torch.zeros_like(batch[key]))
        padded_features = self._features(padded)[0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        empty_features, empty_original, empty_summary = self._features(empty)
        zero_length = dict(batch)
        for key in history_keys:
            zero_length[key] = batch[key][:, :0]
        zero_features, zero_original, zero_summary = self._features(zero_length)
        duplicate_distinct_error = 0.0
        duplicate_event_delta = 0.0
        duplicate_selected_delta = 0.0
        duplicate_context_delta = 0.0
        if mask.shape[1] > 0:
            # Append one copy of the first valid event, per row. Empty rows
            # receive only masked padding. This is a diagnostic, not model input.
            first = mask.long().argmax(-1, keepdim=True)
            duplicate = dict(batch)
            for key in history_keys:
                extra = batch[key].gather(1, first)
                if key == 'history_mask':
                    extra = mask.any(-1, keepdim=True)
                duplicate[key] = torch.cat([batch[key], extra], dim=1)
            _, dc, db, dcommon, dm, _, _, dcontext = self._encode(duplicate)
            dmembership = self._membership(duplicate)[1]
            dcounts = self._key_multiplicities(duplicate, dmembership)
            d_event = self._read(dc, db, dcommon, dm, query, dmembership,
                                 dcounts, 'event')[0]
            d_distinct = self._read(dc, db, dcommon, dm, query, dmembership,
                                    dcounts, 'distinct')[0]
            duplicate_distinct_error = maximum(d_distinct - distinct[0])
            duplicate_event_delta = (d_event - event[0]).square().mean().sqrt()
            d_selected = d_distinct if self.key_measure == 'distinct' else d_event
            duplicate_selected_delta = (d_selected - r['raw']).square().mean().sqrt()
            duplicate_context_delta = (dcontext - context).square().mean().sqrt()
        entropy = -0.5 * ((r['wc'] * r['wc'].clamp_min(1e-9).log()).sum(-1)
                         + (r['wb'] * r['wb'].clamp_min(1e-9).log()).sum(-1))
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'scorer_allocated_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'uses_distinct_key_measure': float(self.key_measure == 'distinct'),
            'uses_target_query': float(self.target_query),
            'original_feature_width': float(original.shape[-1]),
            'appended_feature_width': float(summary.shape[-1]),
            'visible_event_count_mean': n.mean(),
            'nonempty_fraction': active.mean(),
            'occupied_groups_mean': occupied.sum(-1).mean(),
            'distinct_category_keys_mean': unique_c.mean(),
            'distinct_brand_keys_mean': unique_b.mean(),
            'category_redundant_event_fraction_nonempty': ((1-unique_c/n.clamp_min(1))*active).sum()/denom,
            'brand_redundant_event_fraction_nonempty': ((1-unique_b/n.clamp_min(1))*active).sum()/denom,
            'category_repeated_key_event_fraction_nonempty': (((mc>1)&mask).float().sum(-1)/n.clamp_min(1)*active).sum()/denom,
            'brand_repeated_key_event_fraction_nonempty': (((mb>1)&mask).float().sum(-1)/n.clamp_min(1)*active).sum()/denom,
            'category_multiplicity_max': mc.max() if mc.numel() else 1.0,
            'brand_multiplicity_max': mb.max() if mb.numel() else 1.0,
            'all_unique_fraction': all_unique.float().mean(),
            'count_padding_error_max': maximum((mc-1)*(~mask).float())+maximum((mb-1)*(~mask).float()),
            'count_permutation_error_max': maximum(mc-rev_counts[0].flip(1))+maximum(mb-rev_counts[1].flip(1)),
            'partition_coverage_error_max': maximum(semantic.sum(1)-mask.float()),
            'hist_group_size_error_max': maximum(observed.sum((-1,-2))-sizes),
            'hist_action_margin_error_max': maximum(observed.sum(-1)-expected.sum(-1)),
            'hist_age_margin_error_max': maximum(observed.sum(-2)-expected.sum(-2)),
            'weight_mass_error_max': maximum(torch.stack([r['wc'].sum(-1)-active,r['wb'].sum(-1)-active])),
            'group_mass_error_max': maximum(torch.stack([mass_c-prior,mass_b-prior])),
            'masked_weight_abs_max': maximum(torch.stack([r['wc'],r['wb']],1)*(~mask)[:,None,:].float()),
            'field_attention_l1_nonempty': ((r['wc']-r['wb']).abs().sum(-1)*active).sum()/denom,
            'attention_entropy_nonempty': (entropy*active).sum()/denom,
            'distinct_event_weight_l1_nonempty': (0.5*((distinct[1]-event[1]).abs().sum(-1)+(distinct[2]-event[2]).abs().sum(-1))*active).sum()/denom,
            'distinct_event_raw_rms_fixed_query': (distinct[0]-event[0]).square().mean().sqrt(),
            'distinct_event_raw_all_unique_error_max': maximum((distinct[0]-event[0])*all_unique.unsqueeze(-1)),
            'query_replacement_summary_rms_fixed_groups': (self._match_norm(tq,pooled)-self._match_norm(cq,pooled)).square().mean().sqrt(),
            'duplication_distinct_raw_error_max_fixed_query': duplicate_distinct_error,
            'duplication_event_raw_rms_fixed_query': duplicate_event_delta,
            'duplication_selected_raw_rms_fixed_query': duplicate_selected_delta,
            'duplication_context_query_rms_delta': duplicate_context_delta,
            'norm_error_max': maximum(summary.norm(dim=-1)-pooled.norm(dim=-1)),
            'norm_fallback_fraction': (r['raw'].norm(dim=-1)<=1e-6).float().mean(),
            'singleton_summary_error_max': maximum((summary-pooled)*(n==1).unsqueeze(-1)),
            'empty_summary_abs_max': maximum(empty_summary),
            'empty_history_feature_abs_max': maximum(empty_original[:,static.shape[-1]:]),
            'zero_length_summary_abs_max': maximum(zero_summary),
            'zero_length_history_feature_abs_max': maximum(zero_original[:,static.shape[-1]:]),
            'empty_logits_finite': float(torch.isfinite(self.output(empty_features)).all()),
            'zero_length_logits_finite': float(torch.isfinite(self.output(zero_features)).all()),
            'padding_feature_error_max': maximum(features-padded_features),
            'padding_logit_error_max': maximum(self.output(features)-self.output(padded_features)),
            'permutation_feature_error_max': maximum(features-rev_features),
            'permutation_logit_error_max': maximum(self.output(features)-self.output(rev_features)),
            'logits_finite': float(torch.isfinite(self.output(features)).all()),
            'classifier_readout_weight_rms': self.output[0].weight[:,original.shape[-1]:].square().mean().sqrt(),
        }
        return {key: float(value) for key, value in result.items()}
