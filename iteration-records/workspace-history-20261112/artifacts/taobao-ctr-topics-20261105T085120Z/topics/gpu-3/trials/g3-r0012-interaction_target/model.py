"""Direct field-query interactions added to the frozen balanced-target design.

All cells retain paired_match features and a balanced field readout, then
append one d-vector: norm-matched query/value products or an additive replacement.
The 2x2 is target/context query at BOTH read sites x product/addition.
Context queries are independent of target embeddings; match-group partitions
still depend on the target. Inspired by explicit relevance readouts, not a UniDot reproduction.
No measured gain or novelty claim; no input/optimizer/loss/training changes.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': True, 'separate_reads': True, 'group_balanced': True, 'learned_weights': True, 'interaction_target': True, 'interaction_product': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.interaction_target = self.config['interaction_target']
        self.interaction_product = self.config['interaction_product']
        if not (self.config['group_balanced'] and self.config['learned_weights']):
            raise ValueError('All cells fix learned balanced retrieval')
        if self.target_query != self.interaction_target:
            raise ValueError('Query factor must agree at both read sites')
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
        # The complete parent representation and readout are retained.
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
        # All four variants allocate identical shapes; only new feature algebra changes.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + 4 * dim + 4 + 64 + 2 * dim
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


    def _retrieval(self, batch):
        _, category, brand, common, mask, pooled, target, context = self._encode(batch)
        membership = self._membership(batch)[1]
        query = target if self.target_query else context
        raw, wc, wb, gc, gb, bc, bb = self._read(
            category, brand, common, mask, query, membership)
        # Preserve the parent's raw and normalized summaries exactly. The two
        # field values add to raw algebraically (up to reduction roundoff).
        read_c = (wc.unsqueeze(-1) * (category + 0.5 * common)).sum(1)
        read_b = (wb.unsqueeze(-1) * (brand + 0.5 * common)).sum(1)
        return {'summary': self._match_norm(raw, pooled), 'pooled': pooled,
                'raw': raw, 'wc': wc, 'wb': wb, 'global_c': gc, 'global_b': gb,
                'balanced_c': bc, 'balanced_b': bb, 'membership': membership,
                'mask': mask, 'target': target, 'context': context,
                'read_c': read_c, 'read_b': read_b}

    def _interaction(self, retrieval, query, multiply):
        # Both replacements are evaluated in every cell. The query factor is
        # applied at both the inherited scorer and this explicit feature site.
        c, b = retrieval['read_c'], retrieval['read_b']
        product = query[:, 0] * c + query[:, 1] * b
        addition = 0.5 * (query[:, 0] + c + query[:, 1] + b)
        raw = product if multiply else addition
        # Fix scale using the same per-example reference as the parent readout.
        # Direction/gradient geometry still differ and require empirical controls.
        feature = self._match_norm(raw, retrieval['pooled'])
        return raw, feature

    def _features(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        original = torch.cat([static, pools.flatten(1), statistics, histogram], dim=-1)
        retrieval = self._retrieval(batch)
        query = retrieval['target'] if self.interaction_target else retrieval['context']
        _, interaction = self._interaction(retrieval, query, self.interaction_product)
        return (torch.cat([original, retrieval['summary'], interaction], dim=-1),
                original, retrieval['summary'], interaction)

    def forward(self, batch):
        return self.output(self._features(batch)[0]).squeeze(-1)

    def diagnostics(self, batch):
        # Label-free checks on the host's first 512 validation impressions.
        # No perturbation uses labels or changes training. These measurements
        # are candidate supplied; full-population scoring is the host's job.
        def maximum(x):
            return x.abs().max() if x.numel() else 0.0

        def rms(x):
            return x.square().mean().sqrt()

        features, original, parent, interaction = self._features(batch)
        r = self._retrieval(batch)
        mask = r['mask']
        semantic = r['membership']
        sizes = semantic.sum(-1)
        n = mask.sum(-1)
        active = mask.any(-1).float()
        occupied = (sizes > 0).float()
        prior = occupied / occupied.sum(-1, keepdim=True).clamp_min(1)
        target_product_raw, target_product = self._interaction(r, r['target'], True)
        _, context_product = self._interaction(r, r['context'], True)
        _, target_addition = self._interaction(r, r['target'], False)
        _, context_addition = self._interaction(r, r['context'], False)
        selected_raw, selected = self._interaction(
            r, r['target'] if self.interaction_target else r['context'],
            self.interaction_product)
        # Swapping the category/brand value vectors preserves their sum, while
        # matched query-by-value products can differ. This is a source-level
        # limitation probe, not a real-world intervention or a causal estimate.
        swapped_product = self._match_norm(
            r['target'][:, 0] * r['read_b'] + r['target'][:, 1] * r['read_c'],
            r['pooled'])
        selected_query = r['target'] if self.interaction_target else r['context']
        opposite_query = r['context'] if self.interaction_target else r['target']
        opposite = self._interaction(r, opposite_query, self.interaction_product)[1]
        changed_features = torch.cat([original, parent, opposite], dim=-1)
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
        ef, eo, ep, ei = self._features(empty)
        zero = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds', 'history_mask'):
            zero[key] = batch[key][:, :0]
        zf, zo, zp, zi = self._features(zero)
        static_width = len(self.embeddings) * self.config['embedding_dim']
        original_width = original.shape[-1]
        interaction_start = original_width + self.config['embedding_dim']
        ref_norm = r['pooled'].norm(dim=-1)
        singleton = (n == 1).unsqueeze(-1)
        singleton_product = selected_query[:, 0] * r['read_c'] + selected_query[:, 1] * r['read_b']
        singleton_addition = 0.5 * (selected_query.sum(1) + r['read_c'] + r['read_b'])
        return {k: float(v) for k, v in {
            'allocated_parameters': sum(p.numel() for p in self.parameters()),
            'scorer_allocated_parameters': sum(p.numel() for p in self.attention.parameters()),
            'parent_uses_target_query': float(self.target_query),
            'interaction_uses_target_query': float(self.interaction_target),
            'interaction_uses_product': float(self.interaction_product),
            'original_feature_width': original_width,
            'parent_summary_width': parent.shape[-1],
            'extra_feature_width': interaction.shape[-1],
            'total_feature_width': features.shape[-1],
            'nonempty_fraction': active.mean(),
            'singleton_fraction': (n == 1).float().mean(),
            'occupied_groups_mean': occupied.sum(-1).mean(),
            'partition_coverage_error_max': maximum(semantic.sum(1)-mask.float()),
            'selected_weight_mass_error_max': maximum(torch.stack([r['wc'].sum(-1)-active,r['wb'].sum(-1)-active])),
            'equal_group_mass_error_max': maximum(torch.stack([(semantic*r['wc'][:,None,:]).sum(-1)-prior,(semantic*r['wb'][:,None,:]).sum(-1)-prior])),
            'masked_weight_abs_max': maximum(torch.stack([r['wc'],r['wb']],1)*(~mask)[:,None,:].float()),
            'field_attention_l1_mean': (r['wc']-r['wb']).abs().sum(-1).mean(),
            'field_sum_parent_raw_error_max': maximum(r['read_c']+r['read_b']-r['raw']),
            'parent_norm_error_max': maximum(parent.norm(dim=-1)-ref_norm),
            'extra_norm_error_max': maximum(interaction.norm(dim=-1)-ref_norm),
            'extra_feature_rms': rms(interaction),
            'parent_feature_rms': rms(parent),
            'extra_fallback_fraction': (selected_raw.norm(dim=-1)<=1e-6).float().mean(),
            'target_product_fallback_fraction': (target_product_raw.norm(dim=-1)<=1e-6).float().mean(),
            'both_target_fields_missing_fraction': ((batch['static'][:,self.category]==0)&(batch['static'][:,self.brand]==0)).float().mean(),
            'product_query_replacement_rms': rms(target_product-context_product),
            'addition_query_replacement_rms': rms(target_addition-context_addition),
            'target_product_addition_rms': rms(target_product-target_addition),
            'context_product_addition_rms': rms(context_product-context_addition),
            'field_swap_target_product_rms': rms(target_product-swapped_product),
            'selected_feature_identity_error_max': maximum(selected-interaction),
            'query_replacement_logit_rms_fixed_parent': rms(self.output(features)-self.output(changed_features)),
            'singleton_formula_error_max': maximum((selected_raw-(singleton_product if self.interaction_product else singleton_addition))*singleton),
            'empty_extra_abs_max': maximum(ei),
            'empty_parent_abs_max': maximum(ep),
            'empty_original_history_abs_max': maximum(eo[:,static_width:]),
            'zero_length_extra_abs_max': maximum(zi),
            'zero_length_parent_abs_max': maximum(zp),
            'zero_length_original_history_abs_max': maximum(zo[:,static_width:]),
            'empty_logits_finite': torch.isfinite(self.output(ef)).all(),
            'zero_length_logits_finite': torch.isfinite(self.output(zf)).all(),
            'padding_feature_error_max': maximum(features-padded_features),
            'padding_logit_error_max': maximum(self.output(features)-self.output(padded_features)),
            'permutation_feature_error_max': maximum(features-reverse_features),
            'permutation_logit_error_max': maximum(self.output(features)-self.output(reverse_features)),
            'logits_finite': torch.isfinite(self.output(features)).all(),
            'classifier_extra_weight_rms': rms(self.output[0].weight[:,interaction_start:]),
            'classifier_parent_weight_rms': rms(self.output[0].weight[:,original_width:interaction_start]),
        }.items()}
