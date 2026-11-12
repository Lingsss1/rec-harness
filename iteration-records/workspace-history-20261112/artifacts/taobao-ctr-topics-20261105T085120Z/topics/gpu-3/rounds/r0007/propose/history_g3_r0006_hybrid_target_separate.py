"""Complete grouped-history features plus a parallel norm-matched field readout.

Inherited sources: g1-r0006-paired_match (group means/counts/contingencies),
g2-r0001-target_separate (field retrieval). Four cells cross added target versus
history-only queries with separate versus consensus reads. Target independence
applies ONLY to the added query; incumbent groups remain target dependent.
No optimizer/loss changes; no global novelty or measured improvement claim.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'joint_mode': 'observed', 'statistic_mode': 'count', 'grouping_mode': 'match', 'histogram_grouping': 'match', 'histogram_pairing': 'observed', 'target_query': True, 'separate_reads': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.target_query = self.config['target_query']
        self.separate_reads = self.config['separate_reads']
        self.joint_mode = self.config['joint_mode']
        self.statistic_mode = self.config['statistic_mode']
        self.grouping_mode = self.config['grouping_mode']
        if self.joint_mode not in ('observed', 'independent'):
            raise ValueError('Unknown joint_mode')
        if self.statistic_mode not in ('count', 'rate'):
            raise ValueError('Unknown statistic_mode')
        if self.grouping_mode not in ('match', 'rank', 'occupied_global'):
            raise ValueError('Unknown grouping_mode')
        # This factorial changes only the appended retrieval query/readout.
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
        # All four variants allocate identical shapes and execute both reads.
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

    def _read(self, category, brand, common, mask, query):
        if mask.shape[1] == 0:
            zero = category.sum(1)
            weights = mask.float()
            return zero, zero, weights, weights, weights
        category_score = self._score(query[:, 0], category + common)
        brand_score = self._score(query[:, 1], brand + common)
        category_weight = self._weights(category_score, mask)
        brand_weight = self._weights(brand_score, mask)
        # Consensus is a geometric, rather than arithmetic, opinion pool.
        joint_weight = self._weights(0.5 * (category_score + brand_score), mask)
        separate = (
            category_weight.unsqueeze(-1) * category
            + brand_weight.unsqueeze(-1) * brand
            + (0.5 * (category_weight + brand_weight)).unsqueeze(-1) * common
        ).sum(1)
        joint = (joint_weight.unsqueeze(-1) * (category + brand + common)).sum(1)
        return separate, joint, category_weight, brand_weight, joint_weight

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
        query = target if self.target_query else context
        separate, joint, wc, wb, wj = self._read(category, brand, common, mask, query)
        raw = separate if self.separate_reads else joint
        summary = self._match_norm(raw, pooled)
        return summary, pooled, separate, joint, wc, wb, wj, context, target, mask

    def _features(self, batch):
        static, pools, _, statistics, _, semantic, sizes, _ = self._represent(batch)
        histogram = self._histogram(batch, semantic, sizes)[0]
        original = torch.cat([static, pools.flatten(1), statistics, histogram], dim=-1)
        summary = self._retrieval(batch)[0]
        return torch.cat([original, summary], dim=-1), original, summary

    def forward(self, batch):
        return self.output(self._features(batch)[0]).squeeze(-1)

    def diagnostics(self, batch):
        # Label-free candidate-supplied checks, not independently scored evidence.
        static, pools, counts, stats, membership, semantic, sizes, global_mean = self._represent(batch)
        histogram, observed, expected, hist_membership = self._histogram(batch, semantic, sizes)
        summary, pooled, separate, joint, wc, wb, wj, context, target, mask = self._retrieval(batch)
        features, original, _ = self._features(batch)
        n, c, b, both, _ = counts
        active = mask.any(-1).float()
        denom = active.sum().clamp_min(1)
        selected_raw = separate if self.separate_reads else joint
        reference_norm = pooled.norm(dim=-1)
        selected_norm = summary.norm(dim=-1)
        sm = self._match_norm(separate, pooled)
        jm = self._match_norm(joint, pooled)
        cosine = (sm * jm).sum(-1) / (sm.norm(dim=-1) * jm.norm(dim=-1)).clamp_min(1e-9)
        fields, category, brand, common, _, _, _, _ = self._encode(batch)
        cs, cj, cc, cb, _ = self._read(category, brand, common, mask, context)
        ts, tj, _, _, _ = self._read(category, brand, common, mask, target)
        context_summary = self._match_norm(cs if self.separate_reads else cj, pooled)
        target_summary = self._match_norm(ts if self.separate_reads else tj, pooled)
        padded = dict(batch)
        for name in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[name] = torch.where(mask, batch[name], torch.zeros_like(batch[name]))
        padded_features = self._features(padded)[0]
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(mask)
        _, empty_original, empty_summary = self._features(empty)
        rotated = dict(batch)
        rotated_static = batch['static'].clone()
        for index in (self.category, self.brand):
            rotated_static[:, index] = batch['static'][:, index].roll(1, 0)
        rotated['static'] = rotated_static
        rotated_summary = self._retrieval(rotated)[0]
        expected_sizes = torch.stack([n - c - b + both, c - both, b - both, both], -1)
        entropy = -0.5 * ((wc * wc.clamp_min(1e-9).log()).sum(-1)
                         + (wb * wb.clamp_min(1e-9).log()).sum(-1))
        first_weight = self.output[0].weight
        result = {
            'allocated_parameters': float(sum(p.numel() for p in self.parameters())),
            'activated_scorer_parameters': float(sum(p.numel() for p in self.attention.parameters())),
            'added_allocated_readout_parameters': float(self.config['embedding_dim'] * first_weight.shape[0]),
            'neural_event_scorer_executed': 1.0,
            'uses_target_query': float(self.target_query),
            'uses_separate_reads': float(self.separate_reads),
            'original_feature_width': float(original.shape[-1]),
            'appended_feature_width': float(summary.shape[-1]),
            'nonempty_fraction': active.mean(),
            'visible_event_count_mean': n.mean(),
            'visible_count_at_limit_fraction': (n == 128).float().mean(),
            'occupied_groups_mean': (sizes > 0).float().sum(-1).mean(),
            'partition_coverage_error_max': (membership.sum(1) - mask.float()).abs().max() if mask.shape[1] else 0.0,
            'group_count_identity_error_max': (sizes - expected_sizes).abs().max(),
            'global_reconstruction_abs_error_max': ((pools * sizes.unsqueeze(-1)).sum(1) / n.clamp_min(1).unsqueeze(-1) - global_mean).abs().max(),
            'hist_group_size_error_max': (observed.sum((-1, -2)) - sizes).abs().max(),
            'hist_action_margin_error_max': (observed.sum(-1) - expected.sum(-1)).abs().max(),
            'hist_age_margin_error_max': (observed.sum(-2) - expected.sum(-2)).abs().max(),
            'hist_feature_rms': histogram.square().mean().sqrt(),
            'original_feature_rms': original.square().mean().sqrt(),
            'appended_feature_rms': summary.square().mean().sqrt(),
            'reference_norm_mean': reference_norm.mean(),
            'selected_norm_mean': selected_norm.mean(),
            'norm_absolute_error_max': (selected_norm - reference_norm).abs().max(),
            'norm_relative_error_max': ((selected_norm - reference_norm).abs() / reference_norm.clamp_min(1e-6)).max(),
            'norm_fallback_fraction': (selected_raw.norm(dim=-1) <= 1e-6).float().mean(),
            'singleton_summary_error_max': ((summary - pooled).abs() * (n == 1).unsqueeze(-1)).max(),
            'empty_summary_abs_max': empty_summary.abs().max(),
            'empty_history_feature_abs_max': empty_original[:, static.shape[-1]:].abs().max(),
            'padding_feature_abs_error_max': (features - padded_features).abs().max(),
            'padding_logit_abs_error_max': (self.output(features) - self.output(padded_features)).abs().max(),
            'weight_mass_error_max': torch.stack([(wc.sum(-1) - active).abs(), (wb.sum(-1) - active).abs(), (wj.sum(-1) - active).abs()]).max(),
            'masked_weight_abs_max': (wc * (~mask).float()).abs().max() if mask.shape[1] else 0.0,
            'separate_joint_cosine_nonempty': (cosine * active).sum() / denom,
            'separate_joint_rms_delta': (sm - jm).square().mean().sqrt(),
            'field_attention_l1_nonempty': ((wc - wb).abs().sum(-1) * active).sum() / denom,
            'attention_l1_vs_context_nonempty': (0.5 * ((wc - cc).abs().sum(-1) + (wb - cb).abs().sum(-1)) * active).sum() / denom,
            'field_attention_entropy_nonempty': (entropy * active).sum() / denom,
            'target_context_summary_rms_delta': (target_summary - context_summary).square().mean().sqrt(),
            'rotated_target_retrieval_rms_delta': (summary - rotated_summary).square().mean().sqrt(),
            'category_match_present_fraction': (c > 0).float().mean(),
            'brand_match_present_fraction': (b > 0).float().mean(),
            'target_brand_missing_fraction': (batch['static'][:, self.brand] == 0).float().mean(),
            'classifier_original_weight_rms': first_weight[:, :original.shape[-1]].square().mean().sqrt(),
            'classifier_retrieval_weight_rms': first_weight[:, original.shape[-1]:].square().mean().sqrt(),
        }
        return {key: float(value) for key, value in result.items()}
