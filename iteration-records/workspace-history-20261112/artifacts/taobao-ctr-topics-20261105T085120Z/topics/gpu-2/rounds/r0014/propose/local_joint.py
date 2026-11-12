"""Conditional-marginal history sufficiency, with fixed active capacity.

Cross fine-recency localization with pair-concentration readout replacement.
Factorized recency keeps per-group coarse ages and global coarse/fine ages;
product identity keeps complete-support category and brand marginal summaries.
These interventions do not remove target category/brand match membership or
coarse action-age tables. Product mode skips an unused third identity sort.
All cells have identical parameter allocation and initialization, frozen loss
and optimizer. Training and predictive/economic conclusions remain host-pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'content_mode': 'off', 'recency_mode': 'fine', 'recency_localization': 'local', 'identity_mode': 'joint'}

class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = self.config['embedding_dim']
        if dim != 16 or self.config['hidden'] != [128, 64]:
            raise ValueError('Frozen reference dimensions required')
        self.content_mode = self.config['content_mode']
        self.recency_mode = self.config['recency_mode']
        if self.content_mode not in ('pooled', 'off') or self.recency_mode not in ('fine', 'coarse'):
            raise ValueError('Unknown factorial cell')
        # Same allocation order and parameter shapes in every factorial cell.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.category_vocab = int(meta['buckets'][self.category])
        self.brand_vocab = int(meta['buckets'][self.brand])
        if self.category_vocab * self.brand_vocab > (2**63 - 1)//4:
            raise ValueError('Packed identity key would overflow int64')
        self.joint_mode, self.statistic_mode, self.grouping_mode = 'observed', 'count', 'match'
        self.histogram_grouping, self.histogram_pairing = 'match', 'observed'
        self.identity_mode = self.config['identity_mode']
        self.recency_localization = self.config['recency_localization']
        if self.identity_mode not in ('joint', 'independent'):
            raise ValueError('Unknown identity mode')
        if self.recency_localization not in ('local', 'factorized'):
            raise ValueError('Unknown recency localization')
        if self.content_mode != 'off' or self.recency_mode != 'fine':
            raise ValueError('This study fixes content off and fine recency')
        static_width = len(meta['fields']) * dim
        self.input_width = static_width + 4 * dim + 68
        parent_width = static_width + 11 * dim + 68
        parent_output = 128 * (parent_width + 1) + 64 * 129 + 65
        self.reference_active_nonstatic = parent_output + 38 * dim + 464
        # Output has h*(input_width+65)+129 parameters. Count-match the
        # parent INCLUDING active action/recency and identity-head parameters.
        # At F=18 this gives h=158 and +214 active nonstatic parameters (0.276%).
        fixed = 38 * dim + 464 + 129
        denominator = self.input_width + 65
        self.first_hidden = max(1, (self.reference_active_nonstatic - fixed + denominator//2)//denominator)
        self.output = nn.Sequential(
            nn.Linear(self.input_width, self.first_hidden), nn.ReLU(),
            nn.Linear(self.first_hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.identity_head = nn.Sequential(
            nn.Linear(28, 16, bias=False), nn.ReLU(), nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.identity_head[-1].weight)

    def _age_ids(self, batch):
        age = batch['history_age_seconds']
        fine = (torch.log2(age + 1).long() + 1).clamp(0, 32)
        coarse = ((age >= 3600.0).long() + (age >= 86400.0).long() +
                  (age >= 604800.0).long())
        return fine, coarse

    def _fine_counts(self, batch, semantic):
        fine, _ = self._age_ids(batch)
        counts = semantic.new_zeros((semantic.shape[0], 4, 33))
        return counts.scatter_add(2, fine[:, None, :].expand(-1, 4, -1), semantic)

    def _factorized_recency(self, batch, observed):
        # Raw counts, per impression. g=target-match group, s=coarse age,
        # k=log2 age. E[g,s,k] = N[g,s] N[s,k] / N[s]. This keeps N[g,s]
        # and N[s,k], including bins that straddle a coarse-age boundary.
        # No fitted distribution, sequence truncation or foreign-user input.
        fine, coarse = self._age_ids(batch)
        mask = batch['history_mask'].float()
        global_counts = observed.new_zeros((mask.shape[0], 132))
        global_counts = global_counts.scatter_add(1, 33 * coarse + fine, mask)
        global_counts = global_counts.reshape(mask.shape[0], 4, 33)
        group_coarse = observed.sum(-2)
        coarse_mass = global_counts.sum(-1)
        mixture = group_coarse / coarse_mass.clamp_min(1)[:, None, :]
        # No [B,group,coarse,fine] forward temporary is required.
        selected = torch.bmm(mixture, global_counts)
        return selected, global_counts, group_coarse, mixture

    def _context_from_counts(self, counts, observed, sizes):
        return (torch.matmul(observed.sum(-1), self.action.weight[1:5]) +
                torch.matmul(counts, self.recency.weight)) / sizes.clamp_min(1).unsqueeze(-1)

    def _features(self, batch):
        static = torch.cat([e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)], -1)
        _, semantic, sizes = self._membership(batch)
        statistics = self._statistics(self._match_counts(batch))
        histogram, observed = self._histogram(batch, semantic, sizes)[:2]
        if self.recency_localization == 'local':
            counts = self._fine_counts(batch, semantic)
        else:
            counts = self._factorized_recency(batch, observed)[0]
        context = self._context_from_counts(counts, observed, sizes)
        return torch.cat([static, context.flatten(1), statistics, histogram], -1)

    def _direct_context(self, batch, semantic, sizes):
        # Independent event-level reference, diagnostics only.
        if batch['history_mask'].shape[1] == 0:
            return semantic.new_zeros((semantic.shape[0], 4, self.config['embedding_dim']))
        fine, _ = self._age_ids(batch)
        events = self.action(batch['history_type']) + self.recency(fine)
        return torch.bmm(semantic, events) / sizes.clamp_min(1).unsqueeze(-1)

    def diagnostics(self, batch):
        # At most 64 finite scalar checks on the host's first 512 validation rows.
        # These are implementation checks, not independent population evidence.
        _, semantic, sizes = self._membership(batch)
        _, obs = self._histogram(batch, semantic, sizes)[:2]
        local = self._fine_counts(batch, semantic)
        fact, global_counts, group_coarse, mixture = self._factorized_recency(batch, obs)
        by_coarse = mixture.unsqueeze(-1) * global_counts[:, None, :, :]
        local_mean = self._context_from_counts(local, obs, sizes)
        fact_mean = self._context_from_counts(fact, obs, sizes)
        identity = self._identity_tables(batch, semantic, full=True)
        selected = self._identity_features(self._identity_tables(batch, semantic))
        c, b = batch['history_category'], batch['history_brand']
        dense = [self._dense_distribution(key, identity[6], identity[5], identity[3])
                 for key in (c, b, c * self.brand_vocab + b)]
        raw_error = torch.stack([(x-y).abs().max() for x, y in
                                 zip(identity[4], [v for pair in dense for v in pair])]).max()
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'], batch[key], torch.ones_like(batch[key]))
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {k: (v[:, :0] if k.startswith('history_') else v) for k, v in batch.items()}
        reverse = {k: (v.flip(1) if k.startswith('history_') else v) for k, v in batch.items()}
        features = self._features(batch)
        static_width = len(self.embeddings) * self.config['embedding_dim']
        total = sum(p.numel() for p in self.parameters())
        static_count = sum(e.weight.numel() for e in self.embeddings)
        def error(x, y):
            return (x-y).abs().max()
        def selected_identity(x):
            return self._identity_features(self._identity_tables(x))
        result = {
            'allocated_parameters': total,
            'active_parameters': total,
            'active_nonstatic_parameters': total-static_count,
            'reference_active_nonstatic_target': self.reference_active_nonstatic,
            'active_nonstatic_parameter_gap': total-static_count-self.reference_active_nonstatic,
            'actual_first_hidden': self.first_hidden,
            'feature_width': self.input_width,
            'local_fine_recency_enabled': float(self.recency_localization == 'local'),
            'joint_identity_enabled': float(self.identity_mode == 'joint'),
            'identity_sorts_per_forward': float(3 if self.identity_mode == 'joint' else 2),
            'local_fine_count_mass_error': error(local.sum(-1), sizes),
            'factorized_group_mass_error': error(fact.sum(-1), sizes),
            'factorized_global_fine_margin_error': error(fact.sum(1), local.sum(1)),
            'factorized_group_coarse_margin_error': error(by_coarse.sum(-1), group_coarse),
            'factorized_global_coarse_fine_margin_error': error(by_coarse.sum(1), global_counts),
            'fine_histogram_vs_event_context_error': error(local_mean, self._direct_context(batch, semantic, sizes)),
            'local_vs_factorized_context_rms': (local_mean-fact_mean).square().mean().sqrt(),
            'local_vs_factorized_count_l1_mean': (local-fact).abs().sum((1,2)).mean(),
            'sorted_vs_dense_identity_error': raw_error,
            'fast_vs_full_selected_identity_error': error(selected, self._identity_features(identity)),
            'common_identity_coordinates_unchanged_error': error(self._identity_tables(batch)[0], identity[0]),
            'joint_vs_product_entropy_gap_mean': (identity[2][:,:,0]-identity[1][:,:,0]).mean(),
            'joint_vs_product_collision_delta_mean': (identity[1][:,:,1]-identity[2][:,:,1]).mean(),
            'padding_feature_error': error(features, self._features(padded)),
            'padding_identity_error': error(selected, selected_identity(padded)),
            'sequence_reversal_feature_error': error(features, self._features(reverse)),
            'sequence_reversal_identity_error': error(selected, selected_identity(reverse)),
            'all_empty_history_feature_abs_max': self._features(empty)[:, static_width:].abs().max(),
            'zero_length_history_feature_abs_max': self._features(zero)[:, static_width:].abs().max(),
            'all_empty_identity_abs_max': selected_identity(empty).abs().max(),
            'zero_length_identity_abs_max': selected_identity(zero).abs().max(),
            'visible_count_mean': sizes.sum(-1).mean(),
            'empty_history_fraction': (~batch['history_mask'].any(-1)).float().mean(),
            'history_at_limit_fraction': (batch['history_mask'].sum(-1) == 128).float().mean(),
            'identity_residual_rms': self.identity_head(selected).square().mean().sqrt(),
        }
        # A witness for fine-recency localization. Same coarse tables, identities,
        # actions and global fine distribution, but different group fine recency.
        device = c.device
        if self.category_vocab > 3 and self.brand_vocab > 3:
            witness = {'static': batch['static'][:1].expand(2,-1).clone(),
                       'history_category': torch.tensor([[1,2],[1,2]],device=device),
                       'history_brand': torch.tensor([[1,2],[1,2]],device=device),
                       'history_type': torch.ones((2,2),dtype=torch.long,device=device),
                       'history_age_seconds': torch.tensor([[1.,1024.],[1024.,1.]],device=device),
                       'history_mask': torch.ones((2,2),dtype=torch.bool,device=device)}
            witness['static'][:,self.category] = 2
            witness['static'][:,self.brand] = 2
            _, ws, wn = self._membership(witness)
            wh, wo = self._histogram(witness, ws, wn)[:2]
            wl = self._fine_counts(witness, ws)
            wf = self._factorized_recency(witness, wo)[0]
            wi = self._identity_features(self._identity_tables(witness,full=True), 'joint')
            result['recency_witness_coarse_error'] = error(wh[0],wh[1])
            result['recency_witness_local_count_l1_delta'] = (wl[0]-wl[1]).abs().sum()
            result['recency_witness_factorized_error'] = error(wf[0],wf[1])
            result['recency_witness_joint_identity_error'] = error(wi[0],wi[1])
            # A distinct witness for the identity readout. Target groups, margins
            # and all temporal tables coincide; only pair concentration changes.
            witness['history_category'] = torch.tensor([[1,1,2,2],[1,1,2,2]],device=device)
            witness['history_brand'] = torch.tensor([[1,1,2,2],[1,2,1,2]],device=device)
            witness['history_type'] = torch.ones((2,4),dtype=torch.long,device=device)
            witness['history_age_seconds'] = torch.zeros((2,4),device=device)
            witness['history_mask'] = torch.ones((2,4),dtype=torch.bool,device=device)
            witness['static'][:,self.category] = 3
            witness['static'][:,self.brand] = 3
            wt = self._identity_tables(witness,full=True)
            wfeatures = self._features(witness)
            result['identity_witness_common_error'] = error(wt[0][0],wt[0][1])
            result['identity_witness_product_error'] = error(wt[2][0],wt[2][1])
            result['identity_witness_joint_entropy_gap'] = wt[4][4][1,0]-wt[4][4][0,0]
            result['identity_witness_joint_collision_gap'] = wt[4][5][0,0]-wt[4][5][1,0]
            result['identity_witness_nonidentity_feature_error'] = error(wfeatures[0],wfeatures[1])
            result['witnesses_applicable'] = 1.0
        else:
            result['witnesses_applicable'] = 0.0
        for g, name in enumerate(('neither', 'category_only', 'brand_only', 'both')):
            result[name+'_count_mean'] = sizes[:, g].mean()
        return {k: float(v) for k, v in result.items()}

    def forward(self, batch):
        base = self.output(self._features(batch)).squeeze(-1)
        identity = self._identity_features(self._identity_tables(batch))
        return base + self.identity_head(identity).squeeze(-1)

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

    @staticmethod
    def _sorted_distribution(identity, group, eligible, n):
        """Conditional Shannon entropy and with-replacement collision mass.

        Sort BxL packed identity/group keys, then aggregate run counts in BxL
        storage. This never builds a BxLxL training tensor or a vocabulary table.
        Zero category/brand values are excluded from ALL three distributions.
        """
        if identity.shape[1] == 0:
            return torch.zeros_like(n), torch.zeros_like(n)
        key = torch.where(eligible, identity * 4 + group,
                          torch.full_like(identity, -1))
        ordered = key.sort(dim=-1).values
        valid = ordered >= 0
        start = valid.clone()
        start[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
        run = (start.long().cumsum(-1) - 1).clamp_min(0)
        mass = n.new_zeros(identity.shape).scatter_add(1, run, valid.float())
        run_group = torch.zeros_like(identity).scatter_add(
            1, run, torch.where(start, ordered.remainder(4), torch.zeros_like(ordered)))
        sum_square = torch.zeros_like(n).scatter_add(1, run_group, mass.square())
        sum_xlogx = torch.zeros_like(n).scatter_add(
            1, run_group, mass * mass.clamp_min(1).log())
        denom = n.clamp_min(1)
        entropy = torch.where(n > 0, (denom.log() - sum_xlogx / denom).clamp_min(0),
                              torch.zeros_like(n))
        collision = sum_square / denom.square()
        return entropy, collision

    def _identity_tables(self, batch, semantic=None, full=False):
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        # Complete-pair support shared by category, brand and pair summaries.
        # Missing/missing is never interpreted as a repeated product identity.
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        membership = semantic * eligible[:, None, :].float()
        n = membership.sum(-1)
        hc, cc = self._sorted_distribution(c, group, eligible, n)
        hb, cb = self._sorted_distribution(b, group, eligible, n)
        # Analytic product of the two empirical marginals on the SAME support.
        # Self combinations are included. No random shuffle or derangement.
        hi, ci = hc + hb, cc * cb
        # The product cell saves the third identity sort in actual forward.
        # A full table is requested only for diagnostics; no randomness or
        # parameter allocation depends on this branch.
        if self.identity_mode == 'joint' or full:
            hj, cj = self._sorted_distribution(c * self.brand_vocab + b, group, eligible, n)
        else:
            hj, cj = hi, ci
        scale = math.log(129.0)  # fixed supplied history limit, not fitted
        common = torch.stack([torch.log1p(n) / scale, hc / scale, hb / scale, cc, cb], -1)
        joint = torch.stack([hj / scale, cj], -1)
        independent = torch.stack([hi / scale, ci], -1)
        return common, joint, independent, n, (hc, cc, hb, cb, hj, cj), eligible, group

    def _identity_features(self, tables, mode=None):
        mode = self.identity_mode if mode is None else mode
        common, joint, independent = tables[:3]
        selected = joint if mode == 'joint' else independent
        if mode == 'marginal':
            selected = torch.zeros_like(joint)
        features = torch.cat([common, selected], -1).flatten(1)
        return torch.zeros_like(features) if mode == 'zero' else features

    @staticmethod
    def _dense_distribution(identity, group, eligible, n):
        """Independent quadratic reference, DIAGNOSTICS ONLY, never training."""
        if identity.shape[1] == 0:
            return torch.zeros_like(n), torch.zeros_like(n)
        same = ((identity[:, :, None] == identity[:, None, :]) &
                (group[:, :, None] == group[:, None, :]) &
                eligible[:, :, None] & eligible[:, None, :])
        per_event = same.sum(-1).float()
        membership = torch.stack([(group == g) & eligible for g in range(4)], 1).float()
        sum_square = torch.bmm(membership, per_event[:, :, None]).squeeze(-1)
        sum_xlogx = torch.bmm(membership, per_event.clamp_min(1).log()[:, :, None]).squeeze(-1)
        denom = n.clamp_min(1)
        entropy = torch.where(n > 0, (denom.log() - sum_xlogx / denom).clamp_min(0),
                              torch.zeros_like(n))
        return entropy, sum_square / denom.square()
