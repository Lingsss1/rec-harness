"""Action/fine-recency association from exact visible-history count tables.

Both cells preserve the cheaper local-product summary and compute the same
observed and coarse-conditional product counts. Only the selected association
changes before an identical diagonal embedding interaction and classifier.
No event embeddings, labels, I/O, optimizer/loss changes or data selection.
Source checks are local; Torch execution and measured outcomes are host-pending.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'content_mode': 'off', 'recency_mode': 'fine', 'recency_localization': 'local', 'identity_mode': 'independent', 'coupling_mode': 'observed'}

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
        self.coupling_mode = self.config['coupling_mode']
        if self.coupling_mode not in ('observed', 'conditional'):
            raise ValueError('Unknown action/fine-age coupling mode')
        if self.recency_localization != 'local' or self.identity_mode != 'independent':
            raise ValueError('This contrast fixes local recency and product identity')
        static_width = len(meta['fields']) * dim
        self.input_width = static_width + 8 * dim + 68
        parent_width = static_width + 11 * dim + 68
        parent_output = 128 * (parent_width + 1) + 64 * 129 + 65
        self.reference_active_nonstatic = parent_output + 38 * dim + 464
        # Output has h*(input_width+65)+129 parameters. Count-match the
        # parent INCLUDING active action/recency and identity-head parameters.
        # At F=18: input=484, h=139, nonstatic=77512 (reference target=77617).
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


    def _base_features(self, batch):
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


    def _temporal_tables(self, batch, semantic, coarse_observed):
        """Exact observed and coarse-conditional expected group/action/fine counts.

        g = target-match group; a = behavior; s = coarse age; k = log2 age.
        Keep N[g,a,s] and N[g,s,k], including fine bins crossing coarse bounds.
        E[g,a,k] = sum_s N[g,a,s] N[g,s,k] / max(N[g,s],1).
        Both alternatives are computed in every forward; selection changes
        only the supplied association, with no learned count construction.
        """
        fine, coarse = self._age_ids(batch)
        action = (batch['history_type'] - 1).clamp(0, 3)
        shape = (semantic.shape[0], 4, 132)
        observed = semantic.new_zeros(shape).scatter_add(
            2, (33 * action + fine)[:, None, :].expand(-1, 4, -1), semantic)
        coarse_fine = semantic.new_zeros(shape).scatter_add(
            2, (33 * coarse + fine)[:, None, :].expand(-1, 4, -1), semantic)
        observed = observed.reshape(-1, 4, 4, 33)
        coarse_fine = coarse_fine.reshape(-1, 4, 4, 33)
        coarse_mass = coarse_fine.sum(-1)
        allocation = coarse_observed / coarse_mass.clamp_min(1).unsqueeze(-2)
        expected = torch.matmul(allocation, coarse_fine)
        return observed, expected, coarse_fine, allocation

    def _coupling(self, counts, sizes):
        # No event embedding or BxLxL tensor is used in the train/inference path.
        age_by_action = torch.matmul(counts, self.recency.weight)
        diagonal = (age_by_action * self.action.weight[1:5][None, None, :, :]).sum(-2)
        return 0.5 * diagonal / sizes.clamp_min(1).unsqueeze(-1)

    def _feature_parts(self, batch):
        static = torch.cat([e(batch['static'][:, i]) for i, e in enumerate(self.embeddings)], -1)
        _, semantic, sizes = self._membership(batch)
        statistics = self._statistics(self._match_counts(batch))
        histogram, coarse_observed = self._histogram(batch, semantic, sizes)[:2]
        observed, expected, coarse_fine, allocation = self._temporal_tables(
            batch, semantic, coarse_observed)
        # Sum over actions: exact same group-local fine counts as the parent.
        context = self._context_from_counts(observed.sum(-2), coarse_observed, sizes)
        base = torch.cat([static, context.flatten(1), statistics, histogram], -1)
        selected = observed if self.coupling_mode == 'observed' else expected
        coupling = self._coupling(selected, sizes)
        return base, coupling, (observed, expected, coarse_fine, allocation), semantic, sizes

    def _features(self, batch):
        base, coupling = self._feature_parts(batch)[:2]
        return torch.cat([base, coupling.flatten(1)], -1)

    def _direct_coupling(self, batch, semantic, sizes):
        # Independent event-embedding reference. Diagnostics only.
        if batch['history_mask'].shape[1] == 0:
            return semantic.new_zeros((semantic.shape[0], 4, self.config['embedding_dim']))
        fine, _ = self._age_ids(batch)
        events = self.action(batch['history_type']) * self.recency(fine)
        return 0.5 * torch.bmm(semantic, events) / sizes.clamp_min(1).unsqueeze(-1)

    def diagnostics(self, batch):
        # Candidate-supplied implementation checks on the host first512 rows;
        # not an independent performance, uncertainty or attribution estimate.
        base, coupling, tables, semantic, sizes = self._feature_parts(batch)
        observed, expected, coarse_fine, allocation = tables
        coarse = self._histogram(batch, semantic, sizes)[1]
        by_coarse = allocation.unsqueeze(-1) * coarse_fine.unsqueeze(-3)
        # Shape [B,g,a,s,k]; only built for diagnostics, never in forward.
        observed_moment = self._coupling(observed, sizes)
        expected_moment = self._coupling(expected, sizes)
        identity = self._identity_features(self._identity_tables(batch, semantic))
        features = torch.cat([base, coupling.flatten(1)], -1)
        static_width = len(self.embeddings) * self.config['embedding_dim']
        padded = dict(batch)
        for key in ('history_category', 'history_brand', 'history_type', 'history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'], batch[key], torch.ones_like(batch[key]))
        reverse = {k: (v.flip(1) if k.startswith('history_') else v) for k,v in batch.items()}
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {k: (v[:, :0] if k.startswith('history_') else v) for k,v in batch.items()}
        total = sum(p.numel() for p in self.parameters())
        static_count = sum(e.weight.numel() for e in self.embeddings)
        def error(x,y):
            return (x-y).abs().max()
        def ident(x):
            return self._identity_features(self._identity_tables(x))
        result = {
            'allocated_parameters': total,
            'active_nonstatic_parameters': total-static_count,
            'reference_active_nonstatic_target': self.reference_active_nonstatic,
            'active_nonstatic_parameter_gap': total-static_count-self.reference_active_nonstatic,
            'first_hidden': self.first_hidden,
            'feature_width': self.input_width,
            'coupling_coordinates': coupling.shape[1]*coupling.shape[2],
            'observed_coupling_enabled': float(self.coupling_mode == 'observed'),
            'identity_sorts_per_forward': 2.0,
            'event_embedding_consumers_forward': 0.0,
            'parent_feature_error': error(base,self._base_features(batch)),
            'fine_count_margin_error': error(observed.sum(-2),self._fine_counts(batch,semantic)),
            'observed_group_mass_error': error(observed.sum((-2,-1)),sizes),
            'expected_group_mass_error': error(expected.sum((-2,-1)),sizes),
            'expected_fine_margin_error': error(expected.sum(-2),observed.sum(-2)),
            'expected_action_margin_error': error(expected.sum(-1),coarse.sum(-1)),
            'expected_action_coarse_margin_error': error(by_coarse.sum(-1),coarse),
            'expected_coarse_fine_margin_error': error(by_coarse.sum(-3),coarse_fine),
            'expected_aggregation_error': error(by_coarse.sum(-2),expected),
            'direct_event_coupling_error': error(observed_moment,self._direct_coupling(batch,semantic,sizes)),
            'direct_event_context_error': error(self._context_from_counts(observed.sum(-2),coarse,sizes),self._direct_context(batch,semantic,sizes)),
            'observed_expected_count_l1_mean': (observed-expected).abs().sum((1,2,3)).mean(),
            'observed_expected_moment_rms': (observed_moment-expected_moment).square().mean().sqrt(),
            'association_present_fraction': ((observed-expected).abs().sum((1,2,3))>1e-4).float().mean(),
            'coupling_feature_rms': coupling.square().mean().sqrt(),
            'padding_feature_error': error(features,self._features(padded)),
            'sequence_reversal_feature_error': error(features,self._features(reverse)),
            'padding_identity_error': error(identity,ident(padded)),
            'sequence_reversal_identity_error': error(identity,ident(reverse)),
            'all_empty_history_feature_abs_max': self._features(empty)[:,static_width:].abs().max(),
            'zero_length_history_feature_abs_max': self._features(zero)[:,static_width:].abs().max(),
            'all_empty_identity_abs_max': ident(empty).abs().max(),
            'zero_length_identity_abs_max': ident(zero).abs().max(),
            'finite_logit_fraction': torch.isfinite(self.forward(batch)).float().mean(),
            'visible_count_mean': sizes.sum(-1).mean(),
            'empty_history_fraction': (~batch['history_mask'].any(-1)).float().mean(),
            'history_at_limit_fraction': (batch['history_mask'].sum(-1)==128).float().mean(),
        }
        # Two histories share all parent features, category/brand identities,
        # coarse action-age and group/coarse/fine margins. Only action/fine
        # association changes: aligned vs mixed assignments within coarse s=0.
        if self.category_vocab > 3 and self.brand_vocab > 3:
            device = batch['static'].device
            witness = {
                'static':batch['static'][:1].expand(2,-1).clone(),
                'history_category':torch.ones((2,4),dtype=torch.long,device=device),
                'history_brand':torch.ones((2,4),dtype=torch.long,device=device),
                'history_type':torch.tensor([[1,1,2,2],[1,1,2,2]],device=device),
                'history_age_seconds':torch.tensor([[1.,1.,1024.,1024.],[1.,1024.,1.,1024.]],device=device),
                'history_mask':torch.ones((2,4),dtype=torch.bool,device=device),
            }
            witness['static'][:,self.category] = 3
            witness['static'][:,self.brand] = 3
            wb,wc,wt,ws,wn = self._feature_parts(witness)
            wo,we = wt[:2]
            wi = ident(witness)
            wh = self._histogram(witness,ws,wn)[1]
            wm = self._coupling(wo,wn)
            wf = self._coupling(we,wn)
            # Algebraic nonzero-association witness. It is possible for a
            # learned diagonal projection to ignore this particular witness.
            predicted = 0.125*(self.action.weight[1]-self.action.weight[2])*(self.recency.weight[2]-self.recency.weight[11])
            result.update({
                'witness_applicable':1.0,
                'witness_parent_feature_error':error(wb[0],wb[1]),
                'witness_identity_error':error(wi[0],wi[1]),
                'witness_coarse_histogram_error':error(wh[0],wh[1]),
                'witness_coarse_fine_margin_error':error(wt[2][0],wt[2][1]),
                'witness_observed_count_l1_delta':(wo[0]-wo[1]).abs().sum(),
                'witness_conditional_count_error':error(we[0],we[1]),
                'witness_conditional_moment_error':error(wf[0],wf[1]),
                'witness_observed_moment_rms_delta':(wm[0]-wm[1]).square().mean().sqrt(),
                'witness_moment_formula_error':error(wm[0,0]-wm[1,0],predicted),
            })
        else:
            result['witness_applicable'] = 0.0
        for g,name in enumerate(('neither','category_only','brand_only','both')):
            result[name+'_count_mean'] = sizes[:,g].mean()
        return {k:float(v) for k,v in result.items()}


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


    def _direct_context(self, batch, semantic, sizes):
        # Independent event-level reference, diagnostics only.
        if batch['history_mask'].shape[1] == 0:
            return semantic.new_zeros((semantic.shape[0], 4, self.config['embedding_dim']))
        fine, _ = self._age_ids(batch)
        events = self.action(batch['history_type']) + self.recency(fine)
        return torch.bmm(semantic, events) / sizes.clamp_min(1).unsqueeze(-1)


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

