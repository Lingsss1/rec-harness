"""Prospective Shannon-entropy x collision-mass sufficiency experiment.

Inherited from g2-r0017-sketch256, source SHA256
6f7a98e11f5b2e82cc634d8891f0b009d176ce1bf9e0383944873ff96d12f64f.
All trainable modules, allocation order, support rule, and history context are
unchanged. Only marginal diversity coordinates are removed; support remains.
Collision is sum(p**2), INCLUDING self-pairs, for coarsened empirical IDs.
This is not an unbiased estimator or evidence of category-brand event pairing.
Allocated capacity is matched; zeroed inputs reduce effective capacity.
Local source checks only. Numeric validity and predictive outcomes await host.
"""
import math
import torch
from torch import nn

CONFIG = {'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'content_mode': 'off', 'recency_mode': 'fine', 'identity_mode': 'independent', 'recency_localization': 'local', 'recency_weighting': 'log', 'identity_estimator': 'sketch', 'sketch_bins': 256, 'sketch_salt': 0, 'use_entropy': True, 'use_collision': False}

class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.use_entropy = self.config['use_entropy']
        self.use_collision = self.config['use_collision']
        if type(self.use_entropy) is not bool or type(self.use_collision) is not bool:
            raise ValueError('Statistic switches must be booleans')
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
        self.recency_weighting = self.config['recency_weighting']
        self.identity_estimator = self.config['identity_estimator']
        self.sketch_bins = self.config['sketch_bins']
        self.sketch_salt = self.config['sketch_salt']
        if self.identity_estimator != 'sketch':
            raise ValueError('Unknown identity estimator')
        if self.sketch_bins != 256 or self.sketch_salt != 0:
            raise ValueError('Unregistered sketch configuration')
        if self.identity_mode != 'independent':
            raise ValueError('This study fixes product-of-marginals identity')
        if self.recency_weighting != 'log':
            raise ValueError('Unknown recency weighting')
        if self.identity_mode not in ('joint', 'independent'):
            raise ValueError('Unknown identity mode')
        if self.recency_localization != 'local':
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


    @staticmethod
    def _recency_weights(counts, mode):
        # Sum over the supplied bins, not events outside the visible prefix.
        # Both modes have total mass one for nonempty groups and zero for
        # empty groups. Unit mass removes a trivial added-intensity contrast.
        if mode == 'log':
            mass = torch.log1p(counts)
            return mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
        if mode == 'mean':
            return counts / counts.sum(-1, keepdim=True).clamp_min(1)
        raise ValueError('Unknown recency weighting')


    def _context_from_counts(self, counts, observed, sizes):
        if self.recency_weighting == 'mean':
            # Preserve the parent arithmetic, including its rounding order.
            return (torch.matmul(observed.sum(-1), self.action.weight[1:5]) +
                    torch.matmul(counts, self.recency.weight)) / sizes.clamp_min(1).unsqueeze(-1)
        action_mean = (torch.matmul(observed.sum(-1), self.action.weight[1:5]) /
                       sizes.clamp_min(1).unsqueeze(-1))
        recency_mean = torch.matmul(self._recency_weights(counts, 'log'), self.recency.weight)
        return action_mean + recency_mean


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


    def _bucket_ids(self, identity, bins=None, salt=None):
        bins = self.sketch_bins if bins is None else bins
        salt = self.sketch_salt if salt is None else salt
        # Modulo before multiplication bounds intermediates below 2**62.
        # No fitted hash, random draws, vocabulary-sized table or label input.
        multiplier, offset = ((1103515245, 12345) if salt == 0 else (1664525111, 1013904223))
        prime = 2147483647
        return ((identity.remainder(prime) * multiplier + offset)
                .remainder(prime).remainder(bins))


    def _bucket_table(self, identity, group, eligible, n, bins=None, salt=None):
        bins = self.sketch_bins if bins is None else bins
        hashed = self._bucket_ids(identity, bins, salt)
        key = group * bins + hashed
        mass = n.new_zeros((identity.shape[0], 4 * bins))
        mass = mass.scatter_add(1, key, eligible.float())
        return mass.reshape(identity.shape[0], 4, bins), hashed


    def _bucket_distribution(self, identity, group, eligible, n, full=False):
        # Count coarsened identities once. Disabled statistics incur no mass
        # reduction or entropy logarithms. Support-only skips hashing entirely.
        zero = torch.zeros_like(n)
        need_entropy = full or self.use_entropy
        need_collision = full or self.use_collision
        if not (need_entropy or need_collision):
            return zero, zero
        mass, _ = self._bucket_table(identity, group, eligible, n)
        denom = n.clamp_min(1)
        if need_entropy:
            xlogx = (mass * mass.clamp_min(1).log()).sum(-1)
            entropy = torch.where(n > 0,
                                  (denom.log() - xlogx / denom).clamp_min(0), zero)
        else:
            entropy = zero
        collision = mass.square().sum(-1) / denom.square() if need_collision else zero
        return entropy, collision

    def _identity_tables(self, batch, semantic=None, full=False):
        if semantic is None:
            semantic = self._membership(batch)[1]
        c, b = batch['history_category'], batch['history_brand']
        eligible = batch['history_mask'] & (c > 0) & (b > 0)
        group = semantic.argmax(dim=1)
        n = (semantic * eligible[:, None, :].float()).sum(-1)
        hc, cc = self._bucket_distribution(c, group, eligible, n, full)
        hb, cb = self._bucket_distribution(b, group, eligible, n, full)
        hi, ci = hc + hb, cc * cb
        scale = math.log(129.0)
        common = torch.stack([torch.log1p(n)/scale, hc/scale, hb/scale, cc, cb], -1)
        independent = torch.stack([hi/scale, ci], -1)
        # Joint slot is unused by this study; no actual joint identity is computed.
        return common, independent, independent, n, (hc, cc, hb, cb, hi, ci), eligible, group

    def diagnostics(self, batch):
        # Independent dense oracle and invariances only, first 512 impressions.
        # Full-statistic and quadratic calculations NEVER run in forward.
        _, semantic, sizes = self._membership(batch)
        tables = self._identity_tables(batch, semantic)
        full_tables = self._identity_tables(batch, semantic, full=True)
        selected = self._identity_features(tables)
        full = self._identity_features(full_tables)
        n, eligible, group = tables[3], tables[5], tables[6]
        c, b = batch['history_category'], batch['history_brand']
        hc, cc = self._dense_distribution(self._bucket_ids(c), group, eligible, n)
        hb, cb = self._dense_distribution(self._bucket_ids(b), group, eligible, n)
        scale = math.log(129.0)
        dense = torch.stack([torch.log1p(n)/scale, hc/scale, hb/scale,
                             cc, cb, (hc+hb)/scale, cc*cb], -1)
        keep = selected.new_tensor([1.0, float(self.use_entropy), float(self.use_entropy),
                                   float(self.use_collision), float(self.use_collision),
                                   float(self.use_entropy), float(self.use_collision)])
        expected = (dense*keep).flatten(1)
        def err(x,y):
            return (x-y).abs().max()
        def rms(x):
            return x.square().mean().sqrt()
        features = self._features(batch)
        padded = dict(batch)
        for key in ('history_category','history_brand','history_type','history_age_seconds'):
            padded[key] = torch.where(batch['history_mask'],batch[key],torch.ones_like(batch[key]))
        reverse = {key: (value.flip(1) if key.startswith('history_') else value)
                   for key,value in batch.items()}
        empty = dict(batch)
        empty['history_mask'] = torch.zeros_like(batch['history_mask'])
        zero = {key: (value[:,:0] if key.startswith('history_') else value)
                for key,value in batch.items()}
        static_width = len(self.embeddings)*self.config['embedding_dim']
        support = full.reshape(full.shape[0],4,7)*full.new_tensor([1,0,0,0,0,0,0])
        entropy_only = full.reshape(full.shape[0],4,7)*full.new_tensor([1,1,1,0,0,1,0])
        collision_only = full.reshape(full.shape[0],4,7)*full.new_tensor([1,0,0,1,1,0,1])
        total = sum(p.numel() for p in self.parameters())
        static_count = sum(e.weight.numel() for e in self.embeddings)
        # Algebra witness: equal support and collision, unequal Shannon entropy.
        witness = selected.new_tensor([[3,1,1,1],[2,2,2,0]])/6.0
        witness_entropy = -(witness*witness.clamp_min(1e-9).log()).sum(-1)
        witness_collision = witness.square().sum(-1)
        result = {
            'allocated_parameters': total,
            'allocated_nonstatic_parameters': total-static_count,
            'identity_head_parameters': sum(p.numel() for p in self.identity_head.parameters()),
            'identity_nonzero_input_columns_max': 4*(1+3*int(self.use_entropy)+3*int(self.use_collision)),
            'identity_first_layer_zero_input_weights': 16*4*(6-3*int(self.use_entropy)-3*int(self.use_collision)),
            'first_hidden': self.first_hidden,
            'feature_width': self.input_width,
            'entropy_enabled': float(self.use_entropy),
            'collision_enabled': float(self.use_collision),
            'bucket_tables_per_forward': 2.0 if self.use_entropy or self.use_collision else 0.0,
            'bucket_cells_per_field': 4*self.sketch_bins if self.use_entropy or self.use_collision else 0.0,
            'sketch_bins': self.sketch_bins,
            'sketch_salt': self.sketch_salt,
            'dense_hashed_full_stat_error': err(full,dense.flatten(1)),
            'dense_hashed_selected_stat_error': err(selected,expected),
            'selected_full_mask_error': err(selected,(full.reshape(full.shape[0],4,7)*keep).flatten(1)),
            'omitted_coordinate_abs_max': (selected.reshape(selected.shape[0],4,7)*(1-keep)).abs().max(),
            'selected_vs_full_feature_rms': rms(selected-full),
            'same_checkpoint_residual_rms': rms(self.identity_head(selected)),
            'same_checkpoint_full_replacement_residual_rms_delta': rms(self.identity_head(selected)-self.identity_head(full)),
            'same_checkpoint_entropy_removal_residual_rms_delta': rms(self.identity_head(full)-self.identity_head(collision_only.flatten(1))),
            'same_checkpoint_collision_removal_residual_rms_delta': rms(self.identity_head(full)-self.identity_head(entropy_only.flatten(1))),
            'same_checkpoint_support_replacement_residual_rms_delta': rms(self.identity_head(full)-self.identity_head(support.flatten(1))),
            'padding_feature_error': err(features,self._features(padded)),
            'reversal_feature_error': err(features,self._features(reverse)),
            'padding_identity_error': err(selected,self._identity_features(self._identity_tables(padded))),
            'reversal_identity_error': err(selected,self._identity_features(self._identity_tables(reverse))),
            'empty_history_feature_abs_max': self._features(empty)[:,static_width:].abs().max(),
            'zero_length_feature_abs_max': self._features(zero)[:,static_width:].abs().max(),
            'empty_identity_abs_max': self._identity_features(self._identity_tables(empty)).abs().max(),
            'zero_length_identity_abs_max': self._identity_features(self._identity_tables(zero)).abs().max(),
            'zero_input_head_abs_max': self.identity_head(torch.zeros_like(selected)).abs().max(),
            'finite_logit_fraction': torch.isfinite(self.forward(batch)).float().mean(),
            'support_count_mean': n.sum(-1).mean(),
            'missing_pair_count_mean': (sizes.sum(-1)-n.sum(-1)).mean(),
            'empty_history_fraction': (~batch['history_mask'].any(-1)).float().mean(),
            'witness_collision_gap_abs': (witness_collision[0]-witness_collision[1]).abs(),
            'witness_entropy_gap_nats': witness_entropy[0]-witness_entropy[1],
        }
        return {key: float(value) for key,value in result.items()}
