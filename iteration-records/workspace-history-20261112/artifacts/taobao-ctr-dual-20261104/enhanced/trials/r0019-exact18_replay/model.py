"""Recipient allocation controls for complete-row magnitude scaling.

Same half-tangent normalized-product forward, parameter creation order and BCE
as r0017. Let G be the complete supervised row gradient and M its inherited
conflict-conditioned magnitude target. Four prospective arms:
 none: G; total_magnitude: inherited M; rotate_active: redistribute the same
per-row scalar multiset by a deterministic half rotation over nonpadding rows
with ||G||^2>1e-30, in table-index order; uniform_energy: alpha*G on those rows,
where alpha matches M's per-table raw-gradient squared norm on that support.
Both new rules preserve each nonzero row's raw-gradient direction. Uniform
allocation uses conflict only to set one table energy, not recipient identities.
Rotation removes exact recipient assignment, not all conflict dependence: its
scales and the donor support are still training-state dependent. Scalar moments
are matched by rotation; weighted gradient energy and Adam updates need not be.

All arms compute every hypothetical allocation and candidate-authored checks.
No extra learned parameters, RNG/optimizer edits, external state, validation
labels or changes to the scalar BCE. Corrections use detached linear surrogates.
Two actual-backward snapshots are at training calls0/512; running summaries end
at the saved checkpoint. First512 validation checks use inputs only. These are
candidate diagnostics, not independent audit or optimizer-mediation evidence.
No local Torch execution is claimed. Older r0016 projection source is absent;
its mismatch with r0017 stays unresolved. No literature priority is claimed.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'total_magnitude'}


class Model(nn.Module):
    def __init__(self,meta,config=None):
        super().__init__()
        config={**CONFIG,**(config or {})}
        self.config=config
        self.kind=config['kind']
        self.category=meta['category_field_index'];self.brand=meta['brand_field_index']
        dim=config['embedding_dim']
        self.embeddings=nn.ModuleList([nn.Embedding(size,dim,padding_idx=0) for size in meta['buckets']])
        self.action=nn.Embedding(5,dim,padding_idx=0)
        self.recency=nn.Embedding(33,dim,padding_idx=0)
        self.attention=nn.Sequential(nn.Linear(dim*4,64),nn.PReLU(),nn.Linear(64,1))
        layers=[]
        width=len(meta['fields'])*dim+(dim if self.kind!='static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width,hidden),nn.ReLU()]);width=hidden
        layers.append(nn.Linear(width,1))
        self.output=nn.Sequential(*layers)
        if self.category == self.brand:
            raise ValueError('This registered study requires distinct category/brand tables')
        if config['kind'] != 'din' or config['product_pairing'] != 'aligned' or config['product_normalization'] != 'unit':
            raise ValueError('This study fixes target-aware aligned unit products')
        if config['normalization_gradient'] not in ('full', 'scale_half_tangent'):
            raise ValueError('Only the fixed own/half normalization rules are admitted')
        if config['row_intervention'] not in ('none', 'total_magnitude', 'rotate_active', 'uniform_energy'):
            raise ValueError('Only the registered allocation replacements are admitted')
        if config['normalization_gradient'] != 'scale_half_tangent':
            raise ValueError('This bounded study fixes the inherited half-scale rule')
        self._path_cache = None
        self._cached_logits = None
        self._training_loss_calls = 0
        self.register_buffer('_snapshots', torch.zeros(2, 14, dtype=torch.float64))
        self._snapshots[:, 0] = -1
        self.register_buffer('_running', torch.zeros(18, dtype=torch.float64))


    def _encode(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        features = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        history = category + brand
        action = self.action(batch['history_type'])
        history = history + action
        recency = self.recency(ages)
        # Keep the parent's operation order as well as its association.
        history = history + recency
        return fields, features, history, (category, brand, action, recency)

    @staticmethod
    def _weights(scores, mask):
        # Identical frozen softmax arithmetic, including all-masked rows.
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

    def _logits(self, features, summary):
        features = torch.cat([features, summary], dim=-1)
        return self.output(features).squeeze(-1)

    @staticmethod
    def _rotation_donors(inverse, valid_regular, gradient):
        """Integer donor identities; detached keys, stable position tie break.

        No index/age neighborhood is imposed by scale ordering. Circular
        adjacency has one extreme wrap; scalar similarity is a manipulation
        check, not guaranteed for arbitrary short or heavy-tailed groups.
        """
        positions = torch.arange(inverse.shape[-1], device=inverse.device,
                                 dtype=torch.long)[None, :].expand_as(inverse)
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            keys = inverse.detach().masked_fill(~valid_regular, float('inf'))
        elif gradient in ('history_adjacent_tangent', 'history_reassigned_tangent'):
            keys = positions.to(inverse.dtype).masked_fill(~valid_regular, float('inf'))
        else:
            raise ValueError('rotation mode required')
        order = keys.argsort(dim=-1, stable=True)
        rank = torch.zeros_like(order).scatter(-1, order, positions)
        count = valid_regular.sum(-1, keepdim=True)
        shift = 1 if gradient.endswith('adjacent_tangent') else count // 2
        donor_rank = (rank + shift).remainder(count.clamp_min(1))
        donor = order.gather(-1, donor_rank)
        return torch.where(valid_regular, donor, positions)

    @staticmethod
    def _event_scales(inverse, valid_regular, gradient, action_type=None):
        """Detached, within-impression scales; no labels or other rows used."""
        if gradient in ('action_mean_tangent', 'action_reassigned_tangent',
                        'action_adjacent_tangent'):
            if action_type is None:
                raise ValueError('behavior-stratified scales require history_type')
            base_mode = {'action_mean_tangent': 'history_mean_tangent',
                         'action_reassigned_tangent': 'history_reassigned_tangent',
                         'action_adjacent_tangent': 'history_adjacent_tangent'}[gradient]
            result = inverse
            for action in range(1, 5):
                selected = valid_regular & (action_type == action)
                assigned = Model._event_scales(inverse, selected, base_mode)
                result = torch.where(selected, assigned, result)
            return result
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            donor = Model._rotation_donors(inverse, valid_regular, gradient)
            return inverse.gather(-1, donor)
        if gradient == 'full':
            return inverse
        if gradient == 'constant_tangent':
            return torch.ones_like(inverse) * 0.2
        count = valid_regular.sum(-1, keepdim=True)
        values = torch.where(valid_regular, inverse, torch.zeros_like(inverse))
        if gradient == 'history_mean_tangent':
            mean = values.sum(-1, keepdim=True) / count.clamp_min(1)
            return torch.where(valid_regular, mean, inverse)
        if gradient in ('history_reassigned_tangent', 'history_adjacent_tangent'):
            # Pack valid regular events in supplied temporal order. A rotation
            # by floor(n/2) or one has no fixed index for n>=2; n=0/1 is unchanged.
            # Padding and floored events contribute zero and are never donors.
            rank = valid_regular.long().cumsum(-1) - 1
            packed = torch.zeros_like(inverse).scatter_add(
                -1, rank.clamp_min(0), values)
            shift = 1 if gradient == 'history_adjacent_tangent' else count // 2
            donor = (rank + shift).remainder(count.clamp_min(1))
            rotated = packed.gather(-1, donor)
            return torch.where(valid_regular, rotated, inverse)
        raise ValueError('unknown normalization_gradient')

    @staticmethod
    def _normalize_product(product, gradient='full', mask=None, action_type=None):
        squared = product.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-6).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        regular = squared >= 1e-6
        valid_regular = regular.squeeze(-1)
        if mask is not None:
            valid_regular = valid_regular & mask
        inverse = denominator.squeeze(-1).reciprocal().detach()
        target = Model._event_scales(inverse, valid_regular, gradient, action_type)
        # On valid regular events J=target*(I-u*u^T). On floored
        # events retain J=1000*I. Row replacements retain full J on
        # masked events. All cross-event scale derivatives are stopped.
        coefficient = torch.where(regular, denominator * target.unsqueeze(-1),
                                  torch.ones_like(denominator)).detach()
        surrogate = unit * coefficient
        return unit.detach() + (surrogate - surrogate.detach())

    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def _product(self, fields, components, mask):
        category, brand, _, _ = components
        qc, qb = fields[self.category], fields[self.brand]
        if self.training and torch.is_grad_enabled():
            # These four clones are exclusive product-path endpoints. Other
            # static, attention-query, history and value paths use originals.
            qc, qb, category, brand = qc.clone(), qb.clone(), category.clone(), brand.clone()
            self._path_cache = (qc, qb, category, brand)
        product = qc[:, None, :] * category + qb[:, None, :] * brand
        return self._normalize_product(product, self.config['normalization_gradient'], mask)

    def forward(self, batch):
        self._path_cache = None
        self._cached_logits = None
        fields, features, history, components = self._encode(batch)
        product = self._product(fields, components, batch['history_mask'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
        if self._path_cache is not None:
            self._cached_logits = logits
        return logits

    @staticmethod
    def _aggregate(weight, ids, gradient, valid):
        # Accumulate actual product-only gradients across the whole training
        # batch, respecting repeated IDs and each table's padding policy.
        out = torch.zeros_like(weight)
        values = gradient.detach().reshape(-1, gradient.shape[-1])
        values = values * valid.reshape(-1, 1).to(values.dtype)
        out.index_add_(0, ids.reshape(-1), values)
        out[0].zero_()
        return out

    @staticmethod
    def _row_targets(q, h):
        # All operations are detached in training_loss. Absolute squared-norm
        # degeneracy threshold is fixed prospectively, not tuned to outcomes.
        dot = (q * h).sum(-1, keepdim=True)
        q2 = q.square().sum(-1, keepdim=True)
        h2 = h.square().sum(-1, keepdim=True)
        original = q + h
        s2 = original.square().sum(-1, keepdim=True)
        active = (q2 > 1e-30) & (h2 > 1e-30)
        conflict = active & (dot < 0)
        eligible = conflict & (s2 > 1e-30)
        negative_dot = torch.where(eligible, dot, torch.zeros_like(dot))
        projected_q = q - negative_dot / h2.clamp_min(1e-30) * h
        projected_h = h - negative_dot / q2.clamp_min(1e-30) * q
        projected = projected_q + projected_h
        ratio = (projected.square().sum(-1, keepdim=True) /
                 s2.clamp_min(1e-30)).clamp_min(0).sqrt()
        norm_match = torch.where(eligible, original * ratio, original)
        projected = torch.where(eligible, projected, original)
        return original, projected, norm_match, active, conflict, eligible, dot

    @staticmethod
    def _norm(tensors):
        return sum(t.square().sum(dtype=torch.float64) for t in tensors).clamp_min(0).sqrt()

    @staticmethod
    def _total_targets(total, q, h):
        """Detached common-state factorial. Caller stops all input gradients.

        Raw delta is retained verbatim for the projection bridge. All arms
        share the additional G/P norm guard, so degenerate cases cannot be
        silently counted as successful direction/magnitude matching.
        """
        row = Model._row_targets(q, h)
        product, projected = row[:2]
        raw_delta = projected - product
        proposed = total + raw_delta
        g2 = total.square().sum(-1, keepdim=True)
        p2 = proposed.square().sum(-1, keepdim=True)
        common = row[5] & (g2 > 1e-30) & (p2 > 1e-30)
        common = common.clone()
        common[0] = False
        raw_delta = torch.where(common, raw_delta, torch.zeros_like(raw_delta))
        proposed = total + raw_delta
        # Inputs/coefficients are detached. No derivative through these norms.
        direction = proposed * (g2.clamp_min(1e-30) /
                                p2.clamp_min(1e-30)).sqrt()
        magnitude = total * (p2.clamp_min(1e-30) /
                             g2.clamp_min(1e-30)).sqrt()
        direction = torch.where(common, direction, total)
        magnitude = torch.where(common, magnitude, total)
        deltas = {'none': torch.zeros_like(total), 'project': raw_delta,
                  'total_direction': torch.where(common, direction - total,
                                                torch.zeros_like(total)),
                  'total_magnitude': torch.where(common, magnitude - total,
                                                torch.zeros_like(total))}
        # Audit the reconstructible intended gradients, including roundoff.
        direction = total + deltas['total_direction']
        magnitude = total + deltas['total_magnitude']
        return row, common, proposed, direction, magnitude, deltas

    @staticmethod
    def _row_norm(x):
        return x.square().sum(-1, keepdim=True).clamp_min(0).sqrt()

    @staticmethod
    def _relative_norm_error(a, b):
        return ((Model._row_norm(a) - Model._row_norm(b)).abs() /
                Model._row_norm(b).clamp_min(1e-15)).max()

    @staticmethod
    def _allocation_targets(total, audit):
        """Detached targets; support includes complete-gradient other paths.

        Padding and squared norms <=1e-30 are never donors or recipients.
        No CPU decisions, row sampling, RNG, cross-batch state or extra labels.
        """
        g2 = total.square().sum(-1, keepdim=True)
        support = (g2.squeeze(-1) > 1e-30).clone()
        support[0] = False
        common, proposed, magnitude = audit[1], audit[2], audit[4]
        p2 = proposed.square().sum(-1, keepdim=True)
        own = torch.where(common, (p2.clamp_min(1e-30) /
                                  g2.clamp_min(1e-30)).sqrt(),
                          torch.ones_like(g2)).squeeze(-1)
        count = support.long().sum()
        rank = support.long().cumsum(0) - 1
        packed = torch.zeros_like(own).scatter_add(
            0, rank.clamp_min(0), torch.where(support, own, torch.zeros_like(own)))
        donor_rank = (rank + count // 2).remainder(count.clamp_min(1))
        assigned = packed.gather(0, donor_rank)
        assigned = torch.where(support, assigned, torch.ones_like(assigned))
        rotated = total * assigned[:, None]
        rotate_delta = torch.where(support[:, None], rotated - total,
                                   torch.zeros_like(total))
        base_energy = (total.square() * support[:, None]).sum(dtype=torch.float64)
        target_energy = (magnitude.square() * support[:, None]).sum(dtype=torch.float64)
        alpha = torch.where(base_energy > 0,
                            (target_energy / base_energy.clamp_min(1e-30)).sqrt(),
                            torch.ones_like(base_energy)).to(total.dtype)
        uniform_delta = torch.where(support[:, None], total * alpha - total,
                                    torch.zeros_like(total))
        # Keep inherited M correction arithmetic verbatim for its replay.
        deltas = {'none': torch.zeros_like(total),
                  'total_magnitude': audit[5]['total_magnitude'],
                  'rotate_active': rotate_delta, 'uniform_energy': uniform_delta}
        targets = {key: total + delta for key, delta in deltas.items()}
        return support, own, assigned, alpha, deltas, targets

    @staticmethod
    def _allocation_metrics(total, audit, allocation, delta):
        support, own, assigned, alpha, _, targets = allocation
        active = support[:, None]
        energies = [t.square().sum(dtype=torch.float64) for t in
                    (total, targets['total_magnitude'], targets['rotate_active'],
                     targets['uniform_energy'])]
        uniform_error = (energies[3] - energies[1]).abs() / energies[1].clamp_min(1e-30)
        own_sum = (own * support).sum(dtype=torch.float64)
        assigned_sum = (assigned * support).sum(dtype=torch.float64)
        own_sq = (own.square() * support).sum(dtype=torch.float64)
        assigned_sq = (assigned.square() * support).sum(dtype=torch.float64)
        hist1 = (own_sum - assigned_sum).abs() / own_sum.abs().clamp_min(1e-30)
        hist2 = (own_sq - assigned_sq).abs() / own_sq.abs().clamp_min(1e-30)
        moved = support & (assigned != own)
        changed = support & (assigned != 1)
        outside_conflict = changed & ~audit[1].squeeze(-1)
        selected = total + delta
        # Direction check uses a safe scale; tiny excluded rows are unchanged.
        gn = Model._row_norm(total).clamp_min(1e-30)
        sn = Model._row_norm(selected).clamp_min(1e-30)
        direction_error = ((selected / sn - total / gn) * active).abs().max()
        outside = delta.masked_fill(active.expand_as(delta), 0).abs().max()
        extra = (audit[0][5] & ~audit[1]).sum()
        # First twelve are sums; last six are maxima across tables and calls.
        sums = energies + [delta.square().sum(dtype=torch.float64),
                           support.sum(), audit[1].sum(), moved.sum(),
                           changed.sum(), outside_conflict.sum(), extra]
        return sums, [uniform_error, hist1, hist2, delta[0].abs().max(),
                      outside, direction_error]

    def training_loss(self, batch, logits, train_labels):
        labels = train_labels.to(dtype=logits.dtype).reshape_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='mean')
        if not self.training or not torch.is_grad_enabled():
            return loss
        if self._path_cache is None or self._cached_logits is not logits:
            raise RuntimeError('training_loss requires immediately preceding logits')
        paths = self._path_cache
        self._path_cache = None
        self._cached_logits = None
        call = self._training_loss_calls
        self._training_loss_calls += 1
        capture = call in (0, 512)
        params = (self.embeddings[self.category].weight,
                  self.embeddings[self.brand].weight)
        grads = torch.autograd.grad(loss, paths + params, retain_graph=True,
                                    create_graph=False)
        grads = tuple(g.detach() for g in grads)
        total = grads[4:]
        audits, allocations, deltas, metrics = [], [], [], []
        correction = logits.new_zeros(())
        mode = self.config['row_intervention']
        for i, field in enumerate((self.category, self.brand)):
            qids = batch['static'][:, field]
            hids = batch['history_category' if i == 0 else 'history_brand']
            q = self._aggregate(params[i], qids, grads[i], qids != 0)
            h = self._aggregate(params[i], hids, grads[i + 2],
                                (hids != 0) & batch['history_mask'])
            audit = self._total_targets(total[i], q, h)
            allocation = self._allocation_targets(total[i], audit)
            delta = allocation[4][mode].detach()
            audits.append(audit)
            allocations.append(allocation)
            deltas.append(delta)
            metrics.append(self._allocation_metrics(total[i], audit, allocation, delta))
            if mode != 'none':
                correction = correction + (params[i] * delta).sum()
        result = loss if mode == 'none' else loss + (correction - correction.detach())
        sum_values = [sum(m[0][j] for m in metrics) for j in range(11)]
        max_values = [torch.stack([m[1][j] for m in metrics]).max() for j in range(6)]
        running = torch.stack([logits.new_ones(())] + sum_values + max_values).detach().to(torch.float64)
        if not bool(torch.isfinite(running).all()):
            raise RuntimeError('Nonfinite allocation audit; retain failed receipt')
        with torch.no_grad():
            self._running[:12].add_(running[:12])
            self._running[12:].copy_(torch.maximum(self._running[12:], running[12:]))
        if capture:
            actual = torch.autograd.grad(result, params, retain_graph=True,
                                         create_graph=False)
            expected = tuple(g + d for g, d in zip(total, deltas))
            error = self._norm(tuple(a.detach() - e for a, e in zip(actual, expected)))
            total_energy = sum_values[0].clamp_min(1e-30)
            selected_energy = sum(t.square().sum(dtype=torch.float64) for t in expected)
            active_count, eligible_count = sum_values[5], sum_values[6]
            changed_count, outside_count = sum_values[8], sum_values[9]
            values = [logits.new_tensor(float(call)), logits.new_tensor(float(labels.numel())),
                      error / self._norm(expected).clamp_min(1e-30),
                      (result.detach() - loss.detach()).abs(),
                      total_energy.sqrt(), (sum_values[4] / total_energy).sqrt(),
                      (selected_energy / total_energy).sqrt(),
                      eligible_count, active_count,
                      (active_count > 1).to(logits.dtype),
                      sum_values[7] / active_count.clamp_min(1),
                      (changed_count - outside_count) / changed_count.clamp_min(1),
                      max_values[0], torch.maximum(max_values[1], max_values[2])]
            values = torch.stack([v.detach().to(torch.float64) for v in values])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite allocation snapshot; retain failed receipt')
            with torch.no_grad():
                self._snapshots[0 if call == 0 else 1].copy_(values)
        return result

    @staticmethod
    def _synthetic_total_audit():
        # No RNG. Include repeated-ID aggregation, padding0, negative/positive
        # alignment, one-sided roles, exact opposition, and G/P degeneracy.
        q = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 0.],
                          [1., 0.], [1., 0.], [1., 0.], [1e-20, 0.]], dtype=torch.float64)
        h = torch.tensor([[-1., 1.], [-1., 1.], [1., 1.], [1., 0.],
                          [-1., 0.], [-1., 1.], [-1., 1.], [-1e-20, 1e-20]], dtype=torch.float64)
        g = torch.tensor([[1., 2.], [2., 1.], [2., 2.], [1., 0.],
                          [1., 0.], [0., 0.], [-0.5, -0.5], [1., 1.]], dtype=torch.float64)
        row, eligible, p, d, m, deltas = Model._total_targets(g, q, h)
        expected_project = g.clone()
        expected_project[1] = torch.tensor([2.5, 1.5], dtype=g.dtype)
        # All four surrogate gradients should equal their declared targets.
        error = g.new_zeros(())
        scalar_error = g.new_zeros(())
        with torch.enable_grad():
            for delta in deltas.values():
                w = torch.ones_like(g, requires_grad=True)
                base = (w * g).sum()
                term = (w * delta.detach()).sum()
                loss = base + (term - term.detach())
                (actual,) = torch.autograd.grad(loss, w)
                error = torch.maximum(error, (actual - (g + delta)).abs().max())
                scalar_error = torch.maximum(scalar_error, (loss.detach() - base.detach()).abs())
        ids = torch.tensor([0, 1, 1, 2], dtype=torch.long)
        vals = torch.tensor([[9., 9.], [1., 2.], [3., 4.], [7., 8.]], dtype=g.dtype)
        agg = Model._aggregate(torch.zeros(3, 2, dtype=g.dtype), ids, vals,
                               torch.tensor([True, True, True, False]))
        expected_agg = torch.tensor([[0., 0.], [4., 6.], [0., 0.]], dtype=g.dtype)
        direction_formula = g.clone()
        direction_formula[1] = expected_project[1] * (5. / 8.5) ** 0.5
        return {'synthetic_raw_projection_abs_error': float((p - expected_project).abs().max()),
                'synthetic_direction_abs_error': float((d - direction_formula).abs().max()),
                'synthetic_direction_norm_relative_error': float(Model._relative_norm_error(d, g)),
                'synthetic_magnitude_norm_relative_error': float(Model._relative_norm_error(m, p)),
                'synthetic_surrogate_gradient_abs_error': float(error),
                'synthetic_scalar_bce_abs_error': float(scalar_error),
                'synthetic_aggregate_abs_error': float((agg - expected_agg).abs().max()),
                'synthetic_eligibility_abs_error': float((eligible.reshape(-1).to(g.dtype) -
                    torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.], dtype=g.dtype)).abs().max())}

    @staticmethod
    def _synthetic_allocation_audit():
        g = torch.tensor([[0., 0.], [2., 1.], [3., 0.], [0., 4.], [1., 1.]],
                         dtype=torch.float64)
        q = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [0., 0.], [1., 0.]],
                         dtype=g.dtype)
        h = torch.tensor([[0., 0.], [-1., 1.], [1., 0.], [1., 0.], [-1., 0.]],
                         dtype=g.dtype)
        audit = Model._total_targets(g, q, h)
        allocation = Model._allocation_targets(g, audit)
        expected_rotation = g.clone()
        expected_rotation[3, 1] = 4. * (1.7 ** 0.5)
        expected_uniform = g * ((35.5 / 32.) ** 0.5)
        sums, maxima = Model._allocation_metrics(g, audit, allocation,
                                                 allocation[4]['rotate_active'])
        zero = torch.zeros_like(g)
        empty = Model._allocation_targets(zero, Model._total_targets(zero, zero, zero))
        empty_error = torch.stack([t.abs().max() for t in empty[5].values()]).max()
        return {'synthetic_rotation_formula_abs_error': float((allocation[5]['rotate_active'] - expected_rotation).abs().max()),
                'synthetic_uniform_formula_abs_error': float((allocation[5]['uniform_energy'] - expected_uniform).abs().max()),
                'synthetic_uniform_energy_relative_error': float(maxima[0]),
                'synthetic_empty_allocation_abs_error': float(empty_error)}

    def diagnostics(self, batch):
        # 28 training snapshots +18 running +5 validation +12 synthetic =63.
        names = ('call', 'samples', 'actual_gradient_relative_error', 'scalar_bce_abs_error',
                 'complete_gradient_norm', 'correction_ratio', 'selected_norm_ratio',
                 'eligible_rows', 'active_rows', 'multiple_active_rows',
                 'reallocated_fraction', 'rotated_receivers_conflict_fraction',
                 'uniform_energy_relative_error', 'rotation_moment_relative_error')
        out = {}
        for slot in range(2):
            for i, name in enumerate(names):
                out['train_cache' + str(slot) + '_' + name] = float(self._snapshots[slot, i])
        r = self._running
        out.update({'train_through_checkpoint_calls': float(r[0]),
                    'train_energy_G': float(r[1]), 'train_energy_M': float(r[2]),
                    'train_energy_rotated': float(r[3]), 'train_energy_uniform': float(r[4]),
                    'train_correction_rms_ratio': float((r[5] / r[1].clamp_min(1e-30)).sqrt()),
                    'train_active_rows_per_call': float(r[6] / r[0].clamp_min(1)),
                    'train_eligible_rows_per_call': float(r[7] / r[0].clamp_min(1)),
                    'train_reallocated_fraction': float(r[8] / r[6].clamp_min(1)),
                    'train_rotated_receiver_fraction': float(r[9] / r[6].clamp_min(1)),
                    'train_rotated_outside_conflict_fraction': float(r[10] / r[9].clamp_min(1)),
                    'train_extra_degenerate_rows': float(r[11]),
                    'train_uniform_energy_relative_error_max': float(r[12]),
                    'train_rotation_first_moment_relative_error_max': float(r[13]),
                    'train_rotation_second_moment_relative_error_max': float(r[14]),
                    'train_padding_correction_abs_max': float(r[15]),
                    'train_outside_active_correction_abs_max': float(r[16]),
                    'train_raw_direction_abs_error_max': float(r[17])})
        fields, features, history, components = self._encode(batch)
        raw = fields[self.category][:, None, :] * components[0] + fields[self.brand][:, None, :] * components[1]
        product = self._normalize_product(raw, 'full', batch['history_mask'])
        weights = self._weights(self._scores_from_product(fields, history, product), batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        out.update({'validation_samples': float(batch['static'].shape[0]),
                    'validation_forward_abs_max': float((self.forward(batch) - reference).abs().max()),
                    'validation_masked_mass_max': float((weights * (~batch['history_mask']).to(weights.dtype)).sum(-1).max()),
                    'validation_weight_sum_error_max': float((weights.sum(-1) - batch['history_mask'].any(-1).to(weights.dtype)).abs().max()),
                    'validation_empty_summary_norm_max': float((summary.square().sum(-1).sqrt() * (~batch['history_mask'].any(-1))).max())})
        out.update(self._synthetic_total_audit())
        out.update(self._synthetic_allocation_audit())
        return out
