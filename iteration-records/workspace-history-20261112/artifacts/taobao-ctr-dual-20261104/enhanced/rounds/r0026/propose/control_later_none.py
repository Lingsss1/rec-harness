"""Prospective schema5 later-state optimizer propagation probe.

No prediction architecture or live aggregation/optimizer rule is changed.
The exact schema4 _ReferenceModel and selected helpers are inherited from
r0025-shadow_magnitude (ec09311e60c875af8fbbf13e7818bbd40f82968a3c6b931991c7b422a1480eca).
Only calls512/8192 are probed. The two standalone files differ only in CONFIG.
All shadow branches reuse the same retained BCE graph, table parameters,
raw role gradients and complete pre-step Adam table state within a run.
Branches: returned result, rebuilt same-coefficient no-op, repeated original
index_add_, fixed-tree, repeated fixed-tree; sixth branch perturbs cloned
gradients solely to validate the distance measurements. No shadow replaces
live gradients, parameters, state, RNG or global precision/determinism flags.

Scope: ALL elements of the complete category and brand table gradients,
parameters and Adam exp_avg/exp_avg_sq; exact complete state-dict bridges also
check step counters and any extra state. Other model parameters/moments are
outside optimizer-shadow scope. Shapes are checked; rows are exported and
width is frozen16. Distances promote to float64 BEFORE subtraction and report
represented-value differences with no significance/equality threshold.
Effective updates are post-minus-pre parameters in float64, not a mathematical
pre-rounding Adam update. A moment change with zero parameter-update distance
is moment propagation only, not effective-update or AUC mediation.

At each call sensitivity changes just [row1,col0] in EACH cloned table gradient:
g'=g+sign_nonzero(g)*(1e-3+16*abs(g)). This is a fixed positive control, no retry
or adaptive amplitude, unrelated to observed role discrepancies. Exported
sensitivity values are MINIMUM per-table absolute maxima, so positive values
require both tables to respond. This large perturbation checks detector
responsiveness; it does not equate tiny natural differences with this control.
The unperturbed clone is compared bitwise to the actual live parameter, full
state and .grad after the host step. Nonzero no-op/fixed-repeat/bridge errors,
raw-repeat error, failed sensitivity or changed guards invalidate the bounded
smooth-propagation attribution; valid_probe remains0. No failed gate is hidden.
Observer mutations or nonfinite diagnostics fail the job and retain its cost.

Ten 4-bit dtype digits in tensor_dtype_code, least significant first, encode
category then brand: parameter, gradient, exp_avg, exp_avg_sq, step.
Codes: float16=1,bfloat16=2,float32=3,float64=4,int64=5,int32=6.
Step device is not encoded. No other tensors are claimed to be covered.
Two30-scalar training caches plus4 metadata scalars survive the best checkpoint.
Missing calls retain-1; valid_probe is-1 before post-hook completion,0 invalid,
1 passed. These are candidate-owned measurements, not independent verification.
Only source/syntax checks run in the coding jail: no Torch, training or install.
"""
import copy
import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'none'}


class _ReferenceModel(nn.Module):
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
        if config['row_intervention'] not in ('none', 'project', 'total_direction', 'total_magnitude'):
            raise ValueError('Only the registered complete-gradient factorial is admitted')
        if config['normalization_gradient'] != 'scale_half_tangent':
            raise ValueError('This bounded study fixes the inherited half-scale rule')
        self._path_cache = None
        self._cached_logits = None
        self._training_loss_calls = 0
        self.register_buffer('_snapshots', torch.zeros(2, 21, dtype=torch.float64))
        self._snapshots[:, 0] = -1
        self.register_buffer('_running', torch.zeros(9, dtype=torch.float64))


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
                assigned = _ReferenceModel._event_scales(inverse, selected, base_mode)
                result = torch.where(selected, assigned, result)
            return result
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            donor = _ReferenceModel._rotation_donors(inverse, valid_regular, gradient)
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
        target = _ReferenceModel._event_scales(inverse, valid_regular, gradient, action_type)
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
        row = _ReferenceModel._row_targets(q, h)
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
        return ((_ReferenceModel._row_norm(a) - _ReferenceModel._row_norm(b)).abs() /
                _ReferenceModel._row_norm(b).clamp_min(1e-15)).max()

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
        # Complete baseline surrogate row gradients are needed on EVERY batch.
        grads = torch.autograd.grad(loss, paths + params, retain_graph=True,
                                    create_graph=False)
        grads = tuple(g.detach() for g in grads)
        if self._probe_active:
            self._probe_raw(loss, paths, params, grads)
        total = grads[4:]
        audits = []
        deltas = []
        correction = logits.new_zeros(())
        for i, field in enumerate((self.category, self.brand)):
            qids = batch['static'][:, field]
            hids = batch['history_category' if i == 0 else 'history_brand']
            q = self._aggregate(params[i], qids, grads[i], qids != 0)
            h = self._aggregate(params[i], hids, grads[i + 2],
                                (hids != 0) & batch['history_mask'])
            audit = self._total_targets(total[i], q, h)
            if self._probe_active:
                self._probe_table(i, params[i], qids, hids, grads[i], grads[i + 2],
                                  total[i], q, h, audit, batch['history_mask'])
            delta = audit[5][self.config['row_intervention']].detach()
            deltas.append(delta)
            audits.append(audit)
            if self.config['row_intervention'] != 'none':
                correction = correction + (params[i] * delta).sum()
        result = loss if self.config['row_intervention'] == 'none' else (
            loss + (correction - correction.detach()))
        proposed = tuple(a[2] for a in audits)
        direction = tuple(a[3] for a in audits)
        magnitude = tuple(a[4] for a in audits)
        products = tuple(a[0][0] for a in audits)
        eligible = sum(a[1].sum() for a in audits)
        conflict = sum(a[0][4].sum() for a in audits)
        total2 = sum(t.square().sum(dtype=torch.float64) for t in total)
        delta2 = sum(d.square().sum(dtype=torch.float64) for d in deltas)
        conflict_energy = sum((t.square() * a[0][4].to(t.dtype)).sum(
            dtype=torch.float64) for t, a in zip(total, audits))
        direction_error = torch.stack([
            self._relative_norm_error(d, g) for d, g in zip(direction, total)]).max()
        magnitude_error = torch.stack([
            self._relative_norm_error(m, p) for m, p in zip(magnitude, proposed)]).max()
        extra_degenerate = sum((a[0][5] & ~a[1]).sum() for a in audits)
        # Nine bounded running scalars; never retain rows, labels or batches.
        running = torch.stack([logits.new_ones(()), eligible, conflict,
                               conflict_energy, total2, delta2, extra_degenerate,
                               direction_error, magnitude_error]).detach().to(torch.float64)
        if not bool(torch.isfinite(running).all()):
            raise RuntimeError('Nonfinite complete-gradient factorial; retain failed receipt')
        with torch.no_grad():
            self._running[:7].add_(running[:7])
            self._running[7:].copy_(torch.maximum(self._running[7:], running[7:]))
        if capture:
            actual = torch.autograd.grad(result, params, retain_graph=True,
                                         create_graph=False)
            expected = tuple(g + d for g, d in zip(total, deltas))
            magnitude_direction_errors = []
            for g, m in zip(total, magnitude):
                gn = self._row_norm(g).clamp_min(1e-15)
                mn = self._row_norm(m).clamp_min(1e-15)
                magnitude_direction_errors.append((m / mn - g / gn).abs().max())
            inactive_max = torch.stack([
                d.masked_fill(a[1].expand_as(d), 0).abs().max()
                for d, a in zip(deltas, audits)]).max()
            role_active = sum(a[0][3].sum() for a in audits)
            values = [logits.new_tensor(float(call)), logits.new_tensor(float(labels.numel())),
                      role_active, conflict, eligible, extra_degenerate,
                      total2.clamp_min(0).sqrt(),
                      self._norm(tuple(g - p for g, p in zip(total, products))),
                      self._norm(products), self._norm(proposed),
                      delta2.clamp_min(0).sqrt() / total2.clamp_min(1e-30).sqrt(),
                      self._norm(expected) / total2.clamp_min(1e-30).sqrt(),
                      direction_error, magnitude_error,
                      torch.stack(magnitude_direction_errors).max(), inactive_max,
                      torch.stack([d[0].abs().max() for d in deltas]).max(),
                      self._norm(tuple(a.detach() - e for a, e in zip(actual, expected))) /
                          self._norm(expected).clamp_min(1e-30),
                      (result.detach() - loss.detach()).abs(),
                      conflict_energy / total2.clamp_min(1e-30),
                      self._norm(tuple(d - g for d, g in zip(direction, total))) /
                          total2.clamp_min(1e-30).sqrt()]
            values = torch.stack([v.detach().to(torch.float64) for v in values])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite training snapshot; retain failed receipt')
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
        row, eligible, p, d, m, deltas = _ReferenceModel._total_targets(g, q, h)
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
        agg = _ReferenceModel._aggregate(torch.zeros(3, 2, dtype=g.dtype), ids, vals,
                               torch.tensor([True, True, True, False]))
        expected_agg = torch.tensor([[0., 0.], [4., 6.], [0., 0.]], dtype=g.dtype)
        direction_formula = g.clone()
        direction_formula[1] = expected_project[1] * (5. / 8.5) ** 0.5
        return {'synthetic_raw_projection_abs_error': float((p - expected_project).abs().max()),
                'synthetic_direction_abs_error': float((d - direction_formula).abs().max()),
                'synthetic_direction_norm_relative_error': float(_ReferenceModel._relative_norm_error(d, g)),
                'synthetic_magnitude_norm_relative_error': float(_ReferenceModel._relative_norm_error(m, p)),
                'synthetic_surrogate_gradient_abs_error': float(error),
                'synthetic_scalar_bce_abs_error': float(scalar_error),
                'synthetic_aggregate_abs_error': float((agg - expected_agg).abs().max()),
                'synthetic_eligibility_abs_error': float((eligible.reshape(-1).to(g.dtype) -
                    torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.], dtype=g.dtype)).abs().max())}

    def diagnostics(self, batch):
        # 42 snapshot +8 running +5 validation +8 synthetic =63 finite scalars.
        # All train_cache/train_through_checkpoint keys originate in training.
        names = ('call', 'samples', 'role_active_rows', 'role_conflicting_rows',
                 'eligible_rows', 'extra_degenerate_rows', 'complete_gradient_norm',
                 'unchanged_path_norm', 'product_sum_norm', 'raw_project_complete_norm',
                 'correction_complete_ratio', 'new_complete_norm_ratio',
                 'direction_norm_relative_error', 'magnitude_norm_relative_error',
                 'magnitude_direction_abs_error', 'outside_eligible_correction_abs_max',
                 'padding_correction_abs_max', 'intended_gradient_relative_error',
                 'scalar_bce_abs_error', 'conflict_complete_energy_fraction',
                 'direction_change_relative')
        out = {}
        for slot in range(2):
            for i, name in enumerate(names):
                out['train_cache' + str(slot) + '_' + name] = float(self._snapshots[slot, i])
        r = self._running
        out.update({'train_through_checkpoint_calls': float(r[0]),
                    'train_through_checkpoint_eligible_rows_per_call': float(r[1] / r[0].clamp_min(1)),
                    'train_through_checkpoint_conflicting_rows_per_call': float(r[2] / r[0].clamp_min(1)),
                    'train_through_checkpoint_conflict_energy_fraction': float(r[3] / r[4].clamp_min(1e-30)),
                    'train_through_checkpoint_correction_rms_ratio': float((r[5] / r[4].clamp_min(1e-30)).clamp_min(0).sqrt()),
                    'train_through_checkpoint_extra_degenerate_rows': float(r[6]),
                    'train_through_checkpoint_direction_norm_error_max': float(r[7]),
                    'train_through_checkpoint_magnitude_norm_error_max': float(r[8])})
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
        return out


class Model(_ReferenceModel):
    _CALLS = (512, 8192)
    _NAMES = (
        'call', 'raw_repeat_relative', 'role_original_repeat_relative',
        'role_fixed_repeat_relative', 'role_fixed_original_relative',
        'guard_original_mismatches', 'guard_fixed_mismatches',
        'gradient_noop_relative', 'gradient_original_repeat_relative',
        'gradient_fixed_repeat_relative', 'gradient_fixed_original_relative',
        'live_gradient_relative', 'adam_noop_failures',
        'adam_original_repeat_update_absmax', 'adam_fixed_repeat_failures',
        'adam_fixed_original_update_absmax', 'adam_original_repeat_m1_absmax',
        'adam_original_repeat_m2_absmax', 'adam_fixed_original_m1_absmax',
        'adam_fixed_original_m2_absmax', 'adam_step_mismatches',
        'live_bridge_failures', 'observer_failures',
        'sensitivity_gradient_min_table_absmax', 'sensitivity_m1_min_table_absmax',
        'sensitivity_m2_min_table_absmax', 'sensitivity_update_min_table_absmax',
        'tensor_dtype_code', 'eligible_rows', 'valid_probe')

    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        if self.config['row_intervention'] not in ('none', 'total_magnitude'):
            raise ValueError('Schema5 admits only the registered M/N pair')
        if self.config['embedding_dim'] != 16:
            raise ValueError('Schema5 fixes the inherited table width16')
        self.register_buffer('_probe_rows', torch.full((2, len(self._NAMES)), -1., dtype=torch.float64))
        self.register_buffer('_tree_synthetic_error', torch.full((), -1., dtype=torch.float64))
        self._probe_active = False
        self._probe_work = None
        self._pending_step = None
        self._post_step = None

    def _write_probe(self, call, values):
        slot = self._CALLS.index(call)
        with torch.no_grad():
            for name, value in values.items():
                self._probe_rows[slot, self._NAMES.index(name)] = value
            if not bool(torch.isfinite(self._probe_rows[slot]).all()):
                raise RuntimeError('Schema5 nonfinite probe; preserve failed receipt')

    def _read_probe(self, call, name):
        return float(self._probe_rows[self._CALLS.index(call), self._NAMES.index(name)])

    def _finish_probe(self, call, result):
        work = self._probe_work
        tables = work['tables']
        if len(tables) != 2 or [t['index'] for t in tables] != [0, 1]:
            raise RuntimeError('Schema5 missing table probes')
        params = (self.embeddings[self.category].weight,
                  self.embeddings[self.brand].weight)
        loss = work['loss']
        results = (result,) + tuple(self._branch_result(loss, tables, b, params)
                                   for b in (0, 1, 2, 3))
        gradients = tuple(tuple(g.detach() for g in torch.autograd.grad(
            r, params, retain_graph=True, create_graph=False)) for r in results)
        with torch.no_grad():
            roles = lambda key: tuple(v for t in tables for v in t[key])
            r0, rr, rf, rf2 = (roles(k) for k in
                               ('roles', 'repeat_roles', 'fixed_roles', 'fixed2_roles'))
            guards = lambda b: sum((t['audits'][0][1] != t['audits'][b][1]).sum()
                                   for t in tables)
            mutations = float(work['mutations']) + sum(
                int(not self._bits_equal(t, a)) for t, a in zip(work['watched'], work['anchors']))
            scalar_error = torch.stack([(r.detach()-loss.detach()).abs() for r in results]).max()
            vals = {
                'call': float(call), 'raw_repeat_relative': work['raw_repeat'],
                'role_original_repeat_relative': self._distance(r0, rr)[1],
                'role_fixed_repeat_relative': self._distance(rf, rf2)[1],
                'role_fixed_original_relative': self._distance(r0, rf)[1],
                'guard_original_mismatches': guards(1), 'guard_fixed_mismatches': guards(2),
                'gradient_noop_relative': self._distance(gradients[0], gradients[1])[1],
                'gradient_original_repeat_relative': self._distance(gradients[0], gradients[2])[1],
                'gradient_fixed_repeat_relative': self._distance(gradients[3], gradients[4])[1],
                'gradient_fixed_original_relative': self._distance(gradients[0], gradients[3])[1],
                'observer_failures': mutations,
                'eligible_rows': sum(t['audits'][0][1].sum() for t in tables)}
            self._write_probe(call, vals)
            if mutations or float(scalar_error) != 0:
                raise RuntimeError('Schema5 observer/scalar integrity failure')
        self._pending_step = (call, gradients)

    @staticmethod
    def _output_failures(a, b):
        return (sum(int(not Model._bits_equal(x, y)) for x, y in zip(a['params'], b['params'])) +
                sum(Model._state_mismatches(x, y) for x, y in zip(a['state'], b['state'])))

    @staticmethod
    def _state_absmax(a, b, name):
        return Model._distance(tuple(s[name] for s in a['state']),
                               tuple(s[name] for s in b['state']))[0]

    @staticmethod
    def _min_table_absmax(a, b):
        # Both tables must respond; a maximum pooled over tables would hide one failure.
        return torch.stack([Model._distance((x,), (y,))[0] for x, y in zip(a, b)]).min()

    @staticmethod
    def _dtype_scope(params, gradients, states):
        codes = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3,
                 torch.float64: 4, torch.int64: 5, torch.int32: 6}
        packed = 0
        position = 0
        for p, g, state in zip(params, gradients, states):
            if p.ndim != 2 or p.shape[1] != 16 or p.shape[0] < 2 or g.shape != p.shape:
                raise RuntimeError('Schema5 table/gradient scope mismatch')
            if not {'exp_avg', 'exp_avg_sq', 'step'}.issubset(state):
                raise RuntimeError('Schema5 requires initialized later-state Adam')
            if state['exp_avg'].shape != p.shape or state['exp_avg_sq'].shape != p.shape:
                raise RuntimeError('Schema5 moment scope mismatch')
            if not isinstance(state['step'], torch.Tensor) or state['step'].numel() != 1:
                raise RuntimeError('Schema5 scalar tensor step required')
            for tensor in (p, g, state['exp_avg'], state['exp_avg_sq'], state['step']):
                if tensor.dtype not in codes:
                    raise RuntimeError('Schema5 unsupported actual optimizer dtype')
                packed += codes[tensor.dtype] * (16 ** position)
                position += 1
        return float(packed)

    def _adam_shadow(self, group, params, gradients, states, before):
        clones = tuple(nn.Parameter(p.detach().clone()) for p in params)
        options = {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
        options['params'] = list(clones)
        shadow = torch.optim.Adam([options])
        for clone, gradient, state in zip(clones, gradients, states):
            clone.grad = gradient.detach().clone()
            shadow.state[clone] = self._clone_state(state)
        shadow.step()
        return {
            'params': tuple(p.detach().clone() for p in clones),
            'updates': tuple(p.detach().to(torch.float64) - old.to(torch.float64)
                             for p, old in zip(clones, before)),
            'state': tuple(self._clone_state(shadow.state[p]) for p in clones)}

    def _adam_before(self, optimizer, args, kwargs):
        if self._pending_step is None:
            return
        call, gradients = self._pending_step
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
            raise RuntimeError('Schema5 requires inherited standard one-group Adam')
        group = optimizer.param_groups[0]
        params = (self.embeddings[self.category].weight,
                  self.embeddings[self.brand].weight)
        if any(p.grad is None for p in params):
            raise RuntimeError('Schema5 missing live table gradients')
        before = tuple(p.detach().clone() for p in params)
        live_grad = tuple(p.grad.detach().clone() for p in params)
        states = tuple(self._clone_state(optimizer.state.get(p, {})) for p in params)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(params[0].device).clone() if params[0].is_cuda else None
        vals = {'live_gradient_relative': self._distance(gradients[0], live_grad)[1],
                'tensor_dtype_code': self._dtype_scope(params, live_grad, states),
                'adam_step_mismatches': 0.}
        baseline = None
        fixed = None
        for branch, gs in enumerate(gradients):
            out = self._adam_shadow(group, params, gs, states, before)
            if branch == 0:
                baseline = out
            elif branch == 1:
                vals['adam_noop_failures'] = self._output_failures(baseline, out) + sum(
                    int(not self._bits_equal(a, b)) for a, b in zip(gradients[0], gs))
            elif branch in (2, 3):
                prefix = 'adam_original_repeat' if branch == 2 else 'adam_fixed_original'
                vals[prefix+'_update_absmax'] = self._distance(baseline['updates'], out['updates'])[0]
                vals[prefix+'_m1_absmax'] = self._state_absmax(baseline, out, 'exp_avg')
                vals[prefix+'_m2_absmax'] = self._state_absmax(baseline, out, 'exp_avg_sq')
                vals['adam_step_mismatches'] += sum(
                    int(not self._bits_equal(a['step'], b['step']))
                    for a, b in zip(baseline['state'], out['state']))
                if branch == 3:
                    fixed = out
            else:
                vals['adam_fixed_repeat_failures'] = self._output_failures(fixed, out) + sum(
                    int(not self._bits_equal(a, b)) for a, b in zip(gradients[3], gs))
                vals['adam_step_mismatches'] += sum(
                    int(not self._bits_equal(a['step'], b['step']))
                    for a, b in zip(fixed['state'], out['state']))
            del out
        # Predeclared sensitivity perturbation stays strictly on fresh clones.
        sensitive = tuple(g.detach().clone() for g in gradients[0])
        with torch.no_grad():
            for g in sensitive:
                value = g[1, 0].clone()
                sign = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
                g[1, 0] = value + sign * (1e-3 + 16 * value.abs())
        response = self._adam_shadow(group, params, sensitive, states, before)
        vals.update({
            'sensitivity_gradient_min_table_absmax': self._min_table_absmax(gradients[0], sensitive),
            'sensitivity_m1_min_table_absmax': self._min_table_absmax(
                tuple(s['exp_avg'] for s in baseline['state']), tuple(s['exp_avg'] for s in response['state'])),
            'sensitivity_m2_min_table_absmax': self._min_table_absmax(
                tuple(s['exp_avg_sq'] for s in baseline['state']), tuple(s['exp_avg_sq'] for s in response['state'])),
            'sensitivity_update_min_table_absmax': self._min_table_absmax(baseline['updates'], response['updates'])})
        del response, sensitive
        mutations = sum(int(not self._bits_equal(p, a)) for p, a in zip(params, before))
        mutations += sum(int(not self._bits_equal(p.grad, a)) for p, a in zip(params, live_grad))
        mutations += sum(self._state_mismatches(optimizer.state.get(p, {}), a)
                         for p, a in zip(params, states))
        mutations += int(not torch.equal(cpu_rng, torch.get_rng_state()))
        if cuda_rng is not None:
            mutations += int(not torch.equal(cuda_rng, torch.cuda.get_rng_state(params[0].device)))
        vals['observer_failures'] = self._read_probe(call, 'observer_failures') + mutations
        self._write_probe(call, vals)
        if mutations:
            raise RuntimeError('Schema5 shadow mutated live scope/state/RNG')
        self._post_step = (call, baseline, live_grad)
        self._pending_step = None

    def _adam_after(self, optimizer, args, kwargs):
        if self._post_step is None:
            return
        call, expected, expected_grad = self._post_step
        params = (self.embeddings[self.category].weight,
                  self.embeddings[self.brand].weight)
        actual = {'params': params, 'state': tuple(optimizer.state[p] for p in params)}
        mismatches = self._output_failures(expected, actual) + sum(
            int(not self._bits_equal(p.grad, g)) for p, g in zip(params, expected_grad))
        self._write_probe(call, {'live_bridge_failures': float(mismatches)})
        zero_names = ('raw_repeat_relative', 'role_fixed_repeat_relative',
                      'guard_original_mismatches', 'guard_fixed_mismatches',
                      'gradient_noop_relative', 'gradient_fixed_repeat_relative',
                      'live_gradient_relative', 'adam_noop_failures',
                      'adam_fixed_repeat_failures', 'adam_step_mismatches',
                      'live_bridge_failures', 'observer_failures')
        positive_names = ('sensitivity_gradient_min_table_absmax', 'sensitivity_m1_min_table_absmax',
                          'sensitivity_m2_min_table_absmax', 'sensitivity_update_min_table_absmax')
        passed = (all(self._read_probe(call, n) == 0 for n in zero_names) and
                  all(self._read_probe(call, n) > 0 for n in positive_names))
        self._write_probe(call, {'valid_probe': float(passed)})
        self._post_step = None

    def training_loss(self, batch, logits, train_labels):
        call = self._training_loss_calls
        active = self.training and torch.is_grad_enabled() and call in self._CALLS
        if self._pending_step is not None or self._post_step is not None:
            raise RuntimeError('Schema5 requires make_optimizer and one step per batch')
        self._probe_active = active
        if active:
            if call == self._CALLS[0]:
                with torch.no_grad():
                    self._synthetic_tree_check()
            watched = tuple(batch[k] for k in sorted(batch)) + (train_labels, logits) + tuple(self.parameters())
            self._probe_work = {'tables': [], 'mutations': 0, 'watched': watched,
                                'anchors': tuple(t.detach().clone() for t in watched)}
        try:
            result = super().training_loss(batch, logits, train_labels)
            if active:
                self._finish_probe(call, result)
            return result
        finally:
            self._probe_active = False
            self._probe_work = None

    def diagnostics(self, batch):
        out = {'audit_schema': 5., 'training_calls_through_checkpoint': float(self._running[0]),
               'category_table_rows': float(self.embeddings[self.category].weight.shape[0]),
               'brand_table_rows': float(self.embeddings[self.brand].weight.shape[0])}
        for slot in range(2):
            for i, name in enumerate(self._NAMES):
                out['audit'+str(slot)+'_'+name] = float(self._probe_rows[slot, i])
        if len(out) != 64 or not all(math.isfinite(v) for v in out.values()):
            raise RuntimeError('Schema5 64-scalar diagnostic contract violation')
        return out

    @staticmethod
    def _bits_equal(a, b):
        return (a.dtype == b.dtype and a.shape == b.shape and
                torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                            b.detach().contiguous().reshape(-1).view(torch.uint8)))

    @staticmethod
    def _distance(a, b):
        # Promote BEFORE subtraction to measure differences in represented values.
        aa = tuple(t.detach().to(torch.float64) for t in a)
        bb = tuple(t.detach().to(torch.float64) for t in b)
        diff = tuple(y - x for x, y in zip(aa, bb))
        absolute = torch.stack([x.abs().max() for x in diff]).max()
        relative = _ReferenceModel._norm(diff) / _ReferenceModel._norm(aa).clamp_min(1e-30)
        return absolute, relative

    @staticmethod
    def _fixed_tree(weight, ids, gradient, valid):
        # Dense output matches inherited row/mask/padding semantics. Stable sort
        # preserves flattened input order inside each row. Each scan stage has
        # one writer per coordinate; the final copy has unique row indices.
        ids = ids.detach().reshape(-1)
        values = gradient.detach().reshape(-1, gradient.shape[-1])
        values = values * valid.detach().reshape(-1, 1).to(values.dtype)
        out = torch.zeros_like(weight)
        if ids.numel() == 0:
            return out
        order = ids.argsort(stable=True)
        keys = ids[order]
        summed = values[order]
        offset = 1
        while offset < keys.numel():
            same = keys[offset:] == keys[:-offset]
            tail = summed[offset:] + torch.where(
                same[:, None], summed[:-offset], torch.zeros_like(summed[:-offset]))
            summed = torch.cat((summed[:offset], tail), dim=0)
            offset *= 2
        ends = torch.cat((keys[:-1] != keys[1:],
                          torch.ones(1, dtype=torch.bool, device=keys.device)))
        out.index_copy_(0, keys[ends], summed[ends])
        out[0].zero_()
        return out

    def _probe_raw(self, loss, paths, params, original):
        repeat = torch.autograd.grad(loss, paths + params, retain_graph=True,
                                     create_graph=False)
        self._probe_work['loss'] = loss
        self._probe_work['raw_repeat'] = self._distance(original, repeat)[1]

    def _probe_table(self, index, weight, qids, hids, qgrad, hgrad,
                     total, q, h, audit, mask):
        with torch.no_grad():
            fixed_inputs = (weight, qids, hids, qgrad, hgrad, total, q, h, mask)
            anchors = tuple(t.detach().clone() for t in fixed_inputs)
            rq = self._aggregate(weight, qids, qgrad, qids != 0)
            rh = self._aggregate(weight, hids, hgrad, (hids != 0) & mask)
            fq = self._fixed_tree(weight, qids, qgrad, qids != 0)
            fh = self._fixed_tree(weight, hids, hgrad, (hids != 0) & mask)
            fq2 = self._fixed_tree(weight, qids, qgrad, qids != 0)
            fh2 = self._fixed_tree(weight, hids, hgrad, (hids != 0) & mask)
            repeat = self._total_targets(total, rq, rh)
            fixed = self._total_targets(total, fq, fh)
            fixed2 = self._total_targets(total, fq2, fh2)
            self._probe_work['mutations'] += sum(
                int(not self._bits_equal(a, b)) for a, b in zip(fixed_inputs, anchors))
            self._probe_work['tables'].append({
                'index': index, 'roles': (q, h), 'repeat_roles': (rq, rh),
                'fixed_roles': (fq, fh), 'fixed2_roles': (fq2, fh2),
                'total': total, 'audits': (audit, repeat, fixed, fixed2)})

    def _branch_result(self, loss, tables, branch, params):
        if self.config['row_intervention'] == 'none':
            return loss
        correction = loss.new_zeros(())
        for weight, table in zip(params, tables):
            delta = table['audits'][branch][5][self.config['row_intervention']].detach()
            correction = correction + (weight * delta).sum()
        return loss + (correction - correction.detach())

    @staticmethod
    def _clone_state(state):
        return {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k, v in state.items()}

    @staticmethod
    def _state_mismatches(a, b):
        if set(a) != set(b):
            return 1
        return sum(int(not Model._bits_equal(a[k], b[k])) if isinstance(a[k], torch.Tensor)
                   and isinstance(b[k], torch.Tensor) else int(a[k] != b[k]) for k in a)

    def _synthetic_tree_check(self):
        device = self._probe_rows.device
        ids = torch.tensor([0, 2, 1, 2, 1, 2, 3], dtype=torch.long, device=device)
        values = torch.tensor([[9., 9.], [1., 2.], [3., 4.], [5., 6.],
                               [7., 8.], [99., 99.], [10., 11.]], device=device)
        valid = torch.tensor([True, True, True, True, True, False, True], device=device)
        weight = torch.zeros(5, 2, device=device)
        expected = torch.tensor([[0., 0.], [10., 12.], [6., 8.], [10., 11.], [0., 0.]], device=device)
        first = self._fixed_tree(weight, ids, values, valid)
        second = self._fixed_tree(weight, ids, values, valid)
        empty = self._fixed_tree(weight, ids[:0], values[:0], valid[:0])
        error = torch.stack(((first - expected).abs().max(), (first - second).abs().max(),
                             empty.abs().max())).max()
        self._tree_synthetic_error.copy_(error.to(torch.float64))
        if float(error) != 0:
            raise RuntimeError('Schema4 fixed-tree synthetic semantics failure')

def make_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config['lr'])
    optimizer.register_step_pre_hook(model._adam_before)
    optimizer.register_step_post_hook(model._adam_after)
    return optimizer
