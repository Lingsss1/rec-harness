"""Bounded common-state audit of the exact round17/18 source bodies.

The two embedded classes are identifier-renamed copies (AST checked). The
live model follows round17; extra probes occur only at training calls 0/512.
Both source versions and a repeated round17 backward use the SAME live
parameters, supervised batch, mode and Adam moments. No model initialization,
RNG calls, filesystem, network, external state or validation labels are used.
Probes are sequential and can perturb CUDA scheduling; exact replays and the
none audit control quantify that limitation. Candidate-authored checks do not
constitute an independent host audit or prove full-trajectory equivalence.
Only shared category/brand Adam steps are shadowed; all-parameter backward
comparisons cover the rest of the model. The explicit live optimizer is Adam
with declared lr and PyTorch defaults, with unchanged gradients and moments.
Shadow copies never replace live gradients, parameters or optimizer state.
At most 63 finite scalars from two training snapshots and label-free validation.
No local Torch execution is claimed. A failed audit must retain its receipt.
"""
import copy

"""Complete-row gradient direction/magnitude factorial, frozen half-scale forward.

Let q,h be supervised product-role gradients aggregated by shared table row,
G the COMPLETE baseline BCE surrogate row gradient under the fixed half
rule (including all static/query/value paths), and d the inherited symmetric conflict-projection correction.
P=G+d. At common nondegenerate eligible rows the four cells are:
 none: G; project: P; total_direction: P*||G||/||P||;
 total_magnitude: G*||P||/||G||.
Each cell computes all hypothetical targets at its OWN current training state.
This controls complete-row raw-gradient norms, not Adam moments/updates or
whole trajectories. Radial rescaling intentionally also affects the other-path
contribution at eligible rows. Outside eligibility, including padding row0,
all corrections are zero. Role/sum and complete-row squared-norm cutoffs are1e-30.
Common-state scalar mean BCE and forward logits remain unchanged. Detached
linear surrogates replace table gradients; no optimizer/state/RNG edits occur.

Inherited parameter creation order, forward methods, normalization, pooling,
BCE and host Adam/data/seeds are retained from r0016-half_project. Buffers and
extra autograd computation are identical across the four standalone modules;
only literal CONFIG differs. Extra common degeneracy guards mean the project
bridge is conditional on the reported extra-degenerate count being zero.

Diagnostics: two candidate-authored training-cache snapshots at calls0/512,
plus all-call aggregates THROUGH THE SAVED CHECKPOINT, not necessarily through
the completed run. Validation checks use only first512 inputs, no labels.
Host calls diagnostics under no_grad; the synthetic probe enables gradients
only on newly created artificial tensors. At most64 finite scalar outputs.
Torch is unavailable in the coding environment; runtime and derivative checks
are prospective host checks, not claimed local execution. Nearest general
surgery literature is unavailable in registry; no literature priority claimed.
"""
import torch
from torch import nn
_CONFIG17 = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'total_magnitude'}

class _Model17(nn.Module):

    def __init__(self, meta, config=None):
        super().__init__()
        config = {**_CONFIG17, **(config or {})}
        self.config = config
        self.kind = config['kind']
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = config['embedding_dim']
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1))
        layers = []
        width = len(meta['fields']) * dim + (dim if self.kind != 'static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)
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
        fields = [embedding(batch['static'][:, i]) for (i, embedding) in enumerate(self.embeddings)]
        features = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        history = category + brand
        action = self.action(batch['history_type'])
        history = history + action
        recency = self.recency(ages)
        history = history + recency
        return (fields, features, history, (category, brand, action, recency))

    @staticmethod
    def _weights(scores, mask):
        scores = scores.masked_fill(~mask, -10000.0)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-09)

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
        positions = torch.arange(inverse.shape[-1], device=inverse.device, dtype=torch.long)[None, :].expand_as(inverse)
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
        if gradient in ('action_mean_tangent', 'action_reassigned_tangent', 'action_adjacent_tangent'):
            if action_type is None:
                raise ValueError('behavior-stratified scales require history_type')
            base_mode = {'action_mean_tangent': 'history_mean_tangent', 'action_reassigned_tangent': 'history_reassigned_tangent', 'action_adjacent_tangent': 'history_adjacent_tangent'}[gradient]
            result = inverse
            for action in range(1, 5):
                selected = valid_regular & (action_type == action)
                assigned = _Model17._event_scales(inverse, selected, base_mode)
                result = torch.where(selected, assigned, result)
            return result
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            donor = _Model17._rotation_donors(inverse, valid_regular, gradient)
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
            rank = valid_regular.long().cumsum(-1) - 1
            packed = torch.zeros_like(inverse).scatter_add(-1, rank.clamp_min(0), values)
            shift = 1 if gradient == 'history_adjacent_tangent' else count // 2
            donor = (rank + shift).remainder(count.clamp_min(1))
            rotated = packed.gather(-1, donor)
            return torch.where(valid_regular, rotated, inverse)
        raise ValueError('unknown normalization_gradient')

    @staticmethod
    def _normalize_product(product, gradient='full', mask=None, action_type=None):
        squared = product.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-06).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        regular = squared >= 1e-06
        valid_regular = regular.squeeze(-1)
        if mask is not None:
            valid_regular = valid_regular & mask
        inverse = denominator.squeeze(-1).reciprocal().detach()
        target = _Model17._event_scales(inverse, valid_regular, gradient, action_type)
        coefficient = torch.where(regular, denominator * target.unsqueeze(-1), torch.ones_like(denominator)).detach()
        surrogate = unit * coefficient
        return unit.detach() + (surrogate - surrogate.detach())

    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def _product(self, fields, components, mask):
        (category, brand, _, _) = components
        (qc, qb) = (fields[self.category], fields[self.brand])
        if self.training and torch.is_grad_enabled():
            (qc, qb, category, brand) = (qc.clone(), qb.clone(), category.clone(), brand.clone())
            self._path_cache = (qc, qb, category, brand)
        product = qc[:, None, :] * category + qb[:, None, :] * brand
        return self._normalize_product(product, self.config['normalization_gradient'], mask)

    def forward(self, batch):
        self._path_cache = None
        self._cached_logits = None
        (fields, features, history, components) = self._encode(batch)
        product = self._product(fields, components, batch['history_mask'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
        if self._path_cache is not None:
            self._cached_logits = logits
        return logits

    @staticmethod
    def _aggregate(weight, ids, gradient, valid):
        out = torch.zeros_like(weight)
        values = gradient.detach().reshape(-1, gradient.shape[-1])
        values = values * valid.reshape(-1, 1).to(values.dtype)
        out.index_add_(0, ids.reshape(-1), values)
        out[0].zero_()
        return out

    @staticmethod
    def _row_targets(q, h):
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
        ratio = (projected.square().sum(-1, keepdim=True) / s2.clamp_min(1e-30)).clamp_min(0).sqrt()
        norm_match = torch.where(eligible, original * ratio, original)
        projected = torch.where(eligible, projected, original)
        return (original, projected, norm_match, active, conflict, eligible, dot)

    @staticmethod
    def _norm(tensors):
        return sum((t.square().sum(dtype=torch.float64) for t in tensors)).clamp_min(0).sqrt()

    @staticmethod
    def _total_targets(total, q, h):
        """Detached common-state factorial. Caller stops all input gradients.

        Raw delta is retained verbatim for the projection bridge. All arms
        share the additional G/P norm guard, so degenerate cases cannot be
        silently counted as successful direction/magnitude matching.
        """
        row = _Model17._row_targets(q, h)
        (product, projected) = row[:2]
        raw_delta = projected - product
        proposed = total + raw_delta
        g2 = total.square().sum(-1, keepdim=True)
        p2 = proposed.square().sum(-1, keepdim=True)
        common = row[5] & (g2 > 1e-30) & (p2 > 1e-30)
        common = common.clone()
        common[0] = False
        raw_delta = torch.where(common, raw_delta, torch.zeros_like(raw_delta))
        proposed = total + raw_delta
        direction = proposed * (g2.clamp_min(1e-30) / p2.clamp_min(1e-30)).sqrt()
        magnitude = total * (p2.clamp_min(1e-30) / g2.clamp_min(1e-30)).sqrt()
        direction = torch.where(common, direction, total)
        magnitude = torch.where(common, magnitude, total)
        deltas = {'none': torch.zeros_like(total), 'project': raw_delta, 'total_direction': torch.where(common, direction - total, torch.zeros_like(total)), 'total_magnitude': torch.where(common, magnitude - total, torch.zeros_like(total))}
        direction = total + deltas['total_direction']
        magnitude = total + deltas['total_magnitude']
        return (row, common, proposed, direction, magnitude, deltas)

    @staticmethod
    def _row_norm(x):
        return x.square().sum(-1, keepdim=True).clamp_min(0).sqrt()

    @staticmethod
    def _relative_norm_error(a, b):
        return ((_Model17._row_norm(a) - _Model17._row_norm(b)).abs() / _Model17._row_norm(b).clamp_min(1e-15)).max()

    def training_loss(self, batch, logits, train_labels):
        labels = train_labels.to(dtype=logits.dtype).reshape_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
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
        params = (self.embeddings[self.category].weight, self.embeddings[self.brand].weight)
        grads = torch.autograd.grad(loss, paths + params, retain_graph=True, create_graph=False)
        grads = tuple((g.detach() for g in grads))
        total = grads[4:]
        audits = []
        deltas = []
        correction = logits.new_zeros(())
        for (i, field) in enumerate((self.category, self.brand)):
            qids = batch['static'][:, field]
            hids = batch['history_category' if i == 0 else 'history_brand']
            q = self._aggregate(params[i], qids, grads[i], qids != 0)
            h = self._aggregate(params[i], hids, grads[i + 2], (hids != 0) & batch['history_mask'])
            audit = self._total_targets(total[i], q, h)
            delta = audit[5][self.config['row_intervention']].detach()
            deltas.append(delta)
            audits.append(audit)
            if self.config['row_intervention'] != 'none':
                correction = correction + (params[i] * delta).sum()
        result = loss if self.config['row_intervention'] == 'none' else loss + (correction - correction.detach())
        proposed = tuple((a[2] for a in audits))
        direction = tuple((a[3] for a in audits))
        magnitude = tuple((a[4] for a in audits))
        products = tuple((a[0][0] for a in audits))
        eligible = sum((a[1].sum() for a in audits))
        conflict = sum((a[0][4].sum() for a in audits))
        total2 = sum((t.square().sum(dtype=torch.float64) for t in total))
        delta2 = sum((d.square().sum(dtype=torch.float64) for d in deltas))
        conflict_energy = sum(((t.square() * a[0][4].to(t.dtype)).sum(dtype=torch.float64) for (t, a) in zip(total, audits)))
        direction_error = torch.stack([self._relative_norm_error(d, g) for (d, g) in zip(direction, total)]).max()
        magnitude_error = torch.stack([self._relative_norm_error(m, p) for (m, p) in zip(magnitude, proposed)]).max()
        extra_degenerate = sum(((a[0][5] & ~a[1]).sum() for a in audits))
        running = torch.stack([logits.new_ones(()), eligible, conflict, conflict_energy, total2, delta2, extra_degenerate, direction_error, magnitude_error]).detach().to(torch.float64)
        if not bool(torch.isfinite(running).all()):
            raise RuntimeError('Nonfinite complete-gradient factorial; retain failed receipt')
        with torch.no_grad():
            self._running[:7].add_(running[:7])
            self._running[7:].copy_(torch.maximum(self._running[7:], running[7:]))
        if capture:
            actual = torch.autograd.grad(result, params, retain_graph=True, create_graph=False)
            expected = tuple((g + d for (g, d) in zip(total, deltas)))
            magnitude_direction_errors = []
            for (g, m) in zip(total, magnitude):
                gn = self._row_norm(g).clamp_min(1e-15)
                mn = self._row_norm(m).clamp_min(1e-15)
                magnitude_direction_errors.append((m / mn - g / gn).abs().max())
            inactive_max = torch.stack([d.masked_fill(a[1].expand_as(d), 0).abs().max() for (d, a) in zip(deltas, audits)]).max()
            role_active = sum((a[0][3].sum() for a in audits))
            values = [logits.new_tensor(float(call)), logits.new_tensor(float(labels.numel())), role_active, conflict, eligible, extra_degenerate, total2.clamp_min(0).sqrt(), self._norm(tuple((g - p for (g, p) in zip(total, products)))), self._norm(products), self._norm(proposed), delta2.clamp_min(0).sqrt() / total2.clamp_min(1e-30).sqrt(), self._norm(expected) / total2.clamp_min(1e-30).sqrt(), direction_error, magnitude_error, torch.stack(magnitude_direction_errors).max(), inactive_max, torch.stack([d[0].abs().max() for d in deltas]).max(), self._norm(tuple((a.detach() - e for (a, e) in zip(actual, expected)))) / self._norm(expected).clamp_min(1e-30), (result.detach() - loss.detach()).abs(), conflict_energy / total2.clamp_min(1e-30), self._norm(tuple((d - g for (d, g) in zip(direction, total)))) / total2.clamp_min(1e-30).sqrt()]
            values = torch.stack([v.detach().to(torch.float64) for v in values])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite training snapshot; retain failed receipt')
            with torch.no_grad():
                self._snapshots[0 if call == 0 else 1].copy_(values)
        return result

    @staticmethod
    def _synthetic_total_audit():
        q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1e-20, 0.0]], dtype=torch.float64)
        h = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, 1.0], [-1e-20, 1e-20]], dtype=torch.float64)
        g = torch.tensor([[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-0.5, -0.5], [1.0, 1.0]], dtype=torch.float64)
        (row, eligible, p, d, m, deltas) = _Model17._total_targets(g, q, h)
        expected_project = g.clone()
        expected_project[1] = torch.tensor([2.5, 1.5], dtype=g.dtype)
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
        vals = torch.tensor([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [7.0, 8.0]], dtype=g.dtype)
        agg = _Model17._aggregate(torch.zeros(3, 2, dtype=g.dtype), ids, vals, torch.tensor([True, True, True, False]))
        expected_agg = torch.tensor([[0.0, 0.0], [4.0, 6.0], [0.0, 0.0]], dtype=g.dtype)
        direction_formula = g.clone()
        direction_formula[1] = expected_project[1] * (5.0 / 8.5) ** 0.5
        return {'synthetic_raw_projection_abs_error': float((p - expected_project).abs().max()), 'synthetic_direction_abs_error': float((d - direction_formula).abs().max()), 'synthetic_direction_norm_relative_error': float(_Model17._relative_norm_error(d, g)), 'synthetic_magnitude_norm_relative_error': float(_Model17._relative_norm_error(m, p)), 'synthetic_surrogate_gradient_abs_error': float(error), 'synthetic_scalar_bce_abs_error': float(scalar_error), 'synthetic_aggregate_abs_error': float((agg - expected_agg).abs().max()), 'synthetic_eligibility_abs_error': float((eligible.reshape(-1).to(g.dtype) - torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=g.dtype)).abs().max())}

    def diagnostics(self, batch):
        names = ('call', 'samples', 'role_active_rows', 'role_conflicting_rows', 'eligible_rows', 'extra_degenerate_rows', 'complete_gradient_norm', 'unchanged_path_norm', 'product_sum_norm', 'raw_project_complete_norm', 'correction_complete_ratio', 'new_complete_norm_ratio', 'direction_norm_relative_error', 'magnitude_norm_relative_error', 'magnitude_direction_abs_error', 'outside_eligible_correction_abs_max', 'padding_correction_abs_max', 'intended_gradient_relative_error', 'scalar_bce_abs_error', 'conflict_complete_energy_fraction', 'direction_change_relative')
        out = {}
        for slot in range(2):
            for (i, name) in enumerate(names):
                out['train_cache' + str(slot) + '_' + name] = float(self._snapshots[slot, i])
        r = self._running
        out.update({'train_through_checkpoint_calls': float(r[0]), 'train_through_checkpoint_eligible_rows_per_call': float(r[1] / r[0].clamp_min(1)), 'train_through_checkpoint_conflicting_rows_per_call': float(r[2] / r[0].clamp_min(1)), 'train_through_checkpoint_conflict_energy_fraction': float(r[3] / r[4].clamp_min(1e-30)), 'train_through_checkpoint_correction_rms_ratio': float((r[5] / r[4].clamp_min(1e-30)).clamp_min(0).sqrt()), 'train_through_checkpoint_extra_degenerate_rows': float(r[6]), 'train_through_checkpoint_direction_norm_error_max': float(r[7]), 'train_through_checkpoint_magnitude_norm_error_max': float(r[8])})
        (fields, features, history, components) = self._encode(batch)
        raw = fields[self.category][:, None, :] * components[0] + fields[self.brand][:, None, :] * components[1]
        product = self._normalize_product(raw, 'full', batch['history_mask'])
        weights = self._weights(self._scores_from_product(fields, history, product), batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        out.update({'validation_samples': float(batch['static'].shape[0]), 'validation_forward_abs_max': float((self.forward(batch) - reference).abs().max()), 'validation_masked_mass_max': float((weights * (~batch['history_mask']).to(weights.dtype)).sum(-1).max()), 'validation_weight_sum_error_max': float((weights.sum(-1) - batch['history_mask'].any(-1).to(weights.dtype)).abs().max()), 'validation_empty_summary_norm_max': float((summary.square().sum(-1).sqrt() * ~batch['history_mask'].any(-1)).max())})
        out.update(self._synthetic_total_audit())
        return out

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
_CONFIG18 = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'total_magnitude'}

class _Model18(nn.Module):

    def __init__(self, meta, config=None):
        super().__init__()
        config = {**_CONFIG18, **(config or {})}
        self.config = config
        self.kind = config['kind']
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = config['embedding_dim']
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1))
        layers = []
        width = len(meta['fields']) * dim + (dim if self.kind != 'static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)
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
        fields = [embedding(batch['static'][:, i]) for (i, embedding) in enumerate(self.embeddings)]
        features = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        category = self.embeddings[self.category](batch['history_category'])
        brand = self.embeddings[self.brand](batch['history_brand'])
        history = category + brand
        action = self.action(batch['history_type'])
        history = history + action
        recency = self.recency(ages)
        history = history + recency
        return (fields, features, history, (category, brand, action, recency))

    @staticmethod
    def _weights(scores, mask):
        scores = scores.masked_fill(~mask, -10000.0)
        weights = scores.softmax(-1) * mask.float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-09)

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
        positions = torch.arange(inverse.shape[-1], device=inverse.device, dtype=torch.long)[None, :].expand_as(inverse)
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
        if gradient in ('action_mean_tangent', 'action_reassigned_tangent', 'action_adjacent_tangent'):
            if action_type is None:
                raise ValueError('behavior-stratified scales require history_type')
            base_mode = {'action_mean_tangent': 'history_mean_tangent', 'action_reassigned_tangent': 'history_reassigned_tangent', 'action_adjacent_tangent': 'history_adjacent_tangent'}[gradient]
            result = inverse
            for action in range(1, 5):
                selected = valid_regular & (action_type == action)
                assigned = _Model18._event_scales(inverse, selected, base_mode)
                result = torch.where(selected, assigned, result)
            return result
        if gradient in ('scale_adjacent_tangent', 'scale_half_tangent'):
            donor = _Model18._rotation_donors(inverse, valid_regular, gradient)
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
            rank = valid_regular.long().cumsum(-1) - 1
            packed = torch.zeros_like(inverse).scatter_add(-1, rank.clamp_min(0), values)
            shift = 1 if gradient == 'history_adjacent_tangent' else count // 2
            donor = (rank + shift).remainder(count.clamp_min(1))
            rotated = packed.gather(-1, donor)
            return torch.where(valid_regular, rotated, inverse)
        raise ValueError('unknown normalization_gradient')

    @staticmethod
    def _normalize_product(product, gradient='full', mask=None, action_type=None):
        squared = product.square().sum(-1, keepdim=True)
        denominator = squared.clamp_min(1e-06).sqrt()
        unit = product / denominator
        if gradient == 'full':
            return unit
        regular = squared >= 1e-06
        valid_regular = regular.squeeze(-1)
        if mask is not None:
            valid_regular = valid_regular & mask
        inverse = denominator.squeeze(-1).reciprocal().detach()
        target = _Model18._event_scales(inverse, valid_regular, gradient, action_type)
        coefficient = torch.where(regular, denominator * target.unsqueeze(-1), torch.ones_like(denominator)).detach()
        surrogate = unit * coefficient
        return unit.detach() + (surrogate - surrogate.detach())

    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def _product(self, fields, components, mask):
        (category, brand, _, _) = components
        (qc, qb) = (fields[self.category], fields[self.brand])
        if self.training and torch.is_grad_enabled():
            (qc, qb, category, brand) = (qc.clone(), qb.clone(), category.clone(), brand.clone())
            self._path_cache = (qc, qb, category, brand)
        product = qc[:, None, :] * category + qb[:, None, :] * brand
        return self._normalize_product(product, self.config['normalization_gradient'], mask)

    def forward(self, batch):
        self._path_cache = None
        self._cached_logits = None
        (fields, features, history, components) = self._encode(batch)
        product = self._product(fields, components, batch['history_mask'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
        if self._path_cache is not None:
            self._cached_logits = logits
        return logits

    @staticmethod
    def _aggregate(weight, ids, gradient, valid):
        out = torch.zeros_like(weight)
        values = gradient.detach().reshape(-1, gradient.shape[-1])
        values = values * valid.reshape(-1, 1).to(values.dtype)
        out.index_add_(0, ids.reshape(-1), values)
        out[0].zero_()
        return out

    @staticmethod
    def _row_targets(q, h):
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
        ratio = (projected.square().sum(-1, keepdim=True) / s2.clamp_min(1e-30)).clamp_min(0).sqrt()
        norm_match = torch.where(eligible, original * ratio, original)
        projected = torch.where(eligible, projected, original)
        return (original, projected, norm_match, active, conflict, eligible, dot)

    @staticmethod
    def _norm(tensors):
        return sum((t.square().sum(dtype=torch.float64) for t in tensors)).clamp_min(0).sqrt()

    @staticmethod
    def _total_targets(total, q, h):
        """Detached common-state factorial. Caller stops all input gradients.

        Raw delta is retained verbatim for the projection bridge. All arms
        share the additional G/P norm guard, so degenerate cases cannot be
        silently counted as successful direction/magnitude matching.
        """
        row = _Model18._row_targets(q, h)
        (product, projected) = row[:2]
        raw_delta = projected - product
        proposed = total + raw_delta
        g2 = total.square().sum(-1, keepdim=True)
        p2 = proposed.square().sum(-1, keepdim=True)
        common = row[5] & (g2 > 1e-30) & (p2 > 1e-30)
        common = common.clone()
        common[0] = False
        raw_delta = torch.where(common, raw_delta, torch.zeros_like(raw_delta))
        proposed = total + raw_delta
        direction = proposed * (g2.clamp_min(1e-30) / p2.clamp_min(1e-30)).sqrt()
        magnitude = total * (p2.clamp_min(1e-30) / g2.clamp_min(1e-30)).sqrt()
        direction = torch.where(common, direction, total)
        magnitude = torch.where(common, magnitude, total)
        deltas = {'none': torch.zeros_like(total), 'project': raw_delta, 'total_direction': torch.where(common, direction - total, torch.zeros_like(total)), 'total_magnitude': torch.where(common, magnitude - total, torch.zeros_like(total))}
        direction = total + deltas['total_direction']
        magnitude = total + deltas['total_magnitude']
        return (row, common, proposed, direction, magnitude, deltas)

    @staticmethod
    def _row_norm(x):
        return x.square().sum(-1, keepdim=True).clamp_min(0).sqrt()

    @staticmethod
    def _relative_norm_error(a, b):
        return ((_Model18._row_norm(a) - _Model18._row_norm(b)).abs() / _Model18._row_norm(b).clamp_min(1e-15)).max()

    @staticmethod
    def _allocation_targets(total, audit):
        """Detached targets; support includes complete-gradient other paths.

        Padding and squared norms <=1e-30 are never donors or recipients.
        No CPU decisions, row sampling, RNG, cross-batch state or extra labels.
        """
        g2 = total.square().sum(-1, keepdim=True)
        support = (g2.squeeze(-1) > 1e-30).clone()
        support[0] = False
        (common, proposed, magnitude) = (audit[1], audit[2], audit[4])
        p2 = proposed.square().sum(-1, keepdim=True)
        own = torch.where(common, (p2.clamp_min(1e-30) / g2.clamp_min(1e-30)).sqrt(), torch.ones_like(g2)).squeeze(-1)
        count = support.long().sum()
        rank = support.long().cumsum(0) - 1
        packed = torch.zeros_like(own).scatter_add(0, rank.clamp_min(0), torch.where(support, own, torch.zeros_like(own)))
        donor_rank = (rank + count // 2).remainder(count.clamp_min(1))
        assigned = packed.gather(0, donor_rank)
        assigned = torch.where(support, assigned, torch.ones_like(assigned))
        rotated = total * assigned[:, None]
        rotate_delta = torch.where(support[:, None], rotated - total, torch.zeros_like(total))
        base_energy = (total.square() * support[:, None]).sum(dtype=torch.float64)
        target_energy = (magnitude.square() * support[:, None]).sum(dtype=torch.float64)
        alpha = torch.where(base_energy > 0, (target_energy / base_energy.clamp_min(1e-30)).sqrt(), torch.ones_like(base_energy)).to(total.dtype)
        uniform_delta = torch.where(support[:, None], total * alpha - total, torch.zeros_like(total))
        deltas = {'none': torch.zeros_like(total), 'total_magnitude': audit[5]['total_magnitude'], 'rotate_active': rotate_delta, 'uniform_energy': uniform_delta}
        targets = {key: total + delta for (key, delta) in deltas.items()}
        return (support, own, assigned, alpha, deltas, targets)

    @staticmethod
    def _allocation_metrics(total, audit, allocation, delta):
        (support, own, assigned, alpha, _, targets) = allocation
        active = support[:, None]
        energies = [t.square().sum(dtype=torch.float64) for t in (total, targets['total_magnitude'], targets['rotate_active'], targets['uniform_energy'])]
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
        gn = _Model18._row_norm(total).clamp_min(1e-30)
        sn = _Model18._row_norm(selected).clamp_min(1e-30)
        direction_error = ((selected / sn - total / gn) * active).abs().max()
        outside = delta.masked_fill(active.expand_as(delta), 0).abs().max()
        extra = (audit[0][5] & ~audit[1]).sum()
        sums = energies + [delta.square().sum(dtype=torch.float64), support.sum(), audit[1].sum(), moved.sum(), changed.sum(), outside_conflict.sum(), extra]
        return (sums, [uniform_error, hist1, hist2, delta[0].abs().max(), outside, direction_error])

    def training_loss(self, batch, logits, train_labels):
        labels = train_labels.to(dtype=logits.dtype).reshape_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
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
        params = (self.embeddings[self.category].weight, self.embeddings[self.brand].weight)
        grads = torch.autograd.grad(loss, paths + params, retain_graph=True, create_graph=False)
        grads = tuple((g.detach() for g in grads))
        total = grads[4:]
        (audits, allocations, deltas, metrics) = ([], [], [], [])
        correction = logits.new_zeros(())
        mode = self.config['row_intervention']
        for (i, field) in enumerate((self.category, self.brand)):
            qids = batch['static'][:, field]
            hids = batch['history_category' if i == 0 else 'history_brand']
            q = self._aggregate(params[i], qids, grads[i], qids != 0)
            h = self._aggregate(params[i], hids, grads[i + 2], (hids != 0) & batch['history_mask'])
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
        sum_values = [sum((m[0][j] for m in metrics)) for j in range(11)]
        max_values = [torch.stack([m[1][j] for m in metrics]).max() for j in range(6)]
        running = torch.stack([logits.new_ones(())] + sum_values + max_values).detach().to(torch.float64)
        if not bool(torch.isfinite(running).all()):
            raise RuntimeError('Nonfinite allocation audit; retain failed receipt')
        with torch.no_grad():
            self._running[:12].add_(running[:12])
            self._running[12:].copy_(torch.maximum(self._running[12:], running[12:]))
        if capture:
            actual = torch.autograd.grad(result, params, retain_graph=True, create_graph=False)
            expected = tuple((g + d for (g, d) in zip(total, deltas)))
            error = self._norm(tuple((a.detach() - e for (a, e) in zip(actual, expected))))
            total_energy = sum_values[0].clamp_min(1e-30)
            selected_energy = sum((t.square().sum(dtype=torch.float64) for t in expected))
            (active_count, eligible_count) = (sum_values[5], sum_values[6])
            (changed_count, outside_count) = (sum_values[8], sum_values[9])
            values = [logits.new_tensor(float(call)), logits.new_tensor(float(labels.numel())), error / self._norm(expected).clamp_min(1e-30), (result.detach() - loss.detach()).abs(), total_energy.sqrt(), (sum_values[4] / total_energy).sqrt(), (selected_energy / total_energy).sqrt(), eligible_count, active_count, (active_count > 1).to(logits.dtype), sum_values[7] / active_count.clamp_min(1), (changed_count - outside_count) / changed_count.clamp_min(1), max_values[0], torch.maximum(max_values[1], max_values[2])]
            values = torch.stack([v.detach().to(torch.float64) for v in values])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite allocation snapshot; retain failed receipt')
            with torch.no_grad():
                self._snapshots[0 if call == 0 else 1].copy_(values)
        return result

    @staticmethod
    def _synthetic_total_audit():
        q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1e-20, 0.0]], dtype=torch.float64)
        h = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, 1.0], [-1e-20, 1e-20]], dtype=torch.float64)
        g = torch.tensor([[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-0.5, -0.5], [1.0, 1.0]], dtype=torch.float64)
        (row, eligible, p, d, m, deltas) = _Model18._total_targets(g, q, h)
        expected_project = g.clone()
        expected_project[1] = torch.tensor([2.5, 1.5], dtype=g.dtype)
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
        vals = torch.tensor([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [7.0, 8.0]], dtype=g.dtype)
        agg = _Model18._aggregate(torch.zeros(3, 2, dtype=g.dtype), ids, vals, torch.tensor([True, True, True, False]))
        expected_agg = torch.tensor([[0.0, 0.0], [4.0, 6.0], [0.0, 0.0]], dtype=g.dtype)
        direction_formula = g.clone()
        direction_formula[1] = expected_project[1] * (5.0 / 8.5) ** 0.5
        return {'synthetic_raw_projection_abs_error': float((p - expected_project).abs().max()), 'synthetic_direction_abs_error': float((d - direction_formula).abs().max()), 'synthetic_direction_norm_relative_error': float(_Model18._relative_norm_error(d, g)), 'synthetic_magnitude_norm_relative_error': float(_Model18._relative_norm_error(m, p)), 'synthetic_surrogate_gradient_abs_error': float(error), 'synthetic_scalar_bce_abs_error': float(scalar_error), 'synthetic_aggregate_abs_error': float((agg - expected_agg).abs().max()), 'synthetic_eligibility_abs_error': float((eligible.reshape(-1).to(g.dtype) - torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=g.dtype)).abs().max())}

    @staticmethod
    def _synthetic_allocation_audit():
        g = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 0.0], [0.0, 4.0], [1.0, 1.0]], dtype=torch.float64)
        q = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=g.dtype)
        h = torch.tensor([[0.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=g.dtype)
        audit = _Model18._total_targets(g, q, h)
        allocation = _Model18._allocation_targets(g, audit)
        expected_rotation = g.clone()
        expected_rotation[3, 1] = 4.0 * 1.7 ** 0.5
        expected_uniform = g * (35.5 / 32.0) ** 0.5
        (sums, maxima) = _Model18._allocation_metrics(g, audit, allocation, allocation[4]['rotate_active'])
        zero = torch.zeros_like(g)
        empty = _Model18._allocation_targets(zero, _Model18._total_targets(zero, zero, zero))
        empty_error = torch.stack([t.abs().max() for t in empty[5].values()]).max()
        return {'synthetic_rotation_formula_abs_error': float((allocation[5]['rotate_active'] - expected_rotation).abs().max()), 'synthetic_uniform_formula_abs_error': float((allocation[5]['uniform_energy'] - expected_uniform).abs().max()), 'synthetic_uniform_energy_relative_error': float(maxima[0]), 'synthetic_empty_allocation_abs_error': float(empty_error)}

    def diagnostics(self, batch):
        names = ('call', 'samples', 'actual_gradient_relative_error', 'scalar_bce_abs_error', 'complete_gradient_norm', 'correction_ratio', 'selected_norm_ratio', 'eligible_rows', 'active_rows', 'multiple_active_rows', 'reallocated_fraction', 'rotated_receivers_conflict_fraction', 'uniform_energy_relative_error', 'rotation_moment_relative_error')
        out = {}
        for slot in range(2):
            for (i, name) in enumerate(names):
                out['train_cache' + str(slot) + '_' + name] = float(self._snapshots[slot, i])
        r = self._running
        out.update({'train_through_checkpoint_calls': float(r[0]), 'train_energy_G': float(r[1]), 'train_energy_M': float(r[2]), 'train_energy_rotated': float(r[3]), 'train_energy_uniform': float(r[4]), 'train_correction_rms_ratio': float((r[5] / r[1].clamp_min(1e-30)).sqrt()), 'train_active_rows_per_call': float(r[6] / r[0].clamp_min(1)), 'train_eligible_rows_per_call': float(r[7] / r[0].clamp_min(1)), 'train_reallocated_fraction': float(r[8] / r[6].clamp_min(1)), 'train_rotated_receiver_fraction': float(r[9] / r[6].clamp_min(1)), 'train_rotated_outside_conflict_fraction': float(r[10] / r[9].clamp_min(1)), 'train_extra_degenerate_rows': float(r[11]), 'train_uniform_energy_relative_error_max': float(r[12]), 'train_rotation_first_moment_relative_error_max': float(r[13]), 'train_rotation_second_moment_relative_error_max': float(r[14]), 'train_padding_correction_abs_max': float(r[15]), 'train_outside_active_correction_abs_max': float(r[16]), 'train_raw_direction_abs_error_max': float(r[17])})
        (fields, features, history, components) = self._encode(batch)
        raw = fields[self.category][:, None, :] * components[0] + fields[self.brand][:, None, :] * components[1]
        product = self._normalize_product(raw, 'full', batch['history_mask'])
        weights = self._weights(self._scores_from_product(fields, history, product), batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        out.update({'validation_samples': float(batch['static'].shape[0]), 'validation_forward_abs_max': float((self.forward(batch) - reference).abs().max()), 'validation_masked_mass_max': float((weights * (~batch['history_mask']).to(weights.dtype)).sum(-1).max()), 'validation_weight_sum_error_max': float((weights.sum(-1) - batch['history_mask'].any(-1).to(weights.dtype)).abs().max()), 'validation_empty_summary_norm_max': float((summary.square().sum(-1).sqrt() * ~batch['history_mask'].any(-1)).max())})
        out.update(self._synthetic_total_audit())
        out.update(self._synthetic_allocation_audit())
        return out

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'none'}


_AUDIT_NAMES = (
    'call', 'completed', 'cross_logits_abs', 'cross_scalar_abs',
    'repeat_gradient_rel', 'cross_gradient_rel', 'cross_query_role_rel',
    'cross_history_role_rel', 'cross_complete_baseline_rel',
    'cross_project_target_rel', 'cross_correction_rel',
    'cross_eligible_mismatch', 'repeat_adam_update_rel',
    'cross_adam_update_rel', 'actual_gradient17_rel',
    'actual_adam_update17_rel', 'cross_adam_moment1_rel',
    'cross_adam_moment2_rel', 'actual_adam_moment1_rel',
    'actual_adam_moment2_rel', 'padding_correction_abs',
    'outside_eligible_correction_abs', 'scalar_bce_abs',
    'adam_state_present', 'adam_step_before', 'repeat_logits_abs',
    'cross_other_gradient_rel', 'repeat_scalar_abs')


def _relative(xs, ys):
    numerator = sum((x.detach() - y.detach()).square().sum(dtype=torch.float64)
                    for x, y in zip(xs, ys))
    denominator = sum(y.detach().square().sum(dtype=torch.float64) for y in ys)
    return (numerator.clamp_min(0).sqrt() /
            denominator.clamp_min(1e-60).sqrt())


class Model(_Model17):
    # These helpers are called only by the embedded round18 training_loss.
    _allocation_targets = staticmethod(_Model18._allocation_targets)
    _allocation_metrics = staticmethod(_Model18._allocation_metrics)

    def __init__(self, meta, config=None):
        super().__init__(meta, {**CONFIG, **(config or {})})
        if self.config['row_intervention'] not in ('none', 'total_magnitude'):
            raise ValueError('Audit fixes magnitude or unchanged-gradient mode')
        self.register_buffer('_audit_snapshots', torch.zeros(2, 28, dtype=torch.float64))
        self._audit_snapshots[:, 0] = -1
        self.register_buffer('_audit_training_calls', torch.zeros((), dtype=torch.float64))
        self._audit_capture = None
        self._audit_source = 17
        self._audit_pending = None

    def _total_targets(self, total, q, h):
        source = _Model17 if self._audit_source == 17 else _Model18
        result = source._total_targets(total, q, h)
        if self._audit_capture is not None:
            delta = result[5][self.config['row_intervention']]
            self._audit_capture.append(tuple(t.detach().clone() for t in
                (q, h, total, result[2], delta, result[1])))
        return result

    def _probe(self, batch, labels, version):
        # Swap ONLY diagnostic buffers/caches around the exact source loss.
        # No parameter or optimizer state is changed. No RNG is consumed.
        saved = (self._snapshots, self._running, self._training_loss_calls,
                 self._path_cache, self._cached_logits, self._audit_source,
                 self._audit_capture)
        source = _Model17 if version == 17 else _Model18
        self._snapshots = self._snapshots.new_zeros((2, 21 if version == 17 else 14))
        self._snapshots[:, 0] = -1
        self._running = self._running.new_zeros(9 if version == 17 else 18)
        self._training_loss_calls = saved[2] - 1
        self._audit_source = version
        self._audit_capture = []
        try:
            logits = self.forward(batch)
            loss = source.training_loss(self, batch, logits, labels)
            gradients = tuple(g.detach() for g in torch.autograd.grad(
                loss, tuple(self.parameters()), create_graph=False))
            captured = self._audit_capture
            return logits.detach(), loss.detach(), gradients, captured
        finally:
            (self._snapshots, self._running, self._training_loss_calls,
             self._path_cache, self._cached_logits, self._audit_source,
             self._audit_capture) = saved

    def training_loss(self, batch, logits, train_labels):
        if not self.training or not torch.is_grad_enabled():
            return _Model17.training_loss(self, batch, logits, train_labels)
        with torch.no_grad():
            self._audit_training_calls.add_(1)
        call = self._training_loss_calls
        if call not in (0, 512):
            return _Model17.training_loss(self, batch, logits, train_labels)
        if self._audit_pending is not None:
            raise RuntimeError('Prior audit step was not consumed by declared optimizer')
        self._audit_capture = []
        result = _Model17.training_loss(self, batch, logits, train_labels)
        original = self._audit_capture
        self._audit_capture = None
        params = tuple(self.parameters())
        reference = tuple(g.detach() for g in torch.autograd.grad(
            result, params, retain_graph=True, create_graph=False))
        repeat_logits, repeat_loss, repeat_grads, repeat_rows = self._probe(
            batch, train_labels, 17)
        new_logits, new_loss, new_grads, new_rows = self._probe(
            batch, train_labels, 18)
        shared = (self.embeddings[self.category].weight,
                  self.embeddings[self.brand].weight)
        indices = [next(j for j, p in enumerate(params) if p is t) for t in shared]
        other = [j for j in range(len(params)) if j not in indices]
        row = self._audit_snapshots[0 if call == 0 else 1]
        values = {name: row.new_zeros(()) for name in _AUDIT_NAMES}
        values.update(call=row.new_tensor(float(call)),
                      cross_logits_abs=(new_logits - logits.detach()).abs().max(),
                      cross_scalar_abs=(new_loss - result.detach()).abs(),
                      repeat_gradient_rel=_relative(repeat_grads, reference),
                      cross_gradient_rel=_relative(new_grads, reference),
                      cross_other_gradient_rel=_relative(
                          [new_grads[j] for j in other], [reference[j] for j in other]),
                      repeat_logits_abs=(repeat_logits - logits.detach()).abs().max(),
                      repeat_scalar_abs=(repeat_loss - result.detach()).abs(),
                      scalar_bce_abs=(result.detach() -
                          torch.nn.functional.binary_cross_entropy_with_logits(
                              logits.detach(), train_labels.to(logits.dtype).reshape_as(logits),
                              reduction='mean')).abs())
        for k, name in enumerate(('cross_query_role_rel', 'cross_history_role_rel',
                                 'cross_complete_baseline_rel',
                                 'cross_project_target_rel', 'cross_correction_rel')):
            values[name] = _relative([r[k] for r in new_rows], [r[k] for r in original])
        values['cross_eligible_mismatch'] = sum(
            (a[5] != b[5]).sum() for a, b in zip(new_rows, original))
        values['padding_correction_abs'] = torch.stack(
            [a[4][0].abs().max() for a in original + new_rows]).max()
        values['outside_eligible_correction_abs'] = torch.stack([
            a[4].masked_fill(a[5].expand_as(a[4]), 0).abs().max()
            for a in original + new_rows]).max()
        scalars = torch.stack([values[n].detach().to(row) for n in _AUDIT_NAMES])
        if not bool(torch.isfinite(scalars).all()):
            raise RuntimeError('Nonfinite common-state backward audit')
        with torch.no_grad():
            row.copy_(scalars)
        # Only two table gradients per source survive to the optimizer step.
        self._audit_pending = (0 if call == 0 else 1,
            tuple(reference[j] for j in indices),
            tuple(repeat_grads[j] for j in indices),
            tuple(new_grads[j] for j in indices))
        return result

    def diagnostics(self, batch):
        out = {'audit' + str(slot) + '_' + name: float(self._audit_snapshots[slot, j])
               for slot in range(2) for j, name in enumerate(_AUDIT_NAMES)}
        out['audit_completed_snapshots'] = float(self._audit_snapshots[:, 1].sum())
        out['training_calls_through_checkpoint'] = float(self._audit_training_calls)
        # Same five input-only forward checks as the exact sources.
        fields, features, history, components = self._encode(batch)
        raw = (fields[self.category][:, None, :] * components[0] +
               fields[self.brand][:, None, :] * components[1])
        product = self._normalize_product(raw, 'full', batch['history_mask'])
        weights = self._weights(self._scores_from_product(fields, history, product),
                                batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        reference = self._logits(features, summary)
        out.update(validation_samples=float(batch['static'].shape[0]),
                   validation_forward_abs_max=float((self.forward(batch) - reference).abs().max()),
                   validation_masked_mass_max=float((weights * (~batch['history_mask']).to(weights.dtype)).sum(-1).max()),
                   validation_weight_sum_error_max=float((weights.sum(-1) - batch['history_mask'].any(-1).to(weights.dtype)).abs().max()),
                   validation_empty_summary_norm_max=float((summary.square().sum(-1).sqrt() * (~batch['history_mask'].any(-1))).max()))
        return out


class _AuditAdam(torch.optim.Adam):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=model.config['lr'])
        self._audit_model = model

    @staticmethod
    def _shadow(parameter, gradient, state, group):
        # Clone the LIVE moment state, not a hand-derived approximate update.
        p = nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.Adam([p], lr=group['lr'])
        optimizer.param_groups[0].update(
            {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'})
        optimizer.state[p] = copy.deepcopy(state)
        p.grad = gradient.detach().clone()
        optimizer.step()
        s = optimizer.state[p]
        return p.detach(), s['exp_avg'].detach(), s['exp_avg_sq'].detach()

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError('Frozen audit recipe requires ordinary Adam without closure')
        model = self._audit_model
        pending = model._audit_pending
        if pending is None:
            return super().step()
        slot, reference, repeated, cross = pending
        model._audit_pending = None
        shared = (model.embeddings[model.category].weight,
                  model.embeddings[model.brand].weight)
        group = self.param_groups[0]
        expected, repeat_values, cross_values, reference_updates = [], [], [], []
        with torch.no_grad():
            for j, p in enumerate(shared):
                state = self.state[p]
                original = self._shadow(p, reference[j], state, group)
                expected.append(original)
                reference_updates.append(original[0] - p.detach())
                repeat_values.append(self._shadow(p, repeated[j], state, group))
                cross_values.append(self._shadow(p, cross[j], state, group))
            actual_grad_error = _relative([p.grad for p in shared], reference)
            repeat_update_error = _relative(
                [a[0] - p.detach() for a, p in zip(repeat_values, shared)], reference_updates)
            cross_update_error = _relative(
                [a[0] - p.detach() for a, p in zip(cross_values, shared)], reference_updates)
            cross_m1 = _relative([a[1] for a in cross_values], [a[1] for a in expected])
            cross_m2 = _relative([a[2] for a in cross_values], [a[2] for a in expected])
            steps = [self.state[p].get('step', 0) for p in shared]
            step_before = max(float(s) for s in steps)
        result = super().step()
        with torch.no_grad():
            # Relative update error uses the small update norm, not the large
            # embedding norm; momentum comparisons use their own scale.
            difference = [p.detach() - a[0] for p, a in zip(shared, expected)]
            update_error = (_Model17._norm(difference) /
                            _Model17._norm(reference_updates).clamp_min(1e-30))
            actual_m1 = _relative([self.state[p]['exp_avg'] for p in shared],
                                  [a[1] for a in expected])
            actual_m2 = _relative([self.state[p]['exp_avg_sq'] for p in shared],
                                  [a[2] for a in expected])
            updates = dict(completed=1., repeat_adam_update_rel=repeat_update_error,
                           cross_adam_update_rel=cross_update_error,
                           actual_gradient17_rel=actual_grad_error,
                           actual_adam_update17_rel=update_error,
                           cross_adam_moment1_rel=cross_m1,
                           cross_adam_moment2_rel=cross_m2,
                           actual_adam_moment1_rel=actual_m1,
                           actual_adam_moment2_rel=actual_m2,
                           adam_state_present=float(step_before > 0),
                           adam_step_before=step_before)
            row = model._audit_snapshots[slot]
            for name, value in updates.items():
                value = torch.as_tensor(value, dtype=row.dtype, device=row.device)
                if not bool(torch.isfinite(value)):
                    raise RuntimeError('Nonfinite actual/shadow Adam audit')
                row[_AUDIT_NAMES.index(name)].copy_(value)
        return result


def make_optimizer(model):
    return _AuditAdam(model)
