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
import math
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'row_intervention': 'total_magnitude'}


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
    """Candidate-owned bounded trace, identical for M/N apart from CONFIG.

    No optimizer, RNG, loss, gradient replacement, data or selection edits.
    Snapshots are taken BEFORE the host backward/Adam step at calls
    0,1,512,4096,8192,14660. Only snapshots in the selected checkpoint survive.
    Each call rolls checksums of ordered static inputs and training labels;
    snapshots additionally checksum all currently supplied history tensors.
    Prior unsampled history contents and unavailable sample identities are NOT
    certified. Integer checksums are collision-prone sketches, not cryptographic
    identities. Mismatching sketches demonstrate different traced content;
    equal sketches do not prove equality or whole-trajectory reproducibility.
    Gradient probes are two extra backwards on the SAME retained graph, not
    independent forward reconstructions or the later live host backward.
    Instrumentation can perturb scheduling. No independent checker is claimed.
    Extra buffers retain only aggregate checksums/scalars, never example rows.
    """

    def __init__(self, meta, config=None):
        super().__init__(meta, config)
        self.register_buffer('_trace_moduli', torch.tensor(
            [1000000007, 1000000009], dtype=torch.int64))
        self.register_buffer('_trace_prefix', torch.zeros(2, dtype=torch.int64))
        self.register_buffer('_trace_examples', torch.zeros((), dtype=torch.int64))
        self.register_buffer('_trace_rows', torch.zeros(6, 9, dtype=torch.float64))
        self._trace_rows[:, 0] = -1
        self.register_buffer('_trace_repeat_bits', torch.zeros((), dtype=torch.int64))

    def _fingerprint(self, tensors):
        # Chunk sums cannot overflow int64: at most65536 residues below1e9+9.
        # Each multiplication uses residues below1e9+9 (product below1.01e18).
        # Coordinates are reduced BEFORE squaring. No floating-point reductions.
        out = torch.zeros_like(self._trace_moduli)
        offset = 0
        for ordinal, tensor in enumerate(tensors):
            flat = tensor.detach().contiguous().reshape(-1)
            if flat.dtype == torch.float32:
                bits = flat.view(torch.int32)
            elif flat.dtype == torch.float64:
                bits = flat.view(torch.int64)
            elif flat.is_floating_point():
                raise RuntimeError('Trace v1 supports float32/64 only; preserve failure')
            else:
                bits = flat
            out = (out * 1000003 + flat.numel() + ordinal + 1).remainder(
                self._trace_moduli)
            for start in range(0, flat.numel(), 65536):
                stop = min(start + 65536, flat.numel())
                values = bits[start:stop].to(torch.int64)
                pos = torch.arange(offset + start + 1, offset + stop + 1,
                                   device=flat.device, dtype=torch.int64)
                p1, p2 = 1000000007, 1000000009
                w1 = pos.remainder(p1)
                w2 = pos.remainder(p2).square().remainder(p2)
                h1 = (values.remainder(p1) * w1).remainder(p1).sum()
                h2 = (values.remainder(p2) * w2).remainder(p2).sum()
                out = (out + torch.stack((h1, h2))).remainder(self._trace_moduli)
            offset += flat.numel()
        return out

    def training_loss(self, batch, logits, train_labels):
        active = self.training and torch.is_grad_enabled()
        call = self._training_loss_calls
        result = super().training_loss(batch, logits, train_labels)
        if not active:
            return result
        with torch.no_grad():
            static_labels = self._fingerprint((batch['static'], train_labels))
            self._trace_prefix.copy_((self._trace_prefix * 1000003 +
                                     static_labels).remainder(self._trace_moduli))
            self._trace_examples.add_(train_labels.numel())
        calls = (0, 1, 512, 4096, 8192, 14660)
        if call in calls:
            params = tuple(self.parameters())
            # These probes never write .grad or replace any live host gradient.
            first = torch.autograd.grad(result, params, retain_graph=True,
                                        create_graph=False)
            repeat = torch.autograd.grad(result, params, retain_graph=True,
                                         create_graph=False)
            with torch.no_grad():
                input_hash = self._fingerprint((
                    self._trace_prefix, batch['static'], batch['history_category'],
                    batch['history_brand'], batch['history_type'],
                    batch['history_age_seconds'], batch['history_mask'], train_labels))
                parameter_hash = self._fingerprint(params)
                logit_hash = self._fingerprint((logits,))
                gradient_hash = self._fingerprint(first)
                repeat_hash = self._fingerprint(repeat)
                self._trace_repeat_bits.add_((gradient_hash != repeat_hash).any().to(torch.int64))
                relative = self._norm(tuple(a - b for a, b in zip(first, repeat))) / (
                    self._norm(first).clamp_min(1e-30))
                values = torch.stack((
                    relative.new_tensor(float(call)), input_hash[0], input_hash[1],
                    parameter_hash[0], parameter_hash[1], logit_hash[0],
                    gradient_hash[0], gradient_hash[1], relative)).to(torch.float64)
                if not bool(torch.isfinite(values).all()):
                    raise RuntimeError('Nonfinite trace; preserve failed receipt')
                self._trace_rows[calls.index(call)].copy_(values)
        return result

    def diagnostics(self, batch):
        # Exactly64 finite scalars. No validation labels or validation adaptation.
        # All cached trace fields refer to training, through selected checkpoint.
        names = ('call', 'input_checksum1', 'input_checksum2',
                 'parameter_checksum1', 'parameter_checksum2', 'logit_checksum1',
                 'gradient_checksum1', 'gradient_checksum2', 'repeat_gradient_relative')
        out = {}
        for slot in range(6):
            for index, name in enumerate(names):
                out['trace' + str(slot) + '_' + name] = float(self._trace_rows[slot, index])
        selected = self._fingerprint(tuple(self.parameters()))
        out.update({
            'training_calls_through_checkpoint': float(self._running[0]),
            'training_examples_through_checkpoint': float(self._trace_examples),
            'selected_parameter_checksum1': float(selected[0]),
            'selected_parameter_checksum2': float(selected[1]),
            'captured_slots_through_checkpoint': float((self._trace_rows[:, 0] >= 0).sum()),
            'static_label_prefix_checksum1': float(self._trace_prefix[0]),
            'static_label_prefix_checksum2': float(self._trace_prefix[1]),
            'extra_degenerate_rows_through_checkpoint': float(self._running[6]),
            'inherited_two_probe_surrogate_relative_error_max': float(self._snapshots[:, 17].max()),
            'snapshot_repeat_gradient_checksum_mismatches': float(self._trace_repeat_bits)})
        if len(out) != 64 or not all(math.isfinite(v) for v in out.values()):
            raise RuntimeError('Trace diagnostics contract violation')
        return out
