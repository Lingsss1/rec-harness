"""Shared-row product-role projection and a matched-magnitude control.

Prospective attribution experiment, not a verified new literature contribution.
Forward, parameter construction, Adam, mean BCE, data and seed are inherited.
Only gradients into the shared category/brand tables are changed. Per batch,
aggregate product-query and product-history BCE gradients by table row (q,h).
For negative q.h with nondegenerate norms, form symmetric orthogonal removal:
 q' = q - (q.h)/||h||^2 h; h' = h - (q.h)/||q||^2 q.
Projection uses q'+h'; the control keeps q+h direction but matches its norm
at that same state. Other paths O remain unchanged: new gradient O+target.
Rows with ||q||^2, ||h||^2 or ||q+h||^2 <=1e-30 are left unchanged,
including exact cancellation; row0 is never changed. This is single-loss
role-gradient surgery, not multi-task supervision or an Adam update projection.

Extra autograd queries consume compute and may exceed a host cap. Diagnostics
retain two training-call aggregates, never labels/rows; validation diagnostics
use no labels. Buffers describe the frozen best checkpoint's retained history,
not the population or independent seeds. No optimizer state or .grad is edited.
Sources: DIN v4 Eq2/section4.3; local round14 and round15 source/receipts.
General gradient-surgery primary literature is unverified in the authorized
registry (no hits); no originality or exact external reproduction is claimed.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'full', 'row_intervention': 'norm_match'}


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
        if config['row_intervention'] not in ('project', 'norm_match'):
            raise ValueError('Only projection and its norm control are admitted')
        self._path_cache = None
        self._cached_logits = None
        self._training_loss_calls = 0
        self.register_buffer('_snapshots', torch.zeros(2, 23, dtype=torch.float64))
        self._snapshots[:, 0] = -1


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

    def training_loss(self, batch, logits, train_labels):
        labels = train_labels.to(dtype=logits.dtype).reshape_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        if not self.training or not torch.is_grad_enabled():
            return loss
        if self._path_cache is None or self._cached_logits is not logits:
            raise RuntimeError('training_loss must receive the immediately preceding forward logits')
        paths = self._path_cache
        self._path_cache = None
        self._cached_logits = None
        call = self._training_loss_calls
        self._training_loss_calls += 1
        capture = call in (0, 512)
        params = (self.embeddings[self.category].weight, self.embeddings[self.brand].weight)
        targets = paths + params if capture else paths
        grads = torch.autograd.grad(loss, targets, retain_graph=True, create_graph=False)
        grads = tuple(g.detach() for g in grads)
        rows = []
        deltas = []
        correction = logits.new_zeros(())
        for i, field in enumerate((self.category, self.brand)):
            qids = batch['static'][:, field]
            hids = batch['history_category' if i == 0 else 'history_brand']
            q = self._aggregate(params[i], qids, grads[i], qids != 0)
            h = self._aggregate(params[i], hids, grads[i + 2],
                                (hids != 0) & batch['history_mask'])
            row = self._row_targets(q, h)
            original, projected, norm_match = row[:3]
            target = projected if self.config['row_intervention'] == 'project' else norm_match
            delta = (target - original).detach()
            delta[0].zero_()
            if capture:
                rows.append((q, h, row))
                deltas.append(delta)
            correction = correction + (params[i] * delta).sum()
        # Numerically identical scalar BCE. Only the declared table-gradient
        # correction remains in backward; all correction coefficients stop.
        result = loss + (correction - correction.detach())
        if capture:
            actual = torch.autograd.grad(result, params, retain_graph=True, create_graph=False)
            total = grads[4:]
            expected = tuple(g + d for g, d in zip(total, deltas))
            product_sum = tuple(r[2][0] for r in rows)
            unchanged = tuple(g - p for g, p in zip(total, product_sum))
            projection = tuple(r[2][1] for r in rows)
            norm_match = tuple(r[2][2] for r in rows)
            active_rows = sum(r[2][3].sum() for r in rows)
            conflicting = sum(r[2][4].sum() for r in rows)
            eligible_rows = sum(r[2][5].sum() for r in rows)
            qnorm = self._norm(tuple(r[0] for r in rows))
            hnorm = self._norm(tuple(r[1] for r in rows))
            total_norm = self._norm(total)
            match_errors = [((p.square().sum(-1).sqrt() - n.square().sum(-1).sqrt()).abs() /
                             p.square().sum(-1).sqrt().clamp_min(1e-30)).max()
                            for p, n in zip(projection, norm_match)]
            values = [logits.new_tensor(float(call)), logits.new_tensor(float(labels.numel())),
                      labels.detach().sum(), loss.detach(), qnorm, hnorm,
                      self._norm(product_sum), self._norm(unchanged), total_norm,
                      self._norm(deltas), self._norm(deltas) / total_norm.clamp_min(1e-30),
                      active_rows, conflicting, eligible_rows,
                      conflicting.to(torch.float64) / active_rows.clamp_min(1),
                      sum(r[2][6].sum(dtype=torch.float64) for r in rows) /
                          (qnorm * hnorm).clamp_min(1e-30),
                      self._norm(tuple(p - n for p, n in zip(projection, norm_match))) /
                          self._norm(projection).clamp_min(1e-30),
                      torch.stack(match_errors).max(),
                      torch.stack([d[0].abs().max() for d in deltas]).max(),
                      self._norm(tuple(a.detach() - e for a, e in zip(actual, expected))) /
                          self._norm(expected).clamp_min(1e-30),
                      (result.detach() - loss.detach()).abs(),
                      self._norm(expected) / total_norm.clamp_min(1e-30),
                      logits.new_tensor(float((~batch['history_mask'].any(-1)).sum()))]
            values = torch.stack([v.detach().to(dtype=torch.float64) for v in values])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite path-attribution snapshot; preserve failed receipt')
            with torch.no_grad():
                self._snapshots[0 if call == 0 else 1].copy_(values)
        return result

    @staticmethod
    def _synthetic_row_audit():
        # Candidate-supplied host check: negative/positive/zero dot, one-sided,
        # equal opposition, row0 and near-degenerate rows, no RNG or labels.
        q = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [0., 0.],
                          [1., 0.], [1e-20, 0.], [2., 1.]], dtype=torch.float64)
        h = torch.tensor([[0., 0.], [-1., 1.], [1., 1.], [1., 0.],
                          [-1., 0.], [-1e-20, 1e-20], [-1., 3.]], dtype=torch.float64)
        old, project, matched, active, conflict, eligible, _ = Model._row_targets(q, h)
        expected = old.clone()
        expected[1] = torch.tensor([0.5, 1.5], dtype=q.dtype)
        norm_error = (project.square().sum(-1) - matched.square().sum(-1)).abs().max()
        # Independently declared linear surrogate gradient must add delta once.
        with torch.enable_grad():
            w = torch.ones_like(q, requires_grad=True)
            delta = (project - old).detach()
            base = (w * old).sum()
            term = (w * delta).sum()
            augmented = base + (term - term.detach())
            (g,) = torch.autograd.grad(augmented, w)
        return {'synthetic_projection_abs_max_error': float((project - expected).abs().max()),
                'synthetic_matched_squared_norm_abs_error': float(norm_error),
                'synthetic_surrogate_gradient_abs_error': float((g - project).abs().max()),
                'synthetic_unchanged_scalar_abs_error': float((augmented.detach() - base.detach()).abs()),
                'synthetic_eligible_rows': float(eligible.sum()),
                'synthetic_conflicting_rows': float(conflict.sum())}

    def diagnostics(self, batch):
        # Exactly 59 finite scalars: 46 training-cache, 7 scope/forward/mask,
        # 6 candidate-authored synthetic checks. No validation labels or VJPs.
        names = ('call', 'samples', 'positives', 'bce', 'query_norm', 'history_norm',
                 'product_sum_norm', 'unchanged_norm', 'total_embedding_norm',
                 'correction_norm', 'correction_total_ratio', 'active_rows',
                 'conflicting_rows', 'eligible_rows', 'conflicting_fraction',
                 'query_history_cosine', 'projection_control_direction_relative',
                 'matched_norm_relative_max_error', 'padding_correction_abs_max',
                 'intended_gradient_relative_error', 'scalar_bce_abs_error',
                 'new_total_norm_ratio', 'empty_history_samples')
        out = {}
        for slot in range(2):
            for i, name in enumerate(names):
                out['train_' + str(slot) + '_' + name] = float(self._snapshots[slot, i])
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
                    'validation_empty_summary_norm_max': float((summary.square().sum(-1).sqrt() * (~batch['history_mask'].any(-1))).max()),
                    'completed_training_snapshots': float((self._snapshots[:, 0] >= 0).sum()),
                    'projection_enabled': float(self.config['row_intervention'] == 'project')})
        out.update(self._synthetic_row_audit())
        return out
