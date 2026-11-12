"""Bounded training-objective VJPs for the existing scale-half incumbent.

Prospective plan: proposal.json. Same inherited forward and backward rule;
no new optimizer, architecture, donor mode, data input or random seed.
Mean unweighted BCE is declared through the authorized training_loss hook.
At zero-based training-loss calls 0 and 512, probe all six existing rules
on the entire supplied TRAINING batch and current parameters. Probes use
only training labels and autograd.grad; no .grad writes or optimizer steps.
Only detached scalar summaries persist in checkpoint buffers. No validation
label is accepted by diagnostics. Audit-disabled control isolates probe
nonperturbation; ancestor bridges check the explicit loss hook as well.

These are candidate-authored objective VJPs at two training states. They
are not optimizer updates, independent seeds, causal AUC mediation, a new
training mechanism, or evidence of literature priority. Published DIN v4
section4.3 supplies the pooling idea, but omits this adapter's softmax.
Inherited norm floor/mask/rotation/forward methods are copied unchanged.
"""
import torch
from torch import nn

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'product_pairing': 'aligned', 'product_normalization': 'unit', 'normalization_gradient': 'scale_half_tangent', 'audit_enabled': True}


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
        # Numeric diagnostic buffers only; no trainable tensors or RNG calls.
        # Column zero is -1 for an unavailable snapshot; other columns are
        # placeholders until that slot is measured. Never interpret them as
        # zero gradient effects without checking the call-index/count fields.
        self.register_buffer('_audit_table', torch.zeros(2, 29, dtype=torch.float64))
        self._audit_table[:, 0] = -1
        self._training_loss_calls = 0



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


    def _product(self, fields, components, pairing, normalization, mask=None, action_type=None):
        category, brand, _, _ = components
        qc = fields[self.category][:, None, :]
        qb = fields[self.brand][:, None, :]
        if pairing == 'aligned':
            product = qc * category + qb * brand
        elif pairing == 'crossed':
            product = qc * brand + qb * category
        else:
            raise ValueError('product_pairing must be aligned or crossed')
        if normalization == 'unit':
            return self._normalize_product(
                product, self.config.get('normalization_gradient', 'full'), mask, action_type)
        if normalization != 'raw':
            raise ValueError('product_normalization must be raw or unit')
        return product

    def _scores_from_product(self, fields, history, product):
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        inputs = torch.cat([query, history, query - history, product], dim=-1)
        return self.attention(inputs).squeeze(-1)

    def forward(self, batch):
        fields, features, history, components = self._encode(batch)
        product = self._product(fields, components,
                                self.config['product_pairing'],
                                self.config['product_normalization'], batch['history_mask'],
                                batch['history_type'])
        scores = self._scores_from_product(fields, history, product)
        weights = self._weights(scores, batch['history_mask'])
        summary = (weights.unsqueeze(-1) * history).sum(1)
        return self._logits(features, summary)

    @staticmethod
    def _gradient_dot(left, right):
        # Float64 reduction of the actual float32 gradients; no recomputation
        # of the forward in a different precision or hidden normalization.
        return sum((a * b).sum(dtype=torch.float64)
                   for a, b in zip(left, right))

    def _training_objective_probe(self, batch, train_labels, call_index):
        """One fixed full-batch, six-rule, common-parameter objective probe.

        Returned tensor has29 scalars. Raw-product gradient captures that
        channel; embedding gradients include ALL static/query/history paths
        to the two unique category/brand tables. Ratios use a1e-30 floor;
        inspect the reported full norms to identify degenerate comparisons.
        No .backward(), optimizer interaction, RNG, label caching or .grad
        mutation occurs here. A separate graph leaves the real loss intact.
        """
        modes = (('full', 'full'), ('fixed', 'constant_tangent'),
                 ('history_adjacent', 'history_adjacent_tangent'),
                 ('history_half', 'history_reassigned_tangent'),
                 ('scale_adjacent', 'scale_adjacent_tangent'),
                 ('scale_half', 'scale_half_tangent'))
        params = tuple(dict.fromkeys((self.embeddings[self.category].weight,
                                      self.embeddings[self.brand].weight)))
        with torch.enable_grad():
            fields, features, history, components = self._encode(batch)
            mask, actions = batch['history_mask'], batch['history_type']
            raw = self._product(fields, components, 'aligned', 'raw', mask, actions)
            labels = train_labels.detach().to(dtype=raw.dtype).reshape(-1)
            eligible = mask & (raw.detach().square().sum(-1) >= 1e-6)
            comparisons = []
            forward_error = raw.new_zeros((), dtype=torch.float64)
            loss_error = raw.new_zeros((), dtype=torch.float64)
            for index, (_, mode) in enumerate(modes):
                product = self._normalize_product(raw, mode, mask, actions)
                scores = self._scores_from_product(fields, history, product)
                weights = self._weights(scores, mask)
                logits = self._logits(features, (weights.unsqueeze(-1) * history).sum(1))
                objective = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.to(dtype=logits.dtype), reduction='mean')
                grads = torch.autograd.grad(objective, (raw,) + params,
                                             retain_graph=index < len(modes) - 1,
                                             create_graph=False)
                grads = tuple(g.detach() for g in grads)
                product_group, embedding_group = (grads[0],), grads[1:]
                if index == 0:
                    reference_logits = logits.detach()
                    reference_loss = objective.detach().to(torch.float64)
                    reference_product = product_group
                    reference_embedding = embedding_group
                    product_norm = self._gradient_dot(product_group, product_group).clamp_min(0).sqrt()
                    embedding_norm = self._gradient_dot(embedding_group, embedding_group).clamp_min(0).sqrt()
                else:
                    forward_error = torch.maximum(forward_error,
                        (logits.detach() - reference_logits).abs().max().to(torch.float64))
                    loss_error = torch.maximum(loss_error,
                        (objective.detach().to(torch.float64) - reference_loss).abs())
                    product_delta = tuple(a-b for a, b in zip(product_group, reference_product))
                    embedding_delta = tuple(a-b for a, b in zip(embedding_group, reference_embedding))
                    candidate_norm = self._gradient_dot(embedding_group, embedding_group).clamp_min(0).sqrt()
                    comparisons.extend((
                        self._gradient_dot(product_delta, product_delta).clamp_min(0).sqrt() / product_norm.clamp_min(1e-30),
                        candidate_norm / embedding_norm.clamp_min(1e-30),
                        self._gradient_dot(embedding_delta, embedding_delta).clamp_min(0).sqrt() / embedding_norm.clamp_min(1e-30),
                        self._gradient_dot(embedding_group, reference_embedding) /
                            (candidate_norm * embedding_norm).clamp_min(1e-30)))
            def scalar(value):
                return raw.new_tensor(value, dtype=torch.float64)
            values = torch.stack((scalar(call_index), scalar(labels.numel()),
                labels.sum(dtype=torch.float64), eligible.sum().to(torch.float64),
                reference_loss, product_norm, embedding_norm,
                forward_error, loss_error, *comparisons)).detach()
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite training-objective audit; retain failed receipt')
            return values

    def training_loss(self, batch, logits, train_labels):
        """Declared common mean BCE; probe changes no returned loss term.

        Both variants expose exactly this hook. The original host loss body
        is unavailable; exact ancestor metric bridges are required rather
        than assuming that this hook reproduces the historical trajectory.
        Calls made outside training neither probe nor advance the schedule.
        """
        labels = train_labels.to(dtype=logits.dtype).reshape_as(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='mean')
        if self.training:
            call_index = self._training_loss_calls
            self._training_loss_calls += 1
            if self.config.get('audit_enabled', True) and call_index in (0, 512):
                slot = 0 if call_index == 0 else 1
                values = self._training_objective_probe(batch, train_labels, call_index)
                with torch.no_grad():
                    self._audit_table[slot].copy_(values)
        return loss

    def diagnostics(self, batch):
        """Exactly64 finite scalars; labels are never available here.

        Host reads retained buffers at its selected checkpoint. Missing slots
        have call_index=-1 and placeholder zeros; the completed count reports
        coverage. If a checkpoint predates call512 it cannot supply that
        snapshot. No current/previous validation labels or training rows are
        stored. Validation inputs are used only for four forward/mask checks.
        """
        names = ['call_index', 'samples', 'positives', 'eligible_events',
                 'full_bce', 'full_product_gradient_norm',
                 'full_embedding_gradient_norm', 'common_forward_abs_max',
                 'common_loss_abs_max']
        for rule in ('fixed', 'history_adjacent', 'history_half',
                     'scale_adjacent', 'scale_half'):
            names.extend((rule + '_product_relative_difference',
                          rule + '_embedding_norm_ratio',
                          rule + '_embedding_relative_difference',
                          rule + '_embedding_cosine'))
        result = {}
        for slot in range(2):
            for column, name in enumerate(names):
                result['train_snapshot_' + str(slot) + '_' + name] = float(
                    self._audit_table[slot, column].detach().item())
        result['audit_enabled'] = float(bool(self.config.get('audit_enabled', True)))
        result['completed_snapshots'] = float((self._audit_table[:, 0] >= 0).sum().item())
        fields, features, history, components = self._encode(batch)
        mask, actions = batch['history_mask'], batch['history_type']
        raw = self._product(fields, components, 'aligned', 'raw', mask, actions)
        active = self._normalize_product(raw, self.config['normalization_gradient'], mask, actions)
        full = self._normalize_product(raw, 'full', mask, actions)
        weights = self._weights(self._scores_from_product(fields, history, active), mask)
        full_weights = self._weights(self._scores_from_product(fields, history, full), mask)
        summary = (weights.unsqueeze(-1) * history).sum(1)
        full_summary = (full_weights.unsqueeze(-1) * history).sum(1)
        logits, full_logits = self._logits(features, summary), self._logits(features, full_summary)
        result['validation_full_active_logit_abs_max'] = float((logits-full_logits).abs().max().item())
        result['validation_weight_sum_error_max'] = float((weights.sum(-1)-mask.any(-1).float()).abs().max().item())
        result['validation_masked_mass_max'] = float((weights*(~mask).float()).sum(-1).max().item())
        result['validation_empty_summary_norm_max'] = float((summary.square().sum(-1).sqrt()*(~mask.any(-1))).max().item())
        return result
