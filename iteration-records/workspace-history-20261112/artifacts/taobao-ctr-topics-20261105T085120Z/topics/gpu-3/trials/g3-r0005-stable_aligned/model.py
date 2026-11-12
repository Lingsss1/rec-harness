"""Centered auxiliary supervision with a prospective arithmetic integrity audit.

Observation: g3-r0004-expected_control reports a negative training Jensen gap
(-0.00214393), although the declared real-arithmetic gap is nonnegative. Its
saved raw head mean is -132.009. No saved training batches are available, so
rounding/order is a hypothesis, not an established explanation of that receipt.

The mathematical objective is inherited: for nonempty histories u=z-mean(z),
q=mean(y), c=logit(clamp(q)), C=mean(softplus(c+u))-softplus(c),
A=mean((y-q)*u). Aligned loss C-A is BCE(c+u,y)-BCE(c,y) in real
arithmetic; the null C is its exact expectation over training-label permutations.
Here c is added AFTER subtracting the mean, with fp64 scalar arithmetic in
three arms and fp32 scalar arithmetic in the operation-order control. No gap
clamping hides violations. Diagnostics replay legacy/order/precision formulas
on the same training logits without influencing gradients or predictions.

The supervised and expected-label objectives differ in their training signal;
the arithmetic repair itself is not claimed as a new mathematical mechanism.
All claims of effectiveness and general novelty remain unverified. The parent
DIN-style backbone, head initialization, primary loss, optimizer and prediction
graph are inherited. All-empty batches have zero auxiliary encoder gradient.
"""
import torch
from torch import nn
from torch.nn import functional as F

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'aux_mode': 'centered_integrity', 'aux_weight': 0.2, 'aux_hidden': 32,
    'mean_weight': 1.0, 'dispersion_weight': 1.0, 'head_shape': 'learned',
    'encoder_aux_grad': True, 'aux_centering': 'prevalence',
    'prevalence_clip': 0.0001, 'label_alignment': True,
    'arithmetic': 'center_first_fp64',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        config = {**CONFIG, **(config or {})}
        self.config = config
        if config['kind'] != 'din':
            raise ValueError('This study fixes the DIN reference backbone.')
        if config['aux_mode'] != 'centered_integrity':
            raise ValueError('This module implements the centered-loss integrity study.')
        if config['arithmetic'] not in ('center_first_fp64', 'center_first_fp32'):
            raise ValueError('Unknown centered arithmetic condition.')
        if config['head_shape'] != 'learned':
            raise ValueError('This follow-up fixes the learned auxiliary head.')
        if not isinstance(config['label_alignment'], bool):
            raise ValueError('label_alignment must be boolean.')
        if config['mean_weight'] not in (0.0, 1.0) or config['dispersion_weight'] not in (0.0, 1.0):
            raise ValueError('Factorial coefficients must be zero or one.')
        if config['head_shape'] not in ('fixed', 'learned'):
            raise ValueError('Unknown auxiliary head shape condition.')
        if not isinstance(config['encoder_aux_grad'], bool):
            raise ValueError('encoder_aux_grad must be boolean.')
        if config['aux_centering'] != 'prevalence':
            raise ValueError('Unknown centering condition.')
        if not 0.0 < config['prevalence_clip'] < 0.5:
            raise ValueError('prevalence_clip must be between zero and one half.')
        if config['aux_centering'] == 'prevalence' and (
                config['mean_weight'] != 1.0 or config['dispersion_weight'] != 1.0):
            raise ValueError('Centered arms fix the native coefficient flags at one.')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        dim = config['embedding_dim']
        # Keep the reference construction order and initial backbone values.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)
        # Do not advance the CPU RNG used by the host when adding the head.
        with torch.random.fork_rng(devices=[]):
            self.auxiliary = nn.Sequential(
                nn.Linear(dim, config['aux_hidden']), nn.ReLU(),
                nn.Linear(config['aux_hidden'], 1)
            )
        # Same allocated head parameters in every cell. Fixed native arms train
        # only the final bias. Centered arms bypass that redundant bias entirely.
        if config['head_shape'] == 'fixed':
            for parameter in self.auxiliary.parameters():
                parameter.requires_grad_(False)
            self.auxiliary[2].bias.requires_grad_(True)
        # Diagnostic snapshots only; copies consume no RNG and never enter loss.
        self.register_buffer('initial_w1', self.auxiliary[0].weight.detach().clone())
        self.register_buffer('initial_b1', self.auxiliary[0].bias.detach().clone())
        self.register_buffer('initial_w2', self.auxiliary[2].weight.detach().clone())
        self.register_buffer('initial_b2', self.auxiliary[2].bias.detach().clone())
        self._loss_inputs = None
        # Stored training summaries; never use validation labels or enter logits.
        self.register_buffer('train_summary', torch.zeros(19, dtype=torch.float64))

    def _encode(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        static = torch.cat(fields, dim=-1)
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (
            self.embeddings[self.category](batch['history_category']) +
            self.embeddings[self.brand](batch['history_brand']) +
            self.action(batch['history_type']) + self.recency(ages)
        )
        mask = batch['history_mask']
        target = fields[self.category] + fields[self.brand]
        query = target[:, None, :].expand_as(history)
        scores = self.attention(torch.cat([
            query, history, query - history, query * history
        ], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        pooled = (weights.unsqueeze(-1) * history).sum(1)
        return static, pooled, target, weights

    def forward(self, batch):
        static, pooled, target, _ = self._encode(batch)
        # No auxiliary head execution, labels, or batch mixing in inference.
        self._loss_inputs = ((pooled, batch['history_mask'].any(-1))
                             if self.training else None)
        return self.output(torch.cat([static, pooled], dim=-1)).squeeze(-1)

    def _centered_terms(self, z, valid, labels, arithmetic):
        # Promote BEFORE the mean, offset and subtraction. Casting the final
        # loss alone cannot recover information lost in float32 centering.
        dtype = torch.float64 if arithmetic == 'center_first_fp64' else torch.float32
        z = z.to(dtype=dtype)
        labels = labels.to(dtype=dtype)
        mask = valid.to(dtype=dtype)
        n = mask.sum().clamp_min(1)
        present = valid.any().to(dtype=dtype)
        q = ((labels * mask).sum() / n).detach()
        qc = q.clamp(self.config['prevalence_clip'],
                     1.0 - self.config['prevalence_clip'])
        c = qc.log() - torch.log1p(-qc)
        m = (z * mask).sum() / n
        u = z - m
        # Deliberately different from legacy left-associative c + z - m.
        centered = c + u
        gap = (((F.softplus(centered) * mask).sum() / n - F.softplus(c)) * present)
        alignment = (((labels - q) * u * mask).sum() / n) * present
        center_error = (((centered - c) * mask).sum() / n).abs() * present
        return gap, alignment, q, m, u, center_error

    def training_loss(self, batch, logits, train_labels):
        labels = train_labels.reshape(-1).to(dtype=logits.dtype, device=logits.device)
        primary = F.binary_cross_entropy_with_logits(logits, labels)
        if self._loss_inputs is None:
            raise RuntimeError('training_loss requires the current training forward.')
        pooled, valid = self._loss_inputs
        self._loss_inputs = None
        auxiliary_input = pooled if self.config['encoder_aux_grad'] else pooled.detach()
        # Preserve the parent's allocated head, construction order and unused
        # final bias. Only the small scalar loss calculation changes precision.
        z = F.linear(self.auxiliary[1](self.auxiliary[0](auxiliary_input)),
                     self.auxiliary[2].weight, None).squeeze(-1)
        gap, alignment, q, m, u, center_error = self._centered_terms(
            z, valid, labels, self.config['arithmetic'])
        auxiliary = gap - (alignment if self.config['label_alignment'] else 0.0)
        with torch.no_grad():
            # Arithmetic replays use CURRENT TRAINING logits and labels only.
            # They do not replay a trained model or estimate counterfactual AUC.
            # Same replays in every arm, including the blocked-gradient control.
            z32 = z.detach().float()
            mask32 = valid.float()
            n32 = mask32.sum().clamp_min(1)
            present = valid.any().to(dtype=torch.float64)
            q32 = ((labels.float() * mask32).sum() / n32).detach()
            qc32 = q32.clamp(self.config['prevalence_clip'],
                             1.0 - self.config['prevalence_clip'])
            c32 = qc32.log() - torch.log1p(-qc32)
            m32 = (z32 * mask32).sum() / n32
            legacy_centered = c32 + z32 - m32
            legacy_gap = (((F.softplus(legacy_centered) * mask32).sum() / n32 -
                           F.softplus(c32)).double() * present)
            legacy_error = ((((legacy_centered - c32) * mask32).sum() / n32)
                            .abs().double() * present)
            reordered_gap = self._centered_terms(
                z.detach(), valid, labels, 'center_first_fp32')[0].double()
            precise_gap = self._centered_terms(
                z.detach(), valid, labels, 'center_first_fp64')[0]
            mask64 = valid.double()
            count = mask64.sum()
            difference = (legacy_gap - precise_gap).abs()
            values = torch.stack([
                count, gap.detach().double() * count,
                alignment.detach().double() * count,
                auxiliary.detach().double() * count,
                (labels.double() * mask64).sum(),
                (gap.detach().double() < -1e-8).double() * count,
                legacy_gap * count, difference * count,
                m.detach().double().abs() * present,
                center_error.detach().double(), legacy_error,
                gap.detach().double() * present, legacy_gap,
                present, u.detach().double().abs().masked_fill(~valid, 0).max(),
                m.detach().double().abs() * count, difference,
                reordered_gap * count, (reordered_gap - precise_gap).abs() * count,
            ])
            # Sum statistics, max statistics, and nonpositive minima are kept
            # separately. Zero-initialized minima expose only negative values.
            for index in (0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 17, 18):
                self.train_summary[index].add_(values[index])
            for index in (8, 9, 10, 14, 16):
                self.train_summary[index].copy_(
                    torch.maximum(self.train_summary[index], values[index]))
            for index in (11, 12):
                self.train_summary[index].copy_(
                    torch.minimum(self.train_summary[index], values[index]))
        # Autograd casts this tiny fp64 branch back to the fp32 head/backbone.
        # The primary BCE, optimizer, lr and inference graph are unchanged.
        return primary + self.config['aux_weight'] * auxiliary

    def diagnostics(self, batch):
        # Shape measurements use only the host's first 512 validation inputs.
        # Separately named train_* values summarize earlier training batches.
        # Neither group measures validation-label discrimination or causality.
        # All inherited aux_* values describe the RAW head, not the centered
        # training logits. No labels or estimated validation prevalence enter.
        # The zero-offset gap is a common label-free shape measurement.
        _, pooled, target, weights = self._encode(batch)
        valid = batch['history_mask'].any(-1).float()
        count = valid.sum().clamp_min(1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        z = self.auxiliary(pooled).squeeze(-1)
        probability = z.sigmoid()
        mean_z = (z * valid).sum() / count
        present = valid.any().to(dtype=z.dtype)
        variance = ((z - mean_z).square() * valid).sum() / count
        gap = ((F.softplus(z) * valid).sum() / count - F.softplus(mean_z)) * present
        mean_probability = (probability * valid).sum() / count
        # Analytic head Jacobians are label-free and work under host no_grad.
        hidden_pre = self.auxiliary[0](pooled)
        active = (hidden_pre > 0).to(dtype=pooled.dtype)
        jacobian = (active * self.auxiliary[2].weight) @ self.auxiliary[0].weight
        initial_pre = F.linear(pooled, self.initial_w1, self.initial_b1)
        initial_active = (initial_pre > 0).to(dtype=pooled.dtype)
        initial_jacobian = (initial_active * self.initial_w2) @ self.initial_w1
        initial_z = F.linear(F.relu(initial_pre), self.initial_w2, self.initial_b2).squeeze(-1)
        initial_mean = (initial_z * valid).sum() / count
        initial_variance = ((initial_z - initial_mean).square() * valid).sum() / count
        center = pooled - (pooled * valid[:, None]).sum(0, keepdim=True) / count
        covariance = center.transpose(0, 1) @ (center * valid[:, None]) / count
        trace = covariance.diagonal().sum()
        participation = trace.square() / covariance.square().sum().clamp_min(1e-12)
        shape_drift = ((self.auxiliary[0].weight - self.initial_w1).square().sum() +
                       (self.auxiliary[0].bias - self.initial_b1).square().sum() +
                       (self.auxiliary[2].weight - self.initial_w2).square().sum()).sqrt()
        centered_zero_gap = (((F.softplus(z - mean_z) * valid).sum() / count -
                              F.softplus(z.new_zeros(()))) * present)
        pooled_mean = (pooled * valid[:, None]).sum(0) / count
        train_count = self.train_summary[0].clamp_min(1)
        return {
            'arithmetic_fp64_enabled': float(self.config['arithmetic'] == 'center_first_fp64'),
            'train_gap_below_minus_1e_minus8_example_fraction': float((self.train_summary[5] / train_count).item()),
            'train_legacy_gap_mean_same_logits': float((self.train_summary[6] / train_count).item()),
            'train_legacy_vs_fp64_gap_abs_mean': float((self.train_summary[7] / train_count).item()),
            'train_raw_logit_mean_abs_max': float(self.train_summary[8].item()),
            'train_center_error_abs_max': float(self.train_summary[9].item()),
            'train_legacy_center_error_abs_max': float(self.train_summary[10].item()),
            'train_negative_gap_min_or_zero': float(self.train_summary[11].item()),
            'train_legacy_negative_gap_min_or_zero': float(self.train_summary[12].item()),
            'train_nonempty_batches': float(self.train_summary[13].item()),
            'train_centered_logit_abs_max': float(self.train_summary[14].item()),
            'train_raw_logit_mean_abs_mean': float((self.train_summary[15] / train_count).item()),
            'train_legacy_vs_fp64_gap_abs_max': float(self.train_summary[16].item()),
            'train_reordered_fp32_gap_mean_same_logits': float((self.train_summary[17] / train_count).item()),
            'train_reordered_vs_fp64_gap_abs_mean': float((self.train_summary[18] / train_count).item()),
            'label_alignment_enabled': float(self.config['label_alignment']),
            'train_nonempty_examples_accumulated': float(self.train_summary[0].item()),
            'train_centered_gap_mean': float((self.train_summary[1] / train_count).item()),
            'train_label_logit_covariance_mean': float((self.train_summary[2] / train_count).item()),
            'train_auxiliary_mean': float((self.train_summary[3] / train_count).item()),
            'train_nonempty_prevalence': float((self.train_summary[4] / train_count).item()),
            'prevalence_centering_enabled': float(self.config['aux_centering'] == 'prevalence'),
            'centered_jensen_gap_at_zero': float(centered_zero_gap.item()),
            'history_mean_vector_norm_nonempty': float(pooled_mean.norm().item()),
            'head_shape_fixed': float(self.config['head_shape'] == 'fixed'),
            'encoder_aux_grad_enabled': float(self.config['encoder_aux_grad']),
            'head_shape_l2_drift': float(shape_drift.item()),
            'head_output_bias_drift': float((self.auxiliary[2].bias - self.initial_b2).abs().sum().item()),
            'head_jacobian_norm_nonempty': float((jacobian.norm(dim=-1) * valid).sum().div(count).item()),
            'initial_head_jacobian_norm_nonempty': float((initial_jacobian.norm(dim=-1) * valid).sum().div(count).item()),
            'head_active_fraction_nonempty': float((active.mean(-1) * valid).sum().div(count).item()),
            'head_zero_jacobian_fraction_nonempty': float(((jacobian.norm(dim=-1) < 1e-8).float() * valid).sum().div(count).item()),
            'initial_head_logit_variance_nonempty': float(initial_variance.item()),
            'history_covariance_trace_nonempty': float(trace.item()),
            'history_covariance_participation_nonempty': float(participation.item()),
            'aux_trainable_parameter_count': float(sum(p.numel() for p in self.auxiliary.parameters() if p.requires_grad)),
            'nonempty_fraction': float(valid.mean().item()),
            'history_norm_nonempty': float((pooled.norm(dim=-1) * valid).sum().div(count).item()),
            'target_norm_nonempty': float((target.norm(dim=-1) * valid).sum().div(count).item()),
            'attention_entropy_nonempty': float((entropy * valid).sum().div(count).item()),
            'aux_probability_mean_nonempty': float((probability * valid).sum().div(count).item()),
            'aux_logit_mean_nonempty': float((mean_z * present).item()),
            'aux_logit_variance_nonempty': float(variance.item()),
            'aux_jensen_gap_nonempty': float(gap.item()),
            'aux_probability_variance_nonempty': float((((probability - mean_probability).square() * valid).sum() / count).item()),
            'aux_parameter_count': float(sum(p.numel() for p in self.auxiliary.parameters())),
        }
