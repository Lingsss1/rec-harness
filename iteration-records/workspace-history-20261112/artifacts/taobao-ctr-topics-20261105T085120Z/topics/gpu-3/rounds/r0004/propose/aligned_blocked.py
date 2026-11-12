"""Prospective centered label-alignment study; no measured result is claimed.

On nonempty training histories let q=mean(y), c=logit(clamp(q)),
u=z-mean(z). C=mean(softplus(c+u))-softplus(c) is the inherited
centered expected-permutation loss. A=mean((y-q)*u) is empirical
label/logit covariance. C-A equals BCE(c+u,y)-BCE(c,y) exactly.
The null replaces A by zero: it is the exact expectation over all within-
batch label permutations. Both losses project out common-shift gradients.
This does not guarantee calibrated mean predictions for nonconstant u.

All means use nonempty histories; all-empty batches supply zero auxiliary
gradient. Only current training labels enter A or q. Backbone initialization,
primary BCE, data coverage and inference are inherited. The auxiliary head
and training summaries never execute in prediction. Broad novelty unverified.
"""
import torch
from torch import nn
from torch.nn import functional as F

CONFIG = {'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001, 'aux_mode': 'centered_alignment', 'aux_weight': 0.2, 'aux_hidden': 32, 'mean_weight': 1.0, 'dispersion_weight': 1.0, 'head_shape': 'learned', 'encoder_aux_grad': False, 'aux_centering': 'prevalence', 'prevalence_clip': 0.0001, 'label_alignment': True}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        config = {**CONFIG, **(config or {})}
        self.config = config
        if config['kind'] != 'din':
            raise ValueError('This study fixes the DIN reference backbone.')
        if config['aux_mode'] != 'centered_alignment':
            raise ValueError('This module implements centered label alignment.')
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
        self.register_buffer('train_summary', torch.zeros(5, dtype=torch.float64))

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

    def training_loss(self, batch, logits, train_labels):
        labels = train_labels.reshape(-1).to(dtype=logits.dtype, device=logits.device)
        primary = F.binary_cross_entropy_with_logits(logits, labels)
        if self._loss_inputs is None:
            raise RuntimeError('training_loss requires the current training forward.')
        pooled, valid = self._loss_inputs
        self._loss_inputs = None
        auxiliary_input = pooled if self.config['encoder_aux_grad'] else pooled.detach()
        # The final bias cancels in u. Omit it to avoid rounding-driven updates;
        # retain its allocation and initial value to match the previous study.
        z = F.linear(self.auxiliary[1](self.auxiliary[0](auxiliary_input)),
                     self.auxiliary[2].weight, None).squeeze(-1)
        mask = valid.to(dtype=z.dtype)
        n = mask.sum().clamp_min(1)
        present = valid.any().to(dtype=z.dtype)
        q = ((labels * mask).sum() / n).detach()
        qc = q.clamp(self.config['prevalence_clip'],
                     1.0 - self.config['prevalence_clip'])
        c = qc.log() - torch.log1p(-qc)
        m = (z * mask).sum() / n
        u = z - m  # Differentiable centering; no common-shift encoder pressure.
        # Preserve the parent's floating-point operation order for the null.
        centered = c + z - m
        gap = (((F.softplus(centered) * mask).sum() / n - F.softplus(c)) * present)
        alignment = (((labels - q) * u * mask).sum() / n) * present
        # No sampled permutations and no additional forward or backward pass.
        auxiliary = gap - (alignment if self.config['label_alignment'] else 0.0)
        with torch.no_grad():
            # Nonempty-example-weighted TRAINING quantities, up to the saved
            # checkpoint. They measure optimization, not validation prediction.
            count = mask.sum()
            self.train_summary.add_(torch.stack([
                count, gap.detach() * count, alignment.detach() * count,
                auxiliary.detach() * count, (labels * mask).sum(),
            ]).to(dtype=self.train_summary.dtype))
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
