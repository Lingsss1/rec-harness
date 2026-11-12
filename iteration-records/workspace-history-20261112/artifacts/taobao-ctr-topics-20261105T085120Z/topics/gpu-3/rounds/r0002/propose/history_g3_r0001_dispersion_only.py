"""Factorial decomposition of expected shuffled-label auxiliary BCE.

For nonempty histories, z is the auxiliary logit, q the current training
batch prevalence, and m=mean(z). Define M=softplus(m)-q*m and
D=mean(softplus(z))-softplus(m). Then M+D equals BCE(z,q), which
is exactly the permutation expectation of BCE(z, shuffled training labels).
The four arms toggle M and D at the inherited weight, without label matching
or permutation sampling. This algebraic identity is not an empirical claim.
Inference and backbone construction are inherited unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'aux_mode': 'expected_permuted', 'aux_weight': 0.2, 'aux_hidden': 32,
    'mean_weight': 0.0, 'dispersion_weight': 1.0,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        config = {**CONFIG, **(config or {})}
        self.config = config
        if config['kind'] != 'din':
            raise ValueError('This study fixes the DIN reference backbone.')
        if config['aux_mode'] != 'expected_permuted':
            raise ValueError('This module implements the expected-permutation study.')
        if config['mean_weight'] not in (0.0, 1.0) or config['dispersion_weight'] not in (0.0, 1.0):
            raise ValueError('Factorial coefficients must be zero or one.')
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
        self._loss_inputs = None

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
        self._loss_inputs = None  # Release graph references after this step.
        # Auxiliary operations are training-only. No extra encoder pass.
        z = self.auxiliary(pooled).squeeze(-1)
        mask = valid.to(dtype=z.dtype)
        n = mask.sum().clamp_min(1)
        present = valid.any().to(dtype=z.dtype)
        q = ((labels * mask).sum() / n).detach()
        m = (z * mask).sum() / n
        # Empty-history batches produce exactly zero auxiliary gradient.
        mean_calibration = (F.softplus(m) - q * m) * present
        dispersion = ((F.softplus(z) * mask).sum() / n - F.softplus(m)) * present
        auxiliary = (self.config['mean_weight'] * mean_calibration +
                     self.config['dispersion_weight'] * dispersion)
        # The 00 arm retains all parameters and loss computations but supplies
        # zero auxiliary gradients. Adam/lr, data and primary BCE are unchanged.
        return primary + self.config['aux_weight'] * auxiliary

    def diagnostics(self, batch):
        # Descriptive, label-free measurements on the host's first 512 rows.
        # These are not validation probes and do not establish a mechanism.
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
        return {
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
