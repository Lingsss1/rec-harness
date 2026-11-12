"""Fixed target selection versus learned event weighting.

Prospective study: matched-pooling-score-eligibility-factorial-v1.
Two factors: all events versus category-or-brand matches (all-event fallback
when no match exists), and learned normalized scores versus uniform weights.
The scalar count/freshness blocks, summary width, embeddings, head, optimizer,
loss, module construction order and registered shapes are fixed in all cells.
Uniform cells skip the scorer and retain its dormant parameters. Every cell
still embeds all supplied events; fewer eligible events alone do not imply
sparse computation. No measured efficiency or novelty claim is made.

This extends local field-adapted normalized attention, not original DIN's
unnormalized intensity mechanism (DIN v4, primary text lines 266-277).
Fixed matching uses only the supplied category/brand buckets, never item IDs.
"""

import math
import torch
from torch import nn


CONFIG = {
    'kind': 'din',
    'embedding_dim': 16,
    'hidden': [128, 64],
    'lr': 0.001,
    'use_counts': True,
    'use_freshness': True,
    'use_target_groups': True,
    'use_time_values': True,
    'use_attention_summary': True,
    'use_match_gate': False,
    'use_learned_scores': False,
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        if self.config['kind'] != 'din' or self.config['embedding_dim'] != 16:
            raise ValueError('This registered factorial fixes DIN and 16-dimensional slots.')
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.use_counts = self.config['use_counts']
        self.use_freshness = self.config['use_freshness']
        if not self.use_counts or not self.use_freshness:
            raise ValueError('Both evidence slots are fixed on in this study.')
        self.use_target_groups = self.config['use_target_groups']
        self.use_time_values = self.config['use_time_values']
        self.use_attention_summary = self.config['use_attention_summary']
        self.use_match_gate = self.config['use_match_gate']
        self.use_learned_scores = self.config['use_learned_scores']
        if not self.use_attention_summary or not self.use_time_values:
            raise ValueError('This study fixes the summary and scalar time slots on.')
        if not self.use_target_groups:
            raise ValueError('Target groups are fixed on in this study.')
        dim = self.config['embedding_dim']

        # Preserve baseline module construction order up to the widened head.
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        layers = []
        width = len(meta['fields']) * dim + dim + 32
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _evidence(self, batch):
        """Return group-major [B,16] count/freshness blocks and [B,4,4] counts.

        Groups: all, category match, brand match, joint match.
        Within each group: pv, cart, fav, buy. Matches use supplied buckets,
        require nonzero target values, and never include masked events.
        Counts describe only the supplied last 128 events, not lifetime counts.
        T is mean exp(-age/one day), not a decayed sum, so C and T separate
        multiplicity from conditional freshness as far as the zero convention
        permits. Empty cells give C=T=0; T can itself reveal cell presence.
        """
        mask = batch['history_mask']
        target_category = batch['static'][:, self.category].unsqueeze(1)
        target_brand = batch['static'][:, self.brand].unsqueeze(1)
        category_match = (
            mask & target_category.ne(0)
            & batch['history_category'].eq(target_category)
        )
        brand_match = (
            mask & target_brand.ne(0)
            & batch['history_brand'].eq(target_brand)
        )
        groups = torch.stack([
            mask, category_match, brand_match, category_match & brand_match
        ], dim=-1).float()
        actions = torch.stack([
            batch['history_type'].eq(action) for action in range(1, 5)
        ], dim=-1).float()
        # Batched reductions avoid materializing a [B,L,4,4] event tensor.
        group_rows = groups.transpose(1, 2)
        counts = torch.bmm(group_rows, actions)
        ages = torch.where(
            mask, batch['history_age_seconds'],
            torch.zeros_like(batch['history_age_seconds'])
        ).clamp_min(0)
        decay = torch.exp(-ages / 86400.0)
        decayed_counts = torch.bmm(group_rows, actions * decay.unsqueeze(-1))
        count_block = torch.log1p(counts) / math.log(129.0)
        freshness_block = decayed_counts / counts.clamp_min(1.0)
        return count_block.flatten(1), freshness_block.flatten(1), counts

    def _select_evidence(self, count_block, freshness, counts):
        """Keep shapes fixed while replacing information in the scalar slots.

        P=1[n>0] exposes no age value and is already derivable from counts.
        Global activity control repeats group zero (all valid events), without
        using category/brand matches in its direct scalar readout. It still
        retains target-aware learned attention in the separate summary slot.
        """
        presence = counts.gt(0).float().flatten(1)
        time_slot = freshness if self.use_time_values else presence
        if not self.use_target_groups:
            count_block = count_block[:, :4].repeat(1, 4)
            time_slot = time_slot[:, :4].repeat(1, 4)
        return count_block, time_slot

    def _eligibility(self, batch):
        """Exact supplied-bucket matches, with all-history fallback per row.

        Unknown target bucket zero cannot match. Joint matches count once.
        This is a selection rule over the supplied causal prefix, not item
        retrieval or a guarantee of semantic identity before hashing.
        """
        mask = batch['history_mask']
        category = batch['static'][:, self.category].unsqueeze(1)
        brand = batch['static'][:, self.brand].unsqueeze(1)
        category_match = mask & category.ne(0) & batch['history_category'].eq(category)
        brand_match = mask & brand.ne(0) & batch['history_brand'].eq(brand)
        matches = category_match | brand_match
        gated = torch.where(matches.any(-1, keepdim=True), matches, mask)
        return (gated if self.use_match_gate else mask), matches

    def _summary(self, batch, fields):
        # Construction order and dimensions equal the inherited attention/time
        # reference. Uniform cells skip scoring, retaining its dormant weights.
        ages = (
            torch.log2(batch['history_age_seconds'] + 1).long() + 1
        ).clamp(0, 32)
        history = (
            self.embeddings[self.category](batch['history_category'])
            + self.embeddings[self.brand](batch['history_brand'])
            + self.action(batch['history_type']) + self.recency(ages)
        )
        eligible, _ = self._eligibility(batch)
        if self.use_learned_scores:
            query = (fields[self.category] + fields[self.brand])[:, None, :]
            query = query.expand_as(history)
            scores = self.attention(torch.cat([
                query, history, query - history, query * history
            ], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~eligible, -1e4)
            weights = scores.softmax(-1) * eligible.float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        else:
            weights = eligible.float() / eligible.sum(-1, keepdim=True).clamp_min(1)
        return (weights.unsqueeze(-1) * history).sum(1)

    def forward(self, batch):
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        summary = self._summary(batch, fields)
        count_block, freshness, counts = self._evidence(batch)
        count_slot, freshness_slot = self._select_evidence(count_block, freshness, counts)
        features = torch.cat(fields + [summary, count_slot, freshness_slot], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """Label-free implementation checks; first 512 rows are descriptive.

        No checkpoint-time interventions or candidate-side scoring are used.
        Dormant parameters are counted explicitly; registered compression is
        not claimed. All cells have the same registered parameter shapes.
        """
        count_block, freshness, counts = self._evidence(batch)
        eligible, matches = self._eligibility(batch)
        valid_count = batch['history_mask'].sum(-1)
        eligible_count = eligible.sum(-1)
        has_match = matches.any(-1)
        nonempty = valid_count.gt(0)
        fields = [
            embedding(batch['static'][:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        summary = self._summary(batch, fields)
        scorer_parameters = sum(p.numel() for p in self.attention.parameters())
        return {
            'match_gate_enabled': float(self.use_match_gate),
            'learned_scores_enabled': float(self.use_learned_scores),
            'registered_parameters': float(sum(p.numel() for p in self.parameters())),
            'dormant_scorer_parameters': 0.0 if self.use_learned_scores else float(scorer_parameters),
            'active_history_readout_width': 48.0,
            'valid_history_count_mean': float(valid_count.float().mean()),
            'eligible_history_count_mean': float(eligible_count.float().mean()),
            'matched_history_count_mean': float(matches.sum(-1).float().mean()),
            'match_present_fraction': float(has_match.float().mean()),
            'nonempty_fallback_fraction': float((nonempty & ~has_match).float().mean()),
            'empty_history_fraction': float((~nonempty).float().mean()),
            'eligible_fraction_nonempty_mean': float(
                ((eligible_count.float() / valid_count.clamp_min(1)) * nonempty.float()).sum()
                / nonempty.sum().clamp_min(1)
            ),
            'eligibility_outside_history_count': float((eligible & ~batch['history_mask']).sum()),
            'nonempty_with_zero_eligible_count': float((nonempty & eligible_count.eq(0)).sum()),
            'summary_abs_mean': float(summary.abs().mean()),
            'summary_zero_fraction': float(summary.eq(0).float().mean()),
            'count_block_mean': float(count_block.mean()),
            'freshness_block_mean': float(freshness.mean()),
            'category_count_mean': float(counts[:, 1].sum(-1).mean()),
            'brand_count_mean': float(counts[:, 2].sum(-1).mean()),
        }
