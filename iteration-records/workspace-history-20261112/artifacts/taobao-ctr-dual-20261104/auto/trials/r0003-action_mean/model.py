"""Action-specific history summaries with a grouping-by-weighting factorial.

Parent: r0002-support_pair, source SHA256
f11d409d2dfe3653c1fc1c788784f512f2dbc7ab13dbbace4c01f387cf0bada7.
Preserve its global learned-attention vector and all 24 fixed statistics.
Append four D-dimensional summaries with matched parameter shapes:
  action/attention: shared scores, softmax separately inside each action
  global/attention: four copies of the original global attention vector
  action/mean:      arithmetic mean separately inside each action
  global/mean:      four copies of the global arithmetic mean
The original global learned vector remains present in every mode, so mean
controls still train the same relevance scorer. All modes compute every view.
Empty action groups yield zero; actions are pv/cart/fav/buy in that order.
No [B,4,L,D] value tensor, extra events, external I/O, or label access is used.
Allocated parameter shapes and initialization match; backward cost and effective
input rank need not match. This is a local adaptation, not verified novel.
Nearest work inspected: DIN v4 sec. 4.3 (original unnormalized weighting),
EDIN/DINMP v1 sec. 3.2 (partition aggregation), ATRank v2 overview, and
DMBIN AMCIS 2023 primary abstract (full text returned 403). See proposal.json.
"""
import math
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'statistic_target': 'match', 'statistic_scale': 'count',
    'temporal_statistic': 'aligned', 'half_life_seconds': 86400.0,
    'pair_statistic': 'support',
    'summary_grouping': 'action', 'summary_weighting': 'mean',
}


class Model(nn.Module):
    def __init__(self, meta, config=None):
        super().__init__()
        self.config = {**CONFIG, **(config or {})}
        self.category = meta['category_field_index']
        self.brand = meta['brand_field_index']
        self.summary_grouping = self.config['summary_grouping']
        self.summary_weighting = self.config['summary_weighting']
        if self.summary_grouping not in ('action', 'global'):
            raise ValueError('Unknown summary_grouping')
        if self.summary_weighting not in ('attention', 'mean'):
            raise ValueError('Unknown summary_weighting')
        if self.config['pair_statistic'] != 'support':
            raise ValueError('The inherited support branch is frozen')
        self.temporal_statistic = self.config['temporal_statistic']
        self.pair_statistic = self.config['pair_statistic']
        if self.pair_statistic not in ('joint', 'factorized', 'duplicate', 'support'):
            raise ValueError('Unknown pair_statistic')
        if self.config['kind'] != 'din' or self.temporal_statistic != 'aligned':
            raise ValueError('The inherited aligned DIN branch is frozen')
        if self.temporal_statistic not in ('aligned', 'factorized', 'activity', 'duplicate'):
            raise ValueError('Unknown temporal_statistic')
        if (self.config['statistic_target'] != 'match' or
                self.config['statistic_scale'] != 'count'):
            raise ValueError('The inherited count branch is frozen')
        self.half_life = float(self.config['half_life_seconds'])
        if not math.isfinite(self.half_life) or self.half_life <= 0:
            raise ValueError('half_life_seconds must be finite and positive')
        dim = self.config['embedding_dim']
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dim, padding_idx=0) for size in meta['buckets']
        ])
        self.action = nn.Embedding(5, dim, padding_idx=0)
        self.recency = nn.Embedding(33, dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        # Preserve all parent inputs and append four D-dimensional summaries.
        # Every mode constructs the same modules in the same RNG order.
        width = len(meta['fields']) * dim + dim + 24 + 4 * dim
        layers = []
        for hidden in self.config['hidden']:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.output = nn.Sequential(*layers)

    def _statistics(self, batch):
        mask = batch['history_mask']
        types = batch['history_type']
        action_ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == action_ids) & mask.unsqueeze(-1)).float()
        counts = actions.sum(dim=1)
        # The host supplies nonnegative causal ages. Mask padding before exp;
        # no event selection, history extension, learned decay or label access.
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        weighted_actions = actions * decay.unsqueeze(-1)
        matches, supports, decayed_matches, decayed_supports = [], [], [], []
        for column, index in (('history_category', self.category),
                              ('history_brand', self.brand)):
            history_ids = batch[column]
            target_ids = batch['static'][:, index:index + 1]
            valid = mask & (history_ids > 0)
            equal = valid & (target_ids > 0) & (history_ids == target_ids)
            valid_float = valid.unsqueeze(-1).float()
            equal_float = equal.unsqueeze(-1).float()
            supports.append((actions * valid_float).sum(dim=1))
            matches.append((actions * equal_float).sum(dim=1))
            decayed_supports.append((weighted_actions * valid_float).sum(dim=1))
            decayed_matches.append((weighted_actions * equal_float).sum(dim=1))
        # Each statistic is [B,2,4]: category/brand then pv/cart/fav/buy.
        return (counts, torch.stack(matches, dim=1), torch.stack(supports, dim=1),
                torch.stack(decayed_matches, dim=1),
                torch.stack(decayed_supports, dim=1))

    def _evidence_from_statistics(self, statistics):
        _, matches, supports, decayed_matches, decayed_supports = statistics
        factorized = matches * decayed_supports / supports.clamp_min(1.0)
        if self.temporal_statistic == 'aligned':
            extra = decayed_matches
        elif self.temporal_statistic == 'factorized':
            extra = factorized
        elif self.temporal_statistic == 'activity':
            extra = decayed_supports
        else:
            extra = matches
        # Same fixed monotone count transform on both branches and all modes.
        return torch.cat([
            torch.log1p(matches).flatten(start_dim=1),
            torch.log1p(extra).flatten(start_dim=1),
        ], dim=-1) / math.log(129.0)

    def _pair_statistics(self, batch):
        """Counts and decay masses within the SAME pair-valid event population.

        Pair-valid depends only on supplied event-field availability. Target
        equality is separately masked for each field, so ID zero never matches.
        The decayed independence construction uses decay-weighted marginals;
        it is not a uniform age/field permutation or a new event sequence.
        """
        mask = batch['history_mask']
        types = batch['history_type']
        action_ids = torch.arange(1, 5, device=types.device)
        actions = ((types.unsqueeze(-1) == action_ids) & mask.unsqueeze(-1)).float()
        ages = batch['history_age_seconds'].float().masked_fill(~mask, 0.0)
        decay = torch.exp(-math.log(2.0) * ages / self.half_life)
        history_category = batch['history_category']
        history_brand = batch['history_brand']
        target_category = batch['static'][:, self.category:self.category + 1]
        target_brand = batch['static'][:, self.brand:self.brand + 1]
        valid = mask & (history_category > 0) & (history_brand > 0)
        category_equal = valid & (target_category > 0) & (history_category == target_category)
        brand_equal = valid & (target_brand > 0) & (history_brand == target_brand)
        joint_equal = category_equal & brand_equal
        valid_actions = actions * valid.unsqueeze(-1).float()
        weighted_actions = valid_actions * decay.unsqueeze(-1)
        category_match = category_equal.unsqueeze(-1).float()
        brand_match = brand_equal.unsqueeze(-1).float()
        joint_match = joint_equal.unsqueeze(-1).float()
        support = valid_actions.sum(dim=1)
        weighted_support = weighted_actions.sum(dim=1)
        category_count = (valid_actions * category_match).sum(dim=1)
        brand_count = (valid_actions * brand_match).sum(dim=1)
        category_decay = (weighted_actions * category_match).sum(dim=1)
        brand_decay = (weighted_actions * brand_match).sum(dim=1)
        joint_count = (valid_actions * joint_match).sum(dim=1)
        joint_decay = (weighted_actions * joint_match).sum(dim=1)
        independent_count = category_count * (brand_count / support.clamp_min(1.0))
        # Divide before multiply to avoid underflow in Wc*Wb. When U=0,
        # both marginals are zero. A denominator of one gives exact zero
        # without evaluating a division by zero or altering positive masses.
        safe_mass = torch.where(weighted_support > 0, weighted_support,
                                torch.ones_like(weighted_support))
        independent_decay = category_decay * (brand_decay / safe_mass)
        return {
            'support': support, 'weighted_support': weighted_support,
            'category_count': category_count, 'brand_count': brand_count,
            'category_decay': category_decay, 'brand_decay': brand_decay,
            'joint_count': joint_count, 'joint_decay': joint_decay,
            'independent_count': independent_count,
            'independent_decay': independent_decay,
        }

    def _pair_evidence(self, statistics, pairs):
        # Every mode receives all already computed quantities. This selector
        # changes only which eight deterministic scalars enter the same MLP.
        if self.pair_statistic == 'joint':
            count, mass = pairs['joint_count'], pairs['joint_decay']
        elif self.pair_statistic == 'factorized':
            count, mass = pairs['independent_count'], pairs['independent_decay']
        elif self.pair_statistic == 'support':
            count, mass = pairs['support'], pairs['weighted_support']
        else:
            # Original parent category marginals, including events with missing
            # brand. These duplicate existing features and contain no new info.
            count, mass = statistics[1][:, 0, :], statistics[3][:, 0, :]
        return torch.cat([torch.log1p(count), torch.log1p(mass)], dim=-1) / math.log(129.0)

    def _history_views(self, batch, fields):
        """One inherited scorer, with all four summary alternatives computed."""
        ages = (torch.log2(batch['history_age_seconds'] + 1).long() + 1).clamp(0, 32)
        history = (self.embeddings[self.category](batch['history_category']) +
                   self.embeddings[self.brand](batch['history_brand']) +
                   self.action(batch['history_type']) + self.recency(ages))
        mask = batch['history_mask']
        query = (fields[self.category] + fields[self.brand])[:, None, :].expand_as(history)
        scores = self.attention(torch.cat([
            query, history, query - history, query * history
        ], dim=-1)).squeeze(-1)
        # This base pooling is unchanged from the incumbent.
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        global_attention = (weights.unsqueeze(-1) * history).sum(1)
        global_mean_weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        global_mean = (global_mean_weights.unsqueeze(-1) * history).sum(1)

        # [B,4,L] masks/weights; bmm avoids a [B,4,L,D] value expansion.
        action_ids = torch.arange(1, 5, device=mask.device)
        action_mask = ((batch['history_type'][:, None, :] == action_ids[None, :, None]) &
                       mask[:, None, :])
        action_float = action_mask.float()
        action_count = action_float.sum(-1)
        action_scores = scores[:, None, :].expand(-1, 4, -1).masked_fill(
            ~action_mask, torch.finfo(scores.dtype).min)
        action_weights = action_scores.softmax(-1) * action_float
        action_weights = action_weights / action_weights.sum(-1, keepdim=True).clamp_min(1e-9)
        action_mean_weights = action_float / action_count.unsqueeze(-1).clamp_min(1)
        action_attention = torch.bmm(action_weights, history)
        action_mean = torch.bmm(action_mean_weights, history)
        return {
            'global_attention': global_attention, 'global_mean': global_mean,
            'action_attention': action_attention, 'action_mean': action_mean,
            'global_weights': weights, 'global_mean_weights': global_mean_weights,
            'action_weights': action_weights, 'action_mask': action_mask,
            'action_count': action_count,
        }

    def _selected_summary(self, views):
        key = self.summary_grouping + '_' + self.summary_weighting
        selected = views[key]
        if self.summary_grouping == 'global':
            selected = selected[:, None, :].expand(-1, 4, -1)
        # Consistent action-major coordinate order for every mode.
        return selected.reshape(selected.shape[0], -1)

    def forward(self, batch):
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._history_views(batch, fields)
        statistics = self._statistics(batch)
        pairs = self._pair_statistics(batch)
        evidence = self._evidence_from_statistics(statistics)
        pair_evidence = self._pair_evidence(statistics, pairs)
        features = torch.cat(fields + [views['global_attention'], evidence,
                                      pair_evidence, self._selected_summary(views)], dim=-1)
        return self.output(features).squeeze(-1)

    def diagnostics(self, batch):
        """50 finite descriptive scalars, first 512 host-selected validation rows.

        No labels or aggregate validation access. Conditional quantities divide
        by the number of present groups, with zero returned if none are present.
        Attention mass is a description of weights, not causal importance.
        """
        fields = [embedding(batch['static'][:, i])
                  for i, embedding in enumerate(self.embeddings)]
        views = self._history_views(batch, fields)
        counts = views['action_count']
        present = counts > 0
        lengths = batch['history_mask'].sum(-1).float()
        global_mass = (views['global_weights'][:, None, :] *
                       views['action_mask'].float()).sum(-1)
        uniform_mass = (views['global_mean_weights'][:, None, :] *
                        views['action_mask'].float()).sum(-1)
        squared_weights = views['action_weights'].square().sum(-1)
        effective_count = torch.where(present, 1.0 / squared_weights.clamp_min(1e-9),
                                      torch.zeros_like(squared_weights))
        normalization_error = (views['action_weights'].sum(-1) - present.float()).abs()
        action_norm = views['action_attention'].norm(dim=-1)
        content_distance = (views['action_attention'] - views['action_mean']).norm(dim=-1)
        statistics = self._statistics(batch)
        pairs = self._pair_statistics(batch)
        evidence = torch.cat([self._evidence_from_statistics(statistics),
                              self._pair_evidence(statistics, pairs)], dim=-1)
        result = {
            'supplied_history_length_mean': lengths.mean(),
            'supplied_empty_history_fraction': (lengths == 0).float().mean(),
            'supplied_unknown_action_count_mean': (lengths - counts.sum(-1)).mean(),
            'present_action_groups_mean': present.float().sum(-1).mean(),
            'multiple_action_groups_fraction': (present.sum(-1) > 1).float().mean(),
            'global_attention_norm_mean': views['global_attention'].norm(dim=-1).mean(),
            'global_mean_norm_mean': views['global_mean'].norm(dim=-1).mean(),
            'selected_summary_norm_mean': self._selected_summary(views).norm(dim=-1).mean(),
            'evidence_feature_mean': evidence.mean(),
            'evidence_feature_max': evidence.max(),
        }
        for a, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            indicator = present[:, a].float()
            denominator = indicator.sum().clamp_min(1.0)
            result[name + '_count_mean'] = counts[:, a].mean()
            result[name + '_present_fraction'] = indicator.mean()
            result[name + '_global_attention_mass_mean'] = global_mass[:, a].mean()
            result[name + '_global_attention_mass_when_present'] = (
                global_mass[:, a] * indicator).sum() / denominator
            result[name + '_uniform_mass_when_present'] = (
                uniform_mass[:, a] * indicator).sum() / denominator
            result[name + '_local_attention_effective_events_when_present'] = (
                effective_count[:, a] * indicator).sum() / denominator
            result[name + '_local_weight_sum_error_max'] = normalization_error[:, a].max()
            result[name + '_absent_action_pool_norm_max'] = (
                action_norm[:, a] * (1.0 - indicator)).max()
            result[name + '_attention_mean_distance_when_present'] = (
                content_distance[:, a] * indicator).sum() / denominator
            result[name + '_local_pool_norm_when_present'] = (
                action_norm[:, a] * indicator).sum() / denominator
        return result
