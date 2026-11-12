"""Prospective feedback-pairing study of target-routed history compression.

Keep the incumbent eight uniform mean slots and counts. Add four feedback
moment channels per slot, using ONLY the supplied historical task masks.
The joint and factorized variants have identical parameters and compute;
their difference is E[z*b] versus E[z]*E[b] within each target/recency slot.
Composition and widened-parent controls retain the same registered modules.
No new measured outcome or originality claim is implied.
"""
import torch
from torch import nn
from torch.nn import functional as F
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {
    'embedding_dim': 16,
    'slot_dim': 32,
    'attention_hidden_units': [64, 32],
    'tower_hidden_units': [128, 64],
    'grouping': 'target_fields',
    'compression': 'early',
    'recent_window': 16,
    'slot_weighting': 'uniform',
    'include_content': True,
    'include_counts': True,
    'feedback_mode': 'zero',
    'include_feedback_fractions': True,
    'feedback_dim': 8,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='FieldRoutedHistory', gpu=-1,
                 embedding_dim=16, slot_dim=32,
                 attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64),
                 grouping=CONFIG['grouping'], compression=CONFIG['compression'],
                 recent_window=16,
                 slot_weighting=CONFIG['slot_weighting'],
                 include_content=CONFIG['include_content'],
                 include_counts=CONFIG['include_counts'],
                 feedback_mode=CONFIG['feedback_mode'],
                 include_feedback_fractions=CONFIG['include_feedback_fractions'],
                 feedback_dim=8, num_tasks=None,
                 task='binary_classification', **kwargs):
        if feedback_mode not in ('joint', 'factorized', 'zero'):
            raise ValueError('Unknown feedback moment mode')
        if not isinstance(include_feedback_fractions, bool):
            raise ValueError('Feedback fraction switch must be bool')
        if feedback_dim <= 0:
            raise ValueError('feedback_dim must be positive')
        if len(feature_map.labels) != 4:
            raise ValueError('This study requires the four contracted tasks')
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if grouping not in ('target_fields', 'position'):
            raise ValueError('Unknown grouping')
        if compression not in ('early', 'late'):
            raise ValueError('Unknown compression timing')
        if slot_weighting not in ('learned', 'uniform'):
            raise ValueError('Unknown slot weighting')
        if not isinstance(include_content, bool) or not isinstance(include_counts, bool):
            raise ValueError('Information-path switches must be bool')
        if slot_weighting == 'uniform' and compression != 'early':
            raise ValueError('Uniform weighting is defined only for early compression')
        if recent_window <= 0:
            raise ValueError('recent_window must be positive')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.feedback_mode = feedback_mode
        self.include_feedback_fractions = include_feedback_fractions
        self.feedback_dim = feedback_dim
        self.feedback_tasks = len(feature_map.labels)
        self.slot_weighting = slot_weighting
        self.include_content = include_content
        self.include_counts = include_counts
        self.grouping = grouping
        self.compression = compression
        self.recent_window = recent_window
        self.num_slots = 8
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [n for n, s in feature_map.features.items()
                            if s.get('source') == 'item' and s['type'] != 'meta']
        self.user_fields = [n for n, s in feature_map.features.items()
                            if s.get('source') not in ('item', 'action')
                            and s['type'] != 'meta']
        self.action_field = next((n for n, s in feature_map.features.items()
                                  if s.get('source') == 'action'
                                  and s['type'] != 'meta'), None)
        if not {'cate_id', 'brand'}.issubset(self.item_fields):
            raise ValueError('Expected the contracted category and brand fields')
        def width(names):
            return sum(feature_map.features[n].get('embedding_dim', embedding_dim)
                       for n in names)
        self.key_width = width(self.item_fields)
        self.value_width = self.key_width + width(
            [self.action_field] if self.action_field else [])
        layers = []
        dim = 4 * self.key_width
        for hidden in attention_hidden_units:
            # Pointwise ReLU avoids timing-dependent Dice batch statistics.
            layers.extend([nn.Linear(dim, hidden), nn.ReLU()])
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.local_activation = nn.Sequential(*layers)
        self.slot_projection = nn.Linear(self.value_width, slot_dim, bias=False)
        self.feedback_projection = nn.Linear(self.key_width, feedback_dim, bias=False)
        input_width = (width(self.user_fields) + self.key_width
                       + self.num_slots * slot_dim + self.num_slots
                       + self.num_slots * self.feedback_tasks * (feedback_dim + 1))
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            layers = []
            dim = input_width
            for hidden in tower_hidden_units:
                layers.extend([nn.Linear(dim, hidden), Dice(hidden)])
                dim = hidden
            layers.append(nn.Linear(dim, 1))
            self.towers.append(nn.Sequential(*layers))
        self.reset_parameters()
        self.model_to_device()

    def _membership(self, item, mask):
        """Return B x 8 x L membership; each valid event belongs to one slot.

        The last sixteen positions are an ordinal recent band, not a time
        window. Target-conditioned slots encode both/category/brand/neither
        matches in each band. Position controls split each band four ways.
        Token zero never produces a category or brand match.
        """
        batch, length = mask.shape
        pos = torch.arange(length, device=mask.device)[None, :].expand(batch, -1)
        old_length = max(length - self.recent_window, 0)
        recent = pos >= old_length
        if self.grouping == 'target_fields':
            category = item['cate_id']
            brand = item['brand']
            cat_match = ((category[:, :-1] == category[:, -1:])
                         & (category[:, :-1] != 0) & (category[:, -1:] != 0))
            brand_match = ((brand[:, :-1] == brand[:, -1:])
                           & (brand[:, :-1] != 0) & (brand[:, -1:] != 0))
            group = 2 * cat_match.long() + brand_match.long() + 4 * recent.long()
        else:
            old_bin = (pos * 4 // max(old_length, 1)).clamp(max=3)
            recent_bin = ((pos - old_length).clamp(min=0) * 4
                          // max(length - old_length, 1)).clamp(max=3)
            group = torch.where(recent, 4 + recent_bin, old_bin)
        members = F.one_hot(group, num_classes=self.num_slots).transpose(1, 2)
        return members.to(torch.float32) * (mask > 0)[:, None, :]

    def _score(self, target, keys):
        batch, length, _ = keys.shape
        query = target[:, None, :].expand(-1, length, -1)
        interactions = torch.cat((query, keys, query - keys, query * keys), dim=-1)
        return self.local_activation(interactions.reshape(batch * length, -1)).reshape(
            batch, length)

    def forward(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Historical feedback masks are required by this study')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(task_masks) != self.feedback_tasks:
            raise ValueError('Historical feedback mask count mismatch')
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[n][:, -1] for n in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[n][:, :-1] for n in self.item_fields], dim=-1)
        values = keys
        if self.action_field:
            values = torch.cat((values, item_emb[self.action_field][:, :-1]), dim=-1)
        # Zero invalid payloads BEFORE nonlinear scoring and reductions.
        valid = (mask > 0)[:, :, None]
        keys = keys * valid.to(keys.dtype)
        values = values * valid.to(values.dtype)
        context = torch.cat([user_emb[n] for n in self.user_fields], dim=-1)
        members = self._membership(item, mask)
        counts = members.sum(dim=-1)
        divisor = counts.clamp(min=1)[:, :, None]
        if keys.shape[1] == 0:
            slot_values = target.new_zeros((target.shape[0], self.num_slots,
                                           self.value_width))
        elif self.compression == 'early':
            mean_keys = torch.bmm(members.to(keys.dtype), keys) / divisor
            mean_values = torch.bmm(members.to(values.dtype), values) / divisor
            # Target dependence remains in deterministic membership and readout.
            # Keep local_activation registered for identical initialization order,
            # but bypass its computation for the ungated candidate.
            weights = (self._score(target, mean_keys)
                       if self.slot_weighting == 'learned'
                       else torch.ones_like(counts, dtype=mean_values.dtype))
            slot_values = mean_values * weights[:, :, None]
        else:
            weights = self._score(target, keys) * (mask > 0).to(keys.dtype)
            weighted_values = values * weights[:, :, None]
            slot_values = torch.bmm(members.to(weighted_values.dtype),
                                    weighted_values) / divisor
        occupied = (counts > 0)[:, :, None]
        slot_values = slot_values * occupied.to(slot_values.dtype)
        slots = self.slot_projection(slot_values)
        # Retain signed local activation outputs; no extra slot nonlinearity.
        slots = slots * occupied.to(slots.dtype)
        # These removals are applied after identical parent computations to
        # isolate information paths without reallocating tower input columns.
        # Controls deliberately retain upstream work; they are not optimized
        # deployment implementations. Removing explicit counts does not remove
        # occupancy or finite-sample information implicit in the mean content.
        if not self.include_content:
            slots = slots * 0.0
        count_features = counts.to(context.dtype) / 100.0
        if not self.include_counts:
            count_features = count_features * 0.0
        # All variants execute this same branch. Masks can overlap: a single
        # previous exposure may have several feedback types. Never use the
        # target action or current labels, and never interpret masks as four
        # mutually exclusive classes. Invalid events are zeroed before sums.
        feedback = torch.stack([(m > 0) & (mask > 0) for m in task_masks], dim=1)
        feedback = feedback.to(members.dtype)  # B x 4 x L
        batch, _, length = members.shape
        grouped_feedback = members[:, :, None, :] * feedback[:, None, :, :]
        # Denominator is the number of ALL valid events in the parent slot,
        # not the number of positive-feedback events. Thus these are joint
        # moments, with explicit feedback fractions; empty slots remain zero.
        fractions = grouped_feedback.sum(dim=-1) / counts.clamp(min=1)[:, :, None]
        projected = self.feedback_projection(keys)
        if length:
            joint = torch.bmm(
                grouped_feedback.reshape(batch, self.num_slots * self.feedback_tasks,
                                         length).to(projected.dtype), projected)
            joint = joint.reshape(batch, self.num_slots, self.feedback_tasks,
                                  self.feedback_dim) / divisor[:, :, None, :]
            mean_projected = torch.bmm(members.to(projected.dtype), projected) / divisor
        else:
            joint = projected.new_zeros((batch, self.num_slots, self.feedback_tasks,
                                        self.feedback_dim))
            mean_projected = projected.new_zeros((batch, self.num_slots, self.feedback_dim))
        factorized = mean_projected[:, :, None, :] * fractions[:, :, :, None]
        if self.feedback_mode == 'joint':
            moments = joint
        elif self.feedback_mode == 'factorized':
            moments = factorized
        else:
            moments = joint * 0.0
        if not self.include_feedback_fractions:
            fractions = fractions * 0.0
        features = torch.cat((context, target, slots.flatten(start_dim=1),
                              count_features, moments.flatten(start_dim=1),
                              fractions.flatten(start_dim=1)), dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
