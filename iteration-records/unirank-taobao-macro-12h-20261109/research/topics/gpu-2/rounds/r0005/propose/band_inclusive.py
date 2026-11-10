"""Prospective factorial study of the background in compressed feedback gates.

Two factors: all-history versus same-ordinal-band background, and including
versus excluding the slot's own events. Only the product gate changes;
registered parameters, initialization order and raw readout inputs are fixed.
No-background slots fall back to their own rate. Bands are positions, not
elapsed time. All feedback is from the host-masked released prefix only.
Host forward/backward, masking, BF16 and cost checks remain required.
No local quality gain, independent prior or originality is claimed.
"""
import math
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
    'feedback_mode': 'factorized',
    'include_feedback_fractions': True,
    'feedback_dim': 8,
    'rate_interpolation': 'count',
    'shrinkage_strength': 4.0,
    'background_scope': 'band',
    'exclude_slot': False,
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
                 feedback_dim=8,
                 rate_interpolation=CONFIG['rate_interpolation'],
                 shrinkage_strength=CONFIG['shrinkage_strength'],
                 background_scope=CONFIG['background_scope'],
                 exclude_slot=CONFIG['exclude_slot'], num_tasks=None,
                 task='binary_classification', **kwargs):
        if background_scope not in ('global', 'band'):
            raise ValueError('Unknown background scope')
        if not isinstance(exclude_slot, bool):
            raise ValueError('exclude_slot must be bool')
        if rate_interpolation != 'count':
            raise ValueError('This factorial study holds count interpolation fixed')
        if feedback_mode not in ('factorized', 'content', 'global'):
            raise ValueError('Unknown feedback product replacement')
        if not isinstance(include_feedback_fractions, bool):
            raise ValueError('Feedback fraction switch must be bool')
        if feedback_dim <= 0:
            raise ValueError('feedback_dim must be positive')
        if rate_interpolation not in ('count', 'fixed', 'reverse'):
            raise ValueError('Unknown rate interpolation control')
        if not math.isfinite(shrinkage_strength) or shrinkage_strength <= 0:
            raise ValueError('shrinkage_strength must be finite and positive')
        if feedback_mode != 'factorized':
            raise ValueError('This study changes only the factorized product gate')
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
        if grouping != 'target_fields' or compression != 'early' or slot_weighting != 'uniform':
            raise ValueError('This matched study requires uniform early target-field slots')
        if recent_window <= 0:
            raise ValueError('recent_window must be positive')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.background_scope = background_scope
        self.exclude_slot = exclude_slot
        self.feedback_mode = feedback_mode
        self.rate_interpolation = rate_interpolation
        self.shrinkage_strength = float(shrinkage_strength)
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

    def _interpolate_rates(self, fractions, counts, feedback, mask, slot_feedback):
        """Blend local rates with an inclusive or complementary background.

        The eight slots are old[0:4], recent[4:8]. Both global and band
        sufficient statistics are computed in every arm; no new learned
        modules or event projections are introduced. Exclusion removes the
        complete disjoint slot, rather than a single event. This prevents
        self-inclusion but does not make correlated user events independent.
        Empty backgrounds fall back to the local rate, never the other band.
        """
        valid = mask > 0
        total = valid.sum(dim=-1).to(counts.dtype)
        global_successes = feedback.sum(dim=-1)
        length = mask.shape[-1]
        boundary = max(length - self.recent_window, 0)
        band_counts = torch.stack((valid[:, :boundary].sum(dim=-1),
                                   valid[:, boundary:].sum(dim=-1)), dim=1)
        band_counts = band_counts.to(counts.dtype).repeat_interleave(4, dim=1)
        band_successes = torch.stack((feedback[:, :, :boundary].sum(dim=-1),
                                      feedback[:, :, boundary:].sum(dim=-1)), dim=1)
        band_successes = band_successes.repeat_interleave(4, dim=1)
        if self.background_scope == 'global':
            background_count = total[:, None].expand(-1, self.num_slots)
            background_success = global_successes[:, None, :].expand(
                -1, self.num_slots, -1)
        else:
            background_count = band_counts
            background_success = band_successes
        if self.exclude_slot:
            # slot_feedback is the raw masked integer success count, before
            # division. Never reconstruct it from rounded fractions.
            background_count = (background_count - counts).clamp(min=0)
            background_success = (background_success - slot_feedback).clamp(min=0)
        background_rate = background_success / background_count.clamp(min=1)[:, :, None]
        background_rate = torch.where((background_count > 0)[:, :, None],
                                      background_rate, fractions)
        weight = counts / (counts + self.shrinkage_strength)
        return (weight[:, :, None] * fractions
                + (1.0 - weight[:, :, None]) * background_rate)

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
        # Arbitrary valid-vocabulary payloads at masked positions have no path
        # into content, membership, counts or feedback statistics.
        valid = (mask > 0)[:, :, None]
        keys = keys * valid.to(keys.dtype)
        values = values * valid.to(values.dtype)
        context = torch.cat([user_emb[n] for n in self.user_fields], dim=-1)
        members = self._membership(item, mask)
        counts = members.sum(dim=-1)
        divisor = counts.clamp(min=1)[:, :, None]
        batch = mask.shape[0]
        if keys.shape[1]:
            mean_keys = torch.bmm(members.to(keys.dtype), keys) / divisor
            mean_values = torch.bmm(members.to(values.dtype), values) / divisor
        else:
            mean_keys = keys.new_zeros((batch, self.num_slots, self.key_width))
            mean_values = values.new_zeros((batch, self.num_slots, self.value_width))
        occupied = (counts > 0)[:, :, None]
        mean_keys = mean_keys * occupied.to(mean_keys.dtype)
        mean_values = mean_values * occupied.to(mean_values.dtype)
        slots = self.slot_projection(mean_values)
        slots = slots * occupied.to(slots.dtype)
        if not self.include_content:
            slots = slots * 0.0
        count_features = counts.to(context.dtype) / 100.0
        if not self.include_counts:
            count_features = count_features * 0.0

        # Historical feedback masks may overlap; none is a current label.
        # The denominator is ALL valid events in the slot, not positives.
        feedback = torch.stack([(m > 0) & (mask > 0) for m in task_masks], dim=1)
        feedback = feedback.to(members.dtype)  # B x 4 x L
        if keys.shape[1]:
            slot_feedback = torch.bmm(members, feedback.transpose(1, 2)).float()
            fractions = slot_feedback / divisor
        else:
            slot_feedback = members.new_zeros((batch, self.num_slots, self.feedback_tasks))
            fractions = slot_feedback / divisor
        # Reassociation is valid because feedback_projection is bias-free and
        # the routing/denominator do not depend on learned embeddings:
        # W(mean_s x) == mean_s(Wx) in exact arithmetic. BF16 is not bitwise
        # associative. Verify shared-state outputs and gradients on the host.
        projected = self.feedback_projection(mean_keys)
        projected = projected * occupied.to(projected.dtype)
        expanded = projected[:, :, None, :].expand(-1, -1, self.feedback_tasks, -1)
        if self.feedback_mode == 'factorized':
            gate = self._interpolate_rates(fractions, counts, feedback, mask, slot_feedback)
        elif self.feedback_mode == 'content':
            # Identical learned projection and input columns, no product with
            # feedback rates. Explicit slot fractions remain below.
            gate = torch.ones_like(fractions)
        else:
            global_count = (mask > 0).sum(dim=-1).clamp(min=1)
            global_fraction = feedback.sum(dim=-1) / global_count[:, None]
            gate = global_fraction[:, None, :].expand(-1, self.num_slots, -1)
        # All pooled variants execute the same product shape. Local fractions
        # remain explicit even when the PRODUCT uses a different gate.
        moments = expanded * gate[:, :, :, None]
        if not self.include_feedback_fractions:
            fractions = fractions * 0.0
        features = torch.cat((context, target, slots.flatten(start_dim=1),
                              count_features, moments.flatten(start_dim=1),
                              fractions.flatten(start_dim=1)), dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
