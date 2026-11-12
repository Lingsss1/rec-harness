"""Field-selective local retrieval crossed with field/history query feedback.

D=1 gives each original value field its own signed, unnormalized event weight.
D=0 ties those weights using sum/sqrt(F), preserving expected initial variance.
Q toggles only the second query feedback edge. All four cells have identical
registered and active parameters, two reads, masks and task-head widths.
This is a prospective mechanism test, not a measured improvement or novelty claim.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'interaction_width': 128, 'refresh_query': True, 'field_weights': False, 'refinement_gain': 0.5}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='FieldSelectiveRefinement', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), interaction_width=128,
                 refresh_query=True, field_weights=True, refinement_gain=0.5,
                 num_tasks=None, task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.refresh_query = bool(refresh_query)
        self.field_weights = bool(field_weights)
        self.embedding_dim = int(embedding_dim)
        self.refinement_gain = float(refinement_gain)
        if not 0.0 < self.refinement_gain <= 1.0:
            raise ValueError('refinement_gain must be in (0,1]')
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [n for n, s in feature_map.features.items()
                            if s.get('source') == 'item' and s['type'] != 'meta']
        self.user_fields = [n for n, s in feature_map.features.items()
                            if s.get('source') not in ['item', 'action']
                            and s['type'] != 'meta']
        self.action_field = next((n for n, s in feature_map.features.items()
                                  if s.get('source') == 'action'), None)
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim * int(self.action_field is not None)
        self.context_width = len(self.user_fields) * embedding_dim
        self.num_value_fields = len(self.item_fields) + int(self.action_field is not None)
        if not self.item_fields:
            raise ValueError('At least one item field is required')

        # Joint full-width evidence produces one score per original value field.
        # D changes their tying, not the scorer, embeddings or retrieval width.
        layers = []
        width = 4 * self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, self.num_value_fields))
        self.local_activation = nn.Sequential(*layers)

        # Both resulting vectors enter every head even when their feedback edge
        # is disabled, so all learned modules remain active in every cell.
        base_width = self.context_width + self.key_width + self.value_width
        self.field_interaction = nn.Sequential(
            nn.Linear(base_width, interaction_width), nn.GELU())
        self.query_update = nn.Linear(interaction_width, self.key_width)
        self.side_update = nn.Linear(interaction_width, self.key_width)
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            tower = []
            width = base_width + 2 * self.key_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        self.model_to_device()

    def _read(self, query, keys, values, mask):
        batch, length, _ = keys.shape
        if length == 0:
            return values.new_zeros((batch, self.value_width))
        q = query[:, None, :].expand(-1, length, -1)
        interactions = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.local_activation(interactions.reshape(batch * length, -1))
        scores = scores.reshape(batch, length, self.num_value_fields)
        if not self.field_weights:
            # A fixed unit-norm combination of all output heads yields a tied
            # scalar scorer. All heads are active; expected initial per-field
            # score variance matches D=1 for independent Xavier output rows.
            shared = scores.sum(dim=-1, keepdim=True) / math.sqrt(self.num_value_fields)
            scores = shared.expand_as(scores)
        weights = scores * mask[:, :, None].to(scores.dtype)
        # Batched fieldwise dot products avoid an expanded B x T x F x E
        # multiplication temporary. Original embeddings remain the values.
        field_values = values.reshape(batch, length, self.num_value_fields,
                                      self.embedding_dim)
        field_values = field_values.permute(0, 2, 1, 3).reshape(
            batch * self.num_value_fields, length, self.embedding_dim)
        field_weights = weights.transpose(1, 2).reshape(
            batch * self.num_value_fields, 1, length)
        return torch.bmm(field_weights, field_values).reshape(batch, self.value_width)

    def forward(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[n][:, -1] for n in self.item_fields], dim=-1)
        if self.user_fields:
            context = torch.cat([user_emb[n] for n in self.user_fields], dim=-1)
        else:
            context = target.new_zeros((target.shape[0], 0))
        keys = torch.cat([item_emb[n][:, :-1] for n in self.item_fields], dim=-1)
        # Erase invalid history before the batch-statistic-dependent Dice
        # scorer. Changing padded IDs cannot affect a valid row's scores.
        valid = mask.bool()
        keys = keys.masked_fill(~valid[:, :, None], 0.0)
        values = keys
        if self.action_field is not None:
            actions = item_emb[self.action_field][:, :-1]
            actions = actions.masked_fill(~valid[:, :, None], 0.0)
            values = torch.cat([keys, actions], dim=-1)

        first = self._read(target, keys, values, valid)
        state = self.field_interaction(torch.cat([context, target, first], dim=-1))
        query_delta = torch.tanh(self.query_update(state))
        side_state = torch.tanh(self.side_update(state))
        query = target
        if self.refresh_query:
            query = target + self.refinement_gain * query_delta
        second = self._read(query, keys, values, valid)
        # Q0 duplicates the first read (apart from training-time Dice running
        # statistics); both reads execute in every cell for a matched recipe.
        pooled = 0.5 * (first + second)
        features = torch.cat([context, target, pooled, query_delta, side_state], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
