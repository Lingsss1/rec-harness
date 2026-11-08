"""Prospective context/action routing factorial with an unchanged target query.

C gates scorer hidden channels from raw user/context fields; A gates them
from each visible historical action. Both gate summaries also enter every
prediction head, so off-edge controls retain active projection capacity.
The first read is ungated; the second read changes only scorer modulation.
No new performance or novelty claim is made by this implementation.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'interaction_width': 128, 'context_routing': True, 'action_routing': False, 'routing_gain': 0.5}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='ContextActionFieldRouting', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), interaction_width=128,
                 context_routing=True, action_routing=True, routing_gain=0.5,
                 num_tasks=None, task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.context_routing = bool(context_routing)
        self.action_routing = bool(action_routing)
        self.embedding_dim = int(embedding_dim)
        self.routing_gain = float(routing_gain)
        if not 0.0 < self.routing_gain < 1.0:
            raise ValueError('routing_gain must be in (0,1)')
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

        # Keep full-width target/item evidence and original field-valued sums.
        layers = []
        width = 4 * self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        self.local_hidden = nn.Sequential(*layers)
        self.routing_width = width
        self.field_scores = nn.Linear(width, self.num_value_fields)
        self.context_gate = nn.Linear(self.context_width, width)
        self.action_gate = nn.Linear(embedding_dim, width)
        if not self.user_fields or self.action_field is None:
            raise ValueError('Context and historical action fields are required')

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
            width = base_width + 2 * self.key_width + 2 * self.routing_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        self.model_to_device()

    def _read(self, query, keys, values, mask, context_gate, action_gate, routed):
        batch, length, _ = keys.shape
        if length == 0:
            return values.new_zeros((batch, self.value_width))
        q = query[:, None, :].expand(-1, length, -1)
        interactions = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        hidden = self.local_hidden(interactions.reshape(batch * length, -1))
        hidden = hidden.reshape(batch, length, self.routing_width)
        # Compute both routes in all cells; only their information-flow edges
        # differ. The same projections feed every head even with an edge off.
        context_factor = 1.0 + self.routing_gain * context_gate[:, None, :]
        action_factor = 1.0 + self.routing_gain * action_gate
        c_switch = float(routed and self.context_routing)
        a_switch = float(routed and self.action_routing)
        hidden = hidden * (1.0 + c_switch * (context_factor - 1.0))
        hidden = hidden * (1.0 + a_switch * (action_factor - 1.0))
        scores = self.field_scores(hidden).reshape(batch, length, self.num_value_fields)
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

        context_gate = torch.tanh(self.context_gate(context))
        action_gate = torch.tanh(self.action_gate(actions))
        action_gate = action_gate.masked_fill(~valid[:, :, None], 0.0)
        count = valid.sum(dim=1, keepdim=True).to(action_gate.dtype).clamp_min(1.0)
        action_summary = action_gate.sum(dim=1) / count

        first = self._read(target, keys, values, valid,
                           context_gate, action_gate, routed=False)
        state = self.field_interaction(torch.cat([context, target, first], dim=-1))
        query_delta = torch.tanh(self.query_update(state))
        side_state = torch.tanh(self.side_update(state))
        # Retain the parent readout paths while never perturbing the target.
        second = self._read(target, keys, values, valid,
                            context_gate, action_gate, routed=True)
        pooled = 0.5 * (first + second)
        features = torch.cat([context, target, pooled, query_delta, side_state,
                              context_gate, action_summary], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
