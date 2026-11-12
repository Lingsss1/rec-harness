"""Full-width local matching with field/history-conditioned refinement.

Prospective Q x W factorial; see proposal.json for evidence and limits.
All cells perform two shared-parameter DIN reads of the original values and
expose both refinement vectors to identical task heads. Q changes only the
second query; W changes only the second matching keys. Q0/W0 is the matched
stacked control, not an exact reproduction of the smaller original DIN.

No measured result or originality claim. Imports and inputs follow the frozen
host contract. Training, data loading and scoring are host responsibilities.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32],
          'tower_hidden_units': [128, 64], 'interaction_width': 128,
          'refresh_query': True, 'modulate_keys': False,
          'refinement_gain': 0.5}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='NativeMatchingRefinement', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), interaction_width=128,
                 refresh_query=True, modulate_keys=False, refinement_gain=0.5,
                 num_tasks=None, task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.refresh_query = bool(refresh_query)
        self.modulate_keys = bool(modulate_keys)
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

        # Native field-concatenated matching and unnormalized signed activation
        # retain the reference's scoring geometry; there is no latent K/V map.
        layers = []
        width = 4 * self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)

        # Both resulting vectors enter every head even when their feedback edge
        # is disabled, so all learned modules remain active in every cell.
        base_width = self.context_width + self.key_width + self.value_width
        self.field_interaction = nn.Sequential(
            nn.Linear(base_width, interaction_width), nn.GELU())
        self.query_update = nn.Linear(interaction_width, self.key_width)
        self.key_update = nn.Linear(interaction_width, self.key_width)
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
        scores = scores.reshape(batch, length)
        weights = scores * mask.to(scores.dtype)
        return torch.bmm(weights[:, None, :], values).squeeze(1)

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
        key_gate = torch.tanh(self.key_update(state))
        query = target
        if self.refresh_query:
            query = target + self.refinement_gain * query_delta
        match_keys = keys
        if self.modulate_keys:
            # Feature-wise, field/history-conditioned matching only. Original
            # history values are preserved; padded keys remain exactly zero.
            match_keys = keys * (1.0 + self.refinement_gain * key_gate[:, None, :])
        second = self._read(query, match_keys, values, valid)
        # Q0/W0 gives the same two reads in eval mode. No softmax, count
        # multiplier, positional input, task mask or target action is added.
        pooled = 0.5 * (first + second)
        features = torch.cat([context, target, pooled, query_delta, key_gate], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
