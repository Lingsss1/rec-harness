"""Prospective task representation exchange, with no measured efficacy claim.

A 2x2 design compares peer versus self donor representations and detached
versus joint donor gradients. Core architecture and initialization follow the
empirical incumbent. Transfer output layers start at zero, so initial logits
match that core. Detachment only blocks the additional donor path; embeddings
and the history scorer remain shared. No current labels enter forward.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {
    'embedding_dim': 16,
    'attention_hidden_units': [64, 32],
    'tower_hidden_units': [128, 64],
    'attention_normalization': 'none',
    'task_specific': True,
    'action_in_score': True,
    'peer_transfer': False,
    'detach_donors': False,
    'transfer_width': 32,
}


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TaskEvidenceExchange', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 task_specific=True, action_in_score=True, peer_transfer=False,
                 detach_donors=False, transfer_width=32, num_tasks=None,
                 task='binary_classification', **kwargs):
        labels = list(feature_map.labels)
        if len(labels) != 4 or set(labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('Expected all four frozen feedback tasks')
        if num_tasks is not None and num_tasks != len(labels):
            raise ValueError('Task count mismatch')
        if attention_normalization != 'none':
            raise ValueError('This study fixes unnormalized DIN weights')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(labels), task=task, **kwargs)
        self.task_specific = bool(task_specific)
        self.action_in_score = bool(action_in_score)
        if not self.task_specific or not self.action_in_score:
            raise ValueError('This study retains task/action relevance')
        if not tower_hidden_units or transfer_width < 1:
            raise ValueError('Expected a hidden task representation and positive transfer width')
        self.peer_transfer = bool(peer_transfer)
        self.detach_donors = bool(detach_donors)
        self.transfer_width = int(transfer_width)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ['item', 'action']
                            and spec['type'] != 'meta']
        action_fields = [name for name, spec in feature_map.features.items()
                         if spec.get('source') == 'action' and spec['type'] != 'meta']
        if not self.item_fields or len(action_fields) != 1:
            raise ValueError('Expected item fields and one historical action field')
        self.action_field = action_fields[0]
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim

        # Every cell allocates the action columns and four score rows.
        layers = []
        width = 4 * self.key_width + embedding_dim
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, self.num_tasks))
        self.local_activation = nn.Sequential(*layers)

        input_width = len(self.user_fields) * embedding_dim + self.key_width + self.value_width
        self.towers = nn.ModuleList()
        for _ in labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        # Identical initial channels remove a pooling-scale difference between
        # shared averaging and task routing. Task gradients can specialize them.
        score = self.local_activation[-1]
        with torch.no_grad():
            score.weight.copy_(score.weight[:1].clone().expand_as(score.weight))
            score.bias.copy_(score.bias[:1].clone().expand_as(score.bias))
        # Construct added modules AFTER resetting/equalizing the incumbent.
        # This preserves its parameter RNG draws. Do not reset the whole model
        # again: that would reinitialize the core differently from the parent.
        # Restore CPU RNG after adding adapters, preserving any later host
        # random draws as well as the inherited core initialization.
        with torch.random.fork_rng(devices=[]):
            hidden_width = tower_hidden_units[-1]
            self.donor_projections = nn.ModuleList([
                nn.Linear(hidden_width, self.transfer_width) for _ in labels
            ])
            self.receiver_queries = nn.ModuleList([
                nn.Linear(hidden_width, self.transfer_width) for _ in labels
            ])
            self.transfer_outputs = nn.ModuleList([
                nn.Linear(self.transfer_width, 1) for _ in labels
            ])
            for layer in list(self.donor_projections) + list(self.receiver_queries):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in self.transfer_outputs:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.model_to_device()

    def _represent(self, inputs):
        user, item, mask = self.get_inputs(inputs)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        actions = item_emb[self.action_field][:, :-1]
        valid = mask.to(torch.bool)
        # Mask keys/actions before Dice and scores after it. The target query
        # at padded positions still affects Dice statistics; this inherited
        # limitation is held fixed. Padding IDs cannot affect pooled values.
        keys = keys.masked_fill(~valid.unsqueeze(-1), 0)
        actions = actions.masked_fill(~valid.unsqueeze(-1), 0)
        values = torch.cat([keys, actions], dim=-1)
        if self.user_fields:
            context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        else:
            context = target.new_zeros((target.shape[0], 0))
        return context, target, keys, actions, values, valid

    def _pool(self, target, keys, actions, values, valid):
        batch, length, _ = keys.shape
        if length == 0:
            return target.new_zeros((batch, self.num_tasks, self.value_width))
        query = target[:, None, :].expand(-1, length, -1)
        action_input = actions if self.action_in_score else torch.zeros_like(actions)
        interactions = torch.cat([query, keys, query - keys, query * keys, action_input], dim=-1)
        scores = self.local_activation(interactions.reshape(batch * length, -1))
        scores = scores.reshape(batch, length, self.num_tasks)
        if not self.task_specific:
            # Every row gets gradients, but averaging reduces effective capacity.
            scores = scores.mean(dim=-1, keepdim=True).expand(-1, -1, self.num_tasks)
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0)
        # Four summaries cost O(B * T * L * value_width), with no L^2 term.
        # All-masked histories give zero summaries without softmax.
        return torch.bmm(scores.transpose(1, 2), values)

    def _transfer(self, hidden):
        # hidden is a list of task tower activations, each [B,H]. Every receiver
        # uses three donor slots and the same projections in all four cells.
        # Self controls replace source representations, preserving active
        # adapters, their identities, parameter count and dense arithmetic.
        residuals = []
        for receiver in range(self.num_tasks):
            query = torch.tanh(self.receiver_queries[receiver](hidden[receiver]))
            tokens = []
            for donor in range(self.num_tasks):
                if donor == receiver:
                    continue
                source = hidden[donor] if self.peer_transfer else hidden[receiver]
                if self.detach_donors:
                    source = source.detach()
                # Stop only at the donor activation: these trainable adapters
                # still receive recipient loss gradients in detached cells.
                tokens.append(torch.tanh(self.donor_projections[donor](source)))
            tokens = torch.stack(tokens, dim=1)  # B,3,D
            scores = (tokens.float() * query.float()[:, None, :]).sum(dim=-1)
            scores = scores / (self.transfer_width ** 0.5)
            weights = torch.softmax(scores, dim=1).to(tokens.dtype)
            message = (weights[:, :, None] * tokens).sum(dim=1)
            residuals.append(self.transfer_outputs[receiver](message))
        return residuals

    def forward(self, inputs):
        context, target, keys, actions, values, valid = self._represent(inputs)
        pooled = self._pool(target, keys, actions, values, valid)
        hidden = []
        for index, tower in enumerate(self.towers):
            features = torch.cat([context, target, pooled[:, index]], dim=-1)
            # Reuse the existing modules in their original order, including
            # Dice, and execute each tower prefix exactly once.
            for layer in list(tower.children())[:-1]:
                features = layer(features)
            hidden.append(features)
        residuals = self._transfer(hidden)
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            logits = tower[-1](hidden[index]) + residuals[index]
            outputs[label + '_pred'] = torch.sigmoid(logits)
        return outputs
