"""Relation-consistent compression of inherited target-relation rank summaries.

Keep the duplicate-full parent A*M+B*M and add C[r]*M before JointBlock.
Relation sharing uses one C per target-match relation across all five time
bins. Rotated sharing uses C[(r+t)%7] with equal parameters and nominal bank
usage. Shared uses the mean bank; zero retains only the parent function.
All new maps start at zero. No new information, tokens, loss, or host recipe.
This is an unverified local adaptation of heterogeneous parameter sharing,
not a new-method priority claim or a reproduction of HiFormer.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {
    'embedding_dim': 16,
    'attention_hidden_units': [64, 32],
    'tower_hidden_units': [128, 64],
    'token_dim': 32,
    'num_heads': 2,
    'slot_mode': 'recency',
    'representation_mode': 'replicas',
    'content_mode': 'relation',
    'count_mode': 'relation',
    'moment_mode': 'linear',
    'moment_component': 'full',
    'residual_mode': 'copy',
    'bank_mode': 'rotated',
}


class JointBlock(nn.Module):
    """Pre-norm attention with invalid slots excluded as keys and outputs."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens, valid):
        batch, count, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, count, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # FP32 scores/softmax avoid overflow under the frozen BF16 autocast.
        # Explicitly disable autocast: .float() alone does not protect matmul.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.head_width)
            scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
            attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention.to(value.dtype), value)
        mixed = mixed.transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.output(mixed)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens * valid.unsqueeze(-1).to(tokens.dtype)


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TargetRelationStudy', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), token_dim=32, num_heads=2,
                 slot_mode='recency', representation_mode='replicas',
                 content_mode='relation', count_mode='relation', moment_mode='linear',
                 moment_component='full', residual_mode='copy', bank_mode='rotated',
                 num_tasks=None,
                 task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if list(feature_map.labels) != ['is_click', 'cart', 'fav', 'buy']:
            raise ValueError('This study requires the frozen four-task order')
        if slot_mode not in ('pooled', 'recency'):
            raise ValueError('Unknown slot construction')
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError('token_dim must be divisible by num_heads')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        if representation_mode != 'replicas':
            raise ValueError('This factorial study starts from replica_seven')
        if content_mode not in ('relation', 'replicas'):
            raise ValueError('Unknown summary content')
        if count_mode not in ('relation', 'temporal'):
            raise ValueError('Unknown explicit count channel')
        if moment_mode not in ('linear', 'quadratic', 'copy', 'zero'):
            raise ValueError('Unknown position moment')
        if moment_mode != 'linear':
            raise ValueError('Decomposition study freezes the linear rank basis')
        if moment_component not in ('within', 'occupancy', 'full', 'zero'):
            raise ValueError('Unknown relation position component')
        if moment_component != 'full':
            raise ValueError('This study retains the full parent moment')
        if residual_mode not in ('occupancy', 'copy', 'zero'):
            raise ValueError('Unknown residual summary')
        if residual_mode != 'copy':
            raise ValueError('Relation-bank study freezes the duplicate-full parent')
        if bank_mode not in ('relation', 'rotated', 'shared', 'zero'):
            raise ValueError('Unknown relation bank sharing rule')
        self.bank_mode = bank_mode
        self.residual_mode = residual_mode
        self.moment_component = moment_component
        if content_mode != 'relation' or count_mode != 'relation':
            raise ValueError('Position study requires the frozen relation parent')
        self.moment_mode = moment_mode
        self.slot_mode = slot_mode
        self.representation_mode = representation_mode
        self.content_mode = content_mode
        self.count_mode = count_mode
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if not self.item_fields or not self.user_fields or self.action_field is None:
            raise ValueError('Required released feature groups are missing')
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim
        self.static_count = len(self.user_fields) + len(self.item_fields)
        # All moment arms keep the parent's 35 summary tokens and equal parameters.
        value_fields = self.item_fields + [self.action_field]
        expected = {'item_id', 'cate_id', 'campaign_id', 'customer_id',
                    'brand', 'price_bucket', 'action'}
        if set(value_fields) != expected or len(value_fields) != 7:
            raise ValueError('Expected the frozen six item fields plus action')
        self.group_count = len(value_fields)
        self.token_count = self.static_count + 5 * self.group_count
        # Keep the parent's nonrandom buffer and module construction order.
        selection = torch.zeros(self.group_count, self.value_width)
        selection.fill_(1.0 / math.sqrt(self.group_count))
        self.register_buffer('content_selection', selection)
        self.match_fields = list(self.item_fields)
        self.match_padding = {name: feature_map.features[name].get('padding_idx', 0)
                              for name in self.match_fields}
        layers = []
        width = 4 * self.key_width
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)
        self.field_projection = nn.Linear(embedding_dim, token_dim)
        self.slot_projection = nn.Linear(self.value_width + 1, token_dim)
        self.token_identity = nn.Parameter(torch.empty(1, self.token_count, token_dim))
        self.interaction = JointBlock(token_dim, num_heads)
        # Retain raw static fields and the original global DIN sum in all arms.
        input_width = (self.static_count * embedding_dim + self.value_width
                       + self.token_count * token_dim + 5)
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        nn.init.normal_(self.token_identity, std=0.02)
        # Append AFTER every inherited initialization. The new path starts at
        # exactly zero, and its constructor cannot advance the parent's CPU
        # RNG stream (including any later host sampling using that stream).
        with torch.random.fork_rng(devices=[]):
            self.moment_projection = nn.Linear(self.value_width, token_dim, bias=False)
            nn.init.zeros_(self.moment_projection.weight)
        # A separate fork preserves all inherited initialization and the host
        # CPU RNG stream. Every arm has this same extra zero-initialized map.
        with torch.random.fork_rng(devices=[]):
            self.residual_projection = nn.Linear(self.value_width, token_dim, bias=False)
            nn.init.zeros_(self.residual_projection.weight)
        # Append after every inherited initialization; zeros consume no RNG.
        # All arms own exactly seven [value_width, token_dim] matrices.
        self.relation_bank = nn.Parameter(
            torch.zeros(self.group_count, self.value_width, token_dim))
        self.register_buffer('bank_routes', torch.tensor(
            self._bank_assignments(), dtype=torch.long))
        self.model_to_device()

    def _bank_assignments(self):
        """Seven banks, five uses each; rotate by window only in the control.

        A single global bank relabeling cannot undo the rotated assignment:
        relation r uses bank r in the global bin and r+t in window t.
        Only parameter sharing changes. No tokens, values or masks move.
        """
        return tuple(tuple((relation + window) % self.group_count
                           if self.bank_mode == 'rotated' else relation
                           for relation in range(self.group_count))
                     for window in range(5))

    def _project_relation_bank(self, moments):
        """Project existing M with C[r], a balanced rotation, or a shared mean.

        Zero weights start every arm at the duplicate-full parent function.
        The primary capacity-matched contrast is relation versus rotated.
        Shared has equal nominal capacity but a smaller functional subspace;
        zero is a replay check with an inactive path, not an active control.
        """
        bank = self.relation_bank
        if self.bank_mode == 'shared':
            bank = bank.mean(dim=0, keepdim=True).expand(self.group_count, -1, -1)
        routed = bank[self.bank_routes].to(moments.dtype)
        projected = torch.einsum('btgv,tgvd->btgd', moments, routed)
        if self.bank_mode == 'zero':
            projected = projected * 0.0
        return projected

    def _slot_masks(self, valid):
        """Global plus four rank quartiles, or five global copies."""
        all_events = valid[:, None, :]
        if self.slot_mode == 'pooled':
            content = all_events.expand(-1, 5, -1)
        else:
            ranks = valid.long().cumsum(dim=-1) - 1
            length = valid.long().sum(dim=-1, keepdim=True).clamp_min(1)
            bins = torch.div(4 * ranks, length, rounding_mode='floor').clamp(0, 3)
            parts = [(bins == index) & valid for index in range(4)]
            content = torch.cat([all_events, torch.stack(parts, dim=1)], dim=1)
        return content, content.sum(dim=-1)

    def _relation_masks(self, clean_item, valid, temporal_masks):
        """[B,5,7,L]: global plus six overlapping encoded-field matches.

        Zero/padding IDs never create equality evidence. Equality of any other
        encoded ID is not guaranteed to establish true entity identity (OOV or
        publisher missing-value codes may coincide). No action/label matching.
        """
        relations = [valid]
        for name in self.match_fields:
            history = clean_item[name][:, :-1]
            target = clean_item[name][:, -1:]
            match = (history == target) & (history > 0) & (target > 0) & valid
            padding = self.match_padding[name]
            if padding is not None:
                match = match & (history != padding) & (target != padding)
            relations.append(match)
        relations = torch.stack(relations, dim=1)
        return temporal_masks[:, :, None, :] & relations[:, None, :, :]

    def _moment_coefficients(self, temporal_masks):
        """Relative valid rank, independent of padding IDs and absolute offsets.

        Center BEFORE restricting by a target relation, so a relation present
        only late versus early in a bin remains distinguishable. RMS is over
        that temporal bin, not over relation occupancy. The global bin also
        carries a global moment. No real timestamp/delay is inferred.
        """
        visible = temporal_masks.float()
        ranks = temporal_masks.long().cumsum(dim=-1).float() - 1.0
        counts = visible.sum(dim=-1, keepdim=True)
        safe_counts = counts.clamp_min(1.0)
        centered = (2.0 * ranks - counts + 1.0) * visible
        linear_rms = (centered.square().sum(dim=-1, keepdim=True)
                      / safe_counts).sqrt().clamp_min(1e-6)
        linear = centered / linear_rms
        squared = centered.square()
        quadratic = (squared - squared.sum(dim=-1, keepdim=True)
                     / safe_counts) * visible
        quadratic_rms = (quadratic.square().sum(dim=-1, keepdim=True)
                         / safe_counts).sqrt().clamp_min(1e-6)
        quadratic = quadratic / quadratic_rms
        # Always calculate both bases in all arms. In particular, a reversed
        # linear basis is NOT a distinct mechanism; its sign is absorbable.
        if self.moment_mode == 'linear':
            return linear
        if self.moment_mode == 'quadratic':
            return quadratic
        if self.moment_mode == 'copy':
            return visible
        return visible * 0.0

    def _residual_moment(self, moments, occupancy):
        """Choose the input to B; all arms retain A applied to the full M."""
        if self.residual_mode == 'occupancy':
            return occupancy
        if self.residual_mode == 'copy':
            return moments
        return moments * 0.0

    def _position_moments(self, values, weights, relation_masks, temporal_masks):
        """M=W+O with an unweighted matched-event rank centroid.

        Keep the original temporal basis scale. Centering separately within
        a relation removes its mean placement, without dividing by signed
        learned weights or rescaling sparse relations. No-match input is zero;
        singleton/constant weighted-content W vanishes in exact arithmetic.
        """
        batch, length, _ = values.shape
        if not length:
            empty = values.new_zeros((batch, 5, self.group_count, self.value_width))
            return empty, self._residual_moment(empty, empty)
        # Preserve the parent's raw M accumulation for the full replay.
        # Common forward work in every arm; effective backward work can differ.
        with torch.autocast(device_type=values.device.type, enabled=False):
            coefficients = self._moment_coefficients(temporal_masks)
            moment_weights = (relation_masks.float() * weights.float()[:, None, None, :]
                              * coefficients[:, :, None, :])
            moments = torch.bmm(moment_weights.flatten(1, 2), values.float())
            moments = moments.reshape(batch, 5, self.group_count, self.value_width)
            moments = moments / math.sqrt(self.group_count)
            visible = relation_masks.float()
            centroid = (visible * coefficients[:, :, None, :]).sum(dim=-1, keepdim=True)
            centroid = centroid / visible.sum(dim=-1, keepdim=True).clamp_min(1.0)
            content_weights = visible * weights.float()[:, None, None, :]
            content = torch.bmm(content_weights.flatten(1, 2), values.float())
            content = content.reshape(batch, 5, self.group_count, self.value_width)
            content = content / math.sqrt(self.group_count)
            occupancy = centroid * content
            within = moments - occupancy
            if self.moment_component == 'full':
                result = moments
            elif self.moment_component == 'within':
                result = within
            elif self.moment_component == 'occupancy':
                result = occupancy
            else:
                result = moments * 0.0
        return result, self._residual_moment(moments, occupancy)

    def forward(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Frozen historical task masks are required')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(task_masks) != 4:
            raise ValueError('Expected four historical task masks')
        valid = mask.bool()
        batch, length = valid.shape
        # Sanitize padded IDs before embedding and before batch-coupled Dice.
        # Exclude the current action even if an upstream caller changes it.
        clean_item = {}
        for name in self.item_fields + [self.action_field]:
            history = item[name][:, :-1].masked_fill(~valid, 0)
            target = (torch.zeros_like(item[name][:, -1:])
                      if name == self.action_field else item[name][:, -1:])
            clean_item[name] = torch.cat([history, target], dim=1)
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(clean_item)
        static = torch.stack([user_emb[name] for name in self.user_fields]
                             + [item_emb[name][:, -1] for name in self.item_fields], dim=1)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        keys = keys * valid.unsqueeze(-1).to(keys.dtype)
        action = item_emb[self.action_field][:, :-1]
        values = torch.cat([keys, action], dim=-1)
        values = values * valid.unsqueeze(-1).to(values.dtype)
        content_mask, counts = self._slot_masks(valid)
        relation_masks = self._relation_masks(clean_item, valid, content_mask)
        relation_counts = relation_masks.sum(dim=-1)
        if length:
            query = target[:, None, :].expand(-1, length, -1)
            interactions = torch.cat([query, keys, query - keys, query * keys], dim=-1)
            weights = self.local_activation(interactions.reshape(batch * length, -1))
            weights = weights.reshape(batch, length) * valid.to(weights.dtype)
            weighted_masks = content_mask.to(weights.dtype) * weights[:, None, :]
            pooled = torch.bmm(weighted_masks, values)
        else:
            pooled = target.new_zeros((batch, 5, self.value_width))
        slot_valid = counts > 0
        pooled = pooled * slot_valid.unsqueeze(-1).to(pooled.dtype)
        log_counts = torch.log1p(counts.float()) / math.log(101.0)
        log_counts = log_counts.to(pooled.dtype)
        if self.content_mode == 'relation':
            if length:
                relation_weights = (relation_masks.to(weights.dtype)
                                    * weights[:, None, None, :])
                grouped = torch.bmm(relation_weights.flatten(1, 2), values)
                grouped = grouped.reshape(batch, 5, self.group_count, self.value_width)
                grouped = grouped / math.sqrt(self.group_count)
            else:
                grouped = pooled.new_zeros((batch, 5, self.group_count, self.value_width))
        else:
            grouped = (pooled[:, :, None, :]
                       * self.content_selection[None, None, :, :].to(pooled.dtype))
        if self.count_mode == 'relation':
            group_counts = torch.log1p(relation_counts.float()) / math.log(101.0)
            group_counts = group_counts.to(pooled.dtype).unsqueeze(-1)
        else:
            group_counts = log_counts[:, :, None, None].expand(-1, -1, self.group_count, 1)
        slot_tokens = self.slot_projection(torch.cat([grouped, group_counts], dim=-1))
        # The inherited content/count projection and token validity remain
        # intact. The second summary enters the SAME token, before interaction.
        if not length:
            weights = target.new_zeros((batch, 0))
        moments, residual = self._position_moments(values, weights, relation_masks, content_mask)
        moment_tokens = self.moment_projection(moments.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + moment_tokens
        residual_tokens = self.residual_projection(residual.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + residual_tokens
        # Preserve temporal/relation identities and all existing summary paths.
        bank_tokens = self._project_relation_bank(moments.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + bank_tokens
        slot_tokens = slot_tokens.flatten(1, 2)
        # Validity is temporal in EVERY arm, so group occupancy is not leaked
        # by the attention mask when the explicit count input is ablated.
        # A no-match relation has zero content but an active identity token.
        group_valid = slot_valid[:, :, None].expand(-1, -1, self.group_count).flatten(1)
        tokens = torch.cat([self.field_projection(static), slot_tokens], dim=1)
        tokens = tokens + self.token_identity.to(tokens.dtype)
        static_valid = torch.ones((batch, self.static_count), dtype=torch.bool, device=valid.device)
        token_valid = torch.cat([static_valid, group_valid], dim=1)
        tokens = tokens * token_valid.unsqueeze(-1).to(tokens.dtype)
        tokens = self.interaction(tokens, token_valid)
        features = torch.cat([static.flatten(1), pooled[:, 0], tokens.flatten(1), log_counts], dim=-1)
        return {label + '_pred': torch.sigmoid(tower(features))
                for label, tower in zip(self.feature_map.labels, self.towers)}
