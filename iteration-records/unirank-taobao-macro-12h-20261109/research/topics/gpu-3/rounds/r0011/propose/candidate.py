"""Prospective independent content/moment task-routing study.

Retain the shared g0 relation/count/linear-moment tokens and R10 compact
residual scorer, content C, and zero-initialized shared map P(E). Independently
route the scores for C and E as private task scores or their four-task mean.
Cross this separation with linear/copy E, completing the missing cells of
the R10 three-factor study. No new parameters, values, loss or inputs.
Existing matching-route cells are historical controls, not new repetitions.
No predictive, mechanistic, efficiency or originality result is established.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'token_dim': 32, 'num_heads': 2, 'slot_mode': 'recency', 'representation_mode': 'replicas', 'content_mode': 'relation', 'count_mode': 'relation', 'moment_mode': 'linear', 'task_routing': 'private', 'block_click_score_grad': False, 'block_aux_score_grad': False, 'scorer_hidden_width': 64, 'task_extra_basis': 'linear', 'task_extra_routing': 'mean'}

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
    def __init__(self, feature_map, model_id='IndependentContentMomentRoutingStudy', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), token_dim=32, num_heads=2,
                 slot_mode='recency', representation_mode='replicas',
                 content_mode='relation', count_mode='relation', moment_mode='linear',
                 task_routing='private', block_click_score_grad=False,
                 block_aux_score_grad=False, scorer_hidden_width=64,
                 task_extra_basis='linear', task_extra_routing='mean',
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
        if task_routing not in ('private', 'mean', 'parent'):
            raise ValueError('Unknown task readout routing')
        if not isinstance(block_click_score_grad, bool) or not isinstance(block_aux_score_grad, bool):
            raise ValueError('Scorer-input stop switches must be booleans')
        if block_click_score_grad or block_aux_score_grad:
            raise ValueError('Shared-trunk study requires open scorer-input gradients')
        if len(attention_hidden_units) != 2:
            raise ValueError('Shared-trunk study requires two relevance hidden layers')
        if not isinstance(scorer_hidden_width, int) or isinstance(scorer_hidden_width, bool) or scorer_hidden_width <= 0:
            raise ValueError('scorer_hidden_width must be a positive integer')
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
        if content_mode != 'relation' or count_mode != 'relation':
            raise ValueError('Position study requires the frozen relation parent')
        if task_extra_basis not in ('linear', 'copy'):
            raise ValueError('Extra task summary requires linear or copy basis')
        if task_extra_routing not in ('private', 'mean'):
            raise ValueError('Unknown extra-summary score routing')
        self.task_extra_routing = task_extra_routing
        self.task_extra_basis = task_extra_basis
        self.moment_mode = moment_mode
        self.task_routing = task_routing
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
        # Tower widths stay fixed; only the direct global summary is augmented.
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
        # Only the residual scorer architecture changes. Keep every inherited
        # initialization and the host CPU RNG stream intact. Separate head
        # hidden weights and Dice buffers avoid a redundant symmetric mean.
        # All output layers start at zero; the trunk learns after they open.
        with torch.random.fork_rng(devices=[]):
            self.task_trunk = nn.Sequential(
                nn.Linear(4 * self.key_width + embedding_dim, scorer_hidden_width),
                Dice(scorer_hidden_width))
            self.task_scorers = nn.ModuleList([
                nn.Sequential(nn.Linear(scorer_hidden_width, attention_hidden_units[1]),
                              Dice(attention_hidden_units[1]),
                              nn.Linear(attention_hidden_units[1], 1))
                for _ in feature_map.labels])
            for module in self.task_trunk.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)
            for scorer in self.task_scorers:
                for module in scorer.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight)
                        nn.init.zeros_(module.bias)
                nn.init.zeros_(scorer[-1].weight)
                nn.init.zeros_(scorer[-1].bias)
        # Append after every inherited module initialization. This changes
        # neither inherited states nor the host CPU sampling RNG stream.
        # One shared map preserves identical extra capacity in both routing
        # arms; zero bias is structural, so empty history stays exactly zero.
        with torch.random.fork_rng(devices=[]):
            self.task_extra_projection = nn.Linear(
                self.value_width, self.value_width, bias=False)
            nn.init.zeros_(self.task_extra_projection.weight)
        self.model_to_device()

    def _task_pool(self, target, keys, actions, values, valid):
        """Independently route content and extra scores before pooling.

        Both routes use the SAME raw scorer outputs, values and valid mask.
        Each mean uses the inherited FP32 mean followed by the inherited cast.
        Routing changes learned scorer gradients as well as representations;
        these are training interventions, not frozen-weight causal mediation.
        Matching routes reproduce R10 arithmetic; P=0 reproduces its initial
        content source. Empty histories remain zero without projection bias.
        """
        batch, length, _ = keys.shape
        if not length:
            return target.new_zeros((batch, self.num_tasks, self.value_width))
        query = target[:, None, :].expand(-1, length, -1)
        actions = actions.masked_fill(~valid.unsqueeze(-1), 0)
        interactions = torch.cat([query, keys, query - keys, query * keys, actions], dim=-1)
        flat = interactions.reshape(batch * length, -1)
        hidden = self.task_trunk(flat)
        columns = [scorer(hidden) for scorer in self.task_scorers]
        scores = torch.cat(columns, dim=-1).reshape(batch, length, self.num_tasks)
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0)
        # Route C and E independently from the same raw scores. The previous
        # diagonal cells routed both together. Always execute both mixtures;
        # no route adds parameters, normalization or a new hidden state.
        center = scores.float().mean(dim=-1, keepdim=True).to(scores.dtype)
        content_mixing = 1.0 if self.task_routing == 'mean' else 0.0
        extra_mixing = 1.0 if self.task_extra_routing == 'mean' else 0.0
        content_scores = (1.0 - content_mixing) * scores + content_mixing * center
        extra_scores = (1.0 - extra_mixing) * scores + extra_mixing * center
        # Diagonal replay shares the actual tensor/gradient graph, matching
        # R10 accumulation order as well as the forward arithmetic. The
        # second mixture still executes, but is unused on diagonal checks.
        if self.task_extra_routing == self.task_routing:
            extra_scores = content_scores
        # Content path keeps the R9 content control's FP32 arithmetic.
        # Both summaries and the dense projection execute in every arm.
        # Protect signed cancellation and projection/addition from BF16
        # autocast, then restore the inherited value dtype once at the end.
        with torch.autocast(device_type=values.device.type, enabled=False):
            content_basis = valid.float()
            extra_basis = self._moment_coefficients(
                valid[:, None, :], mode=self.task_extra_basis).squeeze(1)
            content_weights = content_scores.float() * content_basis[:, :, None]
            extra_weights = extra_scores.float() * extra_basis[:, :, None]
            content = torch.bmm(content_weights.transpose(1, 2), values.float())
            extra = torch.bmm(extra_weights.transpose(1, 2), values.float())
            result = content + self.task_extra_projection(extra)
        return result.to(values.dtype)

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

    def _moment_coefficients(self, temporal_masks, mode=None):
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
        mode = self.moment_mode if mode is None else mode
        if mode == 'linear':
            return linear
        if mode == 'quadratic':
            return quadratic
        if mode == 'copy':
            return visible
        return visible * 0.0

    def _position_moments(self, values, weights, relation_masks, temporal_masks):
        batch, length, _ = values.shape
        if not length:
            return values.new_zeros((batch, 5, self.group_count, self.value_width))
        # FP32 accumulation protects cancellation in centered signed moments.
        # The added path has the same bmm/projection shapes in every arm.
        with torch.autocast(device_type=values.device.type, enabled=False):
            coefficients = self._moment_coefficients(temporal_masks)
            moment_weights = (relation_masks.float() * weights.float()[:, None, None, :]
                              * coefficients[:, :, None, :])
            moments = torch.bmm(moment_weights.flatten(1, 2), values.float())
            moments = moments.reshape(batch, 5, self.group_count, self.value_width)
            moments = moments / math.sqrt(self.group_count)
        return moments

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
        moments = self._position_moments(values, weights, relation_masks, content_mask)
        moment_tokens = self.moment_projection(moments.to(slot_tokens.dtype))
        slot_tokens = slot_tokens + moment_tokens
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
        task_pooled = pooled[:, 0:1] + self._task_pool(target, keys, action, values, valid)
        # Parent routing is for initial-function/replay diagnostics only. It
        # still allocates and executes the new scorer family. No proposed
        # factorial arm uses this path, and private gradients are unused here.
        if self.task_routing == 'parent':
            task_pooled = pooled[:, 0:1].expand(-1, self.num_tasks, -1)
        outputs = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            features = torch.cat([static.flatten(1), task_pooled[:, index],
                                  tokens.flatten(1), log_counts], dim=-1)
            outputs[label + '_pred'] = torch.sigmoid(tower(features))
        return outputs
