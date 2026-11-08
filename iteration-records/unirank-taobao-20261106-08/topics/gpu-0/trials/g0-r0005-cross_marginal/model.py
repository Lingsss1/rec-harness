"""Prospective 2x2 study of event association in compact history tokens.

Parent: g0-r0004-transport_norm_aux_norm (click AUC .65649492, cart .82757606).
Its longest-history click AUC .65348144 is below DIN .65780310; this is an
exploratory limitation, not evidence that missing event association causes it.

Bank 1 stays the inherited first moment. Bank 2 retains that first moment and
adds either E_w[x * a] or E_w[x] * E_w[a], where * is coordinate-wise product.
The partner a is the mean of the OTHER event fields (including action), or x
itself in dispersion controls. Every factor uses the same field-specific w.
The order contrast is weighted covariance, with no new trainable parameters.

All arms retain retrieval, masks, counts, presence gates, projection widths,
initialization order, attention stack, towers, and path-specific content scale.
The factorial tests explicit cross-field association versus generic dispersion;
it does not identify useful field pairs or establish an event-level causal
mechanism. Raw coordinate products assume aligned learned embedding axes.

Nearest inspected implementations: UniRank HiFormer (pool then tokenize) and
TokenFormer (retain full event tokens). This instead compresses first/second
moments before the inherited stack. DIN already uses multiplicative relevance
features; this is no claim that feature products or moment pooling are novel.
Original TokenFormer paper detail is unavailable in the authorized archive.
Host owns training, information visibility, loss, optimization and scoring.
"""
import math
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice

CONFIG = {'embedding_dim': 16, 'token_dim': 32, 'num_heads': 2, 'num_layers': 2, 'field_retrieval': True, 'feedback_groups': False, 'tower_hidden_units': [128, 64], 'pooling_mode': 'sigmoid_normalized', 'representation_mode': 'scaled_pair', 'click_readout_scale': 'scaled', 'auxiliary_readout_scale': 'normalized', 'transport_scale': 'normalized', 'moment_order': 'marginal', 'moment_partner': 'cross_field'}


class Interaction(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.width = width
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens):
        b, n, d = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        attention = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        mixed = torch.matmul(attention, v).transpose(1, 2).reshape(b, n, d)
        tokens = tokens + self.out(mixed)
        return tokens + self.ffn(self.norm2(tokens))


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='PairedContentVolumeSummary', gpu=-1,
                 embedding_dim=16, token_dim=32, num_heads=2, num_layers=2,
                 field_retrieval=True, feedback_groups=False,
                 pooling_mode='sigmoid_normalized', representation_mode='scaled_pair',
                 tower_hidden_units=(128, 64), num_tasks=None,
                 click_readout_scale='scaled', auxiliary_readout_scale='normalized',
                 transport_scale='normalized', moment_order='marginal',
                 moment_partner='cross_field',
                 task='binary_classification', **kwargs):
        if len(feature_map.labels) != 4 or num_tasks not in (None, 4):
            raise ValueError('This study requires the frozen four tasks')
        if token_dim % num_heads:
            raise ValueError('token_dim must be divisible by num_heads')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=4, task=task, **kwargs)
        if pooling_mode not in ('softmax', 'sigmoid_normalized',
                                'sigmoid_count', 'sigmoid_sum'):
            raise ValueError('Unknown retrieval pooling mode')
        if not field_retrieval or feedback_groups:
            raise ValueError('This study fixes field retrieval and global pooling')
        if pooling_mode != 'sigmoid_normalized':
            raise ValueError('Paired representation study fixes normalized retrieval')
        if representation_mode not in ('split', 'normalized_pair', 'scaled_pair', 'fused_pair'):
            raise ValueError('Unknown paired representation mode')
        if representation_mode != 'scaled_pair':
            raise ValueError('Shared transport is fixed to the incumbent scaled pair')
        if click_readout_scale not in ('scaled', 'normalized') or auxiliary_readout_scale not in ('scaled', 'normalized'):
            raise ValueError('Readout scales must be scaled or normalized')
        if set(feature_map.labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('Expected the frozen four labels')
        if transport_scale not in ('scaled', 'normalized'):
            raise ValueError('Transport scale must be scaled or normalized')
        if moment_order not in ('within_event', 'marginal'):
            raise ValueError('Unknown moment order')
        if moment_partner not in ('cross_field', 'self'):
            raise ValueError('Unknown moment partner')
        self.moment_order = moment_order
        self.moment_partner = moment_partner
        self.transport_scale = transport_scale
        self.click_readout_scale = click_readout_scale
        self.auxiliary_readout_scale = auxiliary_readout_scale
        self.representation_mode = representation_mode
        self.pooling_mode = pooling_mode
        self.field_retrieval = field_retrieval
        self.feedback_groups = feedback_groups
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ['item', 'action']
                            and spec['type'] != 'meta' and name not in feature_map.labels]
        self.action_field = next(name for name, spec in feature_map.features.items()
                                 if spec.get('source') == 'action' and spec['type'] != 'meta')
        fields = self.user_fields + self.item_fields + [self.action_field]
        if any(feature_map.features[name].get('embedding_dim', embedding_dim)
               != embedding_dim for name in fields):
            raise ValueError('Study requires uniform field embedding dimensions')
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        f = len(self.item_fields)
        self.query = nn.ModuleList([nn.Linear(2 * embedding_dim, embedding_dim)
                                    for _ in range(f)])
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        static_count = len(self.user_fields) + f
        self.static_proj = nn.ModuleList([nn.Linear(embedding_dim, token_dim)
                                         for _ in range(static_count)])
        self.summary_proj = nn.ModuleList([
            nn.Linear((f + 1) * embedding_dim, token_dim) for _ in range(2)])
        self.count_proj = nn.Linear(1, token_dim, bias=False)
        self.identities = nn.Parameter(torch.empty(1, static_count + 10 + 4, token_dim))
        self.blocks = nn.ModuleList([Interaction(token_dim, num_heads)
                                     for _ in range(num_layers)])
        # Every arm retains ten summary tokens, identical parameter shapes and
        # identical count inputs. Keep the direct raw/summary/count path.
        input_width = static_count * embedding_dim + 11 * token_dim + 5
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            layers = []
            width = input_width
            for hidden in tower_hidden_units:
                layers.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            layers.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*layers))
        self.reset_parameters()
        nn.init.normal_(self.identities, std=0.02)
        self.model_to_device()

    def _masked_weights(self, scores, mask):
        # Same normalized sigmoid arithmetic as the committed parent.
        # Explicit masking makes an empty retrieval exactly zero.
        valid = mask[:, :, None, :].bool()
        gates = torch.sigmoid(scores[:, None].float()) * valid.to(torch.float32)
        return gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    def _moment_summary(self, history, action, weights, first_moments):
        # values: [B,L,K,D], with action as the final field. Each item field
        # uses its inherited retrieval distribution; action uses their mean.
        # The same distribution weights BOTH factors in the marginal control.
        values = torch.cat([history, action[:, :, None, :]], dim=2)
        value_weights = torch.cat([weights, weights.mean(dim=2, keepdim=True)], dim=2)
        cross_partner = (values.sum(dim=2, keepdim=True) - values) / (values.shape[2] - 1)
        partner = cross_partner if self.moment_partner == 'cross_field' else values
        # Compute both orders in every arm. No learned projection or extra
        # input is added. Difference = weighted coordinate-wise covariance.
        event_product = values * partner
        within = torch.einsum('bgkl,blkd->bgkd', value_weights, event_product)
        partner_mean = torch.einsum('bgkl,blkd->bgkd', value_weights, partner)
        marginal = first_moments * partner_mean
        moment = within if self.moment_order == 'within_event' else marginal
        return moment.flatten(start_dim=2)

    def _pair_summaries(self, summaries, moments, exposure_count):
        # Preserve the original first moment in both banks. Only bank 2 gains
        # a deterministic residual statistic. Scale is applied once AFTER
        # the statistic is formed, never squared with exposure count.
        scale = 0.5 * exposure_count[:, None, None].to(summaries.dtype)
        return summaries * scale, (summaries + moments) * scale

    def _project_banks(self, paired, present, count_features):
        history_banks = []
        for projection, summary in zip(self.summary_proj, paired):
            bank = projection(summary) * present[:, :, None]
            bank = bank + self.count_proj(count_features[:, :, None].to(bank.dtype))
            history_banks.append(bank)
        return torch.cat(history_banks, dim=1)

    def forward(self, inputs):
        if len(inputs) != 4 or inputs[3] is None:
            raise ValueError('Frozen task-specific historical masks are required')
        user, item, mask, multi_masks = self.get_inputs(inputs, return_multi_masks=True)
        if len(multi_masks) != 4 or any(m.shape != mask.shape for m in multi_masks):
            raise ValueError('Expected four historical masks in label order')
        ue = self.embedding_layer(user)
        ie = self.embedding_layer(item)
        targets = [ie[name][:, -1] for name in self.item_fields]
        static = [ue[name] for name in self.user_fields] + targets
        context = torch.stack([ue[name] for name in self.user_fields], dim=1).mean(dim=1)
        history = torch.stack([ie[name][:, :-1] for name in self.item_fields], dim=2)
        action = ie[self.action_field][:, :-1]
        b, length, f, d = history.shape
        # Intersect every host feedback mask with exposure visibility. Groups
        # overlap for composite action tokens; they are not disjoint outcomes.
        groups = torch.stack([mask] + [m * mask for m in multi_masks], dim=1).float()
        counts = groups.sum(dim=-1)
        count_features = torch.log1p(counts) / math.log(101.0)
        present = counts.gt(0).to(history.dtype)
        if length:
            query = torch.stack([torch.tanh(layer(torch.cat([target, context], dim=-1)))
                                 for layer, target in zip(self.query, targets)], dim=1)
            keys = self.key(history)
            scores = (keys * query[:, None]).sum(dim=-1).transpose(1, 2) / math.sqrt(d)
            if not self.field_retrieval:
                # One event score shared by every field, using the same queries.
                scores = scores.mean(dim=1, keepdim=True).expand(-1, f, -1)
            retrieval_masks = groups if self.feedback_groups else mask[:, None].expand(-1, 5, -1)
            weights = self._masked_weights(scores, retrieval_masks).to(history.dtype)
            pooled_fields = torch.einsum('bgfl,blfd->bgfd', weights, history)
            pooled_action = torch.bmm(weights.mean(dim=2), action)
            summaries = torch.cat([pooled_fields.flatten(start_dim=2), pooled_action], dim=-1)
            first_moments = torch.cat([pooled_fields, pooled_action[:, :, None, :]], dim=2)
            moments = self._moment_summary(history, action, weights, first_moments)
        else:
            summaries = history.new_zeros((b, 5, (f + 1) * d))
            moments = torch.zeros_like(summaries)
        # Keep the inherited absence gates and explicit count information.
        summaries = summaries * present[:, :, None]
        moments = moments * present[:, :, None]
        paired = self._pair_summaries(summaries, moments, counts[:, 0])
        # Compute both scale views with the SAME projections in every factorial cell.
        # Only summary content is scaled; counts, biases and absence gates keep
        # their inherited arithmetic.
        history_tokens = self._project_banks(paired, present, count_features)
        # Reuse both learned projections for the unit-mass readout. Scaling is
        # applied before projection: bias/count channels are never divided by n.
        normalized_tokens = self._project_banks(
            (summaries, summaries + moments), present, count_features)
        static_tokens = torch.stack([proj(value) for proj, value in zip(self.static_proj, static)], dim=1)
        task_tokens = static_tokens.new_zeros((b, 4, static_tokens.shape[-1]))
        transport_tokens = (history_tokens if self.transport_scale == 'scaled'
                            else normalized_tokens)
        tokens = torch.cat([static_tokens, transport_tokens, task_tokens], dim=1) + self.identities
        for block in self.blocks:
            tokens = block(tokens)
        common_scaled = torch.cat([
            torch.cat(static, dim=-1), history_tokens.flatten(start_dim=1),
            count_features.to(history_tokens.dtype)], dim=-1)
        common_normalized = torch.cat([
            torch.cat(static, dim=-1), normalized_tokens.flatten(start_dim=1),
            count_features.to(normalized_tokens.dtype)], dim=-1)
        outputs = {}
        for i, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            scale = (self.click_readout_scale if label == 'is_click'
                     else self.auxiliary_readout_scale)
            common = common_scaled if scale == 'scaled' else common_normalized
            outputs[label + '_pred'] = torch.sigmoid(
                tower(torch.cat([common, tokens[:, -4 + i]], dim=-1)))
        return outputs
