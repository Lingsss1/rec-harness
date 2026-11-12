"""DIN reference plus a small residual readout of causal history evidence.

Only the supplied history prefix is used. Evidence comprises generic, category,
brand, and joint matches by action, with total and one-day decayed counts.
CONFIG flags implement the predeclared 2x2 replacement controls.

Inherited: baseline_reference.py; DIN (arXiv:1706.06978v4, section 4.3).
Related aggregation: EDIN/DINMP (arXiv:2104.06312v2, section 3).
This is a local mechanism experiment; priority and transfer are unverified.
"""
import torch
from torch import nn

CONFIG = {
    'kind': 'din', 'embedding_dim': 16, 'hidden': [128, 64], 'lr': 0.001,
    'target_match': False, 'separate_actions': True,
}


class Model(nn.Module):
    def __init__(self,meta,config=None):
        super().__init__()
        config={**CONFIG,**(config or {})}
        self.config=config
        self.kind=config['kind']
        self.category=meta['category_field_index'];self.brand=meta['brand_field_index']
        dim=config['embedding_dim']
        self.embeddings=nn.ModuleList([nn.Embedding(size,dim,padding_idx=0) for size in meta['buckets']])
        self.action=nn.Embedding(5,dim,padding_idx=0)
        self.recency=nn.Embedding(33,dim,padding_idx=0)
        self.attention=nn.Sequential(nn.Linear(dim*4,64),nn.PReLU(),nn.Linear(64,1))
        layers=[]
        width=len(meta['fields'])*dim+(dim if self.kind!='static' else 0)
        for hidden in config['hidden']:
            layers.extend([nn.Linear(width,hidden),nn.ReLU()]);width=hidden
        layers.append(nn.Linear(width,1))
        self.output=nn.Sequential(*layers)
        # Append after the entire reference, preserving its initialization order.
        # Both layers omit bias, so an empty history has zero residual forever.
        self.evidence = nn.Sequential(
            nn.Linear(32, 32, bias=False), nn.ReLU(),
            nn.Linear(32, 1, bias=False),
        )
        # Every cell starts at the reference function; all cells use the same
        # architecture, parameter count, initialization and optimization recipe.
        nn.init.zeros_(self.evidence[-1].weight)

    def forward(self,batch):
        fields=[embedding(batch['static'][:,i]) for i,embedding in enumerate(self.embeddings)]
        features=torch.cat(fields,dim=-1)
        if self.kind!='static':
            ages=(torch.log2(batch['history_age_seconds']+1).long()+1).clamp(0,32)
            history=(self.embeddings[self.category](batch['history_category'])+
                     self.embeddings[self.brand](batch['history_brand'])+
                     self.action(batch['history_type'])+self.recency(ages))
            mask=batch['history_mask']
            if self.kind=='pooled':
                weights=mask.float()/mask.sum(-1,keepdim=True).clamp_min(1)
            else:
                query=(fields[self.category]+fields[self.brand])[:,None,:].expand_as(history)
                scores=self.attention(torch.cat([query,history,query-history,query*history],dim=-1)).squeeze(-1)
                scores=scores.masked_fill(~mask,-1e4)
                weights=scores.softmax(-1)*mask.float()
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
            features=torch.cat([features,(weights.unsqueeze(-1)*history).sum(1)],dim=-1)
        base_logits = self.output(features).squeeze(-1)
        residual = self.evidence(self.evidence_features(batch)).squeeze(-1)
        return base_logits + residual


    def evidence_counts(self, batch):
        """Return [batch, action=4, time=2, match=4] causal count statistics.

        Match channels are all events, category equality, brand equality, and
        equality of both fields in the SAME event. Zero IDs never match.
        Equality is in the host's encoded ID space, which may have collisions.
        """
        mask = batch['history_mask']
        category = batch['static'][:, self.category, None]
        brand = batch['static'][:, self.brand, None]
        cat_match = (batch['history_category'] == category) & (category > 0)
        brand_match = (batch['history_brand'] == brand) & (brand > 0)
        matches = torch.stack([
            torch.ones_like(mask), cat_match, brand_match,
            cat_match & brand_match,
        ], dim=-1).float()
        action_mask = torch.stack([
            mask & (batch['history_type'] == action) for action in (1, 2, 3, 4)
        ], dim=-1).float()
        age = batch['history_age_seconds'].clamp_min(0)
        recent = torch.exp(-age / 86400.0)
        # Fixed kernels are shared by all four cells; no learned age parameter.
        kernels = torch.cat([matches, matches * recent.unsqueeze(-1)], dim=-1)
        counts = torch.bmm(action_mask.transpose(1, 2), kernels)
        return counts.reshape(mask.shape[0], 4, 2, 4)

    def evidence_features(self, batch, counts=None):
        if counts is None:
            counts = self.evidence_counts(batch)
        if not self.config['target_match']:
            # Replace match evidence with generic activity, retaining all 32
            # slots, computation and readout parameters. Effective rank differs.
            counts = counts[:, :, :, :1].expand(-1, -1, -1, 4)
        if not self.config['separate_actions']:
            # Identity-free pooled totals, divided by four before broadcasting
            # to preserve total mass across the four nominal action slots.
            counts = counts.sum(1, keepdim=True).expand(-1, 4, -1, -1) / 4.0
        # Counts concern at most 128 supplied events; no unseen history access.
        # Fixed scaling is shared by all cells, with no validation fitting.
        return (torch.log1p(counts) / 4.859812404361672).reshape(counts.shape[0], 32)

    def diagnostics(self, batch):
        """Descriptive measurements on the host's first 512 validation rows.

        No labels, external state, random perturbations, or persistent caches.
        These small-prefix measurements are not population or causal estimates.
        """
        counts = self.evidence_counts(batch)
        features = self.evidence_features(batch, counts)
        residual = self.evidence(features).squeeze(-1)
        total = counts[:, :, 0, 0].sum(1)
        result = {
            'residual_mean': residual.mean(),
            'residual_abs_mean': residual.abs().mean(),
            'feature_l2_mean': features.norm(dim=-1).mean(),
            'empty_fraction': (total == 0).float().mean(),
            'supplied_event_count_mean': total.mean(),
            'category_match_count_mean': counts[:, :, 0, 1].sum(1).mean(),
            'brand_match_count_mean': counts[:, :, 0, 2].sum(1).mean(),
            'joint_match_count_mean': counts[:, :, 0, 3].sum(1).mean(),
            'recent_joint_count_mean': counts[:, :, 1, 3].sum(1).mean(),
        }
        for i, name in enumerate(('pv', 'cart', 'fav', 'buy')):
            result[name + '_count_mean'] = counts[:, i, 0, 0].mean()
            result[name + '_joint_count_mean'] = counts[:, i, 0, 3].mean()
        # Denominator clamps keep conditional summaries finite on empty slices.
        for name, subset in (
            ('empty', total == 0),
            ('category_present', counts[:, :, 0, 1].sum(1) > 0),
            ('category_absent', counts[:, :, 0, 1].sum(1) == 0),
        ):
            subset = subset.float()
            result[name + '_rows'] = subset.sum()
            result[name + '_residual_abs_mean'] = (
                (residual.abs() * subset).sum() / subset.sum().clamp_min(1)
            )
        return result
