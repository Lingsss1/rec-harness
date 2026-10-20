"""Model contracts accept alternative scoring and still reject invalid behavior."""
import torch
from torch import nn
import pytest

from mle_star_codex.worker import CHECK_VERSION, check


class Model(nn.Module):
    def __init__(self, kind='shared', defect=None):
        super().__init__()
        self.kind, self.defect = kind, defect
        self.item_emb = nn.Embedding(32, 8, padding_idx=0)
        self.output_emb = nn.Embedding(32, 8, padding_idx=0)
        self.projection = nn.Linear(8, 8, bias=False)

    def encode_sequence(self, seq):
        x = self.item_emb(seq).cumsum(1)
        if self.defect == 'causality':
            x = x + x[:, -1:]
        return x

    def targets(self, ids):
        emb = self.output_emb(ids) if self.kind == 'separate' else self.item_emb(ids)
        if self.kind == 'normalized':
            emb = nn.functional.normalize(emb, dim=-1)
        if self.kind == 'projected':
            emb = self.projection(emb)
        return emb

    def forward(self, seq, pos, neg):
        enc = self.encode_sequence(seq)
        p, n = [(enc * self.targets(ids)).sum(-1) for ids in (pos, neg)]
        if self.defect == 'positive_finite':
            p = p * float('nan')
        if self.defect == 'negative_finite':
            n = n * float('inf')
        if self.defect == 'positive_shape':
            p = p[:, :-1]
        if self.defect == 'negative_shape':
            n = n[:, :-1]
        if self.defect == 'role':
            p = p + 1
        if self.defect == 'training_batch':
            p, n = p + seq.shape[0], n + seq.shape[0]
        if self.defect == 'training_causality':
            p, n = p + seq[:, -1:].float(), n + seq[:, -1:].float()
        if self.defect == 'target_set':
            bias = (pos.float().sum() + neg.float().sum()) * 100
            p, n = p + bias, n + bias
        return p, n

    def score_candidates(self, seq, candidates):
        s = self.targets(candidates) @ self.encode_sequence(seq)[0, -1]
        if self.defect == 'candidate_order':
            s = s + 100 * torch.arange(len(candidates))
        if self.defect == 'candidate_set':
            s = s + 100 * len(candidates)
        if self.defect == 'public_finite':
            s = s * float('nan')
        return s


@pytest.mark.parametrize('kind', ['shared', 'separate', 'normalized', 'projected'])
def test_accepts_legitimate_target_parameterizations(kind):
    result = check(Model(kind), 8, 'cpu')
    assert result['check_version'] == CHECK_VERSION == 2
    assert result['training_scores_valid'] and result['training_target_role_independent']
    assert result['shared_embedding_training_scores_match'] == (kind == 'shared')


@pytest.mark.parametrize('defect,message', [
    ('positive_finite', 'positive training score shape/finiteness'),
    ('negative_finite', 'negative training score shape/finiteness'),
    ('positive_shape', 'positive training score shape/finiteness'),
    ('negative_shape', 'negative training score shape/finiteness'),
    ('causality', 'causality'), ('role', 'training target role dependence'),
    ('training_batch', 'training batch dependence'), ('public_finite', 'public scorer'),
    ('training_causality', 'training causality'), ('target_set', 'training target-set dependence'),
    ('candidate_order', 'candidate order dependence'), ('candidate_set', 'candidate set dependence'),
])
def test_rejects_invalid_behavior(defect, message):
    with pytest.raises(AssertionError, match=message):
        check(Model(defect=defect), 8, 'cpu')


def test_checks_do_not_require_internal_item_emb_attribute():
    model = Model('separate')
    # Explicit public methods can use arbitrary internal names.
    model.history = model.item_emb
    del model.item_emb
    model.encode_sequence = lambda seq: model.history(seq).cumsum(1)
    result = check(model, 8, 'cpu')
    assert result['shared_embedding_training_scores_match'] is None
    assert result['tail_readout_matches'] is None
