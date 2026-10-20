"""Verify the target transform and compatibility with D's original incumbent."""
from pathlib import Path

import pytest
import torch

from mle_star_codex.worker import check, load
from mle_star_codex.run import source_parts
from nextitnet_target_norm.model import NextItNetModel


def model(beta):
    torch.manual_seed(2024)
    return NextItNetModel(32, hidden_units=8, num_blocks=2, maxlen=8, target_norm_beta=beta)


@pytest.mark.parametrize('beta', [0., 1., .5, -.5])
def test_normalization_formula_padding_and_v2_contract(beta):
    m = model(beta)
    ids = torch.tensor([0, 1, 2, 3])
    raw = m.item_emb(ids)
    expected = raw if beta == 0 else raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8).pow(beta)
    assert torch.equal(m._target_embeddings(ids), expected)
    assert torch.equal(m._target_embeddings(ids)[0], torch.zeros_like(raw[0]))
    assert check(m, 8, 'cpu')['training_scores_valid']
    if beta == 1:
        torch.testing.assert_close(m._target_embeddings(ids[1:]).norm(dim=-1), torch.ones(3))


def test_norm_remains_in_gradient_graph():
    m = model(1.)
    with torch.no_grad():
        m.item_emb.weight[1] = torch.arange(1., 9.)
    transformed = m._target_embeddings(torch.tensor([1]))
    transformed[0, 0].backward()
    e = m.item_emb.weight[1].detach()
    norm = e.norm()
    expected = -e[0] * e / norm.pow(3)
    expected[0] += 1 / norm
    torch.testing.assert_close(m.item_emb.weight.grad[1], expected)


def test_history_encoder_and_parameter_set_do_not_depend_on_beta():
    original, normalized = model(0.), model(1.)
    original.eval(); normalized.eval()
    assert original.state_dict().keys() == normalized.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, normalized.state_dict()[key])
    seq = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])
    assert torch.equal(original.encode_sequence(seq), normalized.encode_sequence(seq))


def test_matches_original_d_model_when_beta_zero():
    legacy_path = Path(__file__).resolve().parents[1] / 'artifacts/staged-movies-2024-aligned44/study/trajectories/movies_nextitnet/D/1/artifacts/t023/model.py'
    torch.manual_seed(2024)
    _, original = load(legacy_path, 32, 'cpu')
    replacement = NextItNetModel(32, target_norm_beta=0.)
    replacement.load_state_dict(original.state_dict(), strict=True)
    original.eval(); replacement.eval()
    seq = torch.zeros((2, 128), dtype=torch.long)
    seq[:, -3:] = torch.tensor([[1, 2, 3], [2, 3, 4]])
    pos, neg = seq.roll(1, 1), seq.roll(2, 1)
    for a, b in zip(original(seq, pos, neg), replacement(seq, pos, neg)):
        assert torch.equal(a, b)
    candidates = torch.tensor([1, 2, 3, 4])
    assert torch.equal(original.score_candidates(seq[:1], candidates), replacement.score_candidates(seq[:1], candidates))


@pytest.mark.parametrize('beta', [float('nan'), float('inf'), -float('inf')])
def test_reject_nonfinite_exponents(beta):
    with pytest.raises(ValueError, match='finite'):
        model(beta)


def test_model_is_accepted_by_frozen_source_boundary():
    source_parts(Path(__file__).with_name('model.py').read_text())


def test_control_audit_accepts_small_float_drift_but_rejects_rank_changes():
    import numpy as np
    from nextitnet_target_norm.run import compare_control
    scores = np.asarray([[1., 0., 2.]], dtype=np.float32)
    assert compare_control(scores, scores + 4e-5)['all_target_ranks_identical']
    with pytest.raises(RuntimeError, match='rankings'):
        compare_control(scores, np.asarray([[0., 1., 2.]], dtype=np.float32))
