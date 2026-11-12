"""Compare forward/gradient results against frozen author classes on CPU.

Test-only dense/jagged adapters replace fbgemm layout conversions, not attention
math. These adapters are never imported by training code.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

from longseq_bench.models import ModelConfig, STUBlock, HSTUEncoder

REFERENCE = Path(__file__).resolve().parents[1] / "vendor/fuxi-linear"


@unittest.skipUnless(REFERENCE.exists() and importlib.util.find_spec("gin"), "Run scripts/fetch_sources.py and install test dependencies")
class ReferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        sys.path.insert(0, str(REFERENCE))
        def dense_to_jagged(dense, offsets):
            sizes = (offsets[0][1:] - offsets[0][:-1]).tolist()
            return (torch.cat([dense[i, :n] for i, n in enumerate(sizes)]),)
        def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
            length = max_lengths[0]
            bounds = offsets[0].tolist()
            return torch.stack([F.pad(values[a:b], (0, 0, 0, length - b + a), value=padding_value)
                                for a, b in zip(bounds, bounds[1:])])
        torch.ops.fbgemm.dense_to_jagged = dense_to_jagged
        torch.ops.fbgemm.jagged_to_padded_dense = jagged_to_padded_dense
        torch.ops.fbgemm.asynchronous_complete_cumsum = lambda lengths: torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        from generative_recommenders.modeling.sequential.hstu import SequentialTransductionUnitJagged, RelativeBucketedTimeAndPositionBasedBias
        cls.RefBlock, cls.RefBias = SequentialTransductionUnitJagged, RelativeBucketedTimeAndPositionBasedBias

    def test_hstu_forward_and_input_parameter_gradients(self):
        torch.manual_seed(7)
        cfg = ModelConfig(max_history=6, dim=8, heads=2, dqk=3, dv=5,
                          input_dropout=0, linear_dropout=0, query_chunk=2)
        ours = STUBlock(cfg)
        n = cfg.max_history + cfg.reference_output_padding
        bias = self.RefBias(max_seq_len=n, num_buckets=128,
                            bucketization_fn=lambda x: (torch.log(x.abs().clamp(min=1)) / 0.301).long())
        ref = self.RefBlock(embedding_dim=8, linear_hidden_dim=5, attention_dim=3,
                            dropout_ratio=0, attn_dropout_ratio=0, num_heads=2,
                            linear_activation="silu", relative_attention_bias_module=bias)
        with torch.no_grad():
            ref._uvqk.copy_(ours.uvqk)
            ref._o.load_state_dict(ours.output.state_dict())
            bias._ts_w.copy_(ours.bias.ts_weight)
            bias._pos_w.copy_(ours.bias.pos_weight)
        lengths = torch.tensor([6, 4])
        offsets = torch.tensor([0, 6, 10])
        mask = torch.arange(6)[None] < lengths[:, None]
        x = (torch.randn(2, 6, 8) * mask[..., None]).requires_grad_()
        xr = x.detach().clone().requires_grad_()
        timestamps = (torch.arange(n)[None].repeat(2, 1) * 1234).long()
        y = ours(x, timestamps)
        jagged = torch.cat([xr[b, :int(size)] for b, size in enumerate(lengths)])
        yr, _ = ref(jagged, offsets, timestamps, torch.ones(n, n).tril())
        actual = torch.cat([y[b, :int(size)] for b, size in enumerate(lengths)])
        torch.testing.assert_close(actual, yr, atol=2e-6, rtol=2e-5)
        actual.square().sum().backward()
        yr.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad, atol=5e-6, rtol=3e-5)
        torch.testing.assert_close(ours.uvqk.grad, ref._uvqk.grad, atol=2e-5, rtol=5e-5)
        torch.testing.assert_close(ours.bias.ts_weight.grad, bias._ts_w.grad, atol=2e-5, rtol=5e-5)

    def test_complete_hstu_encoder_and_parameter_count_equal_author(self):
        from generative_recommenders.modeling.sequential.hstu import HSTU
        from generative_recommenders.modeling.sequential.embedding_modules import LocalEmbeddingModule
        from generative_recommenders.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
        from generative_recommenders.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
        cfg = ModelConfig(num_items=24, max_history=6, dim=8, blocks=2, heads=2, dqk=3, dv=5,
                          input_dropout=0, linear_dropout=0, query_chunk=2, checkpoint_blocks=False)
        ours = HSTUEncoder(cfg).eval()
        n = cfg.max_history + cfg.reference_output_padding
        embedding = LocalEmbeddingModule(24, 8)
        preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(n, 8, 0)
        ref = HSTU(max_sequence_len=6, max_output_len=11, embedding_dim=8, num_blocks=2,
                   num_heads=2, linear_dim=5, attention_dim=3, normalization="rel_bias",
                   linear_config="uvqk", linear_activation="silu", linear_dropout_rate=0,
                   attn_dropout_rate=0, embedding_module=embedding, similarity_module=torch.nn.Identity(),
                   input_features_preproc_module=preprocessor,
                   output_postproc_module=L2NormEmbeddingPostprocessor(8, eps=1e-6), verbose=False).eval()
        with torch.no_grad():
            embedding._item_emb.weight.copy_(ours.item_embedding.weight)
            preprocessor._pos_emb.weight.copy_(ours.position_embedding.weight)
            for actual, expected in zip(ours.layers, ref._hstu._attention_layers):
                expected._uvqk.copy_(actual.uvqk)
                expected._o.load_state_dict(actual.output.state_dict())
                expected._rel_attn_bias._ts_w.copy_(actual.bias.ts_weight)
                expected._rel_attn_bias._pos_w.copy_(actual.bias.pos_weight)
        self.assertEqual(sum(p.numel() for p in ours.parameters()), sum(p.numel() for p in ref.parameters()))
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [4, 3, 2, 1, 0, 0]])
        lengths = torch.tensor([6, 4])
        times = torch.arange(n)[None].repeat(2, 1) * 1000
        full_ids = F.pad(ids, (0, n - 6))
        actual = ours.encode_all(ids, times)
        expected = ref(past_lengths=lengths, past_ids=full_ids,
                       past_embeddings=embedding.get_item_embeddings(full_ids),
                       past_payloads={"timestamps": times})
        valid = torch.arange(6)[None] < lengths[:, None]
        torch.testing.assert_close(actual[valid], expected[:, :6][valid], atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
