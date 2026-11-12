import sys
import unittest
from pathlib import Path

import torch

from rec_harness.attention_width import (balanced_groups, merged_output, nmse,
                                         paired_interval, response_features, responses, spectrum)


class AttentionWidthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_equal_logit_neurons_merge_exactly_for_both_activations(self):
        q = torch.randn(32, 4, dtype=torch.float64)
        unique = torch.randn(4, 4, dtype=torch.float64)
        k = unique.repeat_interleave(4, 0)
        b = torch.randn(4, dtype=torch.float64).repeat_interleave(4)
        v = torch.randn(16, 7, dtype=torch.float64)
        groups = torch.arange(16).reshape(4, 4)
        for activation in ("silu", "softmax"):
            reference = responses(q, k, b, activation, 19) @ v
            actual = merged_output(q, k, v, b, groups, activation, 19)
            self.assertLess(nmse(actual, reference), 1e-24)

    def test_singleton_groups_are_identity_even_with_distinct_biases(self):
        q, k, v, b = torch.randn(12, 4), torch.randn(8, 4), torch.randn(8, 3), torch.randn(8)
        groups = torch.randperm(8)[:, None]
        for activation in ("silu", "softmax"):
            self.assertLess(nmse(merged_output(q, k, v, b, groups, activation),
                                 responses(q, k, b, activation) @ v), 1e-12)

    def test_changing_bias_can_break_key_only_equivalence(self):
        q = torch.tensor([[-2.], [.5], [3.]])
        k, b, v = torch.ones(2, 1), torch.tensor([-4., 4.]), torch.tensor([[1.], [-2.]])
        exact = responses(q, k, b) @ v
        self.assertGreater(nmse(merged_output(q, k, v, b, torch.tensor([[0, 1]])), exact), .01)

    def test_balanced_groups_cover_each_neuron_once_and_are_nested(self):
        groups = balanced_groups(torch.randn(64, 9), [4, 8, 64])
        for m, group in groups.items():
            self.assertEqual(tuple(group.shape), (m, 64 // m))
            self.assertEqual(group.flatten().sort().values.tolist(), list(range(64)))
        self.assertEqual(groups[8].reshape(4, -1).sort(1).values.tolist(),
                         groups[4].sort(1).values.tolist())

    def test_spectrum_distinguishes_constant_and_independent_responses(self):
        constant = torch.ones(128, 16)
        self.assertAlmostEqual(spectrum(constant)["effective_rank"], 1.0, places=5)
        self.assertEqual(spectrum(constant, centered=True)["energy_rank_99"], 0)
        independent = torch.eye(16)
        self.assertAlmostEqual(spectrum(independent)["effective_rank"], 16.0, places=5)
        self.assertTrue(spectrum(independent)["ceiling_reached"])

    def test_response_feature_projection_is_reproducible(self):
        calibration = torch.randn(64, 32)
        self.assertTrue(torch.equal(response_features(calibration, 42), response_features(calibration, 42)))

    def test_bootstrap_resamples_paired_differences(self):
        result = paired_interval([.125] * 16, repeats=200)
        self.assertEqual(result["low"], .125)
        self.assertEqual(result["high"], .125)

    def test_catalog_scores_preserve_seen_filter_and_tie_policy(self):
        from rec_harness.kuairand_attention_width import task_scores
        items = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 0.], [-1., 0.]])
        contexts = torch.tensor([[1., 0.], [0., 1.]])
        result = task_scores(contexts, items, torch.tensor([1]), 3, torch.tensor([2, 4]), .05)
        self.assertEqual([row["rank"] for row in result], [1, 2])
        repeated = task_scores(contexts, items, torch.tensor([1, 3]), 3, torch.tensor([2, 4]), .05)
        self.assertTrue(all(row["rank"] == 2501 and row["ndcg_at_10"] == 0 for row in repeated))

    def test_native_hstu_last_query_reconstruction_and_full_width_intervention(self):
        bench = Path(__file__).resolve().parents[1] / "artifacts/kuairand-a37797a/benchmarks/kuairand-longseq"
        if not bench.exists():
            self.skipTest("local pinned benchmark not available")
        sys.path.insert(0, str(bench))
        from longseq_bench.models import BenchmarkModel, ModelConfig
        from rec_harness.kuairand_attention_width import capture, replay_last
        cfg = ModelConfig(num_items=50, max_history=16, dim=16, blocks=2, heads=2,
                          dqk=4, dv=8, input_dropout=0, linear_dropout=0)
        model = BenchmarkModel(cfg).eval()
        ids = torch.randint(1, 50, (1, 16))
        times = torch.arange(17)[None] * 1000
        with torch.inference_mode():
            cache, output, _ = capture(model, ids, times)
            self.assertLess(nmse(output, model.encoder.encode_all(ids, times)[:, -1]), 1e-12)
            self.assertLess(nmse(replay_last(model, cache[0], None), output[0]), 1e-12)
            groups = {(layer, head): torch.arange(16)[:, None]
                      for layer in range(2) for head in range(2)}
            self.assertLess(nmse(replay_last(model, cache[0], groups), output[0]), 1e-12)


if __name__ == "__main__":
    unittest.main()
