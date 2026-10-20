"""Causality and exact candidate scoring for the independent CCFormer adapter."""
from dataclasses import replace
import unittest

import torch

from longseq_bench.models import BenchmarkModel, ModelConfig


def config(**changes):
    base = ModelConfig(name="ccformer", num_items=32, max_history=32, dim=8, heads=2,
                       blocks=4, input_dropout=0, linear_dropout=0, query_chunk=4,
                       cc_group_size=4, cc_channel_size=2)
    return replace(base, **changes)


class CCFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_shared_full_sequence_scores_equal_physical_prefixes(self):
        encoder = BenchmarkModel(config()).eval().encoder
        ids = torch.arange(1, 20)[None]
        times = torch.arange(20)[None] * 1000
        candidates = torch.tensor([[25, 26, 27]])
        with torch.no_grad():
            context = encoder.encode_all(ids, times)
            for cutoff in (1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 19):
                rows, positions = torch.tensor([0]), torch.tensor([cutoff - 1])
                expected = encoder.score_queries(context, ids, rows, positions, candidates)
                prefix_ids, prefix_times = ids[:, :cutoff], times[:, :cutoff + 1]
                actual = encoder.score_queries(encoder.encode_all(prefix_ids, prefix_times),
                                               prefix_ids, rows, positions, candidates)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_future_items_and_timestamps_cannot_change_prefix_or_receive_gradients(self):
        encoder = BenchmarkModel(config()).eval().encoder
        ids = torch.arange(1, 17)[None]
        times = torch.arange(17)[None] * 1000
        rows, positions, candidates = torch.tensor([0]), torch.tensor([4]), torch.tensor([[25, 26]])
        original = encoder.score_queries(encoder.encode_all(ids, times), ids, rows, positions, candidates)
        original.sum().backward()
        self.assertEqual(float(encoder.item_embedding.weight.grad[6:17].abs().max()), 0.0)
        changed_ids, changed_times = ids.clone(), times.clone()
        changed_ids[:, 5:] = 30
        # Includes the next target timestamp; this adapter consumes history time only.
        changed_times[:, 5:] = 10 ** 12 + torch.arange(12)[None] * 10 ** 6
        changed = encoder.score_queries(encoder.encode_all(changed_ids, changed_times),
                                         changed_ids, rows, positions, candidates)
        torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-5)
        changed_ids[:, 0] = 24
        different_past = encoder.score_queries(encoder.encode_all(changed_ids, changed_times),
                                                changed_ids, rows, positions, candidates)
        self.assertGreater(float((original - different_past).abs().max()), 1e-6)

    def test_candidates_are_independent_in_forward_and_backward(self):
        encoder = BenchmarkModel(config()).eval().encoder
        ids = torch.arange(1, 12)[None]
        times = torch.arange(12)[None] * 1000
        rows, positions = torch.tensor([0]), torch.tensor([10])
        candidates = torch.tensor([[22, 23, 24, 25]])
        context = encoder.encode_all(ids, times)
        scores = encoder.score_queries(context, ids, rows, positions, candidates)
        scores[:, 0].sum().backward()
        self.assertEqual(float(encoder.item_embedding.weight.grad[23:26].abs().max()), 0.0)
        with torch.no_grad():
            separated = torch.cat([encoder.score_queries(context, ids, rows, positions, candidates[:, i:i + 1])
                                   for i in range(4)], dim=1)
            torch.testing.assert_close(scores, separated, atol=2e-6, rtol=2e-5)
            changed = candidates.clone()
            changed[:, 1:] = 31
            altered = encoder.score_queries(context, ids, rows, positions, changed)
            torch.testing.assert_close(scores[:, 0], altered[:, 0], atol=1e-6, rtol=1e-5)
            self.assertGreater(float((scores[:, 0] - scores[:, 1]).abs().max()), 1e-6)

    def test_checkpointed_and_plain_loss_gradients_agree(self):
        checked = BenchmarkModel(config(input_dropout=.2, linear_dropout=.2), negatives=3, loss_chunk=3)
        plain = BenchmarkModel(config(input_dropout=.2, linear_dropout=.2, checkpoint_blocks=False),
                               negatives=3, loss_chunk=3)
        plain.load_state_dict(checked.state_dict())
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
        batch = {"ids": ids, "times": torch.arange(9)[None] * 1000, "labels": (ids + 1) * ids.ne(0)}
        torch.manual_seed(123)
        loss = checked(batch)
        loss.backward()
        torch.manual_seed(123)
        other = plain(batch)
        other.backward()
        torch.testing.assert_close(loss, other)
        for (name, a), (_, b) in zip(checked.named_parameters(), plain.named_parameters()):
            self.assertIsNotNone(a.grad, name)
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)

    def test_relative_time_changes_history_and_temporal_gradients_stay_finite(self):
        encoder = BenchmarkModel(config()).encoder
        ids = torch.arange(1, 10)[None]
        times = torch.arange(10)[None] * 1000
        first = encoder.encode_all(ids, times)
        second = encoder.encode_all(ids, torch.zeros_like(times))
        # Under stride-2 endpoint anchoring a nine-event prefix has one final
        # coarse token at position zero; later-time effects enter deeper keys
        # and user/target states instead of that single final pooled token.
        self.assertGreater(float((first[4] - second[4]).abs().max()), 1e-6)
        score = encoder.score_queries(first, ids, torch.tensor([0]), torch.tensor([8]), torch.tensor([[25, 26]]))
        score.sum().backward()
        for layer in encoder.layers:
            for name in ("raw_alpha", "raw_decay", "raw_gamma"):
                self.assertTrue(torch.isfinite(getattr(layer.mixer, name).grad).all())

    def test_real_1024_history_geometry_scores_without_truncation(self):
        # Real backbone dimensions and length, with a small catalog for this CPU check.
        cfg = ModelConfig(name="ccformer", num_items=32, input_dropout=0, linear_dropout=0)
        encoder = BenchmarkModel(cfg).eval().encoder
        ids = (torch.arange(1024) % 24 + 1)[None]
        times = torch.arange(1025)[None] * 1000
        with torch.no_grad():
            context = encoder.encode_all(ids, times)
            self.assertEqual([context[4 * i].shape[1] for i in range(4)], [1024, 512, 256, 128])
            scores = encoder.score_queries(context, ids, torch.tensor([0]), torch.tensor([1023]),
                                             torch.tensor([[25, 26, 27]]))
            self.assertEqual(scores.shape, (1, 3))
            self.assertTrue(torch.isfinite(scores).all())

    def test_invalid_grouping_or_compression_is_rejected(self):
        for changes in ({"cc_channel_size": 3}, {"heads": 3}, {"cc_stride": 0},
                        {"cc_kernel_size": 0}, {"cc_time_unit_ms": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)


if __name__ == "__main__":
    unittest.main()
