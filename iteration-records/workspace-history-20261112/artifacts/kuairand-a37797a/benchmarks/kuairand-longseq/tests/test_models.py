import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

from longseq_bench.models import BenchmarkModel, ModelConfig, attention_mask
from longseq_bench.data import AlignedDataset, make_cache
from longseq_bench.evaluation import evaluate
from longseq_bench.train import smoke_input
from torch.utils.data import DataLoader


def small_config(name):
    return ModelConfig(name=name, num_items=24, max_history=16, dim=8, blocks=4,
                       heads=2, dqk=4, dv=8, input_dropout=0, linear_dropout=0,
                       query_chunk=4, checkpoint_blocks=True, sla_local=4,
                       sla_global=2, truncate_after=3, truncate_keep=8, dare_top_k=3,
                       cc_group_size=4, cc_channel_size=2)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
                            [2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        times = torch.arange(17)[None].repeat(2, 1) * 1000
        labels = (ids + 1) * ids.ne(0)
        return dict(ids=ids, times=times, labels=labels, length=torch.tensor([12, 5]), target=torch.tensor([13, 7]))

    def test_all_models_backward_is_finite(self):
        for name in ("hstu", "ultra_hstu", "dare", "ccformer"):
            with self.subTest(name=name):
                torch.manual_seed(42)
                model = BenchmarkModel(small_config(name), negatives=4, loss_chunk=5)
                loss = model(self.batch())
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_transformer_future_items_do_not_change_prefix(self):
        for name in ("hstu", "ultra_hstu"):
            model = BenchmarkModel(small_config(name)).eval()
            batch = self.batch()
            baseline = model.encoder.encode_all(batch["ids"], batch["times"])
            changed = batch["ids"].clone()
            changed[:, 4:] = torch.where(changed[:, 4:] > 0, 20, 0)
            altered = model.encoder.encode_all(changed, batch["times"])
            torch.testing.assert_close(baseline[:, :4], altered[:, :4], atol=1e-6, rtol=1e-5)

    def test_ultra_all_position_training_equals_physical_prefix_truncation(self):
        model = BenchmarkModel(small_config("ultra_hstu")).eval()
        batch = self.batch()
        full = model.encoder.encode_all(batch["ids"], batch["times"])
        current = model.encoder.encode_current(batch["ids"], batch["times"], batch["length"])
        expected = full[torch.arange(2), batch["length"] - 1]
        torch.testing.assert_close(current, expected, atol=1e-6, rtol=1e-5)
        # Check a non-final supervised prefix too, with its own moving cutoff.
        shortened = batch["ids"].clone()
        shortened[0, 9:] = 0
        shortened[1, 3:] = 0
        shorter = model.encoder.encode_current(shortened, batch["times"], torch.tensor([9, 3]))
        torch.testing.assert_close(shorter, full[torch.arange(2), torch.tensor([8, 2])], atol=1e-6, rtol=1e-5)

    def test_dare_is_candidate_conditioned_and_causal(self):
        encoder = BenchmarkModel(small_config("dare")).eval().encoder
        batch = self.batch()
        rows, positions = torch.tensor([0, 1]), torch.tensor([2, 2])
        candidates = torch.tensor([[4, 8, 12], [3, 7, 11]])
        original = encoder.score_queries(encoder.encode_all(batch["ids"], batch["times"]),
                                         batch["ids"], rows, positions, candidates)
        changed = batch["ids"].clone()
        changed[:, 3:] = torch.where(changed[:, 3:] > 0, 23, 0)
        other = encoder.score_queries(encoder.encode_all(changed, batch["times"]), changed, rows, positions, candidates)
        torch.testing.assert_close(original, other)
        self.assertGreater(float((original[:, 0] - original[:, 1]).abs().max()), 1e-7)
        self.assertIsNot(encoder.attention_embedding.weight, encoder.representation_embedding.weight)

    def test_short_dare_prefix_has_no_nan_when_k_exceeds_history(self):
        model = BenchmarkModel(small_config("dare")).eval()
        batch = self.batch()
        scores = model.encoder.score_queries(model.encoder.encode_all(batch["ids"], batch["times"]),
                                             batch["ids"], torch.tensor([0]), torch.tensor([0]), torch.tensor([[1, 4]]))
        self.assertTrue(torch.isfinite(scores).all())

    def test_sla_retrieval_mask(self):
        q, k = torch.tensor([9]), torch.arange(12)
        actual = attention_mask(q, k, local=2, global_size=2)[0].nonzero().flatten().tolist()
        self.assertEqual(actual, [0, 1, 8, 9])
        truncated = attention_mask(q, k, local=2, global_size=2, tail=5)[0].nonzero().flatten().tolist()
        self.assertEqual(truncated, [5, 6, 8, 9])

    def test_collision_only_catalog_has_zero_loss(self):
        for name in ("hstu", "ultra_hstu", "dare", "ccformer"):
            cfg = small_config(name)
            cfg.num_items = 1
            model = BenchmarkModel(cfg, negatives=4)
            batch = self.batch()
            batch["ids"] = batch["ids"].ne(0).long()
            batch["labels"] = batch["labels"].ne(0).long()
            self.assertAlmostEqual(float(model(batch)), 0.0, places=6)

    def test_exhaustive_metrics_are_candidate_chunk_invariant(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = smoke_input(Path(folder))
            loader = DataLoader(AlignedDataset(cache, "eval"), batch_size=3)
            for name in ("hstu", "ultra_hstu", "dare", "ccformer"):
                torch.manual_seed(42)
                model = BenchmarkModel(small_config(name)).eval()
                a = evaluate(model, loader, torch.device("cpu"), 24, candidate_chunk=24)
                b = evaluate(model, loader, torch.device("cpu"), 24, candidate_chunk=5)
                for metric in ("hr@10", "hr@50", "ndcg@10", "ndcg@50", "mrr", "mrr_exact"):
                    self.assertAlmostEqual(a[metric], b[metric], places=6, msg=name + metric)
                self.assertEqual(a["users"], 12)

    def test_reference_mrr_cap_and_seen_target_match_known_ranks(self):
        class ConstantEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.item_embedding = torch.nn.Embedding(2601, 1)
                with torch.no_grad():
                    self.item_embedding.weight.fill_(1)
            def encode_current(self, ids, times, lengths):
                return torch.ones(len(ids), 1)
        model = torch.nn.Module()
        model.cfg = SimpleNamespace(name="hstu")
        model.encoder = ConstantEncoder()
        batch = {"ids": torch.tensor([[1], [1]]), "times": torch.tensor([[1, 2], [1, 2]]),
                 "length": torch.tensor([1, 1]), "target": torch.tensor([2600, 1])}
        metrics = evaluate(model, [batch], torch.device("cpu"), 2600, candidate_chunk=211)
        self.assertAlmostEqual(metrics["mrr"], 1 / 2501, places=7)
        self.assertAlmostEqual(metrics["mrr_exact"], 1 / 2599 / 2, places=7)
        self.assertEqual(metrics["repeated_target_filtered_users"], 1)


if __name__ == "__main__":
    unittest.main()
