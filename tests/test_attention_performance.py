import sys
import unittest
from pathlib import Path

import torch

from rec_harness.attention_width import balanced_groups, nmse, response_features, responses
from rec_harness.kuairand_attention_performance import (compress_memory, native_memory,
                                                       read_memory, tensor_bytes)


class AttentionPerformanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        bench = Path(__file__).resolve().parents[1] / "artifacts/kuairand-a37797a/benchmarks/kuairand-longseq"
        if not bench.exists(): self.skipTest("pinned local benchmark unavailable")
        sys.path.insert(0, str(bench))
        from longseq_bench.models import BenchmarkModel, ModelConfig
        cfg = ModelConfig(num_items=50, max_history=16, dim=16, blocks=2, heads=2,
                          dqk=4, dv=8, input_dropout=0, linear_dropout=0, query_chunk=4)
        self.model = BenchmarkModel(cfg).eval()
        self.ids = torch.randint(1, 50, (1, 16))
        self.times = torch.arange(17)[None] * 1000

    def test_native_prefix_and_raw_cached_read_match_original(self):
        with torch.inference_mode():
            memory = native_memory(self.model, self.ids, self.times)
            reference = self.model.encoder.encode_all(self.ids, self.times)[0, -1]
            self.assertLess(nmse(memory["native_context"], reference), 1e-12)
            self.assertLess(nmse(read_memory(self.model, memory), reference), 1e-12)
            self.assertLess(nmse(native_memory(self.model, self.ids, self.times, retain_cache=False)["native_context"], reference), 1e-12)

    def test_compressed_self_correction_matches_original_intervention(self):
        from rec_harness.kuairand_attention_width import capture, replay_last
        bank = torch.randn(2, 32, 2, 4)
        seed, budget = 81, 4
        with torch.inference_mode():
            memory = native_memory(self.model, self.ids, self.times)
            compressed = compress_memory(memory, bank, budget, seed)
            record, _, _ = capture(self.model, self.ids, self.times)
            assignments = {}
            for li, layer in enumerate(self.model.encoder.layers):
                snapshot = record[0]["layers"][li]
                for hi in range(2):
                    h = responses(bank[li, :, hi], snapshot["key"][:, hi], snapshot["bias"], scale=layer.bias.padded_length)
                    assignments[(li, hi)] = balanced_groups(response_features(h, seed + li * 4 + hi + 16), [budget])[budget]
            reference = replay_last(self.model, record[0], assignments)
            self.assertLess(nmse(read_memory(self.model, compressed), reference), 1e-11)

    def test_singletons_preserve_output_and_compact_storage_does_not_keep_full_kv(self):
        bank = torch.randn(2, 32, 2, 4)
        with torch.inference_mode():
            raw = native_memory(self.model, self.ids, self.times)
            singleton = compress_memory(raw, bank, 16, 81)
            compact = compress_memory(raw, bank, 2, 81)
            self.assertLess(nmse(read_memory(self.model, singleton), read_memory(self.model, raw)), 1e-11)
            self.assertLess(tensor_bytes(compact), tensor_bytes(raw))
            raw_pointers = {layer[key].untyped_storage().data_ptr() for layer in raw["layers"] for key in ["key", "value"]}
            compact_pointers = {layer[key].untyped_storage().data_ptr() for layer in compact["layers"] for key in ["key", "value", "old_key", "old_value"]}
            self.assertFalse(raw_pointers & compact_pointers)

    def test_storage_count_does_not_double_count_views(self):
        tensor = torch.zeros(100)
        self.assertEqual(tensor_bytes(dict(original=tensor, view=tensor[:3])), 400)

    def test_full_catalog_view_scoring_matches_explicit_candidate_scoring(self):
        from rec_harness.kuairand_attention_performance import top_items
        context, items = torch.randn(8), torch.randn(21, 8)
        candidates = torch.arange(1, 21)
        seen = torch.zeros(21, dtype=torch.bool)
        seen[torch.tensor([2, 5, 17])] = True
        reference = (items[candidates] @ context).masked_fill(seen[candidates], -torch.inf).topk(10).indices
        self.assertTrue(torch.equal(top_items(context, items, candidates, seen), reference))


if __name__ == "__main__":
    unittest.main()
