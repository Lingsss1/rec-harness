"""Frozen validation contract checks; optional PyTorch tests run in training env."""
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from rec_harness.adapters.amazon_worker import evaluation_rows, sample_batch, evaluate, correctness
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch environment required')
class AmazonContractTests(unittest.TestCase):
    def test_validation_candidates_independent_of_training_rng(self):
        train = {1: [1, 2, 3], 2: [3, 4, 5]}
        valid = {1: [6], 2: [7]}
        before = evaluation_rows(train, valid, 300, 2, 42)
        for _ in range(1000): random.random()
        after = evaluation_rows(train, valid, 300, 2, 42)
        self.assertEqual(before, after)
        for row in before:
            self.assertEqual(len(set(row['candidates'])), 100)
            self.assertFalse(set(row['history']) & set(row['candidates'][1:]))

    def test_sampler_alignment_and_negative_exclusion(self):
        train = {1: [1, 2, 3, 4]}
        seq, pos, neg = sample_batch(train, [1], 200, 2, 5, random.Random(42))
        self.assertEqual(seq[0].tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(pos[0].tolist(), [0, 0, 2, 3, 4])
        self.assertTrue(all(v not in train[1] for v in neg[:, -3:].flatten()))

    def test_insufficient_negatives_fail_instead_of_hanging(self):
        with self.assertRaises(ValueError):
            evaluation_rows({1: [1, 2]}, {1: [3]}, 50, 1, 42)
        with self.assertRaises(ValueError):
            sample_batch({1: [1, 2]}, [1], 2, 1, 4, random.Random(1))

    def test_frozen_batched_readout_and_causal_checks(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.item_emb = torch.nn.Embedding(301, 4, padding_idx=0)
            def encode_sequence(self, seq):
                return self.item_emb(seq)
            def score_candidates(self, seq, candidates):
                return self.item_emb(candidates) @ self.encode_sequence(seq)[0, -1]
        model = Model()
        self.assertTrue(correctness(model, 8, 300, 'cpu')['tail_readout'])
        rows = evaluation_rows({1: [1, 2, 3], 2: [4, 5]}, {1: [6], 2: [7]}, 300, 2, 42)
        first, ranks = evaluate(model, rows, 8, 'cpu', 1)
        second, ranks2 = evaluate(model, rows, 8, 'cpu', 2)
        self.assertEqual(first, second); self.assertEqual(ranks, ranks2)
        model.score_candidates = lambda seq, candidates: model.item_emb(candidates) @ model.encode_sequence(seq)[0, 0]
        with self.assertRaisesRegex(ValueError, 'right-aligned'):
            correctness(model, 8, 300, 'cpu')


if __name__ == '__main__':
    unittest.main()
