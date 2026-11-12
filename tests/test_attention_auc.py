import unittest

import torch

from rec_harness.kuairand_attention_auc import (bootstrap_multiplicities, pooled_bootstrap,
                                                roc_auc, user_pair_wins, within_user_auc)


class AttentionAUCTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_perfect_reversed_and_tied_predictions(self):
        self.assertEqual(roc_auc(torch.tensor([2., 3.]), torch.tensor([0., 1.])), 1.)
        self.assertEqual(roc_auc(torch.tensor([0., 1.]), torch.tensor([2., 3.])), 0.)
        self.assertEqual(roc_auc(torch.ones(2), torch.ones(5)), .5)

    def test_tie_credit_and_positive_collision_mask(self):
        positives = torch.tensor([.5, .5])
        negatives = torch.tensor([[.1, .5, .9, .5], [.1, .5, .9, .5]])
        valid = torch.tensor([True, True, True, False])
        self.assertTrue(torch.equal(within_user_auc(positives, negatives, valid), torch.tensor([.5, .5], dtype=torch.float64)))

    def test_global_and_within_user_auc_have_different_comparisons(self):
        positive, negative = torch.tensor([10., 1.]), torch.tensor([[9.], [0.]])
        valid = torch.ones(2, 1, dtype=torch.bool)
        self.assertEqual(roc_auc(positive, negative.flatten()), .75)
        wins = user_pair_wins(positive, negative, valid)
        self.assertEqual(float(wins.sum() / 4), .75)
        per_user = (positive[:, None] > negative).double().mean(-1)
        self.assertEqual(float(per_user.mean()), 1.)

    def test_pair_matrix_matches_sorted_auc_with_ties_and_unequal_cluster_sizes(self):
        positive = torch.tensor([2., 1., 1.])
        negative = torch.tensor([[0., 2., 4.], [1., 0., 3.], [2., 1., 0.]])
        valid = torch.tensor([[True, True, False], [True, False, True], [True, True, True]])
        wins = user_pair_wins(positive, negative, valid, chunk=2)
        self.assertAlmostEqual(roc_auc(positive, negative[valid]), float(wins.sum() / (3 * valid.sum())))

    def test_cluster_bootstrap_matches_explicit_user_resampling(self):
        positive = torch.tensor([2., 1., 1.])
        negative = torch.tensor([[0., 2.], [1., 3.], [2., 0.]])
        valid = torch.tensor([[True, False], [True, True], [True, True]])
        wins = user_pair_wins(positive, negative, valid)
        multiplicity = torch.tensor([[2., 0., 1.]], dtype=torch.float64)
        boot = pooled_bootstrap(wins, valid.sum(-1), multiplicity)
        selected = torch.tensor([0, 0, 2])
        exact = roc_auc(positive[selected], negative[selected][valid[selected]])
        self.assertAlmostEqual(float(boot[0]), exact)
        counts = bootstrap_multiplicities(3, repeats=20)
        self.assertTrue(torch.equal(counts.sum(-1), torch.full((20,), 3., dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
