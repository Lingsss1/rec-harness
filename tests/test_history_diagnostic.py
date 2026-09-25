import unittest

from rec_harness.adapters.history_diagnostic import summarize
from rec_harness.core import ContractError


class HistoryDiagnosticTests(unittest.TestCase):
    def test_boundary_bins_empty_groups_and_hit_conservation(self):
        left = {'user_ids': [1, 2, 3, 4, 5], 'ranks': [12, 1, 8, 99, 0]}
        right = {'user_ids': [1, 2, 3, 4, 5], 'ranks': [1, 12, 3, 99, 0]}
        rows = summarize(left, right, {1: 5, 2: 6, 3: 128, 4: 129, 5: 1})
        self.assertEqual([r['users'] for r in rows], [2, 1, 0, 1, 1])
        self.assertIsNone(rows[2]['delta']['HR@10'])
        self.assertEqual(rows[0]['hit_wins'], 1)
        self.assertEqual(rows[1]['hit_losses'], 1)
        for row in rows:
            n = row['users']
            self.assertEqual(n, row['hit_wins'] + row['hit_losses'] + row['hit_unchanged'])
            if n:
                self.assertAlmostEqual(row['delta']['HR@10'], (row['hit_wins'] - row['hit_losses']) / n)

    def test_alignment_missing_history_and_bad_rank_rejected(self):
        left = {'user_ids': [1], 'ranks': [10]}
        for right, lengths in [({'user_ids': [2], 'ranks': [1]}, {1: 5}),
                               (left, {}), ({'user_ids': [1], 'ranks': [100]}, {1: 5})]:
            with self.assertRaises(ContractError):
                summarize(left, right, lengths)
