import unittest

from rec_harness.adapters.amazon_archive import account
from rec_harness.core import ContractError


class ArchiveLedgerTests(unittest.TestCase):
    def events(self):
        return [{'seq': 1, 'kind': 'budget_reserved', 'payload': {'task_id': 'prior', 'amount': 8}},
                {'seq': 2, 'kind': 'budget_settled', 'payload': {'task_id': 'prior', 'actual': 8, 'status': 'retained'}},
                {'seq': 3, 'kind': 'budget_reserved', 'payload': {'task_id': 'job', 'amount': 10}},
                {'seq': 4, 'kind': 'budget_settled', 'payload': {'task_id': 'job', 'actual': 3, 'status': 'success'}},
                {'seq': 5, 'kind': 'budget_reserved', 'payload': {'task_id': 'unknown', 'amount': 10}},
                {'seq': 6, 'kind': 'budget_settled', 'payload': {'task_id': 'unknown', 'actual': None, 'status': 'interrupted'}}]

    def receipt(self):
        return {'job_id': 'job', 'status': 'success', 'allocated_gpu_seconds': 3}

    def test_retains_predecessor_and_unknown_cost_without_success_receipts(self):
        ledger = account(self.events(), [self.receipt()])
        self.assertEqual(sum(r['charged_or_reserved'] for r in ledger), 21)
        self.assertEqual([r['task_id'] for r in ledger if r['receipt_in_archive']], ['job'])
        self.assertIsNone(ledger[-1]['actual'])

    def test_cost_status_and_duplicate_receipt_fail_closed(self):
        for receipt in [{**self.receipt(), 'allocated_gpu_seconds': 0},
                        {**self.receipt(), 'status': 'failure'},
                        {**self.receipt(), 'task_id': 'not-reserved'}]:
            with self.assertRaises(ContractError):
                account(self.events(), [receipt])
        with self.assertRaises(ContractError):
            account(self.events(), [self.receipt(), self.receipt()])

    def test_settlement_without_reservation_fails(self):
        with self.assertRaises(ContractError):
            account(self.events()[1:], [])
