import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from .core import Ledger, BudgetExhausted, read, strict_schema
from .mlestar import validate_block


class AccountingTests(unittest.TestCase):
    def test_pydantic_references_are_closed_in_strict_transport(self):
        original = {'type': 'object', 'properties': {'idea': {'$ref': '#/$defs/Idea'}},
                    'required': ['idea'], '$defs': {'Idea': {'type': 'object',
                    'properties': {'text': {'type': 'string'}, 'note': {'type': 'string', 'default': ''}},
                    'required': ['text']}}}
        result = strict_schema(original)
        self.assertFalse(result['additionalProperties'])
        self.assertFalse(result['$defs']['Idea']['additionalProperties'])
        self.assertEqual(result['$defs']['Idea']['required'], ['text', 'note'])
        self.assertIn({'type': 'null'}, result['$defs']['Idea']['properties']['note']['anyOf'])
        self.assertNotIn('additionalProperties', original)

    def test_parallel_reservations_cannot_overcommit_and_failures_are_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(tmp, 100)
            def reserve(i):
                try:
                    return i, ledger.reserve(str(i), 100, minimum=1, fair_slots=4)
                except BudgetExhausted:
                    return i, None
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = list(pool.map(reserve, range(4)))
            self.assertEqual(len([j for j in jobs if j[1] is not None]), 4)
            self.assertLessEqual(sum(read(ledger.path)['reservations'].values()), 100)
            for i, timeout in jobs:
                ledger.settle(str(i), timeout)
            state = read(ledger.path)
            self.assertEqual(state['reservations'], {})
            self.assertEqual(state['spent_gpu_seconds'], sum(t for _, t in jobs))
            with self.assertRaises(BudgetExhausted):
                ledger.reserve('cannot-fit', 100, minimum=20)

    def test_refinement_cannot_modify_another_function_or_data_path(self):
        before = 'LR = .001\nDATA_DIR = "train"\ndef model():\n    return 1\ndef evaluate():\n    return 2\n'
        validate_block(before, before.replace('return 1', 'return 3'), 'model')
        with self.assertRaises(ValueError):
            validate_block(before, before.replace('return 2', 'return 3'), 'model')
        validate_block(before, before.replace('.001', '.002'), 'hyperparameters')
        with self.assertRaises(ValueError):
            validate_block(before, before.replace('"train"', '"test"'), 'hyperparameters')


if __name__ == '__main__':
    unittest.main()
