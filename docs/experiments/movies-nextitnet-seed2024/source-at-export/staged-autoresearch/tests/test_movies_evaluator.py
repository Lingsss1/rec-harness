"""The agent must receive the real worker exception, not just subprocess failure."""
import sys
import tempfile
import unittest
from pathlib import Path

from staged_research.movies_evaluator import run_logged


class EvaluatorLogTests(unittest.TestCase):
    def test_nested_worker_failure_is_preserved_in_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.log'
            with self.assertRaisesRegex(RuntimeError, 'AssertionError: synthetic causal violation'):
                run_logged([sys.executable, '-c',
                    "raise AssertionError('synthetic causal violation')"], path)
            self.assertIn('Traceback', path.read_text())

    def test_success_retains_worker_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prediction.log'
            run_logged([sys.executable, '-c', "print('synthetic success')"], path)
            self.assertEqual(path.read_text().strip(), 'synthetic success')


if __name__ == '__main__':
    unittest.main()
