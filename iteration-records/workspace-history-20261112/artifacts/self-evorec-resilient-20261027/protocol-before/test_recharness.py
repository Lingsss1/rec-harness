import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .recharness import StateStore, native, matched_group_cost
from gagc.state import ArmState, ThompsonState


class NativeAlignmentTests(unittest.TestCase):
    def test_exact_worker_cost_excludes_zero_gpu_codegen_failures(self):
        self.assertEqual(matched_group_cost([{'wall_time_secs': 0, 'group_wall_time_secs': 500}])[0], 0)
        self.assertAlmostEqual(matched_group_cost([{'wall_time_secs': 7.5}, {'wall_time_secs': 12.5}])[0], 20.)

    def test_native_state_charges_baseline_once_and_persists_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / 'state.json', ThompsonState(
                arms={a: ArmState(a) for a in native._active_arms()}, global_budget=1000))
            fixture = [{'spec': {'dimension': 'baseline', 'arm': 'baseline', 'is_baseline': True},
                        'val_score': .1, 'val_metrics': {}, 'wall_time_secs': 7.5,
                        'timed_out': False, 'oom': False, 'error_message': None,
                        'trial_group_id': 'unit-fixture-baseline'}]
            with patch.object(native, '_STORE', store), patch.object(native, '_LOGS_ROOT', tmp), \
                 patch.object(native, '_trial_group_wall_time_secs', matched_group_cost):
                native.update_thompson_state(json.dumps(fixture), trial_group_id='unit-fixture-baseline')
                state = store.state()
                self.assertTrue(state.baseline_done)
                self.assertEqual(state.global_budget, 992.5)
                self.assertTrue(state.experiment_memory)
                native.update_thompson_state(json.dumps(fixture), trial_group_id='unit-fixture-baseline')
                self.assertEqual(store.state().global_budget, 992.5)
                self.assertEqual(store.state().round_idx, 1)


if __name__ == '__main__':
    unittest.main()
