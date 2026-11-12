import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.core import ContractError, read_json, write_json
from rec_harness.kuairand_collocate import arm, owns_marker, publish, train_argv


class CollocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'baseline'; self.root.mkdir()
        write_json(self.root/'controller.json', {'status': 'running', 'stage': 'formal-training', 'config_hash': 'frozen'})
        for name in ('hstu', 'ultra_hstu'):
            write_json(self.root/name/'latest.json', {'epoch': 25})

    def test_handoff_does_not_touch_original_checkpoints_or_status(self):
        original = read_json(self.root/'controller.json')
        for name in ('hstu', 'ultra_hstu'): (self.root/name/'last.pt').write_bytes(b'unchanged')
        arm(self.root, 'owner')
        self.assertTrue(owns_marker(self.root, 'owner'))
        self.assertEqual(original, read_json(self.root/'controller.json'))
        for name in ('hstu', 'ultra_hstu'): self.assertEqual((self.root/name/'last.pt').read_bytes(), b'unchanged')

    def test_cannot_override_operator_stop_or_launch_duplicate(self):
        (self.root/'STOP').touch()
        with self.assertRaises(FileExistsError): arm(self.root, 'owner')
        self.assertFalse(owns_marker(self.root, 'owner'))
        (self.root/'STOP').unlink()
        write_json(self.root/'jobs/dare/attempt-0001.json', {'status': 'running', 'worker': {'pid': 100}})
        with patch('rec_harness.kuairand_collocate.session_live', return_value=True):
            with self.assertRaises(ContractError): arm(self.root, 'owner')
        self.assertFalse((self.root/'STOP').exists())

    def test_near_completion_is_not_raced(self):
        write_json(self.root/'hstu/latest.json', {'epoch': 99})
        with self.assertRaises(ContractError): arm(self.root, 'owner')
        self.assertFalse((self.root/'STOP').exists())

    def test_all_four_finals_required_for_join_and_operator_stop_preserved(self):
        cfg = {'output': str(self.root), 'benchmark': str(self.root/'benchmark'), 'python': '/usr/bin/python3'}
        arm(self.root, 'owner')
        with self.assertRaises(ContractError): publish(cfg, 'owner', {})
        for name in ('hstu', 'ultra_hstu', 'dare', 'ccformer'): write_json(self.root/name/'final.json', {'epoch': 100})
        with patch('rec_harness.kuairand_collocate.command') as run:
            publish(cfg, 'owner', {'shared_gpu_costs': True})
        self.assertEqual(read_json(self.root/'controller.json')['status'], 'completed')
        self.assertFalse((self.root/'STOP').exists())
        self.assertIn('--require-complete', run.call_args.args[0])
        (self.root/'STOP').touch()
        with patch('rec_harness.kuairand_collocate.command'): publish(cfg, 'owner', {})
        self.assertTrue((self.root/'STOP').exists())

    def test_added_commands_keep_pinned_trainer_and_original_output_resume(self):
        cfg = {'output': str(self.root), 'data_root': str(self.root/'data'), 'python': '/usr/bin/python3'}
        argv = train_argv(cfg, 'dare')
        self.assertIn('longseq_bench.train', argv); self.assertIn('--nproc-per-node=2', argv)
        self.assertNotIn('--smoke', argv); self.assertNotIn('--resume', argv)
        (self.root/'dare').mkdir(); (self.root/'dare/last.pt').write_bytes(b'checkpoint')
        self.assertEqual(train_argv(cfg, 'dare')[-2:], ['--resume', str(self.root/'dare/last.pt')])


if __name__ == '__main__': unittest.main()
