import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from rec_harness.adapters.amazon_study import recover_study, reserve_attempt
from rec_harness.core import ContractError, read_json, write_json
from rec_harness.process_identity import assert_stopped, identity, session_live
from rec_harness.store import Store


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'memory')
        self.addCleanup(self.store.close)
        self.store.configure_budget(100)

    def test_orphan_session_blocks_recovery_and_preserves_reservation(self):
        if identity(os.getpid()).get('observable') is False:
            self.skipTest('requires /proc in the current PID namespace')
        worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            record = identity(worker.pid)
            directory = self.root / 'jobs/a/process'
            write_json(directory / 'launch.json', {'worker': record})
            self.store.reserve('a', 20)
            with self.assertRaisesRegex(ContractError, 'still live'):
                recover_study(self.root, self.store)
            self.assertEqual(self.store.db.execute('SELECT status FROM budget').fetchone()[0], 'running')
            # Recycled PID with a different start time must not count as this worker.
            self.assertFalse(session_live({**record, 'start_ticks': record['start_ticks'] - 1}))
        finally:
            worker.kill(); worker.wait()
        recover_study(self.root, self.store)
        self.assertEqual(self.store.budget()['charged_or_reserved'], 20)
        self.assertEqual(self.store.db.execute('SELECT status FROM budget').fetchone()[0], 'interrupted')

    def test_receipt_recovery_is_idempotent_and_retry_preserves_files(self):
        self.store.reserve('a', 20)
        directory = self.root / 'jobs/a'
        write_json(directory / 'completed.json', {'job_id': 'a', 'status': 'crash', 'allocated_gpu_seconds': 3})
        recover_study(self.root, self.store)
        recover_study(self.root, self.store)
        self.assertEqual(self.store.budget()['charged_or_reserved'], 3)
        retry = reserve_attempt(self.root, self.store, 'a', 20)
        self.assertEqual(retry, 'a-retry-2')
        self.assertFalse(directory.exists())
        self.assertEqual(read_json(self.root / 'attempt-history/a/before-a-retry-2/completed.json')['status'], 'crash')
        self.assertEqual(self.store.budget()['charged_or_reserved'], 23)

    def test_missing_identity_fails_closed(self):
        write_json(self.root / 'launch.json', {'controller': identity(os.getpid())})
        with self.assertRaisesRegex(ContractError, 'unresolved'):
            assert_stopped(self.root)

    def test_namespace_mismatch_blocks_only_unfinished_recovery(self):
        write_json(self.root / 'launch.json', {'worker': {'pid': 123, 'observable': False}})
        with self.assertRaisesRegex(ContractError, 'unavailable'):
            assert_stopped(self.root)
        write_json(self.root / 'process.json', {'status': 'timeout'})
        assert_stopped(self.root)

    def test_real_transport_journals_identity_and_cleans_up(self):
        if identity(os.getpid()).get('observable') is False:
            self.skipTest('requires /proc in the current PID namespace')
        # Run the controller in a separate process, then inspect its actual worker.
        script = ('from rec_harness.agents import run_process; import sys; '
                  'run_process([sys.executable,"-c","import time; time.sleep(60)"],'
                  + repr(str(self.root)) + ',' + repr(str(self.root / 'transport')) + ',timeout=0.5)')
        controller = subprocess.Popen([sys.executable, '-c', script])
        try:
            path = self.root / 'transport/launch.json'
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if path.exists() and 'worker' in read_json(path):
                    break
                time.sleep(0.01)
            record = read_json(path)
            self.assertTrue(session_live(record['worker']))
            controller.wait(timeout=5)
            self.assertEqual(controller.returncode, 0)
            self.assertFalse(session_live(record['worker']))
            self.assertEqual(read_json(path.parent / 'process.json')['status'], 'timeout')
            assert_stopped(path.parent)
        finally:
            if controller.poll() is None:
                controller.kill(); controller.wait()


if __name__ == '__main__':
    unittest.main()
