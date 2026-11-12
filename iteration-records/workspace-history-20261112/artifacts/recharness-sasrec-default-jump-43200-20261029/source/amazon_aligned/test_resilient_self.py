from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .bootstrap import bootstrap
bootstrap()
from .core import read, save, schema, TEXT
from .prepare import ROOT, WORKSPACE
from .resilient_self import ResilientSelfController, TransportUnavailable


class ResilientSelfTests(unittest.TestCase):
    def controller(self, root):
        config = {**read(ROOT / 'config.json'), 'root': str(root), 'gpus': [0, 1, 2, 3],
                  'llm_slot_directory': str(Path(root) / 'slots'), 'global_llm_concurrency': 2,
                  'llm_attempts': 3, 'retry_backoff_seconds': [0, 0]}
        return ResilientSelfController(config, 'self_evolverec')

    def test_timeout_retry_cache_and_restart_preserve_budget(self):
        with tempfile.TemporaryDirectory() as root:
            c = self.controller(root)
            before = read(c.ledger.path)
            with patch.object(c, 'run_attempt', side_effect=[subprocess.TimeoutExpired('probe', 1), {'text': 'ready'}]) as invoke:
                self.assertEqual(c.llm('check'), 'ready')
                self.assertEqual(invoke.call_count, 2)
            restarted = self.controller(root)
            with patch.object(restarted, 'run_attempt', side_effect=AssertionError('cached request must not run')):
                self.assertEqual(restarted.llm('check'), 'ready')
            self.assertEqual(read(c.ledger.path), before)
            with restarted.request_scope('different-iteration'):
                with patch.object(restarted, 'run_attempt', return_value={'text': 'new'}) as invoke:
                    self.assertEqual(restarted.llm('check'), 'new')
                    self.assertEqual(invoke.call_count, 1)

    def test_exhausted_transport_can_retry_same_pending_request(self):
        with tempfile.TemporaryDirectory() as root:
            c = self.controller(root)
            with patch.object(c, 'run_attempt', side_effect=RuntimeError('temporary outage')):
                with self.assertRaises(TransportUnavailable):
                    c.llm('pending request')
            with patch.object(c, 'run_attempt', return_value={'text': 'recovered'}):
                self.assertEqual(c.llm('pending request'), 'recovered')
            self.assertEqual(read(c.ledger.path)['spent_gpu_seconds'], 0)

    def test_training_and_feedback_resume_without_retraining(self):
        with tempfile.TemporaryDirectory() as root:
            c = self.controller(root)
            source = 'print("training fixture")\n'
            def train(key, source, gpu, requested, files, smoke):
                folder = c.root / 'trials' / key
                folder.mkdir(parents=True)
                (folder / 'train.py').write_text(source)
                receipt = {'id': key, 'status': 'success', 'score': .51,
                           'allocated_gpu_seconds': 10, 'output': str(folder)}
                save(folder / 'receipt.json', receipt)
                return receipt
            with c.request_scope('trajectory-0/iteration-1'):
                with patch.object(c, '_trial', side_effect=train) as train_call:
                    receipt = c.trial('candidate-00001', source, 0)
                    self.assertEqual(c.trial('candidate-00099', source, 0), receipt)
                    self.assertEqual(train_call.call_count, 1)
            def feedback(controller, receipt):
                save(Path(receipt['output']) / 'directional-feedback.json', {'feedback': {'diagnosis_comment': 'observed'}})
                return {'diagnosis_comment': 'observed'}
            self.assertEqual(c.feedback(receipt, feedback), {'diagnosis_comment': 'observed'})
            self.assertEqual(c.feedback(receipt, lambda *_: self.fail('feedback must be reused')), {'diagnosis_comment': 'observed'})

    def test_global_slot_limit_across_controllers(self):
        with tempfile.TemporaryDirectory() as root:
            c1 = self.controller(root)
            c2 = self.controller(root)
            counter = [0, 0]
            lock = threading.Lock()
            def one(i):
                with (c1 if i % 2 else c2).llm_slot():
                    with lock:
                        counter[0] += 1
                        counter[1] = max(counter[1], counter[0])
                    time.sleep(.015)
                    with lock:
                        counter[0] -= 1
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(one, range(8)))
            self.assertLessEqual(counter[1], 2)

    def test_native_database_and_pending_selection_restore(self):
        sys.path.insert(0, str(WORKSPACE / 'artifacts/Self-EvolveRec-code'))
        from omegaconf import OmegaConf
        from database2 import Program, ProgramDatabase
        from utils.datatypes import IdeaData
        cfg = OmegaConf.load(WORKSPACE / 'artifacts/Self-EvolveRec-code/configs/config_baseline.yaml')
        rating = {'score': 1, 'positive': 'fixture', 'negative': 'fixture'}
        idea = IdeaData(description='fixture', motivation='fixture', implementation_notes='fixture',
                        pseudocode='fixture', originality=rating, future_potential=rating, code_difficulty=rating)
        with tempfile.TemporaryDirectory() as root:
            c = self.controller(root)
            def engine():
                return SimpleNamespace(workspace=str(Path(root) / 'native'), database=ProgramDatabase(cfg.database),
                                       config=SimpleNamespace(checkpoint_interval=1), _save_checkpoint=lambda _: None)
            e = engine()
            c.configure_engine(e, 0)
            e.database.add(Program(id='root', code='x=1', idea=idea, idea_diagnosis=idea, metrics={'combined_score': .5}))
            parent, inspirations = e.database.sample()
            e2 = engine()
            c.configure_engine(e2, 0)
            parent2, inspirations2 = e2.database.sample()
            self.assertEqual(parent2.id, parent.id)
            self.assertEqual([p.id for p in inspirations2], [p.id for p in inspirations])
            self.assertEqual(e2.database.best_program_id, 'root')
            self.assertEqual(len(e2.database.programs), 1)
            e2.database.add(Program(id='child', code='x=2', idea=idea, idea_diagnosis=idea,
                                   metrics={'combined_score': .6}), iteration=1)
            e2.database.increment_island_generation()
            e2._save_checkpoint(1)
            e3 = engine()
            c.configure_engine(e3, 0)
            self.assertEqual(e3.database.last_iteration, 1)
            self.assertEqual(e3.database.best_program_id, 'child')
            self.assertEqual(e3.database.island_generations, e2.database.island_generations)
            self.assertEqual(set(e3.database.programs), {'root', 'child'})

    def test_timeout_kills_cli_and_its_child_process(self):
        with tempfile.TemporaryDirectory() as root:
            c = self.controller(root)
            c.c['llm_timeout'] = .3
            script = Path(root) / 'codex'
            marker = Path(root) / 'child.pid'
            script.write_text('#!' + sys.executable + '\nimport subprocess,sys,time\n'
                'from pathlib import Path\n'
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                + 'Path(' + repr(str(marker)) + ').write_text(str(p.pid))\n'
                'time.sleep(60)\n')
            script.chmod(0o755)
            attempt = Path(root) / 'attempt'
            attempt.mkdir()
            with patch.dict(os.environ, {'PATH': root + os.pathsep + os.environ['PATH']}):
                with self.assertRaises(subprocess.TimeoutExpired):
                    c.run_attempt(attempt, 'probe', schema({'text': TEXT}), False)
            child = int(marker.read_text())
            for _ in range(20):
                stat = Path(f'/proc/{child}/stat')
                if not stat.exists() or stat.read_text().split()[2] == 'Z':
                    break
                time.sleep(.05)
            else:
                self.fail('CLI child survived timeout process-group cleanup')


if __name__ == '__main__':
    unittest.main()
