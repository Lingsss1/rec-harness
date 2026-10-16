"""Docker-free integration checks using real processes and SYNTHETIC fixtures."""
import copy
import os
import subprocess
import signal
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_protocol import Fixture
from staged_research.evaluate import evaluate_study
from staged_research.local import (LocalRuntime, clean_environment, invoke_process,
                                   make_runtime)
from staged_research.storage import ProtocolError, read, write
from staged_research.study import check_config, freeze, load_registration, template


class LocalTests(Fixture):
    def setUp(self):
        super().setUp()
        self.runtime_files = self.base / "trusted-runtime"
        self.runtime_files.mkdir()
        fixtures = Path(__file__).parent / "fixtures"
        self.agent = self.runtime_files / "synthetic-cli"
        body = (fixtures / "local_agent.py").read_text().split("\n", 1)[1]
        self.agent.write_text("#!" + sys.executable + "\n" + body)
        self.agent.chmod(0o700)
        self.evaluator = self.runtime_files / "synthetic-evaluator.py"
        self.evaluator.write_bytes((fixtures / "local_evaluator.py").read_bytes())
        self.config["execution"] = dict(template("local")["execution"], environment_id="SYNTHETIC-OFFLINE",
            pinned_files=[str(self.agent), str(self.evaluator), sys.executable])
        self.config["agent"].pop("image")
        self.config["agent"]["binary"] = str(self.agent)
        for task in self.config["tasks"].values():
            task.pop("validation_image")
            task.pop("test_image")
            for split in ("validation", "test"):
                task[split + "_argv"] = [sys.executable, str(self.evaluator), "--candidate", "{candidate}",
                    "--data", "{data}", "--result", "{result}"]
                if split == "test":
                    task[split + "_argv"] += ["--checkpoint", "{checkpoint}"]
        self.runtime = make_runtime(self.config, self.root)

    def test_local_configuration_never_requires_docker(self):
        with patch("staged_research.study.shutil.which", return_value=None):
            self.assertEqual(check_config(self.config, self.root, check_images=True), [])
        self.assertIsInstance(self.runtime, LocalRuntime)

    def test_pin_commands_and_reject_leaking_mount_roots(self):
        self.config["execution"]["pinned_files"] = [str(self.agent)]
        self.assertTrue(any("pinned_files" in e for e in check_config(self.config, self.root)))
        self.config["execution"]["readonly_roots"] = [str(self.base)]
        self.assertTrue(any("disjoint" in e for e in check_config(self.config, self.root)))

    def test_freeze_detects_changed_host_evaluator(self):
        self.root.mkdir()
        write(self.root / "study.draft.json", self.config)
        freeze(self.root)
        self.assertTrue(load_registration(self.root)["runtime_files"])
        self.evaluator.write_text("raise RuntimeError('changed evaluator')")
        with self.assertRaisesRegex(ProtocolError, "environment changed"):
            load_registration(self.root)

    def test_check_only_does_not_freeze_or_call_backend(self):
        config_path = self.base / "server.json"
        write(config_path, self.config)
        result = evaluate_study(self.root, self.base / "report", config_path=config_path,
            check_only=True, runtime_factory=lambda *_: self.fail("must not call runtime"))
        self.assertTrue(result["ready"])
        self.assertFalse((self.root / "registration.json").exists())

    def test_real_process_native_session_training_test_report_and_resume(self):
        config_path = self.base / "server.json"
        write(config_path, self.config)
        cells = [{"task": "colored_mnist", "arm": "D", "replicate": 1}]
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "SYNTHETIC-NO-NETWORK"}), \
             patch("staged_research.evaluate.matrix", return_value=cells):
            result = evaluate_study(self.root, self.base / "report", config_path=config_path, progress=lambda _: None)
            self.assertEqual(result["completed_trajectories"], 1)
            jobs = list((self.root / "private_backend_jobs").glob("*.json"))
            self.assertEqual(len(jobs), 101)  # 100 accounted validation calls + one test
            state = read(self.root / "trajectories/colored_mnist/D/1/state.json")
            self.assertEqual(len(state["records"]), 100)
            self.assertEqual(len(state["pre_reviews"]), 10)
            self.assertEqual(len(state["final_reviews"]), 10)
            self.assertEqual(sum(r["diagnostic"] is not None for r in state["records"]), 10)
            self.assertEqual(len(list((self.root / "trajectories/colored_mnist/D/1/private/sessions").iterdir())), 21)
            with patch("staged_research.local.subprocess.Popen", side_effect=AssertionError("finished run must not launch processes")):
                resumed = evaluate_study(self.root, self.base / "report", config_path=config_path, progress=lambda _: None)
            self.assertEqual(resumed["completed_trajectories"], 1)
        report = read(self.base / "report/summary.json")
        self.assertEqual(len(report["trajectories"]), 60)
        self.assertTrue(any("does not enforce" in s for s in report["limitations"]))

    def test_local_failure_is_counted_and_keep_going_exports_partial_report(self):
        write(self.root / "study.draft.json", self.config)
        cells = [{"task": "colored_mnist", "arm": a, "replicate": 1} for a in ("A", "B")]
        from test_protocol import FakeRuntime
        class BrokenRuntime(FakeRuntime):
            def call_agent(self, **kw):
                raise ProtocolError("SYNTHETIC failure before training")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "SYNTHETIC"}), \
             patch("staged_research.evaluate.matrix", return_value=cells):
            result = evaluate_study(self.root, self.base / "report", keep_going=True,
                runtime_factory=lambda *_: BrokenRuntime(), progress=lambda _: None)
        self.assertEqual(len(result["failures"]), 2)
        self.assertEqual(result["completed_trajectories"], 0)
        self.assertEqual(len(list((self.root / "execution_failures").glob("*.json"))), 2)

    def test_different_config_cannot_replace_study(self):
        write(self.root / "study.draft.json", self.config)
        altered = copy.deepcopy(self.config)
        altered["agent"]["model"] = "different-model"
        write(self.base / "other.json", altered)
        with self.assertRaisesRegex(ProtocolError, "differs"):
            evaluate_study(self.root, self.base / "report", config_path=self.base / "other.json", check_only=True)

    def test_gate_does_not_execute_if_pid_receipt_fails(self):
        marker = self.base / "should-not-exist"
        with patch("staged_research.local.write", side_effect=OSError("SYNTHETIC disk full")):
            with self.assertRaises(OSError):
                invoke_process([sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(marker)],
                    5, cwd=self.base, env=clean_environment(self.base), receipt=self.base / "job.json")
        self.assertFalse(marker.exists())

    def test_timeout_stops_child_process_too(self):
        marker = self.base / "orphan"
        child = "import time; from pathlib import Path; time.sleep(0.8); Path(%r).touch()" % str(marker)
        parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(20)" % child
        result = invoke_process([sys.executable, "-c", parent], 0.2, cwd=self.base,
                                env=clean_environment(self.base), receipt=self.base / "job.json")
        self.assertTrue(result["timed_out"])
        time.sleep(1)
        self.assertFalse(marker.exists())

    def test_sigterm_cleans_worker_and_preserves_signal_handler(self):
        from staged_research.cli import main
        marker = self.base / "should-not-survive"
        code = "import time; from pathlib import Path; time.sleep(0.7); Path(%r).touch()" % str(marker)
        previous = signal.getsignal(signal.SIGTERM)
        def run_until_terminated(*args, **kwargs):
            invoke_process([sys.executable, "-c", code], 10, cwd=self.base,
                env=clean_environment(self.base), receipt=self.base / "signal-job.json")
        timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
        try:
            with patch("staged_research.cli.evaluate_study", side_effect=run_until_terminated):
                timer.start()
                code = main(["evaluate", "--root", str(self.root), "--output", str(self.base / "out")])
        finally:
            timer.cancel()
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        time.sleep(0.8)
        self.assertFalse(marker.exists())
        self.assertIn("finished_at", read(self.base / "signal-job.json"))

    def test_interrupted_pid_identity_cannot_kill_another_job(self):
        write(self.runtime.jobs / "job.json", {"pid": 123, "boot_id": "boot", "starttime": "old"})
        with patch("staged_research.local.boot_identity", return_value="boot"), \
             patch("staged_research.local.process_identity", return_value="new"), \
             patch("staged_research.local.kill_group") as kill:
            with self.assertRaisesRegex(ProtocolError, "reused"):
                self.runtime.stop("job")
        kill.assert_not_called()

    def test_validation_argv_has_no_hidden_data_or_checkpoint(self):
        task = self.config["tasks"]["openood"]
        argv = self.runtime.evaluation_command(task, self.base / "candidate", self.base / "result", "validation")
        self.assertIn(task["validation_data"], argv)
        self.assertNotIn(task["test_data"], argv)
        self.assertNotIn("--checkpoint", argv)
        self.assertNotIn("ANTHROPIC_API_KEY", clean_environment(self.base, "0"))
        with self.assertRaises(ProtocolError):
            self.runtime.evaluation_command(task, self.base, self.base / "out", "test")

    def test_bubblewrap_constructs_explicit_no_test_agent_view(self):
        self.runtime.execution["isolation"] = "bubblewrap"
        with patch("staged_research.local.sys.platform", "linux"):
            argv = self.runtime.agent_command(self.base / "work", self.base / "evidence", self.base / "session", "s", False, "review", {})
            eval_argv = self.runtime.evaluation_command(self.config["tasks"]["openood"], self.base / "candidate", self.base / "out", "validation")
        self.assertIn("--unshare-net", eval_argv)
        self.assertNotIn("--unshare-net", argv)
        self.assertIn("--ro-bind", argv)
        self.assertNotIn(self.config["tasks"]["openood"]["test_data"], argv + eval_argv)
        self.assertNotIn(str(self.root), argv + eval_argv)

    def test_shell_entrypoint_is_callable_from_another_directory(self):
        entry = Path(__file__).resolve().parents[1] / "evaluate.sh"
        result = subprocess.run([str(entry), "--help"], cwd=self.base,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-only", result.stdout)


if __name__ == "__main__":
    unittest.main()
