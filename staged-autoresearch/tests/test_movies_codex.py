"""Offline contract checks for the Movies profile and native Codex transport."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from staged_research.codex import CodexRuntime, remove_optional_nulls, service_schema
from staged_research.contracts import EXPERIMENT, PLAN, obj, TEXT, validate_shape
from staged_research.report import report
from staged_research.storage import ProtocolError, read, write
from staged_research.study import MOVIES_PROFILE, matrix, template


class MoviesCodexTests(unittest.TestCase):
    def test_movies_uses_full_design_and_default_fml_is_unchanged(self):
        self.assertEqual(len(matrix(template())), 60)
        movies = template("local", MOVIES_PROFILE)
        self.assertEqual(list(movies["tasks"]), ["movies_nextitnet"])
        self.assertEqual(len(matrix(movies)), 20)
        self.assertEqual(sum(c["trial_budget"] for c in matrix(movies)), 2000)
        self.assertEqual(movies["protocol"]["trial_budget"], 100)
        self.assertEqual(movies["protocol"]["stage_size"], 10)
        self.assertEqual(movies["protocol"]["review_after"], 8)

    def test_optional_plan_is_nullable_at_service_boundary_only(self):
        schema = dict(EXPERIMENT, properties=dict(EXPERIMENT["properties"], plan_revision=PLAN))
        converted = service_schema(schema)
        self.assertIn("plan_revision", converted["required"])
        self.assertEqual(converted["properties"]["plan_revision"]["anyOf"][-1], {"type": "null"})
        value = {"hypothesis": "test", "change_description": "test", "expected_observation": "test", "plan_revision": None}
        clean = remove_optional_nulls(value, schema)
        self.assertNotIn("plan_revision", clean)
        validate_shape(clean, schema)
        self.assertNotIn("plan_revision", schema["required"])

    def test_movies_report_does_not_claim_sixty_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "study"
            c = template("local", MOVIES_PROFILE)
            summary = report(root, {"registration_id": "synthetic", "config": c}, base / "report")
            self.assertEqual(summary["expected_trajectories"], 20)
            self.assertEqual(summary["completed_trajectories"], 0)
            self.assertTrue(all(a["expected"] == 5 for a in summary["aggregates"]))

    def test_native_thread_mapping_resume_and_failure_are_not_silent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home, work, evidence = [root / s for s in ("home", "work", "evidence")]
            work.mkdir(); evidence.mkdir()
            runtime = CodexRuntime({"binary": "/synthetic/codex", "model": "SYNTHETIC",
                "effort": "low", "timeout_seconds": 10, "auth_root": str(root / "no-auth")},
                {"isolation": "process", "agent_env_passthrough": []}, root)
            captured = []
            def fake(argv, timeout, **kw):
                captured.append((argv, kw["env"]))
                answer = Path(argv[argv.index("--output-last-message") + 1])
                write(answer, {"answer": "synthetic"})
                events = [{"type": "thread.started", "thread_id": "native-thread"},
                    {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1, "cached_input_tokens": 3}}]
                return {"returncode": 0, "timed_out": False, "elapsed_seconds": 1,
                    "stdout": "\n".join(json.dumps(e) for e in events), "stderr": ""}
            kwargs = dict(work=work, evidence=evidence, session_home=home, session="logical-id",
                role="experiment", schema=obj({"answer": TEXT}), prompt="synthetic")
            with patch.dict(os.environ, {"CODEX_API_KEY": "SYNTHETIC"}), patch("staged_research.codex.invoke_process", side_effect=fake):
                runtime.call_agent(**kwargs, resume=False, receipt_dir=root / "call-1")
                runtime.call_agent(**kwargs, resume=True, receipt_dir=root / "call-2")
                with self.assertRaises(ProtocolError):
                    runtime.call_agent(**kwargs, resume=False, receipt_dir=root / "call-3")
            self.assertNotIn("--ephemeral", captured[0][0])
            self.assertIn("resume", captured[1][0])
            self.assertIn("native-thread", captured[1][0])
            self.assertEqual(captured[0][1]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(read(home / "native-session.json")["completed_calls"], 2)
            self.assertEqual(read(root / "call-2/usage.json")["cache_read_input_tokens"], 3)
            write(home / "pending-call.json", {"synthetic": True})
            with self.assertRaisesRegex(ProtocolError, "Interrupted native Codex"):
                runtime.call_agent(**kwargs, resume=True, receipt_dir=root / "call-4")


if __name__ == "__main__":
    unittest.main()
