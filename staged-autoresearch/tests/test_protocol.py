"""Offline protocol tests. Scores and native responses are SYNTHETIC fixtures."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from staged_research.contracts import (direction_id, score_result, validate_plan,
                                        validate_review)
from staged_research.engine import Trajectory
from staged_research.report import normalized, report, trajectory_summary
from staged_research.prepare import prepare_fml
from staged_research.runtime import DockerRuntime, invoke, usage_from_response
from staged_research.storage import ProtocolError, digest, inventory, locked, read, write
from staged_research.study import (check_config, freeze, initialize, load_registration,
                                  matrix, template)


def make_plan(direction=None):
    return {"direction": direction or {"bottleneck": "synthetic bottleneck", "mechanism": "synthetic intervention"},
            "hypothesis": "synthetic hypothesis", "falsification": "synthetic falsification",
            "observables": ["synthetic score"], "revision_reason": "Initial prospective plan",
            "steps": [{"slot": i, "instruction": "synthetic edit", "condition": "always",
                       "expected_observation": "synthetic observation"} for i in range(1, 11)]}


def make_review(direction, diagnostics=False):
    return {"facts": [{"claim": "synthetic observation", "trial_refs": [0]}],
            "interpretations": [{"explanation": "synthetic explanation", "trial_refs": [0]}],
            "uncertainties": ["synthetic uncertainty"], "decision": "continue",
            "next_direction": direction, "reason_refs": [0],
            "diagnostics": [{"kind": "paired", "parent_trial": 0, "paired_with": 0,
                             "instruction": "synthetic intervention", "uncertainty": "could change next direction",
                             "decision_if_supported": "continue", "decision_if_refuted": "switch"}] if diagnostics else []}


class FakeRuntime:
    def __init__(self, diagnostics=True, failures=(), interrupt=False, invalid=False, ties=False):
        self.calls, self.evaluations, self.stopped = [], [], []
        self.diagnostics, self.failures = diagnostics, failures
        self.interrupt, self.invalid, self.ties = interrupt, invalid, ties

    def call_agent(self, **kw):
        packet = read(kw["evidence"] / "packet.json")
        self.calls.append({"role": kw["role"], "session": kw["session"], "resume": kw["resume"],
                           "home": kw["session_home"], "packet": packet})
        write(kw["receipt_dir"] / "usage.json", {"input_tokens": 10, "output_tokens": 5,
              "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
              "total_cost_usd": 0.0, "elapsed_seconds": 0.01}, exclusive=True)
        if kw["role"] == "plan":
            return make_plan(packet["request"]["expected_direction"])
        if kw["role"] == "review":
            return make_review(packet["request"]["current_direction"], self.diagnostics and packet["request"]["diagnostics_allowed"])
        for name in packet["task"]["editable"]:
            path = kw["work"] / name
            path.write_text(str(packet["request"]["trial"]))
        return {"hypothesis": "synthetic only", "change_description": "synthetic edit",
                "expected_observation": "synthetic only"}

    def evaluate(self, task, artifact, output, split, **kw):
        self.evaluations.append({"split": split, "artifact": artifact, **kw})
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt("simulated host interruption")
        output.mkdir(parents=True, exist_ok=False)
        trial = int(artifact.name[1:])
        transport = {"returncode": 1 if trial in self.failures else 0, "timed_out": False,
                     "stdout": "synthetic training log", "stderr": "", "elapsed_seconds": 0.1}
        metric = 0.2 if self.ties else 0.2 + trial / 1000
        if self.invalid:
            metric = 5
        payload = {"metric": metric, "constraints_ok": True, "training_seconds": 0.05,
                   "constraint_checks": {"target_epsilon": 8, "target_delta": 1e-5,
                                         "actual_epsilon": 7.99, "actual_delta": 1e-5}}
        if split == "test":
            assert kw["checkpoint"].is_dir()
        else:
            (output / "checkpoint").mkdir()
            (output / "checkpoint" / "weights.fixture").write_text("SYNTHETIC ONLY")
        write(output / "metrics.json", payload)
        write(output / "transport.json", transport)
        return transport

    def stop(self, name):
        self.stopped.append(name)


class Fixture(unittest.TestCase):
    def setUp(self):
        # Durability ordering is tested below; do not benchmark the host's fsync latency.
        sync = patch("staged_research.storage.os.fsync", return_value=None)
        sync.start()
        self.addCleanup(sync.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "study"
        self.config = template()
        self.config["sources"]["fml_commit"] = "f" * 40
        self.config["agent"].update(image="agent@sha256:" + "a" * 64,
                                     model="synthetic-fixed-model", version="synthetic-cli")
        trees = {}
        for index, (name, task) in enumerate(self.config["tasks"].items()):
            for key in ("public_source", "validation_data", "test_data", "baseline_checkpoint"):
                path = self.base / name / key
                path.mkdir(parents=True)
                task[key] = str(path)
                (path / "fixture.txt").write_text("SYNTHETIC ONLY " + key)
            for name_ in task["editable"]:
                path = Path(task["public_source"]) / name_
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}" if name_ == "split_config.json" else "0")
            task.update(validation_image="val@sha256:" + str(index) * 64,
                        test_image="test@sha256:" + str(index + 3) * 64,
                        baseline_validation=0.2, baseline_test=0.2,
                        attestation={"test_labels_absent_from_public_and_validation": True,
                                     "evaluator_outside_candidate_process": True,
                                     "protocol": "SYNTHETIC TEST FIXTURE", "evidence": "SYNTHETIC TEST FIXTURE"})
            if "split_config.json" in task["editable"]:
                task["split_config_sha256"] = inventory(task["public_source"])["split_config.json"]
            trees[name] = {k: inventory(task[k]) for k in ("public_source", "validation_data", "test_data", "baseline_checkpoint")}
        self.registration = {"registration_id": "synthetic-registration", "config": self.config, "trees": trees}

    def trajectory(self, arm="D", runtime=None, task="colored_mnist"):
        return Trajectory(self.root, self.registration, task, arm, 1, runtime or FakeRuntime())


class StudyTests(Fixture):
    def test_prepare_fml_exports_only_declared_source_without_execution(self):
        self.root.mkdir()
        write(self.root / "study.draft.json", template())
        checkout = self.base / "fml-checkout"
        for task in self.config["tasks"].values():
            directory = checkout / "ml_tasks" / task["fml_name"]
            directory.mkdir(parents=True)
            repository = checkout / "workspace" / task["fml_name"]
            for item in task["editable"]:
                path = repository / item
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("0")
            (repository / "secret-test-labels.txt").write_text("must never be exported")
            write(directory / "config.json", {"repo_dir": str(repository.relative_to(checkout)), "target_files": task["editable"]})
            write(directory / "prompt.json", {"task_description": "public task only"})
        from subprocess import CompletedProcess
        with patch("staged_research.prepare.subprocess.run", side_effect=[
            CompletedProcess([], 0, "f" * 40 + "\n", ""), CompletedProcess([], 0, "", "")]) as mocked:
            draft = prepare_fml(self.root, checkout, self.base / "assets")
        self.assertEqual(mocked.call_count, 2)  # git inspection only, no setup/training.
        self.assertEqual(draft["sources"]["fml_commit"], "f" * 40)
        for task in draft["tasks"].values():
            self.assertEqual(set(inventory(task["public_source"])), set(task["editable"]) | {"TASK.md"})

    def test_prepare_fml_rejects_rebinding_a_frozen_study(self):
        self.root.mkdir()
        write(self.root / "registration.json", {})
        with self.assertRaises(ProtocolError):
            prepare_fml(self.root, self.base / "checkout", self.base / "assets")

    def test_matrix_has_60_cells_and_6000_calls(self):
        rows = matrix(self.config)
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(r["trial_budget"] for r in rows), 6000)
        self.assertEqual(len({(r["task"], r["arm"], r["replicate"]) for r in rows}), 60)
        self.assertEqual(rows, matrix(self.config))

    def test_init_is_draft_only_and_no_overwrite(self):
        path = initialize(self.root)
        self.assertFalse((self.root / "registration.json").exists())
        self.assertTrue(check_config(read(path), self.root))
        with self.assertRaises(FileExistsError):
            initialize(self.root)

    def test_valid_fixture_and_reject_moving_model(self):
        self.assertEqual(check_config(self.config, self.root), [])
        self.config["agent"]["model"] = "sonnet"
        self.assertTrue(check_config(self.config, self.root))

    def test_split_drift_and_data_overlap(self):
        task = self.config["tasks"]["openood"]
        task["split_config_sha256"] = "wrong"
        self.assertTrue(check_config(self.config, self.root))
        task["validation_data"] = task["test_data"]
        self.assertTrue(any("disjoint" in e for e in check_config(self.config, self.root)))

    def test_cross_task_private_assets_cannot_enter_public_source(self):
        self.config["tasks"]["openood"]["test_data"] = self.config["tasks"]["colored_mnist"]["public_source"]
        self.assertTrue(any("Cross-task" in e for e in check_config(self.config, self.root)))

    def test_reject_public_transcripts_and_symlinks(self):
        task = self.config["tasks"]["openood"]
        (Path(task["public_source"]) / "CLAUDE.md").write_text("not allowed")
        self.assertTrue(check_config(self.config, self.root))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "escape").symlink_to(self.base)
            with self.assertRaises(ProtocolError):
                inventory(tmp)

    def test_freeze_and_detect_data_and_registration_drift(self):
        self.root.mkdir()
        write(self.root / "study.draft.json", self.config)
        with patch("staged_research.study.check_config", return_value=[]):
            frozen = freeze(self.root)
        self.assertEqual(load_registration(self.root)["registration_id"], frozen["registration_id"])
        with self.assertRaises(ProtocolError):
            freeze(self.root)
        path = Path(self.config["tasks"]["openood"]["test_data"]) / "fixture.txt"
        path.write_text("changed")
        with self.assertRaises(ProtocolError):
            load_registration(self.root)
        changed = read(self.root / "registration.json")
        changed["config"]["protocol"]["trial_budget"] = 200
        write(self.root / "registration.json", changed)
        with self.assertRaises(ProtocolError):
            load_registration(self.root, verify_data=False)


class ContractTests(unittest.TestCase):
    def test_plan_requires_ten_unique_slots(self):
        plan = make_plan()
        validate_plan(plan)
        plan["steps"][0]["slot"] = 2
        with self.assertRaises(ProtocolError):
            validate_plan(plan)

    def test_future_evidence_and_confounded_pair_rejected(self):
        plan = make_plan()
        review = make_review(plan["direction"], True)
        review["facts"][0]["trial_refs"] = [900]
        with self.assertRaises(ProtocolError):
            validate_review(review, [], plan["direction"], {}, True)
        review = make_review(plan["direction"], True)
        review["diagnostics"][0]["paired_with"] = 1
        with self.assertRaises(ProtocolError):
            validate_review(review, [{"trial": 1, "valid": True}], plan["direction"], {}, True)

    def test_only_D_pre_tail_can_request_diagnostics(self):
        plan = make_plan()
        review = make_review(plan["direction"], True)
        for allowed, final in ((False, False), (True, True)):
            with self.assertRaises(ProtocolError):
                validate_review(review, [], plan["direction"], {}, allowed, final)

    def test_continue_preserves_direction_identity(self):
        plan = make_plan()
        review = make_review({"bottleneck": "different", "mechanism": "different"})
        with self.assertRaises(ProtocolError):
            validate_review(review, [], plan["direction"], {}, False)

    def test_revisit_requires_known_distinct_direction(self):
        plan = make_plan()
        other = {"bottleneck": "different", "mechanism": "other"}
        review = make_review(other)
        review["decision"] = "revisit"
        with self.assertRaises(ProtocolError):
            validate_review(review, [], plan["direction"], {}, False)
        validate_review(review, [], plan["direction"], {direction_id(other): other}, False)

    def test_score_scale_finiteness_constraints(self):
        task = {"fml_name": "other"}
        for metric in (float("nan"), float("inf"), 83, True):
            with self.assertRaises(ProtocolError):
                score_result({"metric": metric, "constraints_ok": True}, task)
        with self.assertRaises(ProtocolError):
            score_result({"metric": 0.8, "constraints_ok": False}, task)

    def test_opacus_requires_external_privacy_accounting(self):
        with self.assertRaises(ProtocolError):
            score_result({"metric": 0.8, "constraints_ok": True}, {"fml_name": "Privacy_opacus"})


class EngineTests(Fixture):
    def test_budget_reservation_is_durable_before_execution(self):
        runtime = FakeRuntime()
        trajectory = self.trajectory("A", runtime)
        state = trajectory.prepare()
        original = runtime.evaluate
        def check_reserved(*args, **kwargs):
            stored = read(trajectory.state_path)
            self.assertEqual(stored["pending"]["trial"], 1)
            events = sorted((trajectory.root / "events").glob("*.json"))
            self.assertEqual(read(events[-1])["kind"], "trial_reserved")
            return original(*args, **kwargs)
        runtime.evaluate = check_reserved
        trajectory.trial(state)
        self.assertEqual(len(state["records"]), 1)

    def test_native_failure_does_not_launch_unlogged_training(self):
        runtime = FakeRuntime()
        def fail(**kwargs):
            raise ProtocolError("synthetic CLI failure")
        runtime.call_agent = fail
        trajectory = self.trajectory("A", runtime)
        state = trajectory.prepare()
        with self.assertRaises(ProtocolError):
            trajectory.trial(state)
        self.assertEqual(len(runtime.evaluations), 0)
        self.assertEqual(len(state["records"]), 0)
        self.assertEqual(state["agent_calls"], 1)

    def test_all_four_arms_complete_with_expected_contexts_and_budget(self):
        for arm in "ABCD":
            with self.subTest(arm=arm):
                runtime = FakeRuntime(failures=(1,))
                trajectory = self.trajectory(arm, runtime)
                state = trajectory.run()
                self.assertEqual(len(state["records"]), 100)
                self.assertEqual(len([e for e in runtime.evaluations if e["split"] == "validation"]), 100)
                self.assertEqual(len([e for e in runtime.evaluations if e["split"] == "test"]), 1)
                self.assertEqual(state["records"][0]["status"], "crash")
                self.assertEqual(state["records"][1]["parent_trial"], 0)
                self.assertEqual(len(state["plans"]), 0 if arm == "A" else 10)
                self.assertEqual(len(state["final_reviews"]), 0 if arm == "A" else 10)
                sessions = {c["session"] for c in runtime.calls}
                self.assertEqual(len(sessions), 21 if arm in "CD" else 1)
                diagnostic_records = [r for r in state["records"] if r["diagnostic"]]
                self.assertEqual(len(diagnostic_records), 10 if arm == "D" else 0)
                for r in diagnostic_records:
                    self.assertEqual(r["slot"], 9)
                    self.assertEqual(r["parent_trial"], 0)
                trajectory.verify_events(state)
                trajectory.run()  # idempotent completed-run resumption
                self.assertEqual(len(runtime.evaluations), 101)
                packet = trajectory.build_packet(state, "review", {})
                text = json.dumps(packet)
                for secret in (self.config["tasks"]["colored_mnist"]["test_data"], "synthetic training log", '"test_metric"', '"session"'):
                    self.assertNotIn(secret, text)

    def test_fresh_review_cannot_mount_old_session_or_controller(self):
        runtime = FakeRuntime()
        trajectory = self.trajectory("C", runtime)
        state = trajectory.prepare()
        trajectory.plan(state, 1)
        old = state["session"]
        for _ in range(8):
            trajectory.trial(state)
        trajectory.review(state, 1, False)
        call = runtime.calls[-1]
        self.assertFalse(call["resume"])
        self.assertNotEqual(call["session"], old)
        evidence = trajectory.root / "visible_evidence"
        docker = DockerRuntime(self.config["agent"])
        command = docker.agent_command(trajectory.root / "work", evidence, call["home"], call["session"], False, "review", {})
        text = " ".join(command)
        self.assertNotIn(old, text)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("Bash", text)
        self.assertIn("--bare", command)
        self.assertIn("--session-id", command)
        self.assertNotIn(str(trajectory.state_path), text)

    def test_reserved_interruption_consumes_slot_without_reexecution(self):
        runtime = FakeRuntime(interrupt=True)
        trajectory = self.trajectory("A", runtime)
        with self.assertRaises(KeyboardInterrupt):
            trajectory.run()
        state = trajectory.prepare()
        self.assertEqual(state["pending"]["trial"], 1)
        self.assertEqual(len(runtime.evaluations), 1)
        trajectory.recover_pending(state)
        self.assertEqual(len(state["records"]), 1)
        self.assertFalse(state["records"][0]["valid"])
        self.assertEqual(len(runtime.evaluations), 1)
        self.assertEqual(len(runtime.stopped), 1)

    def test_no_early_hidden_test(self):
        trajectory = self.trajectory()
        with self.assertRaises(ProtocolError):
            trajectory.finalize(trajectory.prepare())

    def test_ties_discard_and_incumbent_unchanged(self):
        trajectory = self.trajectory("A", FakeRuntime(ties=True))
        state = trajectory.prepare()
        trajectory.trial(state)
        self.assertEqual(state["records"][0]["status"], "discard")
        self.assertEqual(state["best_trial"], 0)
        prompt = read(trajectory.root / "private" / "calls" / "00001" / "request.json")["prompt"]
        self.assertNotIn("## Prospective plans", prompt)
        self.assertNotIn("## Reviews", prompt)

    def test_invalid_metric_does_not_become_scientific_failure(self):
        trajectory = self.trajectory("A", FakeRuntime(invalid=True))
        state = trajectory.prepare()
        trajectory.trial(state)
        self.assertEqual(state["records"][0]["status"], "crash")
        self.assertEqual(state["best_trial"], 0)

    def test_protected_changes_rejected_before_validation(self):
        runtime = FakeRuntime()
        original = runtime.call_agent
        def malicious(**kw):
            result = original(**kw)
            if kw["role"] == "experiment":
                (kw["work"] / "fixture.txt").write_text("tamper")
            return result
        runtime.call_agent = malicious
        trajectory = self.trajectory("A", runtime)
        state = trajectory.prepare()
        with self.assertRaises(ProtocolError):
            trajectory.trial(state)
        self.assertEqual(len(runtime.evaluations), 0)
        self.assertIsNone(state["pending"])

    def test_split_changes_rejected_before_validation(self):
        runtime = FakeRuntime()
        original = runtime.call_agent
        def edit_split(**kw):
            result = original(**kw)
            (kw["work"] / "split_config.json").write_text('{"val_seed":999}')
            return result
        runtime.call_agent = edit_split
        trajectory = self.trajectory("A", runtime, "openood")
        state = trajectory.prepare()
        with self.assertRaises(ProtocolError):
            trajectory.trial(state)
        self.assertEqual(len(runtime.evaluations), 0)

    def test_plan_revision_is_prospective_and_versioned(self):
        trajectory = self.trajectory("B")
        state = trajectory.prepare()
        trajectory.plan(state, 1)
        trajectory.trial(state)
        revision = make_plan()
        revision["steps"][1]["instruction"] = "revised future experiment"
        revision["revision_reason"] = "new evidence at trial 1"
        trajectory.revise(state, 1, revision)
        self.assertEqual(len(state["plans"]["1"]), 2)
        self.assertEqual(state["plans"]["1"][-1]["registered_before_trial"], 2)
        revision["steps"][0]["instruction"] = "rewrite the past"
        with self.assertRaises(ProtocolError):
            trajectory.revise(state, 1, revision)

    def test_state_and_artifact_tampering_detected(self):
        trajectory = self.trajectory("A")
        state = trajectory.prepare()
        trajectory.trial(state)
        bad = copy.deepcopy(state)
        bad["best_metric"] = 0.99
        with self.assertRaises(ProtocolError):
            trajectory.verify_events(bad)
        bad = copy.deepcopy(state)
        bad["records"][0]["metric"] = 0.99
        with self.assertRaises(ProtocolError):
            trajectory.verify_events(bad)
        (trajectory.artifact(1) / "domainbed" / "algorithms.py").write_text("modified")
        with self.assertRaises(ProtocolError):
            trajectory.verify_events(state)

    def test_interrupted_hidden_test_is_never_rerun(self):
        trajectory = self.trajectory("A")
        state = trajectory.run()
        state["test"] = {"status": "started"}
        with self.assertRaises(ProtocolError):
            trajectory.finalize(state)

    def test_exact_repeat_probe_preserves_code_and_anchor(self):
        runtime = FakeRuntime(diagnostics=False)
        trajectory = self.trajectory("D", runtime)
        state = trajectory.prepare()
        trajectory.plan(state, 1)
        for _ in range(8):
            trajectory.trial(state)
        trajectory.review(state, 1, False)
        probe = make_review(make_plan()["direction"], True)["diagnostics"][0]
        probe.update(kind="repeat", parent_trial=8, paired_with=8)
        state["pre_reviews"]["1"]["diagnostics"] = [probe]
        calls = len(runtime.calls)
        trajectory.trial(state)
        self.assertEqual(len(runtime.calls), calls)
        self.assertEqual(state["records"][-1]["diff"], {})
        self.assertEqual(state["records"][-1]["parent_trial"], 8)


class ReportAndRuntimeTests(Fixture):
    def test_deliberate_diagnostic_repeat_is_not_counted_as_search_waste(self):
        state = {"records": [
            {"trial": i, "parent_trial": 0, "artifact_hash": "same", "diff": {},
             "best_metric": 0.2, "valid": True, "diagnostic": {"kind": "repeat"},
             "training_seconds": 0, "elapsed_seconds": 0} for i in (1, 2)],
            "pre_reviews": {}, "final_reviews": {}, "test": None, "task": "colored_mnist",
            "arm": "D", "replicate": 1, "status": "searching"}
        result = trajectory_summary(state, self.config["tasks"]["colored_mnist"], self.base)
        self.assertIsNone(result["exact_repeat_fraction"])
        self.assertEqual(result["diagnostic_fraction"], 1)

    def test_timeout_kills_container_not_just_client(self):
        import subprocess
        process = Mock(returncode=-9)
        process.communicate.side_effect = [subprocess.TimeoutExpired("docker", 1), ("out", "err")]
        with patch("staged_research.runtime.subprocess.Popen", return_value=process), patch("staged_research.runtime.subprocess.run") as stop:
            result = invoke(["docker", "run", "--name", "synthetic-eval"], 1)
        self.assertTrue(result["timed_out"])
        self.assertEqual(stop.call_args.args[0], ["docker", "kill", "synthetic-eval"])
        process.kill.assert_called_once()

    def test_interrupt_kills_container_and_propagates(self):
        process = Mock(returncode=-9)
        process.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with patch("staged_research.runtime.subprocess.Popen", return_value=process), patch("staged_research.runtime.subprocess.run") as stop:
            with self.assertRaises(KeyboardInterrupt):
                invoke(["docker", "run", "--name", "synthetic-eval"], 1)
        self.assertEqual(stop.call_args.args[0], ["docker", "kill", "synthetic-eval"])
        process.kill.assert_called_once()

    def test_validation_has_no_test_mount_and_test_uses_selected_checkpoint(self):
        runtime = DockerRuntime(self.config["agent"])
        task = self.config["tasks"]["openood"]
        validation = runtime.evaluation_command(task, self.base / "candidate", self.base / "result", "validation")
        text = " ".join(validation)
        self.assertIn("--network none", text)
        self.assertIn(task["validation_data"], text)
        self.assertNotIn(task["test_data"], text)
        self.assertNotIn("ANTHROPIC_API_KEY", text)
        with self.assertRaises(ProtocolError):
            runtime.evaluation_command(task, self.base / "candidate", self.base / "result", "test")
        test = runtime.evaluation_command(task, self.base / "candidate", self.base / "result", "test", checkpoint=self.base / "selected-checkpoint")
        self.assertIn("dst=/checkpoint,readonly", " ".join(test))
        self.assertNotIn(task["validation_data"], " ".join(test))

    def test_usage_missing_not_zero_and_cache_separate(self):
        self.assertIsNone(usage_from_response({})["input_tokens"])
        usage = usage_from_response({"usage": {"input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": 5}})
        self.assertEqual(usage["input_tokens"], 3)
        self.assertEqual(usage["cache_read_input_tokens"], 5)

    def test_report_retains_missing_cells_and_no_macro_until_complete(self):
        trajectory = self.trajectory("A")
        state = trajectory.run()
        result = report(self.root, self.registration, self.base / "report")
        self.assertEqual(len(result["trajectories"]), 60)
        self.assertEqual(result["completed_trajectories"], 1)
        self.assertTrue(all(x["macro_normalized_test_improvement"] is None for x in result["aggregates"]))
        self.assertEqual(len((self.base / "report" / "curves.csv").read_text().splitlines()), 101)
        summary = trajectory_summary(state, trajectory.task, trajectory.root)
        self.assertEqual(summary["agent_usage_observed"]["input_tokens"], 1000)
        self.assertIsNotNone(summary["validation_auc"])
        self.assertEqual(normalized(0.1, 0.2, trajectory.task), 0)

    def test_report_cannot_write_inside_study(self):
        with self.assertRaises(ValueError):
            report(self.root, self.registration, self.root / "reports")

    def test_concurrent_owner_rejected(self):
        with locked(self.base / "lock"):
            with self.assertRaises(ProtocolError):
                with locked(self.base / "lock"):
                    pass


if __name__ == "__main__":
    unittest.main()
