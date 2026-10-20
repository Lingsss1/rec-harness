"""One owner, one durable budget, native edits, immutable scientific receipts."""
from __future__ import annotations

import csv
import copy
import hashlib
import io
import shutil
import time
import uuid
from pathlib import Path

from .contracts import (EXPERIMENT, PLAN, REVIEW, direction_id, score_result,
                        validate_plan, validate_review, validate_shape)
from .storage import ProtocolError, digest, inventory, locked, now, read, write
from .study import ARMS


def copy_tree(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


class Trajectory:
    def __init__(self, study_root, registration, task, arm, replicate, runtime):
        if task not in registration["config"]["tasks"] or arm not in ARMS or replicate not in range(1, 6):
            raise ProtocolError("Unknown matrix cell")
        self.study_root = Path(study_root).resolve()
        self.registration = registration
        self.task_name, self.arm, self.replicate = task, arm, replicate
        self.task = registration["config"]["tasks"][task]
        self.budget = registration["config"]["protocol"]["trial_budget"]
        self.policy = ARMS[arm]
        self.root = self.study_root / "trajectories" / task / arm / str(replicate)
        self.runtime = runtime
        self.state_path = self.root / "state.json"

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            state = read(self.state_path)
            if state["registration_id"] != self.registration["registration_id"]:
                raise ProtocolError("Trajectory belongs to a different registration")
            self.verify_events(state)
            return state
        copy_tree(Path(self.task["public_source"]), self.root / "artifacts" / "t000")
        state = {"registration_id": self.registration["registration_id"], "task": self.task_name,
                 "arm": self.arm, "replicate": self.replicate, "created_at": now(),
                 "trial_budget": self.budget, "records": [], "plans": {}, "pre_reviews": {}, "final_reviews": {},
                 "directions": {}, "best_trial": 0, "best_metric": self.task["baseline_validation"],
                 "session": None, "resume": False, "pending": None, "event_count": 0,
                 "event_head": None, "agent_calls": 0, "status": "searching", "test": None}
        self.save(state)
        return state

    def save(self, state):
        write(self.state_path, state)

    def event(self, state, kind, payload):
        sequence = state["event_count"] + 1
        event = {"sequence": sequence, "previous": state["event_head"], "time": now(),
                 "registration_id": state["registration_id"], "trials_consumed": len(state["records"]),
                 "kind": kind, "payload": payload}
        event["id"] = digest(event)
        write(self.root / "events" / ("%06d.json" % sequence), event, exclusive=True)
        state["event_count"], state["event_head"] = sequence, event["id"]
        self.save(state)
        return event["id"]

    def verify_events(self, state):
        head = None
        paths = sorted((self.root / "events").glob("*.json"))
        if len(paths) != state["event_count"]:
            raise ProtocolError("Uncommitted or missing event; preserve run for manual audit, do not overwrite")
        for i, path in enumerate(paths, 1):
            event = read(path)
            if event["sequence"] != i or event["previous"] != head or event["registration_id"] != state["registration_id"] or digest({k: v for k, v in event.items() if k != "id"}) != event["id"]:
                raise ProtocolError("Event history was modified")
            head = event["id"]
        if head != state["event_head"]:
            raise ProtocolError("Event head mismatch")
        # Records and artifacts are independently linked to immutable events.
        trials = [read(p)["payload"] for p in paths if read(p)["kind"] == "trial_completed"]
        if trials != state["records"]:
            raise ProtocolError("Trial records differ from the append-only audit log")
        if inventory(self.artifact(0)) != self.registration["trees"][self.task_name]["public_source"]:
            raise ProtocolError("Baseline artifact differs from frozen registration")
        expected_best, expected_trial = self.task["baseline_validation"], 0
        for record in state["records"]:
            if digest(inventory(self.artifact(record["trial"]))) != record["artifact_hash"]:
                raise ProtocolError("Historical artifact was modified")
            if record["valid"]:
                if digest(inventory(self.checkpoint(record["trial"]))) != record["checkpoint_hash"]:
                    raise ProtocolError("Historical checkpoint was modified")
                better = record["metric"] > expected_best if self.task["direction"] == "higher" else record["metric"] < expected_best
                if better:
                    expected_best, expected_trial = record["metric"], record["trial"]
        if (state["best_metric"], state["best_trial"]) != (expected_best, expected_trial):
            raise ProtocolError("Incumbent was changed outside the acceptance rule")
        plans, pre, final = {}, {}, {}
        test = None
        for path in paths:
            event = read(path)
            payload = event["payload"]
            if event["kind"] in ("plan_registered", "plan_revised"):
                plans.setdefault(str(payload["stage"]), []).append(payload["plan"])
            elif event["kind"] == "pre_tail_review":
                pre[str(payload["stage"])] = payload["review"]
            elif event["kind"] == "final_review":
                final[str(payload["stage"])] = payload["review"]
            elif event["kind"] in ("hidden_test_reserved", "hidden_test_finished"):
                test = payload
        if (plans, pre, final, test) != (state["plans"], state["pre_reviews"], state["final_reviews"], state["test"]):
            raise ProtocolError("Plan, review, or test state differs from immutable receipts")

    def artifact(self, trial):
        return self.root / "artifacts" / ("t%03d" % trial)

    def checkpoint(self, trial):
        return Path(self.task["baseline_checkpoint"]) if trial == 0 else self.root / "validation" / ("t%03d" % trial) / "checkpoint"

    def build_packet(self, state, role, extra):
        # Explicit allowlist: no test metrics, test paths, raw CLI output, or session directories.
        task = {k: self.task[k] for k in ("description", "metric", "direction", "editable", "baseline_validation")}
        task["editable"] = [p for p in task["editable"] if p != "split_config.json"]
        task["fixed_split"] = True
        packet = {"task": task, "budget_remaining": self.budget - len(state["records"]),
                  "best_trial": state["best_trial"], "best_metric": state["best_metric"],
                  "experiments": state["records"], "request": extra}
        if self.policy["planning"]:
            packet.update(plans=state["plans"], pre_reviews=state["pre_reviews"],
                          final_reviews=state["final_reviews"], known_directions=state["directions"])
        return packet

    def materialize_evidence(self, state, packet):
        evidence = self.root / "visible_evidence"
        evidence.mkdir(exist_ok=True)
        write(evidence / "packet.json", packet)
        program = Path(__file__).parent / "resources" / "karpathy-program.md"
        (evidence / "karpathy-original.md").write_bytes(program.read_bytes())
        # These are executed code and training logs, never native-agent conversations.
        for trial in [0] + [r["trial"] for r in state["records"]]:
            source = self.artifact(trial)
            destination = evidence / "artifacts" / ("t%03d" % trial)
            for item in self.task["editable"]:
                target = destination / item
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source / item, target)
            transport = self.root / "validation" / ("t%03d" % trial) / "transport.json"
            log_target = evidence / "trial_logs" / ("t%03d.json" % trial)
            if transport.exists() and not log_target.exists():
                result = read(transport)
                write(log_target,
                      {k: result.get(k) for k in ("stdout", "stderr", "returncode", "timed_out")})
        return evidence

    def fresh_context(self, state, reason):
        old = state["session"]
        state["session"], state["resume"] = str(uuid.uuid4()), False
        self.event(state, "context_reset", {"reason": reason, "previous_session": old,
                                            "new_session": state["session"]})

    def call(self, state, role, schema, extra, work=None):
        if state["session"] is None:
            self.fresh_context(state, "initial")
        if work is None:
            work = self.root / "work"
            copy_tree(self.artifact(state["best_trial"]), work)
        packet = self.build_packet(state, role, extra)
        evidence = self.materialize_evidence(state, packet)
        prompt = (Path(__file__).parent / "resources" / "protocol.md").read_text().replace("100-call budget", str(self.budget) + "-call budget")
        if not self.policy["planning"]:
            # Do not teach the baseline the treatment's planning/review strategy.
            prompt = prompt.split("## Prospective plans", 1)[0]
        prompt += "\nROLE=" + role + "\nRead /evidence/packet.json and return the required JSON.\n"
        if not self.policy["planning"]:
            prompt += "ARM A: use the original autonomous propose/edit/keep-or-discard strategy. No imposed stage plan or review.\n"
        if role == "experiment":
            prompt += "Edit the allowed files in /work using native editing tools. Training is run by the controller after this call. Do not serialize conversation history in code.\n"
        state["agent_calls"] += 1
        receipt_dir = self.root / "private" / "calls" / ("%05d" % state["agent_calls"])
        write(receipt_dir / "request.json", {"role": role, "session": state["session"],
              "resume": state["resume"], "packet_hash": digest(packet), "prompt": prompt}, exclusive=True)
        self.save(state)
        output = self.runtime.call_agent(work=work, evidence=evidence,
            session_home=self.root / "private" / "sessions" / state["session"],
            session=state["session"], resume=state["resume"], role=role,
            schema=schema, prompt=prompt, receipt_dir=receipt_dir)
        validate_shape(output, schema)
        write(receipt_dir / "response.json", output, exclusive=True)
        state["resume"] = True
        self.event(state, "agent_response", {"role": role, "call": state["agent_calls"],
                                             "response_hash": digest(output)})
        return output

    def plan(self, state, stage):
        previous = state["final_reviews"].get(str(stage - 1))
        expected = previous["next_direction"] if previous else None
        plan = self.call(state, "plan", PLAN, {"stage": stage,
                         "expected_direction": expected, "requirement": "Prospectively specify all ten slots, including conditional branches and optional diagnostics in slots 9-10."})
        validate_plan(plan, expected)
        plan["anchor_trial"] = state["best_trial"]
        plan["registered_before_trial"] = len(state["records"]) + 1
        plan["version"] = 1
        state["plans"][str(stage)] = [plan]
        state["directions"][direction_id(plan["direction"])] = plan["direction"]
        self.event(state, "plan_registered", {"stage": stage, "plan": plan})

    def review(self, state, stage, final):
        if self.policy["flush"]:
            self.fresh_context(state, "stage_%d_%s_review" % (stage, "final" if final else "pre_tail"))
        current = state["plans"][str(stage)][-1]["direction"]
        review = self.call(state, "review", REVIEW, {"stage": stage, "final": final,
            "diagnostics_allowed": self.policy["diagnostics"] and not final,
            "current_direction": current,
            "requirement": "Separate facts, interpretations, and uncertainty; reference only completed trials. A diagnostic must resolve uncertainty consequential to the next direction decision. Final reviews request no further diagnostics."})
        validate_review(review, state["records"], current, state["directions"], self.policy["diagnostics"], final)
        key = "final_reviews" if final else "pre_reviews"
        state[key][str(stage)] = review
        self.event(state, "final_review" if final else "pre_tail_review", {"stage": stage, "review": review})

    def revise(self, state, stage, plan):
        previous = state["plans"][str(stage)][-1]
        validate_plan(plan, previous["direction"])
        completed = len(state["records"]) % 10
        for old in previous["steps"]:
            if old["slot"] <= completed and old != next(x for x in plan["steps"] if x["slot"] == old["slot"]):
                raise ProtocolError("Cannot rewrite the plan for an experiment already observed")
        updated = dict(copy.deepcopy(plan), anchor_trial=previous["anchor_trial"],
                       registered_before_trial=len(state["records"]) + 1, version=previous["version"] + 1)
        state["plans"][str(stage)].append(updated)
        self.event(state, "plan_revised", {"stage": stage, "plan": updated})

    def trial(self, state):
        count = len(state["records"])
        if count >= self.budget or state["pending"]:
            raise ProtocolError("No available trial slot")
        stage, slot, trial = count // 10 + 1, count % 10 + 1, count + 1
        plan = state["plans"].get(str(stage), [None])[-1]
        probes = state["pre_reviews"].get(str(stage), {}).get("diagnostics", [])
        probe = probes[slot - 9] if slot >= 9 and slot - 9 < len(probes) else None
        if probe and not self.policy["diagnostics"]:
            raise ProtocolError("Diagnostic outside D")
        parent = probe["parent_trial"] if probe else state["best_trial"]
        work = self.root / "work"
        copy_tree(self.artifact(parent), work)
        if probe and probe["kind"] == "repeat":
            proposal = {"hypothesis": probe["uncertainty"], "change_description": "Exact repeat of comparator artifact", "expected_observation": probe["instruction"]}
        else:
            schema = dict(EXPERIMENT, properties=dict(EXPERIMENT["properties"], plan_revision=PLAN))
            request = {"trial": trial, "parent_trial": parent, "diagnostic": probe,
                       "instruction": probe["instruction"] if probe else (next(x for x in plan["steps"] if x["slot"] == slot) if plan else "Propose and implement the next experimental idea.")}
            proposal = self.call(state, "experiment", schema, request, work=work)
            revision = proposal.pop("plan_revision", None)
            if revision is not None:
                if not plan or probe:
                    raise ProtocolError("Cannot revise a stage plan here")
                self.revise(state, stage, revision)
                plan = state["plans"][str(stage)][-1]
        baseline_files, candidate_files = inventory(self.artifact(0)), inventory(work)
        mutable = set(self.task["editable"]) - {"split_config.json"}
        changed = {p for p in set(baseline_files) | set(candidate_files) if baseline_files.get(p) != candidate_files.get(p)}
        if not changed <= mutable or not set(baseline_files) <= set(candidate_files):
            raise ProtocolError("Agent changed protected files, fixed split, or deleted source files")
        copy_tree(work, self.artifact(trial))
        before_files = inventory(self.artifact(parent))
        actual_diff = {p: {"before": before_files.get(p), "after": candidate_files.get(p)}
                       for p in set(before_files) | set(candidate_files) if before_files.get(p) != candidate_files.get(p)}
        pending = {"trial": trial, "stage": stage, "slot": slot, "parent_trial": parent,
                   "direction": plan["direction"] if plan else None,
                   "direction_id": direction_id(plan["direction"]) if plan else None,
                   "plan_version": plan["version"] if plan else None, "proposal": proposal,
                   "diagnostic": probe, "diff": actual_diff,
                   "artifact_hash": digest(candidate_files), "started_at": now(),
                   "container_name": "sar-eval-" + uuid.uuid4().hex}
        state["pending"] = pending
        # Reserve durably BEFORE launching. A host crash never refunds a training call.
        self.event(state, "trial_reserved", pending)
        output = self.root / "validation" / ("t%03d" % trial)
        try:
            transport = self.runtime.evaluate(self.task, self.artifact(trial), output, "validation", name=pending["container_name"])
        except Exception as exc:
            transport = {"returncode": -1, "stdout": "", "stderr": str(exc), "timed_out": False, "elapsed_seconds": None}
            output.mkdir(parents=True, exist_ok=True)
            if not (output / "transport.json").exists():
                write(output / "transport.json", transport, exclusive=True)
        self.finish_trial(state, transport)

    def finish_trial(self, state, transport):
        pending = state["pending"]
        payload_path = self.root / "validation" / ("t%03d" % pending["trial"]) / "metrics.json"
        score, payload, error, checkpoint_hash = None, None, None, None
        try:
            if transport["timed_out"]:
                raise ProtocolError("Evaluation timed out: " + (transport.get("stderr") or "")[-4000:])
            if transport["returncode"] != 0:
                details = transport.get("stderr") or "No worker diagnostic returned; inspect visible trial logs"
                raise ProtocolError("Evaluation failed (exit %s): %s" %
                                    (transport["returncode"], details[-4000:]))
            payload = read(payload_path)
            score = score_result(payload, self.task)
            files = inventory(self.checkpoint(pending["trial"]))
            if not files:
                raise ProtocolError("Evaluator did not preserve a trained checkpoint")
            checkpoint_hash = digest(files)
        except (OSError, ValueError, KeyError) as exc:
            error, score = str(exc), None
        # Never infer a negative scientific result from an invalid execution.
        valid = score is not None
        better = valid and (score > state["best_metric"] if self.task["direction"] == "higher" else score < state["best_metric"])
        status = "keep" if better else ("discard" if valid else "crash")
        if better:
            state["best_metric"], state["best_trial"] = score, pending["trial"]
        record = dict(pending, valid=valid, metric=score, status=status, error=error,
                      checkpoint_hash=checkpoint_hash,
                      best_metric=state["best_metric"], best_trial=state["best_trial"],
                      training_seconds=payload.get("training_seconds") if payload and valid else None,
                      elapsed_seconds=transport.get("elapsed_seconds"), completed_at=now())
        # Private operational identifiers are not needed in scientific records.
        record.pop("container_name")
        state["records"].append(record)
        state["pending"] = None
        self.event(state, "trial_completed", record)
        table = io.StringIO()
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["commit", "metric", "status", "description"])
        for r in state["records"]:
            writer.writerow([r["artifact_hash"][:12], r["metric"] if r["valid"] else "", r["status"], r["proposal"]["change_description"]])
        (self.root / "results.tsv").write_text(table.getvalue())

    def recover_pending(self, state):
        pending = state["pending"]
        if pending is None:
            return
        self.runtime.stop(pending["container_name"])
        path = self.root / "validation" / ("t%03d" % pending["trial"]) / "transport.json"
        transport = read(path) if path.exists() else {"returncode": -1, "timed_out": False,
            "elapsed_seconds": None, "stdout": "", "stderr": "Interrupted: outcome unknown; budget remains spent"}
        self.finish_trial(state, transport)

    def finalize(self, state):
        if len(state["records"]) != self.budget or state["pending"]:
            raise ProtocolError("Hidden test is only allowed after the registered budget is exhausted")
        if self.policy["planning"] and str((self.budget + 9) // 10) not in state["final_reviews"]:
            raise ProtocolError("Final direction review is missing")
        if state["test"] is not None:
            if state["test"]["status"] == "started":
                if state["test"].get("container_name"):
                    self.runtime.stop(state["test"]["container_name"])
                raise ProtocolError("Hidden-test outcome unknown after interruption; never rerun it")
            return
        state["test"] = {"status": "started", "selected_trial": state["best_trial"], "started_at": now(),
                         "container_name": "sar-test-" + uuid.uuid4().hex}
        self.event(state, "hidden_test_reserved", state["test"])
        output = self.root / "hidden_test"
        try:
            transport = self.runtime.evaluate(self.task, self.artifact(state["best_trial"]), output, "test", name=state["test"]["container_name"], checkpoint=self.checkpoint(state["best_trial"]))
            if transport["returncode"] != 0 or transport["timed_out"]:
                raise ProtocolError("Hidden test execution failed")
            payload = read(output / "metrics.json")
            metric = score_result(payload, self.task)
            state["test"].update(status="completed", metric=metric)
        except Exception as exc:
            state["test"].update(status="failed", metric=None, error=str(exc))
        state["status"] = "completed"
        state["completed_at"] = now()
        self.event(state, "hidden_test_finished", state["test"])

    def run(self):
        with locked(self.root / ".owner.lock"):
            state = self.prepare()
            self.recover_pending(state)
            while True:
                n = len(state["records"])
                if self.policy["planning"] and n and n % 10 == 0 and str(n // 10) not in state["final_reviews"]:
                    self.review(state, n // 10, final=True)
                    continue
                if n == self.budget:
                    final_stage = (self.budget + 9) // 10
                    if self.policy["planning"] and str(final_stage) not in state["final_reviews"]:
                        self.review(state, final_stage, final=True)
                        continue
                    self.finalize(state)
                    return state
                stage = n // 10 + 1
                if self.policy["planning"] and str(stage) not in state["plans"]:
                    self.plan(state, stage)
                if self.policy["planning"] and n % 10 == 8 and str(stage) not in state["pre_reviews"]:
                    self.review(state, stage, final=False)
                self.trial(state)
