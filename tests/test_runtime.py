"""Contract and failure tests, plus an actual CPU research-loop integration test."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from rec_harness.agents import CodexAgent, run_process
from rec_harness.core import Artifacts, ContractError, digest, read_json, write_json
from rec_harness.evolution import Evolution, validate_policy
from rec_harness.migration import import_legacy
from rec_harness.replay import Replay, build_world
from rec_harness.runtime import Research, node, edge
from rec_harness.store import Store

ROOT = Path(__file__).resolve().parents[1]


class Temporary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


class GraphTests(Temporary):
    def setUp(self):
        super().setUp()
        self.store = Store(self.path / "memory")
        self.addCleanup(self.store.close)

    def patch(self, nodes, edges=None):
        return self.store.patch({"base_version": self.store.version, "nodes": nodes, "edges": edges or []})

    def test_reject_stale_hidden_dangling_and_cross_protocol(self):
        a = node("a", "Observation", {}, "t", "p")
        b = node("b", "GapHypothesis", {}, "t", "q", "agent_hypothesis")
        self.patch([a, b])
        with self.assertRaises(ContractError):
            self.store.patch({"base_version": 0, "nodes": [], "edges": []})
        private = node("hidden", "Observation", {"hidden_score": 1}, "t", "p")
        with self.assertRaises(ContractError):
            self.patch([private])
        edge = {"id": "e", "source": "a", "target": "b", "relation": "SUPPORTS", "attrs": {"scope": "t", "evidence_refs": ["a"], "origin": "local_evidence", "status": "provisional", "created_at": "now", "revision_id": "1"}}
        with self.assertRaises(ContractError):
            self.patch([], [edge])
        edge["target"] = "missing"
        with self.assertRaises(ContractError):
            self.patch([], [edge])
        self.assertEqual(self.store.version, 1)
        self.assertEqual(self.store.verify()["graph_version"], 1)

    def test_claim_independence_and_immutable_facts(self):
        runs = [node(k, "Experiment", {"valid": True, "independent_run_id": "same", "effect_gate_passed": True}, "t", "p") for k in ["r1", "r2"]]
        self.patch(runs)
        claim = node("claim", "Claim", {"status": "supported", "scope": "t", "evidence_refs": ["r1", "r2"]}, "t", "p")
        with self.assertRaises(ContractError):
            self.patch([claim])
        runs[0]["attrs"]["valid"] = False
        with self.assertRaises(ContractError):
            self.patch([runs[0]])
        self.patch([node("r3", "Experiment", {"valid": True, "independent_run_id": "different", "effect_gate_passed": True}, "t", "p")])
        claim["attrs"]["evidence_refs"] = ["r1", "r3"]
        self.patch([claim])
        claim["attrs"]["status"] = "confirmed_in_scope"
        with self.assertRaises(ContractError):
            self.patch([claim])

    def test_snapshot_rollback_rebuild_and_prefix(self):
        self.patch([node("old", "GapHypothesis", {"evidence_dependency_ids": ["a"]}, "t", "p")])
        self.patch([node("future", "GapHypothesis", {"evidence_dependency_ids": ["b"]}, "t", "p")])
        bundle = self.store.bundle("t", "p", "evidence", revealed=["a"])
        self.assertEqual([n["id"] for n in bundle["nodes"]], ["old"])
        snapshot = self.store.rollback(1)
        self.assertEqual(snapshot["version"], 3)
        self.assertEqual(len(snapshot["nodes"]), 1)
        self.assertEqual(len(self.store.snapshot(2)["nodes"]), 2)
        self.store.db.execute("DELETE FROM nodes")
        self.store.rebuild()
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0], 1)

    def test_development_and_unqualified_controls_cannot_confirm_claim(self):
        def experiment(key, **extra):
            return node(key, "Experiment", {"valid": True, "independent_run_id": key,
                        "effect_gate_passed": True, "partition": "confirmation", **extra}, "t", "p")
        self.patch([experiment("confirm1"), experiment("development", partition="development"),
                    experiment("failed-control", effect_gate_passed=False, mechanism_control=True)])
        claim = node("claim", "Claim", {"status": "supported", "scope": "t",
                     "evidence_refs": ["confirm1", "development", "failed-control"]}, "t", "p")
        with self.assertRaisesRegex(ContractError, "two independent"):
            self.patch([claim])
        self.patch([experiment("confirm2")])
        claim["attrs"]["evidence_refs"].append("confirm2")
        self.patch([claim])
        claim["attrs"]["status"] = "confirmed_in_scope"
        with self.assertRaisesRegex(ContractError, "mechanism control"):
            self.patch([claim])
        self.patch([experiment("good-control", mechanism_control=True)])
        claim["attrs"]["evidence_refs"].append("good-control")
        self.patch([claim])

    def test_reported_legacy_validity_cannot_upgrade_claim(self):
        runs = [node(k, "Experiment", {"valid": True, "independent_run_id": k,
                     "effect_gate_passed": True}, "t", "p", "legacy_unverified") for k in ["a", "b"]]
        self.patch(runs)
        claim = node("claim", "Claim", {"status": "supported", "scope": "t",
                     "evidence_refs": ["a", "b"]}, "t", "p")
        with self.assertRaisesRegex(ContractError, "two independent"):
            self.patch([claim])

    def test_budget_failure_unknown_and_no_double_charge(self):
        self.store.configure_budget(5)
        self.store.reserve("crash", 2)
        self.store.settle("crash", 2, "crash")
        self.store.reserve("interrupted", 3)
        self.assertEqual(self.store.recover(), ["interrupted"])
        self.assertEqual(self.store.budget()["remaining"], 0)
        with self.assertRaises(ContractError):
            self.store.reserve("new", 1)
        with self.assertRaises(ContractError):
            self.store.reserve("crash", 0)
        with self.assertRaises(ContractError):
            self.store.configure_budget(6)

    def test_legacy_idempotent_unverified(self):
        source = self.path / "legacy"
        source.mkdir()
        write_json(source / "lessons.json", [{"lesson": "X works", "confidence": "confirmed"}])
        self.assertEqual(import_legacy(self.store, source, "t")["imported"], 1)
        self.assertEqual(import_legacy(self.store, source, "t")["imported"], 0)
        self.assertEqual(self.store.snapshot()["nodes"][0]["attrs"]["status"], "legacy_unverified")

    def test_counterevidence_is_reserved(self):
        self.patch([node("support-" + str(i), "Method", {"description": "matching"}, "t", "p") for i in range(10)] + [node("counter", "Claim", {"status": "disputed", "scope": "t", "evidence_refs": []}, "t", "p")])
        bundle = self.store.bundle("t", "p", "hypothesis", query="matching", limit=3)
        self.assertIn("counter", [n["id"] for n in bundle["nodes"]])

    def test_role_priorities_keep_counterevidence_and_visibility_boundary(self):
        self.patch([node('obs', 'Observation', {}, 't', 'p'),
                    node('method', 'Method', {}, 't', 'p'),
                    node('counter', 'Failure', {}, 't', 'p'),
                    node('future', 'Observation', {}, 't', 'p'),
                    node('other-task', 'Observation', {}, 'other', 'p')])
        visible = ['obs', 'method', 'counter']
        evidence = self.store.bundle('t', 'p', 'evidence', limit=2, revealed=visible)
        hypothesis = self.store.bundle('t', 'p', 'hypothesis', limit=2, revealed=visible)
        self.assertEqual([n['id'] for n in evidence['nodes']], ['counter', 'obs'])
        self.assertEqual([n['id'] for n in hypothesis['nodes']], ['counter', 'method'])
        self.assertEqual(evidence, self.store.bundle('t', 'p', 'evidence', limit=2, revealed=visible))

    def test_second_hop_precedes_unrelated_fill(self):
        nodes = [node('seed', 'Observation', {'description': 'needle'}, 't', 'p'),
                 node('zz-first', 'Method', {}, 't', 'p'),
                 node('zz-second', 'Component', {}, 't', 'p')]
        nodes += [node('unrelated-' + str(i), 'Observation', {}, 't', 'p') for i in range(5)]
        self.patch(nodes, [edge('e1', 'seed', 'zz-first', 'ABOUT', [], 't', 1),
                           edge('e2', 'zz-first', 'zz-second', 'ABOUT', [], 't', 1)])
        bundle = self.store.bundle('t', 'p', 'evidence', query='needle', limit=3)
        self.assertEqual([n['id'] for n in bundle['nodes']], ['seed', 'zz-first', 'zz-second'])
        self.assertEqual(bundle['retrieval']['selection_reasons']['zz-second'], 'hop_2')


class ArtifactTests(Temporary):
    def test_hash_restore_and_path_escape(self):
        source = self.path / "source"
        source.mkdir()
        (source / "model.py").write_text("public")
        artifacts = Artifacts(self.path / "objects")
        snapshot = artifacts.snapshot(source, ["model.py"])
        artifacts.restore(snapshot, self.path / "restored")
        self.assertEqual((self.path / "restored/model.py").read_text(), "public")
        with self.assertRaises(ContractError):
            artifacts.snapshot(source, ["../source/model.py"])
        (source / "link").symlink_to(source / "model.py")
        with self.assertRaises(ContractError):
            artifacts.snapshot(source, ["link"])
        (artifacts.root / snapshot["model.py"]["sha256"]).write_text("corrupt")
        with self.assertRaises(ContractError):
            artifacts.verify(snapshot["model.py"])


class ReplayTests(unittest.TestCase):
    def nodes(self):
        return [{"node_id": key, "primary_parent_id": parent, "creation_seq": i + 1, "dependencies": deps,
                 "observation": {"score": score, "status": status}, "cost": cost, "protocol_hash": "p"}
                for i, (key, parent, deps, score, status, cost) in enumerate([
                    ("a", "root", [], 0.6, "success", 1), ("b", "root", ["a"], 0.9, "success", 2),
                    ("a2", "a", ["b"], None, "failure", 3)])]

    def test_prefix_dependencies_cost_selection_and_exhaustion(self):
        world = build_world(self.nodes(), workers=2)
        self.assertEqual(world["qualification"], "partial_replay")
        replay = Replay(world, 10)
        self.assertNotIn("0.9", json.dumps(replay.view()))
        replay.step(["root"])
        unsupported = replay.step(["a"])
        self.assertEqual(unsupported["outcomes"][0]["status"], "unsupported")
        replay.step(["root"])
        replay.select("b")
        replay.step(["a"])
        self.assertEqual(replay.report()["historical_cost"], 6)
        self.assertEqual(replay.report()["best_so_far"], 0.9)
        self.assertEqual(replay.report()["selected_score"], 0.9)
        self.assertEqual(replay.step(["a2"])["outcomes"][0]["status"], "exhausted")
        self.assertEqual(replay.step(["root"], action_kind="new_tool")["status"], "unsupported")
        replay.step([])
        self.assertEqual(replay.report()["termination_reason"], "policy_stop")

    def test_general_dag_never_silently_pruned(self):
        nodes = self.nodes()
        nodes.append({**nodes[-1], "node_id": "a3", "creation_seq": 4})
        world = build_world(nodes)
        self.assertEqual(world["topology"], "general_dag")
        with self.assertRaises(ContractError):
            Replay(world, 20)

    def test_batch_visibility_and_budget_atomicity(self):
        replay = Replay(build_world(self.nodes(), workers=2), 2)
        replay.step(["root"])
        replay.step(["root", "a"])
        self.assertEqual(replay.revealed, {"root", "a"})
        self.assertEqual(replay.spent, 1)
        self.assertEqual(replay.stopped, "budget")

    def test_future_dependencies_and_world_tamper(self):
        nodes = self.nodes()
        nodes[0]["dependencies"] = ["a2"]
        with self.assertRaises(ContractError):
            build_world(nodes)
        world = build_world(self.nodes())
        world["nodes"][0]["cost"] = 0
        with self.assertRaises(ContractError):
            Replay(world, 3)


class ScriptedAgent:
    """Deterministic transport substitute; actual model training/evaluation still run."""
    def run(self, role, context, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "request.json", {"role": role, "context": context})
        if role == "evidence":
            from rec_harness.core import CARD_FIELDS
            out = {k: "unknown" for k in CARD_FIELDS["gap"].split()}
            out.update(gap_id="interaction", model_version=context["model_version"], protocol_hash=context["protocol_hash"], status="proposed", observation_refs=[], affected_components=["features"], alternative_explanations=["training"], predicted_observations=["residual interaction"])
        elif role == "hypothesis":
            from rec_harness.core import CARD_FIELDS
            method = {k: "unknown" for k in CARD_FIELDS["method"].split()}
            method.update(method_id="product", origin="agent_hypothesis", sources=[])
            proposal = {k: "unknown" for k in CARD_FIELDS["proposal"].split()}
            proposal.update(proposal_id="p", gap_ids=["interaction"], method_ids=["product"], base_model_id=context["model_version"], action="add", train_protocol_hash=context["protocol_hash"], seeds=context["protocol"]["seeds"], budget=4, controls=["paired baseline"], correctness_checks=["frozen evaluator"])
            out = {"method": method, "proposal": proposal}
        elif role == "experiment":
            path = directory / "model/model.py"
            path.write_text(path.read_text().replace('return [1.0, row[0], row[1]]', 'return [1.0, row[0], row[1], row[0] * row[1]]'))
            out = {"changed_files": ["model.py"], "implementation_checks": ["frozen evaluator"], "limitations": ["toy only"]}
        elif role == "knowledge":
            out = {"patch_json": json.dumps({"base_version": context["evidence"]["graph_snapshot_id"], "nodes": [], "edges": []}), "rationale": "No scientific claim without mechanism control"}
        else:
            out = {"selection": "keep", "conclusion_status": "provisional", "rationale": "paired quality improved", "evidence_refs": [], "unresolved_explanations": ["capacity control missing"]}
        write_json(directory / "response.json", out)
        return out


class RuntimeTests(Temporary):
    def config(self):
        config = read_json(ROOT / "examples/toy/campaign.json")
        config["source"] = str(ROOT / "examples/toy")
        return config

    def test_knowledge_interruption_preserves_committed_decision_without_retraining(self):
        class InterruptKnowledge(ScriptedAgent):
            def run(self,role,context,directory):
                if role=='knowledge':raise KeyboardInterrupt('controller lost')
                return super().run(role,context,directory)
        campaign=self.path/'campaign';Research.init(campaign,self.config())
        research=Research(campaign,InterruptKnowledge());self.addCleanup(research.store.close)
        with self.assertRaises(KeyboardInterrupt):research.run()
        count=sum(e['kind']=='evaluation_completed' for e in research.store.events())
        research.agent=object()  # Recovery may not call any agent.
        recovered=research.run()
        self.assertEqual(recovered['state']['incumbent'],'attempt-1')
        self.assertGreater(recovered['state']['baseline']['score'],0.98)
        self.assertEqual(count,sum(e['kind']=='evaluation_completed' for e in research.store.events()))
        self.assertEqual(recovered['budget']['charged_or_reserved'],10)
        self.assertEqual(read_json(campaign/'research_archive/attempts/attempt-1/decision.json')['selection'],'keep')
        self.assertEqual(research.run(),recovered)

    def test_actual_cpu_loop_archive_budget_and_resume(self):
        campaign = self.path / "campaign"
        Research.init(campaign, self.config())
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        result = research.run(1)
        self.assertEqual(result["state"]["incumbent"], "attempt-1")
        self.assertGreater(result["state"]["baseline"]["score"], 0.98)
        self.assertEqual(result["budget"]["charged_or_reserved"], 10)
        decision = read_json(campaign / "research_archive/attempts/attempt-1/decision.json")
        self.assertEqual(decision["conclusion_status"], "provisional")
        self.assertEqual(len(list((campaign / "tasks").glob("*"))), 6)
        # Reconstruct a stale state projection from the committed completion event.
        old_state = next(e["payload"]["state"] for e in research.store.events() if e["kind"] == "round_started")
        write_json(campaign / "state.json", old_state)
        recovered = research.run(0)
        self.assertEqual(recovered["state"]["incumbent"], "attempt-1")
        self.assertEqual(recovered["budget"]["charged_or_reserved"], 10)
        for event in research.store.events():
            for ref in event["payload"].get("archive", {}).values():
                research.artifacts.verify(ref)

    def test_frozen_evaluator_rejected(self):
        campaign = self.path / "campaign"
        Research.init(campaign, self.config())
        (campaign / "evaluator/evaluate.py").write_text("print('tamper')")
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        with self.assertRaisesRegex(ContractError, "baseline failed"):
            research.run(1)
        self.assertEqual(research.store.budget()["charged_or_reserved"], 2)

    def test_flat_presentation_reaches_all_roles_with_correct_role_retrieval(self):
        from rec_harness.memory_view import restore
        config = self.config()
        config['memory_presentation'] = 'flat'
        campaign = self.path / 'flat'
        Research.init(campaign, config)
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        result = research.run(1)
        self.assertEqual(result['state']['incumbent'], 'attempt-1')
        requests = [read_json(p) for p in (campaign / 'tasks').glob('*/request.json')]
        self.assertEqual(len(requests), 6)
        for request in requests:
            evidence = request['context']['evidence']
            self.assertEqual(evidence['presentation']['mode'], 'flat')
            self.assertNotIn('nodes', evidence)
            self.assertEqual(restore(evidence)['role'], request['role'])

    def test_timeout_retains_cost_logs_and_group_cleanup(self):
        import sys
        result = run_process([sys.executable, "-c", "import time; time.sleep(3)"], self.path, self.path / "timeout", timeout=0.05)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue((self.path / "timeout/process.json").exists())

    def test_codex_cli_fake_executable_structured_output_and_failure(self):
        fake = self.path / "codex"
        fake.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--output-last-message")+1])\np.write_text(json.dumps({"selection":"provisional","conclusion_status":"unknown","rationale":"transport","evidence_refs":[],"unresolved_explanations":[]}))\nprint(json.dumps({"type":"turn.completed","usage":{"input_tokens":4,"output_tokens":5}}))\n')
        fake.chmod(0o755)
        agent = CodexAgent(executable=str(fake), timeout=2)
        result = agent.run("evaluation", {}, self.path / "call")
        self.assertEqual(result["selection"], "provisional")
        self.assertEqual(read_json(self.path / "call/resources.json")["token_usage"]["output_tokens"], 5)


class EvolutionTests(Temporary):
    def test_stable_kernel_rejected(self):
        with self.assertRaises(ContractError):
            validate_policy({"budget": 999})
        with self.assertRaises(ContractError):
            validate_policy({"routing": "skip-evaluation"})
        validate_policy({"prompts": {"evidence": "Show counterevidence"}, "retrieval_limit": 20})

    def test_no_replay_or_overlap_as_online_utility(self):
        evolution = Evolution(self.path)
        self.addCleanup(evolution.store.close)
        plan = {"development_ids": ["dev"], "identity": {"base_llm": "fixed"}, "stages": {"utility": {"minimum_gain": 0.1, "maximum_regression": 0, "maximum_total_cost": 10}}}
        report = {"passed": True, "gain": 0.2, "regression": 0, "total_cost": 5, "evaluation_ids": ["new"], "identity": plan["identity"], "comparison_groups": ["no_tool", "existing_tool", "candidate_tool"], "online": False}
        with self.assertRaises(ContractError):
            evolution._check_report({"kind": "tool"}, plan, "utility", report)
        report["online"] = True
        report["evaluation_ids"] = ["dev"]
        with self.assertRaises(ContractError):
            evolution._check_report({"kind": "tool"}, plan, "utility", report)
        report["evaluation_ids"] = ["new"]
        evolution._check_report({"kind": "tool"}, plan, "utility", report)


class AdditionalRegressionTests(Temporary):
    def test_rebuild_without_snapshots_and_event_tamper(self):
        store = Store(self.path / "memory")
        self.addCleanup(store.close)
        store.patch({"base_version": 0, "nodes": [node("m", "ModelVersion", {}, "t", "p")], "edges": []})
        store.db.execute("DELETE FROM snapshots")
        self.assertEqual(store.rebuild()["graph_version"], 1)
        store.db.execute("UPDATE events SET payload='{}' WHERE seq=1")
        with self.assertRaises(ContractError):
            store.rebuild()

    def test_complete_runtime_to_replay_preserves_root_quality(self):
        from rec_harness.replay import world_from_campaign
        config = read_json(ROOT / "examples/toy/campaign.json")
        config["source"] = str(ROOT / "examples/toy")
        campaign = self.path / "campaign"
        Research.init(campaign, config)
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        research.run(1)
        self.assertGreater(research.verify()["verified_artifacts"], 10)
        world = world_from_campaign(campaign)
        self.assertEqual(world["qualification"], "replay_ready")
        replay = Replay(world, 24)
        self.assertGreater(replay.report()["selected_score"], 0.8)
        self.assertLess(replay.report()["selected_score"], 0.9)
        replay.step(["root"])
        self.assertEqual(replay.report()["historical_cost"], 10)
        replay.select("attempt-1")
        self.assertGreater(replay.report()["selected_score"], 0.98)

    def test_unknown_replay_cost_is_not_free(self):
        nodes = ReplayTests().nodes()
        nodes[0]["cost"] = None
        replay = Replay(build_world(nodes), 3)
        response = replay.step(["root"])
        self.assertEqual(response["outcomes"][0]["reason"], "unknown_resource_cost")
        self.assertEqual(replay.report()["reveals"], 0)
        self.assertEqual(replay.report()["unsupported_ratio"], 1)

    def test_knowledge_retry_does_not_retrain(self):
        config = read_json(ROOT / "examples/toy/campaign.json")
        config["source"] = str(ROOT / "examples/toy")
        campaign = self.path / "campaign"
        Research.init(campaign, config)
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        research.run(1)
        count = len([e for e in research.store.events() if e["kind"] == "evaluation_completed"])
        research.consolidate("attempt-1")
        self.assertEqual(count, len([e for e in research.store.events() if e["kind"] == "evaluation_completed"]))
        self.assertEqual(research.store.budget()["charged_or_reserved"], 11)

    def test_incomplete_attempt_keeps_identity_dependencies_and_cost_ledger(self):
        from rec_harness.replay import world_from_campaign
        config = read_json(ROOT / 'examples/toy/campaign.json')
        config['source'] = str(ROOT / 'examples/toy')
        campaign = self.path / 'incomplete'
        Research.init(campaign, config)
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        research.run(1)
        state = read_json(campaign / 'state.json')
        research.store.event('round_started', {'attempt': 'attempt-2', 'state': state})
        research.store.reserve('attempt-2-evidence-6', 1)
        world = world_from_campaign(campaign)
        self.assertEqual(world['qualification'], 'audit_only')
        self.assertEqual(world['incomplete_attempt_ids'], ['attempt-2'])
        missing = world['nodes'][-1]
        self.assertEqual(missing['dependencies'], ['attempt-1'])
        self.assertEqual(missing['primary_parent_id'], 'attempt-1')
        self.assertEqual(missing['observation'], {'score': None, 'status': 'incomplete'})
        self.assertIsNone(missing['cost'])
        self.assertEqual(sum(r['reserved'] if r['actual'] is None else r['actual'] for r in world['resource_ledger']), 11)
        with self.assertRaisesRegex(ContractError, 'audit-only'):
            Replay(world, 24)

    def test_pre_round_failed_baseline_cost_is_not_lost(self):
        from rec_harness.replay import world_from_campaign
        config = read_json(ROOT / 'examples/toy/campaign.json')
        config['source'] = str(ROOT / 'examples/toy')
        campaign = self.path / 'failed-baseline'
        Research.init(campaign, config)
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        research.store.reserve('eval-root-42', 2)
        research.store.settle('eval-root-42', 2, 'failure')
        world = world_from_campaign(campaign)
        self.assertEqual(world['initial_cost'], 2)
        self.assertEqual(world['initial_cost_task_ids'], ['eval-root-42'])
        self.assertIsNone(world['root_observation'])
        with self.assertRaisesRegex(ContractError, 'initial cost'):
            Replay(world, 1)

    def test_evolution_complete_ladder_and_plan_tamper(self):
        from rec_harness.evolution import STAGES
        from rec_harness.harness_activation import load_active_policy
        class Optimizer:
            def __init__(self, limit=25):
                self.limit = limit
            def run(self, role, context, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                return {"patch_id": "policy", "failure_pattern": "missing counterevidence", "trace_refs": ["trace"],
                        "causal_hypothesis": "context truncation", "editable_scope": ["retrieval_limit"],
                        "patch_diff": json.dumps({'retrieval_limit': self.limit}), "expected_decision_change": "recall contradiction",
                        "regression_risks": ["cost"], "validation_plan": "frozen", "rollback_version": "base"}
        identity = {k: "fixed" for k in ["base_llm", "tools", "start", "graph", "budget", "protocol"]}
        validators = {}
        for stage in STAGES["harness"]:
            path = self.path / (stage + ".py")
            # These are synthetic test fixtures for the gate, NOT measured online utility.
            report = {"passed": True, "gain": 1, "regression": 0, "total_cost": 4, "evaluation_ids": [stage], "identity": identity, "comparison_groups": ["baseline", "candidate"], "online": True}
            path.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(' + repr(json.dumps(report)) + ')\n')
            validators[stage] = {"command": ["{python}", str(path), "{output}"], "timeout": 5, "cost": 1, "minimum_gain": 0.1, "maximum_regression": 0, "maximum_total_cost": 8}
        plan = {"budget": 20, "agent_cost": 1, "development_ids": ["source"], "stages": validators, "identity": identity, "rollback_version": "base"}
        evolution = Evolution(self.path / "evolution")
        self.addCleanup(evolution.store.close)
        evolution.propose("candidate", "harness", {}, plan, agent=Optimizer())
        with self.assertRaises(ContractError):
            evolution.promote("candidate")
        with self.assertRaises(ContractError):
            evolution.validate("candidate", "confirmation")
        for stage in STAGES["harness"]:
            evolution.validate("candidate", stage)
        self.assertEqual(evolution.promote("candidate")["candidate_id"], "candidate")
        budget_before=evolution.store.budget();head_before=evolution.store.verify()
        for stage in STAGES['harness']:evolution.validate('candidate',stage)
        # Recover a lost deployment pointer without publishing a second version.
        (evolution.root/'active-harness.json').unlink()
        self.assertEqual(evolution.promote('candidate')['candidate_id'],'candidate')
        self.assertEqual(evolution.store.budget(),budget_before)
        self.assertEqual(evolution.store.verify(),head_before)
        policy, provenance = load_active_policy(evolution.root, identity)
        self.assertEqual(policy['retrieval_limit'], 25)
        self.assertEqual(provenance['candidate_id'], 'candidate')
        with self.assertRaisesRegex(ContractError, 'validated conditions'):
            load_active_policy(evolution.root, {})
        config = read_json(ROOT / 'examples/toy/campaign.json')
        config.update(source=str(ROOT / 'examples/toy'), harness_registry=str(evolution.root), harness_identity=identity)
        config.pop('research_policy', None)
        campaign = self.path / 'activated'
        Research.init(campaign, config)
        research = Research(campaign, ScriptedAgent())
        self.addCleanup(research.store.close)
        self.assertEqual(research.config['research_policy']['retrieval_limit'], 25)
        self.assertNotIn('harness_registry', research.config)
        evolution.propose("candidate2", "harness", {}, plan, agent=Optimizer(26))
        for stage in STAGES["harness"]:
            evolution.validate("candidate2", stage)
        evolution.promote("candidate2")
        self.assertEqual(load_active_policy(evolution.root, identity)[0]['retrieval_limit'], 26)
        # Existing campaign stays frozen even after registry changes.
        self.assertEqual(research.context('evidence', read_json(campaign / 'state.json'))['evidence']['retrieval']['limit'], 25)
        self.assertEqual(evolution.rollback("harness")["candidate_id"], "candidate")
        self.assertEqual(load_active_policy(evolution.root, identity)[0]['retrieval_limit'], 25)
        pointer = read_json(evolution.root / 'active-harness.json')
        write_json(evolution.root / 'active-harness.json', {**pointer, 'candidate_id': 'candidate2'})
        with self.assertRaisesRegex(ContractError, 'pointer differs'):
            load_active_policy(evolution.root, identity)
        write_json(evolution.root / 'active-harness.json', pointer)
        plan["stages"]["confirmation"]["minimum_gain"] = 0
        write_json(self.path / "evolution/candidates/candidate/plan.json", plan)
        with self.assertRaises(ContractError):
            evolution.promote("candidate")


if __name__ == "__main__":
    unittest.main()
