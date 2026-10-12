"""Frozen historical worlds. Only observed outcomes, never counterfactual scores."""
from __future__ import annotations

from .core import ContractError, digest, number, public_only, required


def build_world(nodes, *, root_id="root", workers=1, artifacts=None, root_observation=None, initial_cost=0):
    if type(workers) is not int or workers < 1:
        raise ContractError("workers must be a positive integer")
    number(initial_cost, "initial historical cost", 0)
    public_only(root_observation)
    public_only(nodes)
    by_id, children = {}, {}
    qualification, reasons = "replay_ready", []
    for node in nodes:
        required(node, ["node_id", "primary_parent_id", "creation_seq", "dependencies", "observation", "protocol_hash"])
        if "cost" not in node:
            raise ContractError("missing cost field; unknown must be null")
        if node["node_id"] in by_id or node["node_id"] == root_id:
            raise ContractError("duplicate replay node")
        if type(node["creation_seq"]) is not int:
            raise ContractError("invalid creation sequence")
        by_id[node["node_id"]] = node
        children.setdefault(node["primary_parent_id"], []).append(node)
        if node["cost"] is None:
            qualification = "partial_replay"
            reasons.append("unknown_resource_cost")
        else:
            number(node["cost"], "historical cost", 0)
        if node["observation"].get("score") is not None:
            number(node["observation"]["score"], "public score")
        if node.get("request_ref") is None or node.get("workspace_before") is None or node.get("workspace_after") is None or node.get("context_dependencies_complete") is not True:
            qualification = "partial_replay"
            reasons.append("missing_request_workspace_or_dependencies")
        if artifacts is None:
            qualification = "partial_replay"
            reasons.append("artifact_integrity_not_verified")
        else:
            for ref in [node.get("request_ref"), *node.get("workspace_before", {}).values(), *node.get("workspace_after", {}).values()]:
                if ref is not None:
                    artifacts.verify(ref)
    if len({n["protocol_hash"] for n in nodes}) > 1:
        raise ContractError("world mixes protocols")
    if len({n["creation_seq"] for n in nodes}) != len(nodes):
        raise ContractError("ambiguous creation order")
    for node in nodes:
        parent = node["primary_parent_id"]
        if parent != root_id and parent not in by_id:
            raise ContractError("missing parent")
        dependencies = set(node["dependencies"])
        if not dependencies.issubset(set(by_id) | {root_id}):
            raise ContractError("missing dependency")
        for previous in dependencies | {parent}:
            if previous != root_id and by_id[previous]["creation_seq"] >= node["creation_seq"]:
                raise ContractError("cycle or future dependency")
    topology = "general_dag" if any(parent != root_id and len(group) > 1 for parent, group in children.items()) else "branch_chain"
    if topology == "general_dag":
        qualification = "audit_only"
        reasons.append("internal_multi_successor_requires_explicit_adapter")
    body = {"root_id": root_id, "root_observation": root_observation, "initial_cost": initial_cost, "workers": workers, "nodes": sorted(nodes, key=lambda n: n["creation_seq"]), "topology": topology,
            "qualification": qualification, "reasons": sorted(set(reasons)), "builder_version": "1", "batch_semantics": "synchronous"}
    return {**body, "hash": digest(body)}


class Replay:
    def __init__(self, world, budget, direction="maximize"):
        if digest({k: v for k, v in world.items() if k != "hash"}) != world["hash"]:
            raise ContractError("world hash mismatch")
        if world["topology"] != "branch_chain":
            raise ContractError("general DAG is audit-only; no silent branch pruning")
        if world['qualification'] == 'audit_only':
            raise ContractError('audit-only world cannot execute policy replay')
        if direction not in {"maximize", "minimize"}:
            raise ContractError("invalid quality direction")
        number(budget, "replay budget", 0)
        self.world, self.budget, self.direction = world, budget, direction
        self.root = world["root_id"]
        self.revealed, self.frontier = {self.root}, set()
        self.spent, self.batches, self.unsupported, self.actions = world.get("initial_cost", 0), 0, 0, 0
        self.requested_transitions = 0
        if self.spent > budget:
            raise ContractError("budget below frozen initial cost")
        self.stopped, self.selected = None, None
        self.observations = {self.root: world["root_observation"]} if world.get("root_observation") is not None else {}
        if self.observations:
            self.selected = self.root
        self.history = []

    def view(self):
        return {"revealed_node_ids": sorted(self.revealed), "legal_parents": [self.root] + sorted(self.frontier),
                "observations": dict(self.observations), "remaining_budget": self.budget - self.spent,
                "pending": [], "termination_reason": self.stopped, "selected": self.selected}

    def step(self, parents, *, action_kind="continue"):
        if self.stopped:
            raise ContractError("replay already stopped")
        self.actions += 1
        self.requested_transitions += max(1, len(parents)) if action_kind != "continue" else len(parents)
        if action_kind != "continue":
            self.unsupported += 1
            return {"status": "unsupported", "reason": "new_prompt_tool_or_code_requires_online_execution"}
        if not parents:
            self.stopped = "policy_stop"
            return {"status": "stopped"}
        if len(parents) > self.world["workers"] or len(parents) != len(set(parents)):
            raise ContractError("invalid synchronous batch")
        before = set(self.revealed)
        selected = []
        for parent in parents:
            if parent not in before or (parent != self.root and parent not in self.frontier):
                raise ContractError("action is not root/frontier")
            successors = [n for n in self.world["nodes"] if n["primary_parent_id"] == parent and n["node_id"] not in before]
            if not successors:
                selected.append({"parent": parent, "status": "exhausted", "reason": "no_recorded_continuation"})
                continue
            node = successors[0]
            if not set(node["dependencies"]).issubset(before):
                self.unsupported += 1
                selected.append({"parent": parent, "status": "unsupported", "reason": "unrevealed_dependency"})
                continue
            if node["cost"] is None:
                self.unsupported += 1
                selected.append({"parent": parent, "status": "unsupported", "reason": "unknown_resource_cost"})
                continue
            selected.append({"parent": parent, "status": "ready", "node": node})
        cost = sum(s["node"]["cost"] for s in selected if s["status"] == "ready")
        if self.spent + cost > self.budget:
            self.stopped = "budget"
            return {"status": "budget_exhausted"}
        self.batches += 1
        outcomes = []
        for item in selected:
            if item["status"] != "ready":
                outcomes.append(item)
                continue
            node = item["node"]
            self.revealed.add(node["node_id"])
            self.frontier.discard(item["parent"])
            self.frontier.add(node["node_id"])
            self.observations[node["node_id"]] = node["observation"]
            self.spent += node["cost"]
            outcomes.append({"status": "revealed", "node_id": node["node_id"], "observation": node["observation"], "historical_cost": node["cost"]})
        result = {"status": "completed", "outcomes": outcomes}
        self.history.append({"parents": parents, "result": result})
        return result

    def select(self, node_id):
        if node_id not in self.observations or self.observations[node_id].get("status") != "success":
            raise ContractError("cannot select unrevealed or invalid model")
        self.selected = node_id

    def report(self):
        scores = [o["score"] for o in self.observations.values() if o.get("status") == "success" and o.get("score") is not None]
        best = (max(scores) if self.direction == "maximize" else min(scores)) if scores else None
        return {"best_so_far": best, "selected_score": self.observations[self.selected].get("score") if self.selected else None,
                "reveals": len(self.revealed) - 1, "historical_cost": self.spent, "batches": self.batches,
                "unsupported_count": self.unsupported, "requested_batches": self.actions,
                "requested_transitions": self.requested_transitions, "unsupported_ratio": self.unsupported / self.requested_transitions if self.requested_transitions else 0,
                "termination_reason": self.stopped, "world_qualification": self.world["qualification"]}


def world_from_campaign(directory, workers=1):
    """Conservative conversion; preserve incomplete attempts as partial/audit archives."""
    from pathlib import Path
    from .core import Artifacts, read_json
    from .store import Store
    directory = Path(directory)
    artifacts = Artifacts(directory / "artifacts" / "objects")
    store = Store(directory / "memory")
    try:
        integrity = store.verify()
        events = store.events()
        reservations = {e['payload']['task_id']: e['payload']['amount'] for e in events if e['kind'] == 'budget_reserved'}
        settled = {e["payload"]["task_id"]: e["payload"] for e in events if e["kind"] == "budget_settled"}
        started = [e for e in events if e["kind"] == "round_started"]
        nodes = []
        earlier = []
        incomplete, assigned = [], set()
        for order, event in enumerate(started, 1):
            attempt = event["payload"]["attempt"]
            path = directory / "research_archive" / "attempts" / attempt
            complete = (path / 'node.json').exists() and (path / 'observations.json').exists()
            if complete:
                metadata = read_json(path / 'node.json')
                observation = read_json(path / 'observations.json')
            else:
                incomplete.append(attempt)
                metadata = {'primary_parent_id': event['payload']['state']['incumbent'],
                            'workspace_before': event['payload']['state']['workspace'],
                            'workspace_after': {}, 'protocol_hash': read_json(directory / 'campaign.json')['protocol_hash']}
                observation = {'score': None, 'valid': False}
            keys = {key for key in reservations if key.startswith(attempt + '-') or key.startswith('eval-' + attempt + '-')}
            assigned.update(keys)
            costs = [settled.get(key, {'actual': None}) for key in keys]
            cost = sum(c['actual'] for c in costs) if costs and all(c['actual'] is not None for c in costs) else None
            decision_context = directory / "research_archive" / "decisions" / attempt / "context.json"
            node = {"node_id": attempt, "primary_parent_id": metadata["primary_parent_id"], "creation_seq": order,
                    # Full-history dependencies are conservative and prevent cross-branch leakage.
                    "dependencies": list(earlier), "context_dependencies_complete": complete and decision_context.exists(),
                    "workspace_before": metadata["workspace_before"], "workspace_after": metadata["workspace_after"],
                    "request_ref": artifacts.put(decision_context) if decision_context.exists() else None,
                    "protocol_hash": metadata["protocol_hash"], "cost": cost,
                    "observation": {"score": observation["score"], "status": ('success' if observation['valid'] else 'failure') if complete else 'incomplete'},
                    "source_task_ids": sorted(keys)}
            nodes.append(node)
            earlier.append(attempt)
        baseline = started[0]["payload"]["state"].get("baseline") if started else None
        root_observation = {"score": baseline["score"], "status": "success"} if baseline and baseline["valid"] else None
        unassigned = set(reservations) - assigned
        initial_cost = sum(settled[key]['actual'] if key in settled and settled[key]['actual'] is not None else reservations[key] for key in unassigned)
        world = build_world(nodes, workers=workers, artifacts=artifacts, root_observation=root_observation, initial_cost=initial_cost)
        if incomplete:
            world['qualification'] = 'audit_only'
            world['reasons'].append('incomplete_attempts_have_no_observed_transition')
        world['incomplete_attempt_ids'] = incomplete
        world['initial_cost_task_ids'] = sorted(unassigned)
        world['resource_ledger'] = [{'task_id': key, 'reserved': reserved,
            'actual': settled.get(key, {}).get('actual'),
            'status': settled.get(key, {}).get('status', 'running')} for key, reserved in sorted(reservations.items())]
        if store.verify() != integrity:
            raise ContractError('campaign changed during world conversion')
        world["source_event_head"] = integrity['head']
        world["hash"] = digest({k: v for k, v in world.items() if k != "hash"})
        return world
    finally:
        store.close()
