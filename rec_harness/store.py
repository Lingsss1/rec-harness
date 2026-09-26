"""Transactional event source and rebuildable property graph (SQLite, stdlib)."""
from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .retrieval import select
from .core import ContractError, canonical, digest, identifier, number, public_only, required, write_json

NODE_TYPES = set("Context ModelVersion Component Observation GapHypothesis Method Proposal Experiment Claim Failure MeasurementGap ToolCandidate ToolValidation DecisionTrace DecisionFailureHypothesis HarnessVersion HarnessPatch HarnessValidation ResearchMethodVersion Source ToolVersion LegacyRecord".split())
RELATIONS = set("HAS_COMPONENT FLOWS_TO DERIVED_FROM EVALUATES COMPARED_WITH PRODUCES SUPPORTS CONTRADICTS LOCATED_AT ALTERNATIVE_TO TARGETS USES MODIFIES TESTS APPLIES_IN ABOUT SUPERSEDES HAS_FAILURE EXPOSES BLOCKS ADDRESSES BASED_ON PROMOTED_FROM MEASURED_BY GENERATED_BY SUPPORTED_BY".split())


class Store:
    def __init__(self, root, *, read_only=False):
        self.root = Path(root)
        self.read_only=read_only
        if read_only:
            path=(self.root/'memory.sqlite').resolve()
            self.db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30,isolation_level=None)
            self.db.row_factory=sqlite3.Row
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "memory.sqlite", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, kind TEXT NOT NULL,
                payload TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshots(version INTEGER PRIMARY KEY, hash TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, type TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS edges(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS budget(id TEXT PRIMARY KEY, reserved REAL NOT NULL,
                actual REAL, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        if not self.db.execute("SELECT 1 FROM snapshots LIMIT 1").fetchone():
            empty = {"nodes": [], "edges": []}
            self.db.execute("INSERT OR IGNORE INTO snapshots VALUES(0,?,?)", (digest(empty), canonical(empty)))
        self.export_events()  # recover an interrupted JSONL projection from committed SQL events

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        if self.read_only:raise ContractError('read-only store cannot mutate events or projections')
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.export_events()

    def _event(self, kind, payload):
        public_only(payload)
        row = self.db.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        seq, previous = (row["seq"] + 1, row["hash"]) if row else (1, "0" * 64)
        timestamp = datetime.now(timezone.utc).isoformat()
        body = {"seq": seq, "kind": kind, "payload": payload, "previous_hash": previous, "created_at": timestamp}
        sha = digest(body)
        self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (seq, kind, canonical(payload), previous, sha, timestamp))
        return {**body, "hash": sha}

    def event(self, kind, payload):
        with self.transaction():
            return self._event(kind, payload)

    def events(self):
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in self.db.execute("SELECT * FROM events ORDER BY seq")]

    def export_events(self):
        if self.read_only:raise ContractError('read-only store cannot export projections')
        # Hold a database write lock while projecting, so concurrent writers cannot regress the view.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.events()
            target = self.root / "events.jsonl"
            temporary = target.with_suffix(".tmp")
            temporary.write_text("".join(canonical(r) + "\n" for r in rows))
            temporary.replace(target)
        finally:
            self.db.execute("COMMIT")

    def verify(self):
        if self.db.in_transaction:return self._verify_snapshot()
        self.db.execute('BEGIN')
        try:return self._verify_snapshot()
        finally:self.db.execute('ROLLBACK')

    def _verify_snapshot(self):
        previous = "0" * 64
        count=0
        for seq, row in enumerate(self.db.execute('SELECT * FROM events ORDER BY seq'), 1):
            event={**dict(row),'payload':json.loads(row['payload'])}
            sha = event.pop("hash")
            if event["seq"] != seq or event["previous_hash"] != previous or digest(event) != sha:
                raise ContractError("event chain is corrupt")
            previous = sha
            count=seq
        for row in self.db.execute("SELECT * FROM snapshots"):
            if digest(json.loads(row["body"])) != row["hash"]:
                raise ContractError("graph snapshot is corrupt")
        return {"events": count, "head": previous, "graph_version": self.version}

    @property
    def version(self):
        return self.db.execute("SELECT MAX(version) FROM snapshots").fetchone()[0]

    def snapshot(self, version=None):
        version = self.version if version is None else version
        row = self.db.execute("SELECT * FROM snapshots WHERE version=?", (version,)).fetchone()
        if row is None:
            raise ContractError("unknown graph version")
        return {"version": version, "hash": row["hash"], **json.loads(row["body"])}

    def patch(self, patch, *, accompanying_events=()):
        """Commit graph changes and their controller receipts in one transaction."""
        public_only(patch)
        required(patch, ["base_version", "nodes", "edges"])
        try:
            with self.transaction():
                if patch["base_version"] != self.version:
                    raise ContractError("stale graph patch")
                old = self.snapshot()
                nodes = {n["id"]: n for n in old["nodes"]}
                edges = {e["id"]: e for e in old["edges"]}
                for node in patch["nodes"]:
                    required(node, ["id", "type", "attrs"])
                    identifier(node["id"])
                    if node["type"] not in NODE_TYPES:
                        raise ContractError("unknown node type")
                    attrs = node["attrs"]
                    required(attrs, ["task_id", "protocol_hash", "origin"])
                    if attrs["origin"] not in {"paper_claim", "agent_hypothesis", "local_evidence", "legacy_unverified"}:
                        raise ContractError("unknown origin")
                    previous = nodes.get(node["id"])
                    if previous and previous != node and previous["type"] in {"Experiment", "Observation", "Failure", "DecisionTrace"}:
                        raise ContractError("facts are immutable; append a new record")
                    if previous and previous["type"] != node["type"]:
                        raise ContractError("node identity cannot change type")
                    nodes[node["id"]] = node
                for edge in patch["edges"]:
                    required(edge, ["id", "source", "target", "relation", "attrs"])
                    identifier(edge["id"])
                    if edge["relation"] not in RELATIONS or edge["source"] not in nodes or edge["target"] not in nodes:
                        raise ContractError("invalid graph relation or dangling reference")
                    required(edge["attrs"], ["scope", "evidence_refs", "origin", "status", "created_at", "revision_id"])
                    for ref in edge["attrs"]["evidence_refs"]:
                        if ref not in nodes:
                            raise ContractError("missing edge evidence")
                    if edge["relation"] in {"SUPPORTS", "CONTRADICTS", "COMPARED_WITH"}:
                        a, b = nodes[edge["source"]]["attrs"], nodes[edge["target"]]["attrs"]
                        if (a["task_id"], a["protocol_hash"]) != (b["task_id"], b["protocol_hash"]):
                            raise ContractError("incomparable evidence")
                    edges[edge["id"]] = edge
                for node in nodes.values():
                    if node["type"] == "Claim":
                        self._claim(node, nodes)
                body = {"nodes": sorted(nodes.values(), key=lambda n: n["id"]), "edges": sorted(edges.values(), key=lambda e: e["id"])}
                version = self.version + 1
                self.db.execute("INSERT INTO snapshots VALUES(?,?,?)", (version, digest(body), canonical(body)))
                self._project(body)
                self._event("graph_committed", {"version": version, "before_hash": old["hash"], "after_hash": digest(body), "patch": patch})
                for kind, payload in accompanying_events:
                    self._event(kind, payload)
            return self.snapshot(version)
        except ContractError as error:
            self.event("graph_rejected", {"patch_hash": digest(patch), "reason": str(error)})
            raise

    def _claim(self, node, nodes):
        attrs = node["attrs"]
        required(attrs, ["status", "scope", "evidence_refs"])
        statuses = {"provisional", "supported", "confirmed_in_scope", "disputed", "demoted", "superseded", "retracted", "legacy_unverified"}
        if attrs["status"] not in statuses:
            raise ContractError("invalid claim status")
        experiments = []
        for ref in attrs["evidence_refs"]:
            if ref not in nodes:
                raise ContractError("claim evidence missing")
            evidence = nodes[ref]
            if evidence["type"] == "Experiment":
                e = evidence["attrs"]
                if e.get("valid") is True and e.get("protocol_hash") == attrs["protocol_hash"] and e.get("task_id") == attrs["task_id"]:
                    experiments.append(e)
        if attrs["status"] in {"supported", "confirmed_in_scope"}:
            # Explicit development evidence remains available for interpretation,
            # but cannot be spent again as independent confirmation. Older local
            # protocols without partitions retain their existing run contract.
            qualified = [e for e in experiments
                         if e.get("effect_gate_passed") is True
                         and e.get("origin") == "local_evidence"
                         and e.get("partition") in {None, "confirmation"}]
            runs = {e.get("independent_run_id") for e in qualified}
            runs.discard(None)
            if len(runs) < 2:
                raise ContractError("claim requires two independent comparable effect-qualified runs")
            if attrs["status"] == "confirmed_in_scope" and not any(
                    e.get("mechanism_control") is True and e.get("independent_run_id") is not None
                    for e in qualified):
                raise ContractError("mechanism control required")

    def _project(self, body):
        self.db.execute("DELETE FROM nodes")
        self.db.execute("DELETE FROM edges")
        self.db.executemany("INSERT INTO nodes VALUES(?,?,?)", [(n["id"], n["type"], canonical(n)) for n in body["nodes"]])
        self.db.executemany("INSERT INTO edges VALUES(?,?)", [(e["id"], canonical(e)) for e in body["edges"]])

    def rollback(self, version):
        with self.transaction():
            snapshot = self.snapshot(version)
            body = {k: snapshot[k] for k in ("nodes", "edges")}
            new_version = self.version + 1
            self.db.execute("INSERT INTO snapshots VALUES(?,?,?)", (new_version, digest(body), canonical(body)))
            self._project(body)
            self._event("graph_rollback", {"restored_version": version, "version": new_version})
        return self.snapshot(new_version)

    def rebuild(self):
        # Events, not existing graph tables/snapshots, are authoritative.
        versions = {0: {"nodes": [], "edges": []}}
        previous = "0" * 64
        for seq, event in enumerate(self.events(), 1):
            claimed = event.pop("hash")
            if event["seq"] != seq or event["previous_hash"] != previous or digest(event) != claimed:
                raise ContractError("cannot rebuild from corrupt event chain")
            previous = claimed
            payload = event["payload"]
            if event["kind"] == "graph_committed":
                patch = payload["patch"]
                base = versions[patch["base_version"]]
                nodes = {n["id"]: n for n in base["nodes"]}
                edges = {e["id"]: e for e in base["edges"]}
                nodes.update({n["id"]: n for n in patch["nodes"]})
                edges.update({e["id"]: e for e in patch["edges"]})
                body = {"nodes": sorted(nodes.values(), key=lambda n: n["id"]), "edges": sorted(edges.values(), key=lambda e: e["id"])}
                if digest(base) != payload["before_hash"] or digest(body) != payload["after_hash"]:
                    raise ContractError("graph event hash mismatch")
                versions[payload["version"]] = body
            elif event["kind"] == "graph_rollback":
                versions[payload["version"]] = versions[payload["restored_version"]]
        with self.transaction():
            self.db.execute("DELETE FROM snapshots")
            self.db.executemany("INSERT INTO snapshots VALUES(?,?,?)", [(v, digest(body), canonical(body)) for v, body in versions.items()])
            self._project(versions[max(versions)])
        return self.verify()

    def bundle(self, task_id, protocol_hash, role, *, version=None, query="", limit=24, revealed=None):
        if limit < 1:
            raise ContractError("context limit must be positive")
        snapshot = self.snapshot(version)
        nodes = {n["id"]: n for n in snapshot["nodes"] if n["attrs"]["task_id"] == task_id and n["attrs"]["protocol_hash"] == protocol_hash}
        if revealed is not None:
            visible = set(revealed)
            nodes = {key: n for key, n in nodes.items() if set(n["attrs"].get("evidence_dependency_ids", [key])).issubset(visible)}
        edges = [e for e in snapshot["edges"] if e["source"] in nodes and e["target"] in nodes]
        counter = {e["source"] for e in edges if e["relation"] in {"CONTRADICTS", "ALTERNATIVE_TO"}}
        counter |= {n["id"] for n in nodes.values() if n["attrs"].get("status") in {"disputed", "demoted"} or n["type"] == "Failure" or n["attrs"].get("counterevidence") is True}
        selected, audit = select(nodes, edges, counter, role, query, limit, canonical)
        return {"graph_snapshot_id": snapshot["version"], "graph_hash": snapshot["hash"], "role": role,
                "task_id": task_id, "protocol_hash": protocol_hash, "nodes": [nodes[k] for k in selected],
                "edges": [e for e in edges if e["source"] in selected and e["target"] in selected],
                "retrieval": {"query": query, "limit": limit, "retrieved_node_ids": selected, "counterevidence_available": len(counter), "counterevidence_omitted": len(counter - set(selected)), **audit}}

    def configure_budget(self, limit):
        number(limit, "budget", 0)
        with self.transaction():
            row = self.db.execute("SELECT value FROM settings WHERE key='budget'").fetchone()
            if row and float(row[0]) != limit:
                raise ContractError("budget is frozen")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('budget',?)", (str(limit),))

    def budget(self):
        row = self.db.execute("SELECT value FROM settings WHERE key='budget'").fetchone()
        if not row:
            raise ContractError("budget not configured")
        used = self.db.execute("SELECT COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE actual END),0) FROM budget").fetchone()[0]
        return {"limit": float(row[0]), "charged_or_reserved": used, "remaining": float(row[0]) - used}

    def reserve(self, task_id, amount):
        identifier(task_id)
        number(amount, "reservation", 0)
        with self.transaction():
            if self.db.execute("SELECT 1 FROM budget WHERE id=?", (task_id,)).fetchone():
                raise ContractError("task already charged; retry with a new ID")
            if amount > self.budget()["remaining"]:
                raise ContractError("budget exhausted")
            self.db.execute("INSERT INTO budget VALUES(?,?,NULL,'running')", (task_id, amount))
            self._event("budget_reserved", {"task_id": task_id, "amount": amount})

    def settle(self, task_id, actual, status):
        if actual is not None:
            number(actual, "cost", 0)
        with self.transaction():
            row = self.db.execute("SELECT * FROM budget WHERE id=?", (task_id,)).fetchone()
            if not row or row["status"] != "running":
                raise ContractError("reservation not running")
            self.db.execute("UPDATE budget SET actual=?,status=? WHERE id=?", (actual, status, task_id))
            self._event("budget_settled", {"task_id": task_id, "actual": actual, "status": status, "reserved": row["reserved"]})

    def recover(self):
        tasks = [r[0] for r in self.db.execute("SELECT id FROM budget WHERE status='running'")]
        for task in tasks:
            self.settle(task, None, "interrupted")
        return tasks

    def export_compat(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        write_json(destination / "graph.json", snapshot)
        write_json(destination / "lessons.json", [n for n in snapshot["nodes"] if n["type"] == "Claim"])
        with (destination / "results.tsv").open("w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["experiment_id", "protocol_hash", "status", "score"])
            for n in snapshot["nodes"]:
                if n["type"] == "Experiment":
                    a = n["attrs"]
                    writer.writerow([n["id"], a["protocol_hash"], a.get("status", "unknown"), a.get("score", "unknown")])
