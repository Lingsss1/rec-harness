"""Canonical encoding, artifact storage, public input boundaries and contracts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


class ContractError(ValueError):
    pass


class RoleTransportFailure(ContractError):
    """Host/service failure requiring explicit repair before further roles."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical(value) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ContractError(f"invalid identifier: {value!r}")
    return value


def number(value, name="value", minimum=None):
    try:finite=isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
    except OverflowError:finite=False
    if not finite:
        raise ContractError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    return value


def required(obj, fields):
    if not isinstance(obj, dict):
        raise ContractError("expected an object")
    for key in fields:
        if key not in obj or obj[key] is None or obj[key] == "":
            raise ContractError(f"missing {key}")


def public_only(value):
    """Reject explicit private labels. This is validation, not an OS security boundary."""
    if isinstance(value, dict):
        if value.get("visibility", "public") != "public":
            raise ContractError("only public evidence is accepted")
        for key, child in value.items():
            if key.lower() in {"hidden", "hidden_score", "hidden_metrics", "test_score", "secret", "api_key"}:
                raise ContractError(f"forbidden evidence field: {key}")
            public_only(child)
    elif isinstance(value, list):
        for child in value:
            public_only(child)
    canonical(value)  # rejects nonfinite JSON


def safe_file(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ContractError("path must be relative and confined")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError("symlinks are not public artifacts")
    if not current.is_file():
        raise ContractError(f"missing file: {relative}")
    return current


class Artifacts:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ContractError("artifact must be a regular file")
        sha = file_hash(path)
        target = self.root / sha
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".pending-")
            try:
                with path.open("rb") as src, os.fdopen(fd, "wb") as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                if file_hash(temporary) != sha:
                    raise ContractError("source artifact changed while archiving")
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass  # Another writer published the same content.
            finally:
                os.unlink(temporary)
        if file_hash(target) != sha:
            raise ContractError("artifact hash mismatch")
        return {"sha256": sha, "size": target.stat().st_size, "availability": "available"}

    def verify(self, ref):
        sha = ref.get("sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ContractError("invalid content hash")
        path = self.root / sha
        if not path.is_file() or path.stat().st_size != ref["size"] or file_hash(path) != sha:
            raise ContractError(f"unavailable/corrupt artifact: {sha}")
        return path

    def snapshot(self, root, paths):
        return {name: self.put(safe_file(root, name)) for name in sorted(set(paths))}

    def restore(self, snapshot, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name, ref in snapshot.items():
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ContractError("invalid snapshot path")
            target = destination / rel
            current = destination
            for part in rel.parts:
                current = current / part
                if current.is_symlink():
                    raise ContractError("symlink in restore destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.verify(ref).read_bytes())


CARD_FIELDS = {
    "gap": "gap_id model_version task_context protocol_hash observation_refs suspected_capability affected_components alternative_explanations predicted_observations cheapest_discriminating_test status",
    "method": "method_id mechanism prerequisites insertion_points tensor_constraints cost origin sources",
    "proposal": "proposal_id gap_ids method_ids base_model_id action target_component_ids mechanism insertion_contract expected_primary_and_slice_effects alternative_explanations falsification_condition controls correctness_checks train_protocol_hash seeds tuning_allowance budget guardrails",
    "decision": "selection conclusion_status rationale evidence_refs unresolved_explanations",
    "measurement_gap": "measurement_gap_id related_gap_ids experiment_refs blocked_decision competing_explanations current_tools_and_versions observed_limitation required_information available_artifacts cheapest_test development_budget acceptance_contract",
    "harness_patch": "patch_id failure_pattern trace_refs causal_hypothesis editable_scope patch_diff expected_decision_change regression_risks validation_plan rollback_version",
}


METHOD_SOURCE_CONTRACT = {
    'schema':'locatable-method-sources-v1',
    'requirement':'Each paper_claim source must be a nonempty citation string containing a specific HTTP(S) URL, DOI or arXiv identifier. Include version/section where available. Missing sources require agent_hypothesis, not an invented citation.',
    'qualification':'Identifier syntax only; source existence, primary-source status, content, version and applicability remain unverified.'}


def validate_method_sources(card):
    from urllib.parse import urlsplit
    sources=card['sources']
    if not isinstance(sources,list) or any(not isinstance(s,str) or not s.strip() for s in sources):
        raise ContractError('method sources must be a list of nonempty citation strings')
    if card['origin']!='paper_claim':return
    if not sources:raise ContractError('paper_claim requires locatable sources; otherwise use agent_hypothesis')
    for citation in sources:
        located=bool(re.search(r'\b10\.\d{4,9}/[^\s]+',citation,re.I) or
                     re.search(r'\barxiv\s*:\s*(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]*/\d{7}(?:v\d+)?)\b',citation,re.I))
        for raw in re.findall(r'https?://[^\s<>]+',citation,re.I):
            try:
                url=urlsplit(raw.rstrip('.,;)]'))
                located=located or bool(url.hostname and not url.username and url.path.strip('/'))
            except ValueError:continue
        if not located:raise ContractError('paper_claim source lacks a specific URL, DOI or arXiv identifier')


def validate_card(kind, card, *, require_method_sources=False):
    required(card, CARD_FIELDS[kind].split())
    public_only(card)
    if kind == "gap" and card["status"] not in {"proposed", "supported", "contradicted", "mitigated", "unknown"}:
        raise ContractError("invalid gap status")
    if kind == "method" and card["origin"] not in {"paper_claim", "agent_hypothesis", "local_evidence"}:
        raise ContractError("invalid method provenance")
    if kind == 'method' and require_method_sources:validate_method_sources(card)
    if kind == "proposal":
        if card["action"] not in {"add", "remove", "replace", "move", "reconnect", "objective", "training_method"}:
            raise ContractError("invalid proposal action")
        for field in ("gap_ids", "method_ids", "controls", "correctness_checks", "seeds"):
            if not isinstance(card[field], list) or not card[field]:
                raise ContractError(f"{field} must be a nonempty list")
        number(card["budget"], "proposal budget", 0)
    if kind == "decision":
        if card["selection"] not in {"keep", "discard", "provisional", "crash", "invalid"}:
            raise ContractError("invalid model decision")
        if card["conclusion_status"] not in {"provisional", "supported", "contradicted", "unknown"}:
            raise ContractError("invalid conclusion status")
    return card
