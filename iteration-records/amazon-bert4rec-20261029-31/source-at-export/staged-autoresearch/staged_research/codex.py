"""Native Codex editing with per-logical-session homes and real exec resume.

This adapter is for trusted process mode. Workspace-write/read-only permissions
constrain edits, but do not constitute an OS boundary against reading host data.
Credentials and native conversations remain in private session homes, never in
the scientific evidence packet.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
import shutil
import uuid

from .contracts import validate_shape
from .local import LocalRuntime, clean_environment, invoke_process
from .storage import ProtocolError, now, read, write


def service_schema(schema):
    result = copy.deepcopy(schema)
    if result.get("type") == "object":
        required = set(result.get("required", result["properties"]))
        for name, value in result["properties"].items():
            child = service_schema(value)
            if name not in required:
                child = {"anyOf": [child, {"type": "null"}]}
            result["properties"][name] = child
        result["required"] = list(result["properties"])
    elif result.get("type") == "array":
        result["items"] = service_schema(result["items"])
    return result


def remove_optional_nulls(value, schema):
    if schema.get("type") == "object" and isinstance(value, dict):
        required = set(schema.get("required", schema["properties"]))
        return {k: remove_optional_nulls(v, schema["properties"][k])
                for k, v in value.items() if not (v is None and k not in required)}
    if schema.get("type") == "array" and isinstance(value, list):
        return [remove_optional_nulls(v, schema["items"]) for v in value]
    return value


class CodexRuntime(LocalRuntime):
    def call_agent(self, *, work, evidence, session_home, session, resume, role, schema, prompt, receipt_dir):
        if self.isolated:
            raise ProtocolError("Codex runtime currently requires trusted process mode")
        session_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        session_home.chmod(0o700)
        profile = session_home / ".codex"
        profile.mkdir(mode=0o700, exist_ok=True)
        mapping_path = session_home / "native-session.json"
        pending = session_home / "pending-call.json"
        if pending.exists():
            raise ProtocolError("Interrupted native Codex call: inspect its private receipt and session before retrying")
        mapping = read(mapping_path) if mapping_path.exists() else None
        if resume:
            if not mapping or mapping["logical_session"] != session:
                raise ProtocolError("Missing or mismatched native Codex session for resume")
        elif mapping is not None:
            raise ProtocolError("Fresh context cannot reuse a populated Codex session home")
        # Auth and the public model catalog are staged. Conversations/config/instructions are not
        # copied into a new native home, so C/D get an actual new native thread.
        if not (profile / "auth.json").exists():
            auth = Path(self.agent_config.get("auth_root", "/root/.codex")) / "auth.json"
            if auth.exists():
                shutil.copyfile(auth, profile / "auth.json")
                (profile / "auth.json").chmod(0o600)
            elif not os.environ.get("CODEX_API_KEY"):
                raise ProtocolError("Codex login or CODEX_API_KEY is required")
        catalog = Path(self.agent_config.get("auth_root", "/root/.codex")) / "models_cache.json"
        if catalog.exists() and not (profile / catalog.name).exists():
            shutil.copyfile(catalog, profile / catalog.name)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        schema_path, answer = receipt_dir / "service-schema.json", receipt_dir / "native-answer.json"
        write(schema_path, service_schema(schema), exclusive=True)
        mode = "workspace-write" if role == "experiment" else "read-only"
        argv = [self.agent_config["binary"], "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="' + mode + '"', "-c", 'web_search="disabled"',
            "-c", 'model_reasoning_effort="' + self.agent_config["effort"] + '"', "exec"]
        if resume:
            argv.append("resume")
        argv += ["--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
                 "--model", self.agent_config["model"], "--output-schema", str(schema_path.resolve()),
                 "--output-last-message", str(answer.resolve())]
        if resume:
            argv.append(mapping["thread_id"])
        argv.append("-")
        env = clean_environment(session_home, gpu="")
        env["CODEX_HOME"] = str(profile.resolve())
        env["PATH"] = str(Path(self.agent_config["binary"]).parent) + ":" + env["PATH"]
        for name in ("CODEX_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                     "http_proxy", "https_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE"):
            if name in os.environ:
                env[name] = os.environ[name]
        for name in self.execution.get("agent_env_passthrough", []):
            if name in os.environ:
                env[name] = os.environ[name]
        # Replace virtual path tokens, not matching suffixes inside existing
        # absolute paths (e.g. /study/trajectory/work/model.py).
        paths = {"evidence": str(evidence.resolve()), "work": str(work.resolve())}
        prompt = re.sub(r"(?<![\w/.-])/(evidence|work)(?=/|\b)", lambda m: paths[m.group(1)], prompt)
        prompt = prompt.replace("Use native Read/Glob/Grep/Edit/Write tools for coding. No shell, GPU, dataset or previous\nsession directories are available.",
            "Use native Codex file editing and read-only file inspection commands. Do not run training, tests, scripts, "
            "or inspect datasets, credentials, private receipts or other sessions. This is trusted process mode; "
            "filesystem isolation is not enforced. GPU visibility is empty in the agent environment.")
        prompt += "\nOptional structured fields may be null when unused. Supply plan_revision=null unless a future plan genuinely needs revision.\n"
        write(pending, {"logical_session": session, "resume": resume, "role": role,
                        "receipt": str(receipt_dir), "started_at": now()}, exclusive=True)
        write(receipt_dir / "native-command.json", {"argv": argv, "logical_session": session,
              "native_thread": mapping["thread_id"] if mapping else None,
              "resume": resume, "sandbox": mode, "profile": str(profile)}, exclusive=True)
        result = invoke_process(argv, self.agent_config["timeout_seconds"], cwd=work, env=env,
            stdin=prompt, receipt=self.jobs / "agents" / (uuid.uuid4().hex + ".json"))
        write(receipt_dir / "transport.json", result, exclusive=True)
        events = []
        for line in result["stdout"].splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        threads = [e["thread_id"] for e in events if e.get("type") == "thread.started"]
        usages = [e.get("usage", {}) for e in events if e.get("type") == "turn.completed"]
        usage = usages[-1] if usages else {}
        write(receipt_dir / "usage.json", {"input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"), "cache_read_input_tokens": usage.get("cached_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_write_input_tokens"), "total_cost_usd": None,
            "elapsed_seconds": result["elapsed_seconds"]}, exclusive=True)
        if result["returncode"] != 0 or result["timed_out"] or not usages or any(e.get("type") == "turn.failed" for e in events):
            raise ProtocolError("Native Codex call failed; preserve private session for audit, no training launched")
        if len(set(threads)) != 1 or not threads:
            raise ProtocolError("Codex did not report exactly one native thread identity")
        if resume and threads[-1] != mapping["thread_id"]:
            raise ProtocolError("Codex resumed a different native thread")
        output = remove_optional_nulls(read(answer), schema)
        validate_shape(output, schema)
        write(mapping_path, {"logical_session": session, "thread_id": threads[-1],
            "completed_calls": (mapping["completed_calls"] if mapping else 0) + 1,
            "last_receipt": str(receipt_dir), "updated_at": now()})
        pending.unlink()
        return output
