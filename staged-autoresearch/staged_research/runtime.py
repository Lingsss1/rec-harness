"""Native Claude Code editing; training and test run in separate Docker containers.

No host shell, GPU, dataset, controller directory or old session home is mounted
into the coding agent. The trusted scoring-image contract is documented in README.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from .storage import ProtocolError, read, write


def mount(source, target, readonly=True):
    if "," in str(source) or "," in target:
        raise ProtocolError("Docker bind paths cannot contain commas")
    return ["--mount", "type=bind,src=" + str(Path(source).resolve()) + ",dst=" + target + (",readonly" if readonly else "")]


def container_base(name):
    return ["docker", "run", "--name", name, "--rm", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "512", "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
            "--user", str(os.getuid()) + ":" + str(os.getgid())]


def invoke(argv, timeout, stdin=None):
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Killing only the Docker client leaves training running: terminate the container first.
        name = argv[argv.index("--name") + 1]
        try:
            subprocess.run(["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        finally:
            process.kill()
        stdout, stderr = process.communicate()
    except BaseException:
        name = argv[argv.index("--name") + 1]
        try:
            subprocess.run(["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        finally:
            process.kill()
            process.communicate()
        raise
    return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started}


def usage_from_response(response):
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        for item in response.get("modelUsage", {}).values():
            for source, target in (("inputTokens", "input_tokens"), ("outputTokens", "output_tokens"),
                                   ("cacheReadInputTokens", "cache_read_input_tokens"),
                                   ("cacheCreationInputTokens", "cache_creation_input_tokens")):
                usage[target] = usage.get(target, 0) + item.get(source, 0)
    known = bool(usage)
    return {key: usage.get(key) if known else None for key in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}


def native_agent_command(cfg, session, resume, role, schema):
    tools = "Read,Glob,Grep,Edit,Write" if role == "experiment" else "Read,Glob,Grep"
    return [cfg["binary"], "--bare", "--print", "--model", cfg["model"],
            "--effort", cfg["effort"], "--output-format", "json", "--tools", tools,
            "--allowedTools", tools, "--permission-mode", "dontAsk",
            "--setting-sources", "", "--disable-slash-commands", "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
            "--max-budget-usd", str(cfg["max_budget_usd_per_call"]),
            "--json-schema", json.dumps(schema), "--resume" if resume else "--session-id", session]


class DockerRuntime:
    def __init__(self, agent_config, gpu_device="0"):
        self.agent_config = agent_config
        self.gpu_device = gpu_device

    def agent_command(self, work, evidence, session_home, session, resume, role, schema):
        cfg = self.agent_config
        argv = container_base("sar-agent-" + uuid.uuid4().hex)
        argv += ["--network", "bridge", "--memory", "4g", "--cpus", "2",
                 "--env", "ANTHROPIC_API_KEY", "--env", "HOME=/agent",
                 "--env", "CLAUDE_CONFIG_DIR=/agent/config", "--workdir", "/work"]
        # Native editors may atomically replace files. Admit only allowlisted changes
        # after the call, before anything is executed; reviewers receive a read-only tree.
        argv += mount(work, "/work", role != "experiment") + mount(evidence, "/evidence") + mount(session_home, "/agent", False)
        native = native_agent_command(cfg, session, resume, role, schema)
        return argv + ["--entrypoint", native[0], cfg["image"]] + native[1:]

    def call_agent(self, *, work, evidence, session_home, session, resume, role, schema, prompt, receipt_dir):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProtocolError("ANTHROPIC_API_KEY is required for bare native-agent execution")
        session_home.mkdir(parents=True, exist_ok=True)
        argv = self.agent_command(work, evidence, session_home, session, resume, role, schema)
        result = self.invoke_agent(argv, prompt, work, evidence, session_home, receipt_dir)
        # Raw conversations/CLI output never enter an evidence packet or a later fresh home.
        write(receipt_dir / "transport.json", result, exclusive=True)
        try:
            response = json.loads(result["stdout"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProtocolError("Native CLI returned invalid JSON; inspect private transport receipt") from exc
        write(receipt_dir / "usage.json", dict(usage_from_response(response),
              total_cost_usd=response.get("total_cost_usd"), elapsed_seconds=result["elapsed_seconds"]), exclusive=True)
        if result["returncode"] != 0 or result["timed_out"] or response.get("is_error"):
            raise ProtocolError("Native CLI failed; no validation was launched")
        if response.get("session_id") != session:
            raise ProtocolError("Native CLI session identity mismatch")
        output = response.get("structured_output")
        if not isinstance(output, dict):
            raise ProtocolError("Native CLI did not return structured_output")
        return output

    def invoke_agent(self, argv, prompt, work, evidence, session_home, receipt_dir):
        return invoke(argv, self.agent_config["timeout_seconds"], prompt)

    def evaluation_command(self, task, artifact, output, split, name=None, checkpoint=None):
        if split not in ("validation", "test"):
            raise ProtocolError("Unknown evaluation split")
        argv = container_base(name or "sar-eval-" + uuid.uuid4().hex)
        argv += ["--network", "none", "--gpus", "device=" + self.gpu_device, "--workdir", "/tmp"]
        argv += mount(artifact, "/candidate") + mount(task[split + "_data"], "/data") + mount(output, "/result", False)
        if split == "test":
            if checkpoint is None:
                raise ProtocolError("Test requires the validation-selected checkpoint")
            argv += mount(checkpoint, "/checkpoint")
        # Image entrypoint is a trusted external evaluator, never agent-controlled shell text.
        command = task[split + "_argv"]
        return argv + ["--entrypoint", command[0], task[split + "_image"]] + command[1:]

    def evaluate(self, task, artifact, output, split, name=None, checkpoint=None):
        output.mkdir(parents=True, exist_ok=False)
        result = invoke(self.evaluation_command(task, artifact, output, split, name=name, checkpoint=checkpoint), task["timeout_seconds"])
        write(output / "transport.json", result, exclusive=True)
        return result

    def stop(self, name):
        subprocess.run(["docker", "kill", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)
