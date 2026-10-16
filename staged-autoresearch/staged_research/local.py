"""Docker-free POSIX process backend, optionally isolated with Linux bubblewrap.

Commands are trusted registration argv, never agent-generated shell text. Process
mode is a trusted host deployment, not a filesystem or network security boundary.
"""
from __future__ import annotations

import hashlib
import contextlib
import os
import signal
import subprocess
import sys
import time
import tempfile
import uuid
from pathlib import Path

from .runtime import DockerRuntime, native_agent_command
from .storage import ProtocolError, now, read, relative, write


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_environment(home, gpu=None):
    # Do not hand dataset credentials, proxies, PYTHONPATH, CLI settings or API
    # keys to candidate training. Agent credentials are admitted separately.
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
           "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def boot_identity():
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.exists() else None


def process_identity(pid):
    path = Path("/proc") / str(pid) / "stat"
    try:
        # comm may contain spaces or parentheses; starttime is field 22.
        return path.read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def invoke_process(argv, timeout, *, cwd, env, stdin=None, receipt=None):
    """Bound entire process group; persist identity before waiting on a worker."""
    started = time.monotonic()
    # The gate waits for a byte from this parent before executing any evaluator.
    # A crash between Popen and durable PID registration closes stdin: no train.
    gated = receipt is not None
    command = [sys.executable, str(Path(__file__).with_name("worker_gate.py"))] + argv if gated else argv
    process = subprocess.Popen(command, cwd=str(cwd), env=env, start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None or gated else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        if receipt is not None:
            write(receipt, {"pid": process.pid, "boot_id": boot_identity(),
                  "starttime": process_identity(process.pid), "started_at": now()}, exclusive=True)
        try:
            stdout, stderr = process.communicate(("\n" + (stdin or "")) if gated else stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_group(process.pid)
            stdout, stderr = process.communicate()
        # A successful parent must not leave GPU-consuming children running.
        kill_group(process.pid)
    except BaseException:
        kill_group(process.pid)
        process.communicate()
        if receipt is not None and Path(receipt).exists():
            try:
                write(receipt, dict(read(receipt), finished_at=now()))
            except OSError:
                pass  # preserve original failure; uncommitted cleanup requires audit
        raise
    if receipt is not None:
        job = read(receipt)
        write(receipt, dict(job, finished_at=now()))
    return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started}


def bubblewrap_command(argv, bindings, *, cwd, network, roots, files, gpu=False):
    if sys.platform != "linux":
        raise ProtocolError("bubblewrap requires Linux")
    result = ["bwrap", "--die-with-parent", "--new-session", "--unshare-user",
              "--unshare-pid", "--unshare-ipc", "--unshare-uts"]
    if not network:
        result += ["--unshare-net"]
    # PyTorch DataLoader workers and POSIX semaphores require a writable shm
    # mount even though the enclosing host /dev is intentionally hidden.
    result += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp"]
    for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ssl", "/etc/ld.so.cache", "/etc/resolv.conf", "/etc/hosts"):
        if Path(path).exists():
            result += ["--ro-bind", path, path]
    for path in roots:
        result += ["--ro-bind", path, path]
    for path in files:
        result += ["--ro-bind", path, path]
    if gpu:
        for path in sorted(Path("/dev").glob("nvidia*")):
            result += ["--dev-bind", str(path), str(path)]
    for source, target, readonly in bindings:
        result += ["--ro-bind" if readonly else "--bind", str(source), target]
    return result + ["--chdir", cwd, "--"] + argv


class LocalRuntime(DockerRuntime):
    """Same scientific runtime interface, with separate local task environments."""
    def __init__(self, agent_config, execution, study_root, gpu_device="0"):
        super().__init__(agent_config, gpu_device)
        self.execution = execution
        self.jobs = Path(study_root).resolve() / "private_backend_jobs"

    @property
    def isolated(self):
        return self.execution["isolation"] == "bubblewrap"

    def wrap(self, argv, bindings, cwd, network, gpu=False):
        if not self.isolated:
            return argv
        return bubblewrap_command(argv, bindings, cwd=cwd, network=network,
            roots=self.execution["readonly_roots"], files=self.execution["pinned_files"], gpu=gpu)

    def agent_command(self, work, evidence, session_home, session, resume, role, schema):
        # Keep the native CLI flags identical across execution backends.
        argv = native_agent_command(self.agent_config, session, resume, role, schema)
        if not self.isolated:
            argv += ["--add-dir", str(evidence.resolve())]
        return self.wrap(argv, [(work, "/work", role != "experiment"),
            (evidence, "/evidence", True), (session_home, "/agent", False)], "/work", True)

    def invoke_agent(self, argv, prompt, work, evidence, session_home, receipt_dir):
        for previous in (self.jobs / "agents").glob("*.json"):
            self.stop_receipt(previous)
        home = "/agent" if self.isolated else str(session_home.resolve())
        env = clean_environment(home)
        env["PATH"] = str(Path(self.agent_config["binary"]).parent) + ":" + env["PATH"]
        env.update(ANTHROPIC_API_KEY=os.environ["ANTHROPIC_API_KEY"], CLAUDE_CONFIG_DIR=home + "/config")
        for name in self.execution.get("agent_env_passthrough", []):
            if name in os.environ:
                env[name] = os.environ[name]
        if not self.isolated:
            prompt = prompt.replace("/evidence", str(evidence.resolve())).replace("/work", str(work.resolve()))
            prompt = prompt.replace("No shell, GPU, dataset or previous\nsession directories are available.",
                "No shell tools are enabled. This is trusted process mode; filesystem isolation is not enforced.")
        return invoke_process(argv, self.agent_config["timeout_seconds"], cwd=work,
            env=env, stdin=prompt, receipt=self.jobs / "agents" / (uuid.uuid4().hex + ".json"))

    def evaluation_command(self, task, artifact, output, split, name=None, checkpoint=None):
        if split not in ("validation", "test"):
            raise ProtocolError("Unknown evaluation split")
        if split == "test" and checkpoint is None:
            raise ProtocolError("Test requires the validation-selected checkpoint")
        scratch = output / "scratch"
        paths = {"candidate": "/candidate" if self.isolated else str(artifact.resolve()),
                 "data": "/data" if self.isolated else task[split + "_data"],
                 "result": "/result" if self.isolated else str(output.resolve()),
                 "scratch": "/scratch" if self.isolated else str(scratch.resolve())}
        if checkpoint is not None:
            paths["checkpoint"] = "/checkpoint" if self.isolated else str(checkpoint.resolve())
        argv = []
        for word in task[split + "_argv"]:
            for key, value in paths.items():
                word = word.replace("{" + key + "}", value)
            if "{" in word or "}" in word:
                raise ProtocolError("Unknown or unavailable evaluator placeholder: " + word)
            argv.append(word)
        bindings = [(artifact, "/candidate", True), (Path(task[split + "_data"]), "/data", True),
                    (output, "/result", False), (scratch, "/scratch", False)]
        if checkpoint is not None:
            bindings.append((checkpoint, "/checkpoint", True))
        return self.wrap(argv, bindings, "/scratch", False, gpu=True)

    def evaluate(self, task, artifact, output, split, name=None, checkpoint=None):
        name = name or "sar-local-" + uuid.uuid4().hex
        relative(name)
        if Path(name).name != name:
            raise ProtocolError("Invalid backend job identity")
        output.mkdir(parents=True, exist_ok=False)
        scratch = output / "scratch"
        scratch.mkdir()
        argv = self.evaluation_command(task, artifact, output, split, checkpoint=checkpoint)
        env = clean_environment("/scratch" if self.isolated else scratch, self.gpu_device)
        env["PATH"] = str(Path(task[split + "_argv"][0]).parent) + ":" + env["PATH"]
        env["TMPDIR"] = "/scratch" if self.isolated else str(scratch)
        result = invoke_process(argv, task["timeout_seconds"], cwd=scratch, env=env,
                                receipt=self.jobs / (name + ".json"))
        write(output / "transport.json", result, exclusive=True)
        return result

    def stop(self, name):
        relative(name)
        if Path(name).name != name:
            raise ProtocolError("Invalid backend job identity")
        receipt = self.jobs / (name + ".json")
        self.stop_receipt(receipt)

    def stop_receipt(self, receipt):
        if not receipt.exists():
            return  # interruption before process launch
        job = read(receipt)
        if job.get("finished_at") or boot_identity() != job["boot_id"]:
            return
        identity = process_identity(job["pid"])
        if boot_identity() == job["boot_id"] and identity == job["starttime"] and identity is not None:
            kill_group(job["pid"])
            write(receipt, dict(job, finished_at=now(), recovered=True))
        elif identity is not None:
            raise ProtocolError("Worker PID was reused; refuse to signal an unrelated process")
        else:
            raise ProtocolError("Interrupted worker identity is unavailable; audit remaining processes before resuming")


def make_runtime(config, root):
    execution = config.get("execution", {"backend": "docker"})
    if execution["backend"] == "docker":
        return DockerRuntime(config["agent"], config["protocol"]["gpu_device"])
    if execution["backend"] == "local":
        return LocalRuntime(config["agent"], execution, root, execution.get("gpu_device", config["protocol"]["gpu_device"]))
    raise ProtocolError("Unknown execution backend")


@contextlib.contextmanager
def gpu_lease(config):
    from .storage import digest, locked
    lock_root = Path(tempfile.gettempdir()) / ("staged-research-gpu-" + str(os.getuid()))
    lock_root.mkdir(mode=0o700, exist_ok=True)
    gpu = config.get("execution", {}).get("gpu_device", config["protocol"]["gpu_device"])
    with locked(lock_root / (digest(gpu)[:16] + ".lock")):
        yield
