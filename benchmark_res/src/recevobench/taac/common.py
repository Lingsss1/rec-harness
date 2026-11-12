"""Explicit identities, atomic records and bounded local subprocesses."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }


def verify_data(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest["identity"] != digest(manifest["files"]):
        raise ValueError("Prepared data manifest identity changed")
    for name, expected in manifest["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or file_hash(p) != expected:
            raise ValueError("Prepared artifact changed: " + name)
    return manifest


def bounded_run(argv, cwd, log, timeout, env=None):
    if not argv or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("Command and positive finite timeout required")
    started = time.monotonic()
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        write(
            Path(log).with_suffix(".launch.json"),
            {"pid": process.pid, "pgid": process.pid, "argv": argv, "cwd": str(cwd)},
        )
        try:
            code = process.wait(timeout=timeout)
            status = "success" if code == 0 else "crash"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "timeout"
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return {
        "status": status,
        "exit_code": code,
        "wall_seconds": time.monotonic() - started,
        "argv": argv,
    }


def save_emb(emb, path):
    """Compatibility hook for the upstream model's optional binary exporter."""
    import struct

    with Path(path).open("wb") as stream:
        stream.write(struct.pack("II", *emb.shape))
        emb.tofile(stream)
