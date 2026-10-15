"""Durable receipts, content identities, and exclusive ownership of a trajectory."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


class ProtocolError(ValueError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (encode(value) + "\n").encode()
    if exclusive:
        with path.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    else:
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    # fsync the directory as well: a reserved trial must survive a host crash.
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def relative(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ProtocolError("Expected a non-empty relative path without '..'")
    return path


def inventory(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("Expected a real directory: " + str(root))
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProtocolError("Symlinks are not admitted: " + str(path))
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            result[path.relative_to(root).as_posix()] = h.hexdigest()
        elif not path.is_dir():
            raise ProtocolError("Special files are not admitted: " + str(path))
    return result


def assert_disjoint(paths):
    paths = [Path(p).resolve() for p in paths]
    for i, first in enumerate(paths):
        for second in paths[i + 1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ProtocolError("Public, validation, test, and controller paths must be disjoint")


@contextlib.contextmanager
def locked(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Another process owns this trajectory") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
