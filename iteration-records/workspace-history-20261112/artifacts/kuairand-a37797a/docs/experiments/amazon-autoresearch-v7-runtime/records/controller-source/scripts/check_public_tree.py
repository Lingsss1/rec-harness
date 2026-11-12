#!/usr/bin/env python3
"""Reject internal model artifacts before they enter the public repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


FORBIDDEN_PREFIXES = (
    "examples/unirank-autoresearch/",
    "design/architecture/",
)

FORBIDDEN_PATH_PARTS = (
    "kai-v2-",
    "kangshuwen-",
    "search-rec-v",
    "two-model-construction",
    "lr-shelf-timeline",
)

# Join fragments so the checker does not contain the exact strings it rejects.
FORBIDDEN_CONTENT = (
    "kai_v2_" + "model.py",
    "origin/" + "songzeyu",
    "origin/" + "kangshuwen",
    "dsp_" + "szy_agent",
    "item_card_" + "version",
)

TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yaml", ".yml", ".html", ".svg", ".ts", ".tsx", ".js", ".mjs"}


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [entry.decode("utf-8") for entry in result.stdout.split(b"\0") if entry]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    violations: list[str] = []
    for relative in tracked_files(root):
        if relative == "scripts/check_public_tree.py":
            continue
        if relative.startswith(FORBIDDEN_PREFIXES) or any(part in relative for part in FORBIDDEN_PATH_PARTS):
            violations.append(f"forbidden path: {relative}")
            continue
        path = root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in FORBIDDEN_CONTENT:
            if marker in content:
                violations.append(f"forbidden internal marker in {relative}: {marker}")
    if violations:
        print("Public-tree validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("public-tree boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
