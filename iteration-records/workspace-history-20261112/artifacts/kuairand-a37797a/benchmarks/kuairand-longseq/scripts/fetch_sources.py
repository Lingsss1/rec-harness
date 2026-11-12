#!/usr/bin/env python3
"""Fetch the fixed Apache-2.0 reference for optional numerical parity tests."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "5a704061a7ebf0b81afb465f12f93f2747ba4667"


def main():
    target = ROOT / "vendor/fuxi-linear"
    if target.exists():
        manifest = json.loads((target / "SOURCE_MANIFEST.json").read_text())
        if manifest["commit"] != COMMIT:
            raise ValueError("Existing reference has the wrong commit")
        print("Pinned reference already exists:", target)
        return
    url = "https://codeload.github.com/USTC-StarTeam/fuxi-linear/zip/" + COMMIT
    request = urllib.request.Request(url, headers={"User-Agent": "longseq-kuairand-bench"})
    raw = urllib.request.urlopen(request, timeout=120).read()
    archive = zipfile.ZipFile(io.BytesIO(raw))
    if archive.comment.decode() != COMMIT:
        raise ValueError("Archive commit does not match pin")
    prefix = archive.namelist()[0].split("/")[0]
    target.mkdir(parents=True)
    for entry in archive.infolist():
        relative = Path(entry.filename).relative_to(prefix)
        if entry.is_dir() or not relative.parts:
            continue
        if ".." in relative.parts:
            raise ValueError("Unsafe archive entry")
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(entry))
    (target / "SOURCE_MANIFEST.json").write_text(json.dumps({"commit": COMMIT,
        "source": url, "archive_sha256": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n")
    print("Fetched", COMMIT)


if __name__ == "__main__":
    main()
