"""Verified, resumable KuaiRand-27K download and selective atomic extraction."""
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

URL = "https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz"
MD5 = "3e3c799a24e2d23a4d2c757fbf9adf59"
SOURCE = "https://zenodo.org/records/10439422"
LOGS = (
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
)
BLOCK = 8 * 1024 * 1024


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            md5.update(block)
            sha.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def download_archive(path, url=URL, expected_md5=MD5, attempts=5, report=print):
    """Resume a .part with HTTP Range; never publish unverified archive bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashes(path)
        if digest["md5"] != expected_md5:
            raise ValueError("Existing archive has the wrong MD5: " + str(path))
        report("Verified existing archive: " + str(path))
        return digest
    partial = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        report("Downloading KuaiRand-27K from byte %d (attempt %d/%d)" % (offset, attempt + 1, attempts))
        headers = {"User-Agent": "longseq-kuairand-bench/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes=%d-" % offset
        try:
            try:
                response = urlopen(Request(url, headers=headers), timeout=60)
            except HTTPError as error:
                # A complete .part left before rename is a normal restart case.
                if error.code == 416 and partial.exists() and hashes(partial)["md5"] == expected_md5:
                    digest = hashes(partial)
                    os.replace(partial, path)
                    return digest
                if error.code == 416:
                    partial.unlink(missing_ok=True)
                raise
            with response:
                status = response.getcode()
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith("bytes %d-" % offset):
                        raise ValueError("Server returned a mismatched Content-Range")
                    mode = "ab"
                elif status == 200:
                    # Some servers ignore Range. Start over, do not append.
                    offset, mode = 0, "wb"
                else:
                    raise ValueError("Unexpected HTTP status: " + str(status))
                declared = response.headers.get("Content-Length")
                received, last_report = 0, time.monotonic()
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(BLOCK)
                        if not block:
                            break
                        handle.write(block)
                        received += len(block)
                        if time.monotonic() - last_report > 30:
                            report("Downloaded %.2f GiB" % ((offset + received) / 2 ** 30))
                            last_report = time.monotonic()
                if declared is not None and received != int(declared):
                    raise OSError("Incomplete HTTP response; will resume")
            digest = hashes(partial)
            if digest["md5"] != expected_md5:
                partial.unlink()
                raise ValueError("Downloaded archive failed the official MD5; discarded corrupt partial")
            os.replace(partial, path)
            return digest
        except (OSError, HTTPError, HTTPException) as error:
            if attempt + 1 == attempts:
                raise
            report("Download interrupted (%s); retrying with Range" % error)
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("Download attempts exhausted")


def extract_logs(archive, destination, archive_hashes, report=print):
    """Read the tar sequentially and copy only four known regular log files."""
    destination = Path(destination)
    if destination.exists():
        manifest = json.loads((destination / "download.json").read_text())
        if manifest["archive_hashes"] != archive_hashes or set(manifest["files"]) != set(LOGS):
            raise ValueError("Raw logs belong to a different archive")
        for name, expected in manifest["files"].items():
            if hashes(destination / name)["sha256"] != expected:
                raise ValueError("Changed raw log: " + name)
        report("Verified extracted logs")
        return manifest
    stage = destination.with_name(destination.name + ".extracting")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    found = {}
    try:
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                parts = PurePosixPath(member.name).parts
                name = parts[-1] if parts else ""
                if name not in LOGS:
                    continue
                if (not member.isfile() or member.size <= 0 or name in found
                        or ".." in parts or PurePosixPath(member.name).is_absolute()):
                    raise ValueError("Unsafe or duplicate log member: " + member.name)
                report("Extracting " + name)
                digest, count = hashlib.sha256(), 0
                with bundle.extractfile(member) as source, (stage / name).open("wb") as output:
                    for block in iter(lambda: source.read(BLOCK), b""):
                        output.write(block)
                        digest.update(block)
                        count += len(block)
                if count != member.size:
                    raise ValueError("Truncated log member: " + name)
                found[name] = digest.hexdigest()
        if set(found) != set(LOGS):
            raise ValueError("Archive is missing standard-policy logs: " + str(set(LOGS) - set(found)))
        manifest = {"source": SOURCE, "download_url": URL, "archive_hashes": archive_hashes,
                    "files": found, "extracted": "standard-policy logs only"}
        atomic_json(stage / "download.json", manifest)
        os.replace(stage, destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
