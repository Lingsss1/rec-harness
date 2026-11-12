"""Download only task-required official files, pinned and SHA256-verified."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True, exist_ok=True)


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(job):
    url, target, expected, size = job
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (size is None or target.stat().st_size == size):
        if not expected or sha256(target) == expected:
            return str(target.relative_to(ROOT)), target.stat().st_size, sha256(target)
    for attempt in range(6):
        temporary = target.with_name(target.name + ".partial")
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(4 << 20), b""):
                    stream.write(chunk)
                    h.update(chunk)
            actual = h.hexdigest()
            if expected and actual != expected:
                raise ValueError("Publisher SHA256 mismatch: " + str(target))
            if size is not None and temporary.stat().st_size != size:
                raise ValueError("Publisher file size mismatch: " + str(target))
            os.replace(temporary, target)
            return str(target.relative_to(ROOT)), target.stat().st_size, actual
        except Exception as error:
            print(json.dumps({"retry": attempt + 1, "file": str(target), "error": str(error)}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


jobs, sources = [], {}
for kind, repo, revision, destination in [
    ("datasets", "TAAC2025/TencentGR-1M", "4be8735e2a7e9fc11fe7eb2a5e1e06e27c57e453", "raw"),
    ("models", "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", "encoder"),
]:
    metadata = fetch_json(f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true")
    assert metadata["sha"] == revision
    files = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if kind == "datasets":
            selected = name in {"README.md", "indexer.pkl"} or (
                name.endswith(".parquet") and name.startswith(("seq/", "candidate/", "item_feat/", "user_feat/", "mm_emb/emb_81_32_parquet/"))
            )
        else:
            selected = "/" not in name and name not in {".gitattributes", "pytorch_model.bin"}
        if not selected:
            continue
        files.append(item)
        prefix = "datasets/" if kind == "datasets" else ""
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}?download=true"
        expected = item.get("lfs", {}).get("sha256")
        if item.get("lfs") and not expected:
            raise ValueError("Missing publisher SHA256: " + name)
        jobs.append((url, ROOT / destination / name, expected, item.get("size")))
    sources[destination] = {"repo": repo, "revision": revision, "files": files}
(ROOT / "publisher-manifests.json").write_text(json.dumps(sources, indent=2))
print(json.dumps({"files": len(jobs), "bytes": sum(j[3] or 0 for j in jobs)}), flush=True)
completed = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for future in concurrent.futures.as_completed([pool.submit(download, job) for job in jobs]):
        result = future.result()
        completed.append({"file": result[0], "bytes": result[1], "sha256": result[2]})
        print(json.dumps({"completed": len(completed), "total": len(jobs), **completed[-1]}), flush=True)
(ROOT / "downloads-verified.json").write_text(json.dumps(completed, indent=2))
