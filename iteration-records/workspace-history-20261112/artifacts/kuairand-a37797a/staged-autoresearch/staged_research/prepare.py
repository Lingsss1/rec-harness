"""Read an existing FML checkout and export only public editable code. No setup/train."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .storage import ProtocolError, inventory, read, relative, write
from .study import TASKS


def prepare_fml(root, checkout, assets):
    root, checkout, assets = Path(root).resolve(), Path(checkout).resolve(), Path(assets).resolve()
    if (root / "registration.json").exists():
        raise ProtocolError("Study is already frozen; do not rebind task assets")
    if root == assets or root in assets.parents or assets in root.parents:
        raise ProtocolError("Assets and private study must be disjoint")
    if assets.exists():
        raise ProtocolError("Use a new asset directory; never overwrite an exported baseline")
    draft = read(root / "study.draft.json")
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    clean = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--", "ml_tasks", "configs/tasks"], capture_output=True, text=True, check=True)
    if clean.stdout.strip():
        raise ProtocolError("FML task definitions have local changes; pin a clean checkout")
    exports = []
    for key, card in TASKS.items():
        directory = checkout / "ml_tasks" / card["fml_name"]
        config = read(directory / "config.json")
        if set(config["target_files"]) != set(card["editable"]):
            raise ProtocolError(key + ": upstream editable scope changed; review adaptation explicitly")
        repository = (checkout / relative(config["repo_dir"])).resolve()
        if checkout not in repository.parents:
            raise ProtocolError("FML repository escaped checkout")
        for item in card["editable"]:
            source = repository / item
            if repository not in source.resolve().parents or not source.is_file() or source.is_symlink():
                raise ProtocolError("Prepare the FML baseline workspace first: missing " + str(source))
            exports.append((source, assets / key / "public" / item))
        prompt = read(directory / "prompt.json")
        task_text = prompt["task_description"]
        draft["tasks"][key]["public_source"] = str(assets / key / "public")
        draft["tasks"][key]["attestation"]["protocol"] = "FML checkout " + commit + "; upstream task " + card["fml_name"]
        exports.append((task_text, assets / key / "public" / "TASK.md"))
    # Validation above is read-only; export only after all task scopes have passed.
    for source, destination in exports:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes() if isinstance(source, Path) else source.encode())
    draft["sources"]["fml_commit"] = commit
    for key, task in draft["tasks"].items():
        if "split_config.json" in task["editable"]:
            task["split_config_sha256"] = inventory(task["public_source"])["split_config.json"]
    write(root / "study.draft.json", draft)
    return draft
