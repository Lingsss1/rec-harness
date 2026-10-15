"""Registration is separate from execution; templates are deliberately not runnable."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

from .storage import ProtocolError, assert_disjoint, digest, inventory, now, read, relative, write

TASKS = {
    "colored_mnist": {
        "fml_name": "Generalization_domainbed", "regime": "sparse",
        "metric": "in_acc_mean", "direction": "higher",
        "description": "Improve ERM on ColoredMNIST under color/label correlation shift; target env2.",
        "editable": ["domainbed/algorithms.py", "domainbed/networks.py"],
    },
    "openood": {
        "fml_name": "Robustness_openood", "regime": "sparse",
        "metric": "auroc_mean", "direction": "higher",
        "description": "Improve CIFAR-10 OOD detection. Validation OOD=CIFAR-100; hidden test OOD=SVHN.",
        "editable": ["algorithm.py", "split_config.json"],
    },
    "opacus": {
        "fml_name": "Privacy_opacus", "regime": "dense",
        "metric": "test_acc_mean", "direction": "higher",
        "description": "Improve CIFAR-10 DP-SGD accuracy at fixed epsilon=8, delta=1e-5.",
        "editable": ["algorithm.py", "split_config.json"],
    },
}
ARMS = {
    "A": {"planning": False, "flush": False, "diagnostics": False},
    "B": {"planning": True, "flush": False, "diagnostics": False},
    "C": {"planning": True, "flush": True, "diagnostics": False},
    "D": {"planning": True, "flush": True, "diagnostics": True},
}
IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")


def template():
    tasks = {}
    for key, card in TASKS.items():
        tasks[key] = dict(card, public_source="", validation_data="", test_data="", baseline_checkpoint="",
            validation_image="", test_image="", timeout_seconds=2400,
            validation_argv=["python", "/evaluator/validate.py"],
            test_argv=["python", "/evaluator/test.py"],
            baseline_validation=None, baseline_test=None, metric_best=1.0, metric_worst=0.0,
            split_config_sha256=None,
            attestation={"test_labels_absent_from_public_and_validation": False,
                         "evaluator_outside_candidate_process": False,
                         "protocol": "", "evidence": ""})
    return {
        "schema_version": 1, "created_at": now(),
        "sources": {"fml_commit": "", "fml_repository": "https://github.com/qrzou/FML-bench",
                    "karpathy_commit": "228791fb499afffb54b46200aca536f79142f117"},
        "agent": {"image": "", "binary": "claude", "model": "", "effort": "high",
                  "timeout_seconds": 1800, "max_budget_usd_per_call": 5.0,
                  "version": ""},
        "protocol": {"arms": list(ARMS), "replicates": 5, "trial_budget": 100,
                     "stage_size": 10, "review_after": 8,
                     "training_seed_policy": "task-native; agent repetitions are independent",
                     "gpu_device": "0", "score_scale": "fraction, not percentage"},
        "tasks": tasks,
    }


def initialize(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write(root / "study.draft.json", template(), exclusive=True)
    return root / "study.draft.json"


def check_config(config, root, check_images=False):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(config.get("schema_version") == 1, "Unsupported schema version")
    require(config.get("protocol") == template()["protocol"], "Protocol must match the registered 4-arm, 100-trial, 5-repeat design")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", config.get("sources", {}).get("fml_commit", ""))), "Pin the FML checkout commit")
    agent = config.get("agent", {})
    require(bool(IMAGE.fullmatch(agent.get("image", ""))), "agent.image must use an immutable sha256 digest")
    require(bool(agent.get("version")), "Record the native CLI version")
    require(bool(agent.get("model")) and agent.get("model") not in ("sonnet", "opus", "haiku", "latest"), "Pin an explicit model ID, not a moving alias")
    require(agent.get("binary") == "claude", "This adapter supports the Claude Code CLI")
    require(agent.get("effort") in ("low", "medium", "high", "xhigh", "max"), "Invalid effort")
    for key in ("timeout_seconds", "max_budget_usd_per_call"):
        value = agent.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0, "Invalid agent " + key)
    require(set(config.get("tasks", {})) == set(TASKS), "All three predeclared tasks must be present")
    for name, task in config.get("tasks", {}).items():
        label = name + ": "
        if name not in TASKS:
            continue
        for key, value in TASKS[name].items():
            require(task.get(key) == value, label + "changed fixed task card " + key)
        for key in ("metric_best", "metric_worst"):
            require(task.get(key) == (1.0 if key == "metric_best" else 0.0), label + "invalid normalization")
        for key in ("baseline_validation", "baseline_test"):
            v = task.get(key)
            require(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1, label + "supply measured " + key)
        limit = task.get("timeout_seconds")
        require(isinstance(limit, int) and not isinstance(limit, bool) and limit > 0, label + "invalid timeout")
        for key in ("validation_image", "test_image"):
            require(bool(IMAGE.fullmatch(task.get(key, ""))), label + "pin " + key)
        require(task.get("validation_image") != task.get("test_image"), label + "validation/test images must be separate")
        for key in ("validation_argv", "test_argv"):
            argv = task.get(key)
            require(isinstance(argv, list) and bool(argv) and all(isinstance(x, str) and x for x in argv), label + "invalid " + key)
        att = task.get("attestation", {})
        require(att.get("test_labels_absent_from_public_and_validation") is True, label + "attest data isolation")
        require(att.get("evaluator_outside_candidate_process") is True, label + "attest trusted external scoring")
        require(bool(att.get("protocol")) and bool(att.get("evidence")), label + "document evaluator contract and isolation audit evidence")
        paths = [task.get(x, "") for x in ("public_source", "validation_data", "test_data", "baseline_checkpoint")]
        if not all(paths):
            errors.append(label + "bind public/validation/test/baseline-checkpoint directories")
            continue
        require(all(Path(p).is_absolute() for p in paths), label + "asset paths must be absolute")
        try:
            assert_disjoint(paths + [str(root)])
            public = inventory(paths[0])
            require(bool(public), label + "empty public source")
            require(not any(p.startswith((".git/", ".claude/", "records/", "sessions/")) or Path(p).name in ("CLAUDE.md", "AGENTS.md") for p in public), label + "remove VCS, instruction, conversation, and evidence files from public source")
            for item in task["editable"]:
                relative(item)
                require(item in public, label + "missing editable file " + item)
            for item in paths[1:]:
                require(bool(inventory(item)), label + "empty evaluation data directory")
            if "split_config.json" in task["editable"]:
                require(task.get("split_config_sha256") == public.get("split_config.json"), label + "pin the original split_config.json hash; split edits are prohibited in this study")
        except (OSError, ProtocolError) as exc:
            errors.append(label + str(exc))
    # A public source for one task must not contain another task's private assets.
    tasks = list(config.get("tasks", {}).values())
    for public_task in tasks:
        source = public_task.get("public_source")
        if not source:
            continue
        for private_task in tasks:
            for key in ("validation_data", "test_data", "baseline_checkpoint"):
                other = private_task.get(key)
                if other:
                    try:
                        assert_disjoint([source, other])
                    except ProtocolError:
                        errors.append("Cross-task source/private-asset overlap: " + key)
    if check_images:
        if not shutil.which("docker"):
            errors.append("Docker is unavailable; real execution requires an isolated Docker GPU host")
        else:
            images = {agent.get("image", "")}
            for task in config.get("tasks", {}).values():
                images.update([task.get("validation_image", ""), task.get("test_image", "")])
            for image in sorted(images):
                if IMAGE.fullmatch(image):
                    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20)
                    require(result.returncode == 0, "Image not locally available: " + image)
    return errors


def freeze(root):
    root = Path(root).resolve()
    if (root / "registration.json").exists():
        raise ProtocolError("Study already frozen; create a new registration instead of overwriting")
    config = read(root / "study.draft.json")
    errors = check_config(config, root, check_images=True)
    if errors:
        raise ProtocolError("Preflight failed:\n- " + "\n- ".join(errors))
    trees = {}
    for name, task in config["tasks"].items():
        trees[name] = {k: inventory(task[k]) for k in ("public_source", "validation_data", "test_data", "baseline_checkpoint")}
    native = Path(__file__).parent / "resources" / "karpathy-program.md"
    implementation = inventory(Path(__file__).parent)
    implementation = {k: v for k, v in implementation.items() if "__pycache__" not in k}
    value = {"config": config, "trees": trees, "implementation": implementation,
             "native_program_sha256": __import__("hashlib").sha256(native.read_bytes()).hexdigest(), "frozen_at": now()}
    value["registration_id"] = digest(value)
    write(root / "registration.json", value, exclusive=True)
    return value


def load_registration(root, verify_data=True):
    value = read(Path(root) / "registration.json")
    unsigned = {k: v for k, v in value.items() if k != "registration_id"}
    if digest(unsigned) != value["registration_id"]:
        raise ProtocolError("Registration changed after freeze")
    implementation = {k: v for k, v in inventory(Path(__file__).parent).items() if "__pycache__" not in k}
    if implementation != value["implementation"]:
        raise ProtocolError("Controller implementation changed after freeze")
    if verify_data:
        for name, task in value["config"]["tasks"].items():
            for key in ("public_source", "validation_data", "test_data", "baseline_checkpoint"):
                if inventory(task[key]) != value["trees"][name][key]:
                    raise ProtocolError(name + ": frozen source/data drift: " + key)
    return value


def matrix(config):
    import random
    rng = random.Random(20261015)
    result = []
    for rep in range(1, config["protocol"]["replicates"] + 1):
        for task in sorted(config["tasks"]):
            arms = list(config["protocol"]["arms"])
            rng.shuffle(arms)
            result.extend({"task": task, "arm": arm, "replicate": rep, "trial_budget": 100} for arm in arms)
    return result
