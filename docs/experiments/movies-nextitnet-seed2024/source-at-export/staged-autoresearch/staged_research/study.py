"""Registration is separate from execution; templates are deliberately not runnable."""
from __future__ import annotations

import math
import os
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
MOVIES_PROFILE = "movies-nextitnet-v1"
MOVIES_TASKS = {
    "movies_nextitnet": {
        "fml_name": "RecHarness_Movies_NextItNet", "regime": "dense",
        "metric": "HR@10", "direction": "higher",
        "description": "Optimize causal NextItNet on Movies_and_TV; training seed=2024, evaluation seed=42. "
            "Frozen BPR/Adam, 200 epochs, 200 batches per epoch, evaluation every 5 epochs, patience 20, "
            "200 validation users, 99 unique negatives. Histories are right aligned; the raw baseline's "
            "count_nonzero-1 readout may index padding. Treat correctness fixes separately from architecture gains. "
            "Shared, separate, projected or normalized target embeddings are allowed. "
            "Training scores must be finite, have the target shape, and be invariant to positive/negative "
            "argument role and batch composition; each position must use only its history prefix and own target. "
            "Public scores must be finite and independent of candidate "
            "order and candidate-set composition. The encoder must remain causal and batch-independent. "
            "Edit only the two nn.Module classes and eight numeric hyperparameters; imports, training and evaluation are fixed.",
        "editable": ["model.py"],
    },
}
IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")


def task_cards(config):
    return MOVIES_TASKS if config.get("study_profile") == MOVIES_PROFILE else TASKS


def template(backend="docker", profile=None):
    if profile not in (None, MOVIES_PROFILE):
        raise ProtocolError("Unknown study profile")
    tasks = {}
    for key, card in (MOVIES_TASKS if profile == MOVIES_PROFILE else TASKS).items():
        tasks[key] = dict(card, public_source="", validation_data="", test_data="", baseline_checkpoint="",
            validation_image="", test_image="", timeout_seconds=2400,
            validation_argv=["python", "/evaluator/validate.py"],
            test_argv=["python", "/evaluator/test.py"],
            baseline_validation=None, baseline_test=None, metric_best=1.0, metric_worst=0.0,
            split_config_sha256=None,
            attestation={"test_labels_absent_from_public_and_validation": False,
                         "evaluator_outside_candidate_process": False,
                         "protocol": "", "evidence": ""})
    config = {
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
    if profile == MOVIES_PROFILE:
        config["study_profile"] = profile
        config["sources"]["rec_harness_commit"] = ""
    if backend == "local":
        config["execution"] = {"backend": "local", "isolation": "process",
            "environment_id": "", "gpu_device": "0", "pinned_files": [], "readonly_roots": [],
            "agent_env_passthrough": []}
        config["agent"].pop("image")
        config["agent"]["binary"] = ""
        for task in tasks.values():
            task.pop("validation_image")
            task.pop("test_image")
            for split in ("validation", "test"):
                task[split + "_argv"] = ["/absolute/venv/bin/python", "/absolute/evaluators/" + split + ".py",
                    "--candidate", "{candidate}", "--data", "{data}", "--result", "{result}"]
                if split == "test":
                    task[split + "_argv"] += ["--checkpoint", "{checkpoint}"]
    elif backend != "docker":
        raise ProtocolError("Unknown backend")
    return config


def initialize(root, backend="docker", profile=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write(root / "study.draft.json", template(backend, profile), exclusive=True)
    return root / "study.draft.json"


def check_config(config, root, check_images=False):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(config.get("schema_version") == 1, "Unsupported schema version")
    require(config.get("study_profile") in (None, MOVIES_PROFILE), "Unknown study profile")
    expected_protocol = template()["protocol"]
    if config.get("study_profile") == MOVIES_PROFILE and config.get("protocol", {}).get("trial_budget") == 44:
        expected_protocol.update(trial_budget=44, replicates=1)
    require(config.get("protocol") == expected_protocol, "Protocol must match the full design or Movies 44-trial, one-repeat comparison")
    revision_key = "rec_harness_commit" if config.get("study_profile") == MOVIES_PROFILE else "fml_commit"
    require(bool(re.fullmatch(r"[0-9a-f]{40}", config.get("sources", {}).get(revision_key, ""))), "Pin the task checkout commit: " + revision_key)
    agent = config.get("agent", {})
    execution = config.get("execution", {"backend": "docker"})
    backend = execution.get("backend")
    require(backend in ("docker", "local"), "Unknown execution backend")
    local = backend == "local"
    provider = agent.get("provider", "claude")
    require(provider in ("claude", "codex"), "Unknown native agent provider")
    require(provider != "codex" or local, "Codex provider currently requires the local process backend")
    if not local:
        require(bool(IMAGE.fullmatch(agent.get("image", ""))), "agent.image must use an immutable sha256 digest")
    require(bool(agent.get("version")), "Record the native CLI version")
    require(bool(agent.get("model")) and agent.get("model") not in ("sonnet", "opus", "haiku", "latest"), "Pin an explicit model ID, not a moving alias")
    if not local:
        require(agent.get("binary") == "claude", "This adapter supports the Claude Code CLI")
    else:
        require(bool(agent.get("binary")) and Path(agent["binary"]).is_absolute(), "local agent.binary must be an absolute native CLI executable")
    require(agent.get("effort") in ("low", "medium", "high", "xhigh", "max"), "Invalid effort")
    for key in (("timeout_seconds",) if provider == "codex" else ("timeout_seconds", "max_budget_usd_per_call")):
        value = agent.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0, "Invalid agent " + key)
    if provider == "codex":
        require(agent.get("max_budget_usd_per_call") is None, "Codex has no enforced per-call USD cap; set max_budget_usd_per_call=null")
    cards = task_cards(config)
    require(set(config.get("tasks", {})) == set(cards), "All predeclared tasks for this profile must be present")
    for name, task in config.get("tasks", {}).items():
        label = name + ": "
        if name not in cards:
            continue
        for key, value in cards[name].items():
            require(task.get(key) == value, label + "changed fixed task card " + key)
        for key in ("metric_best", "metric_worst"):
            require(task.get(key) == (1.0 if key == "metric_best" else 0.0), label + "invalid normalization")
        for key in ("baseline_validation", "baseline_test"):
            v = task.get(key)
            require(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1, label + "supply measured " + key)
        limit = task.get("timeout_seconds")
        require(isinstance(limit, int) and not isinstance(limit, bool) and limit > 0, label + "invalid timeout")
        if not local:
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
        if "builtin_fml" in task:
            from .fml.data import verify_data
            from .fml.specs import DATA_PROTOCOL
            require(task["builtin_fml"].get("protocol") == DATA_PROTOCOL and task["builtin_fml"].get("profile") == "full", label + "only full built-in FML data may be registered")
            for phase in ("validation", "test"):
                try:
                    verify_data(task.get(phase + "_data", ""), name, phase)
                except (OSError, ProtocolError, ValueError, KeyError) as exc:
                    errors.append(label + "prepared dataset: " + str(exc))
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
    if local:
        errors.extend(check_local_config(config, root, check_images))
    elif check_images:
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


def check_local_config(config, root, check_environment):
    from .local import bubblewrap_command
    errors = []
    execution = config["execution"]
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(execution.get("isolation") in ("process", "bubblewrap"), "local isolation must be process or bubblewrap")
    require(bool(execution.get("environment_id")), "Record a fixed local environment_id")
    require(bool(re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", str(execution.get("gpu_device", "0")))), "Select one local GPU ordinal or GPU UUID")
    devices = execution.get("gpu_devices", [execution.get("gpu_device", "0")])
    require(isinstance(devices, list) and bool(devices) and len(set(map(str, devices))) == len(devices)
        and all(bool(re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", str(d))) for d in devices), "Invalid GPU pool")
    require(str(execution.get("gpu_device", "0")) in list(map(str, devices)), "Primary GPU must belong to the registered pool")
    if config["agent"].get("provider") == "codex":
        require(execution.get("isolation") == "process", "Codex adapter currently supports trusted process mode only")
    files = execution.get("pinned_files", [])
    roots = execution.get("readonly_roots", [])
    if not isinstance(files, list) or not all(isinstance(p, str) and Path(p).is_absolute() and Path(p).is_file() for p in files):
        errors.append("pinned_files must list existing absolute runtime/evaluator files")
        files = []
    require(bool(files), "Pin local CLI, evaluator and environment manifest files")
    if not isinstance(roots, list) or not all(isinstance(p, str) and Path(p).is_absolute() and Path(p).is_dir() for p in roots):
        errors.append("readonly_roots must list existing absolute dependency directories")
        roots = []
    allowed_env = {"ANTHROPIC_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE"}
    passthrough = execution.get("agent_env_passthrough", [])
    require(isinstance(passthrough, list) and all(p in allowed_env for p in passthrough), "Unsupported agent environment passthrough")
    required_files = [config["agent"].get("binary", "")]
    for task in config.get("tasks", {}).values():
        for split in ("validation", "test"):
            argv = task.get(split + "_argv", [])
            if not isinstance(argv, list) or not argv or not all(isinstance(w, str) for w in argv):
                continue
            require(Path(argv[0]).is_absolute() and os.access(argv[0], os.X_OK), "Evaluator argv[0] must be an absolute executable")
            joined = " ".join(argv)
            for key in ("candidate", "data", "result") + (("checkpoint",) if split == "test" else ()):
                require("{" + key + "}" in joined, "local " + split + " command must bind {" + key + "}")
            require(split == "test" or "{checkpoint}" not in joined, "Validation cannot bind hidden-test checkpoint")
            rest = re.sub(r"\{(?:candidate|data|result|scratch|checkpoint)\}", "", joined)
            require("{" not in rest and "}" not in rest, "Unknown evaluator placeholder")
            required_files.extend(w for w in argv if Path(w).is_absolute() and Path(w).is_file())
    require(os.access(config["agent"].get("binary", ""), os.X_OK), "Native CLI is not executable")
    for path in required_files:
        require(path in files, "Add executable/script to execution.pinned_files: " + path)
    private_paths = [str(root)] + [task.get(key, "") for task in config.get("tasks", {}).values()
        for key in ("public_source", "validation_data", "test_data", "baseline_checkpoint")]
    # Read-only roots are visible to BOTH roles. Never expose data or transcripts
    # through a broad dependency mount, even though it is read-only.
    exposed = roots + files
    if execution.get("isolation") == "bubblewrap":
        exposed += [p for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ssl") if Path(p).exists()]
    for visible in exposed:
        for private in private_paths:
            if private:
                try:
                    assert_disjoint([visible, private])
                except ProtocolError:
                    errors.append("Runtime paths must be disjoint from study/data/source: " + visible)
    if check_environment and execution.get("isolation") == "bubblewrap":
        if not shutil.which("bwrap") or __import__("sys").platform != "linux":
            errors.append("bubblewrap backend requires Linux and bwrap; process mode is available without it")
        elif not errors:
            try:
                result = subprocess.run(bubblewrap_command(["/usr/bin/true"], [], cwd="/tmp", network=False,
                    roots=roots, files=files), capture_output=True, timeout=20)
                require(result.returncode == 0, "bubblewrap/user namespaces are unavailable: " + result.stderr.decode(errors="replace")[:500])
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append("bubblewrap preflight failed: " + str(exc))
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
    if config.get("execution", {}).get("backend") == "local":
        from .local import file_hash
        value["runtime_files"] = {p: file_hash(p) for p in config["execution"]["pinned_files"]}
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
    if "runtime_files" in value:
        from .local import file_hash
        if {p: file_hash(p) for p in value["runtime_files"]} != value["runtime_files"]:
            raise ProtocolError("Local executable/evaluator/environment changed after freeze")
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
            result.extend({"task": task, "arm": arm, "replicate": rep, "trial_budget": config["protocol"]["trial_budget"]} for arm in arms)
    return result
