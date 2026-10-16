"""Server bootstrap: fixed environments, upstream code, data, baselines, study.

Default run does real baseline training and then the preregistered matrix. plan
and setup do not call a model or train. Nothing installs on import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from ..evaluate import evaluate_study
from ..local import LocalRuntime, file_hash, gpu_lease, invoke_process
from ..storage import ProtocolError, digest, inventory, locked, now, read, write
from ..study import TASKS, check_config, template
from .specs import (CLAUDE_VERSION, DATA_PROTOCOL, DOMAINBED_COMMIT, ENVIRONMENTS,
                    FML_COMMIT, MICROMAMBA_SHA256, MICROMAMBA_VERSION)

PROJECT = Path(__file__).resolve().parents[2]


def command(argv, *, cwd=None, log=None, env=None):
    env = {k: v for k, v in (os.environ if env is None else env).items()
           if not k.endswith("API_KEY") and k != "ANTHROPIC_AUTH_TOKEN"}
    result = invoke_process([str(x) for x in argv], 7200, cwd=cwd or PROJECT, env=env)
    if log is not None:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write("\nCOMMAND " + shlex.join([str(x) for x in argv]) + "\n")
            stream.write(result["stdout"] + result["stderr"])
    if result["returncode"] or result["timed_out"]:
        raise ProtocolError("Setup command failed; " + ("inspect " + str(log) if log else result["stderr"][-2000:]))
    return result["stdout"].strip()


def download(url, path, sha256):
    path = Path(path)
    if path.exists() and file_hash(path) == sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if file_hash(temporary) != sha256:
        raise ProtocolError("Download checksum mismatch: " + url)
    temporary.replace(path)


def ensure_mamba(prefix):
    binary = prefix / "tools" / "micromamba"
    if binary.exists():
        if command([binary, "--version"]) != MICROMAMBA_VERSION:
            raise ProtocolError("Unexpected micromamba version")
        return binary
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ProtocolError("Automatic environment installation targets Linux x86_64; use existing task environments on other hosts")
    archive = prefix / "downloads" / ("micromamba-" + MICROMAMBA_VERSION + ".tar.bz2")
    download("https://micro.mamba.pm/api/micromamba/linux-64/" + MICROMAMBA_VERSION, archive, MICROMAMBA_SHA256)
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        member = source.getmember("bin/micromamba")
        if not member.isfile():
            raise ProtocolError("Invalid micromamba archive")
        with source.extractfile(member) as stream, binary.open("wb") as output:
            shutil.copyfileobj(stream, output)
    binary.chmod(0o755)
    return binary


def checkout(url, commit, directory):
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--no-checkout", url, directory], log=directory.parent / (directory.name + ".clone.log"))
        command(["git", "-C", directory, "checkout", "--detach", commit])
    actual = command(["git", "-C", directory, "rev-parse", "HEAD"])
    if actual != commit or command(["git", "-C", directory, "status", "--porcelain"]):
        raise ProtocolError("Upstream checkout is not clean at the pinned commit: " + str(directory))
    return directory


def code_identity():
    return digest({k: v for k, v in inventory(PROJECT / "staged_research").items() if "__pycache__" not in k})


def ensure_environment(prefix, task, mamba, cpu=False):
    directory = prefix / "envs" / task
    python = directory / "bin" / "python"
    marker = directory / "staged-environment.json"
    specification = {"task": task, "dependencies": ENVIRONMENTS[task], "cpu": cpu, "controller": code_identity()}
    if marker.exists():
        if read(marker)["specification"] != specification:
            raise ProtocolError("Environment definition changed; use a new prefix instead of mutating " + str(directory))
        return python
    print("Preparing fixed Python environment: " + task, flush=True)
    env = dict(os.environ, MAMBA_ROOT_PREFIX=str(prefix / "mamba-cache"))
    log = prefix / "logs" / (task + "-environment.log")
    spec = ENVIRONMENTS[task]
    if not python.exists():
        command([mamba, "create", "-y", "-p", directory, "-c", "conda-forge",
                 "python=" + spec["python"], "pip=25.0.1"], log=log, env=env)
    command([python, "-m", "pip", "install", "setuptools==75.8.0", "wheel==0.45.1"], log=log)
    index = "cpu" if cpu else spec["cuda"]
    command([python, "-m", "pip", "install", "torch==" + spec["torch"] + "+" + index,
             "torchvision==" + spec["torchvision"] + "+" + index,
             "--index-url", "https://download.pytorch.org/whl/" + index], log=log)
    extras = ["huggingface-hub==0.23.5", "safetensors==0.4.3", "PyYAML==6.0.1"] if task == "colored_mnist" else []
    command([python, "-m", "pip", "install"] + spec["packages"] + extras, log=log)
    command([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", PROJECT], log=log)
    command([python, "-m", "pip", "check"], log=log)
    frozen = command([python, "-m", "pip", "freeze"])
    (directory / "pip-freeze.txt").write_text(frozen + "\n")
    command([mamba, "list", "-p", directory, "--explicit"], log=prefix / "logs" / (task + "-conda-explicit.log"), env=env)
    versions = command([python, "-c", "import torch,torchvision,numpy,json; print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'cuda_available':torch.cuda.is_available()}))"])
    write(marker, {"specification": specification, "versions": json.loads(versions), "installed_at": now()}, exclusive=True)
    return python


def ensure_agent(prefix, existing=None):
    if existing is not None or shutil.which("claude"):
        binary = Path(existing or shutil.which("claude")).resolve()
        version = command([binary, "--version"])
        with binary.open("rb") as stream:
            header = stream.read(128)
        if header.startswith(b"#!") and b"node" in header.split(b"\n", 1)[0]:
            node = shutil.which("node")
            if not node:
                raise ProtocolError("Existing npm CLI needs node; pass a standalone --agent-binary or install node")
            node = Path(node).resolve()
            wrapper = prefix / "tools" / "claude"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(str(node)) + " " + shlex.quote(str(binary)) + ' "$@"\n')
            wrapper.chmod(0o755)
            return wrapper, version, [str(wrapper), str(binary), str(node)], [str(binary.parent), str(node.parent.parent)]
        return binary, version, [str(binary)], []
    mamba = ensure_mamba(prefix)
    node_env = prefix / "envs" / "agent-node"
    node = node_env / "bin" / "node"
    if not node.exists():
        command([mamba, "create", "-y", "-p", node_env, "-c", "conda-forge", "nodejs=22.14.0"],
                log=prefix / "logs" / "agent-install.log", env=dict(os.environ, MAMBA_ROOT_PREFIX=str(prefix / "mamba-cache")))
    home = prefix / "agent-code"
    package = home / "node_modules" / "@anthropic-ai" / "claude-code"
    if not (package / "package.json").exists():
        env = dict(os.environ, PATH=str(node_env / "bin") + ":" + os.environ.get("PATH", ""))
        command([node_env / "bin" / "npm", "install", "--prefix", home,
                 "@anthropic-ai/claude-code@" + CLAUDE_VERSION], env=env, log=prefix / "logs" / "agent-install.log")
    metadata = read(package / "package.json")
    if metadata["version"] != CLAUDE_VERSION:
        raise ProtocolError("Unexpected installed native CLI version")
    entry = package / (metadata["bin"]["claude"] if isinstance(metadata["bin"], dict) else metadata["bin"])
    binary = prefix / "tools" / "claude"
    binary.write_text("#!/bin/sh\nexec " + shlex.quote(str(node)) + " " + shlex.quote(str(entry)) + ' "$@"\n')
    binary.chmod(0o755)
    return binary, command([binary, "--version"]), [str(binary), str(node), str(entry), str(package / "package.json")], [str(home), str(node_env)]


def public_sources(prefix, fml, domainbed):
    outputs = {}
    for task, card in TASKS.items():
        directory = prefix / "public" / task
        identity_file = prefix / "source-manifests" / (task + ".json")
        if directory.exists():
            if not identity_file.exists() or inventory(directory) != read(identity_file):
                raise ProtocolError("Public baseline source changed or incomplete: " + str(directory))
            outputs[task] = directory
            continue
        directory.mkdir(parents=True)
        task_dir = fml / "ml_tasks" / card["fml_name"]
        if task == "colored_mnist":
            # Source-only export, no git/data/tests or upstream setup execution.
            for source in (domainbed / "domainbed").rglob("*.py"):
                target = directory / source.relative_to(domainbed)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            replacements = {"hparams_registry.py": "domainbed/hparams_registry.py", "misc.py": "domainbed/lib/misc.py",
                            "fast_data_loader.py": "domainbed/lib/fast_data_loader.py"}
            for source, target in replacements.items():
                shutil.copyfile(task_dir / "original_file_backup" / source, directory / target)
            shutil.copyfile(domainbed / "LICENSE", directory / "DOMAINBED_LICENSE")
        else:
            for name in ("algorithm.py", "split_config.json"):
                shutil.copyfile(task_dir / name, directory / name)
        text = read(task_dir / "prompt.json")["task_description"]
        text += "\n\nControlled study: split_config.json is FIXED. No training/evaluation code or data may be changed. "
        text += "Checkpoints must retain all learned inference state in model.state_dict(); hidden test only reloads the selected model."
        if task == "colored_mnist":
            text += " The external trainer calls ERM.update on two training domains for 5001 steps (batch 64 per domain), then scores model.predict; public scripts/train.py is not the execution entry."
        (directory / "TASK.md").write_text(text + "\n")
        shutil.copyfile(fml / "LICENSE", directory / "FML_LICENSE")
        write(identity_file, inventory(directory), exclusive=True)
        outputs[task] = directory
    return outputs


def build_config(prefix, model, binary, version, agent_files, agent_roots, pythons, public, gpu, cpu, isolation):
    config = template("local")
    config["sources"]["fml_commit"] = FML_COMMIT
    config["agent"].update(binary=str(binary), model=model, version=version)
    pinned = list(agent_files)
    roots = list(agent_roots)
    for task, python in pythons.items():
        environment = python.parent.parent
        package = Path(command([python, "-c", "import staged_research,pathlib; print(pathlib.Path(staged_research.__file__).parent)"]))
        pinned += [str(python), str(environment / "pip-freeze.txt"), str(environment / "staged-environment.json")]
        pinned += [str(p) for p in package.rglob("*.py")]
        roots.append(str(environment))
        item = config["tasks"][task]
        item.update(public_source=str(public[task]), validation_data=str(prefix / "data" / task / "validation"),
            test_data=str(prefix / "data" / task / "test"), baseline_checkpoint=str(prefix / "calibration" / task / "validation" / "checkpoint"),
            timeout_seconds=7200, builtin_fml={"protocol": DATA_PROTOCOL, "profile": "full", "device": "cpu" if cpu else "cuda"})
        for phase in ("validation", "test"):
            item[phase + "_argv"] = [str(python), "-m", "staged_research.fml.evaluator", "--task", task,
                "--phase", phase, "--candidate", "{candidate}", "--data", "{data}", "--result", "{result}",
                "--device", "cpu" if cpu else "cuda"]
            item[phase + "_argv"] += ["--worker-isolation", isolation]
            if phase == "test":
                item[phase + "_argv"] += ["--checkpoint", "{checkpoint}"]
        if "split_config.json" in item["editable"]:
            item["split_config_sha256"] = inventory(public[task])["split_config.json"]
        item["attestation"] = {"test_labels_absent_from_public_and_validation": True,
            "evaluator_outside_candidate_process": True,
            "protocol": DATA_PROTOCOL + "; fixed FML split, external prediction scoring; process mode is not an OS sandbox",
            "evidence": str(prefix / "data" / task) + "/{validation,test}/data.json; bundled evaluator/worker code and checksums"}
    config["execution"].update(environment_id=DATA_PROTOCOL + "-" + code_identity()[:16], gpu_device=str(gpu),
        isolation=isolation, pinned_files=sorted(set(pinned)), readonly_roots=sorted(set(roots)),
        agent_env_passthrough=["ANTHROPIC_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE"])
    return config


def calibrate(prefix, config):
    from ..contracts import score_result
    runtime = LocalRuntime(config["agent"], config["execution"], prefix / "calibration_backend", config["execution"]["gpu_device"])
    with gpu_lease(config):
        for task, card in config["tasks"].items():
            directory = prefix / "calibration" / task
            for phase in ("validation", "test"):
                reservation = directory / (phase + ".reserved.json")
                output = directory / phase
                if reservation.exists():
                    if not (output / "transport.json").exists():
                        runtime.stop(read(reservation)["job"])
                        raise ProtocolError("Calibration outcome unknown; no automatic retry. Audit " + str(directory))
                else:
                    print("Calibrating baseline: %s %s" % (task, phase), flush=True)
                    import uuid
                    job = "calibration-" + uuid.uuid4().hex
                    write(reservation, {"started_at": now(), "job": job, "source": digest(inventory(card["public_source"]))}, exclusive=True)
                    runtime.evaluate(card, Path(card["public_source"]), output, phase, name=job,
                        checkpoint=Path(card["baseline_checkpoint"]) if phase == "test" else None)
                transport = read(output / "transport.json")
                if transport["returncode"] != 0 or transport["timed_out"]:
                    raise ProtocolError("Baseline calibration failed; inspect " + str(output))
                payload = read(output / "metrics.json")
                if payload.get("source_sha256") != digest(inventory(card["public_source"])) or payload.get("data_id") != read(Path(card[phase + "_data"]) / "data.json")["id"]:
                    raise ProtocolError("Calibration belongs to a different baseline source/data")
                if payload.get("checkpoint_sha256") != digest(inventory(card["baseline_checkpoint"])):
                    raise ProtocolError("Baseline checkpoint changed after calibration")
                score = score_result(payload, card)
                card["baseline_validation" if phase == "validation" else "baseline_test"] = score
    return config


def plan(prefix, cpu):
    return {"prefix": str(prefix), "fml_commit": FML_COMMIT, "domainbed_commit": DOMAINBED_COMMIT,
        "environments": ENVIRONMENTS, "cpu": cpu, "tasks": list(TASKS),
        "baseline_preparation": "3 train/validation calls and 3 baseline tests, recorded separately",
        "study": "60 independent trajectories, 6000 validation calls, up to 60 final tests",
        "actions_performed": "none; no download, installation, model call or training"}


def main(argv=None):
    from ..cli import interruptible
    return interruptible(dispatch, argv)


def dispatch(argv=None):
    parser = argparse.ArgumentParser(description="Built-in FML setup and one-command evaluation; setup only downloads/prepares, run trains.")
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--model", help="Explicit native-agent model ID; required for first run")
    parser.add_argument("--agent-binary", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Raw torchvision dataset cache; missing datasets are downloaded")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true", help="Use fixed CPU wheels for all arms (slow); no silent GPU fallback")
    parser.add_argument("--isolation", choices=("process", "bubblewrap"), default="process")
    parser.add_argument("--stage", choices=("plan", "setup", "data", "smoke", "calibrate", "run"), default="run")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    prefix = args.prefix.resolve()
    if args.stage == "plan":
        print(json.dumps(plan(prefix, args.cpu), indent=2))
        return 0
    try:
        if not shutil.which("git"):
            raise ProtocolError("git is required for pinned upstream sources")
        with locked(prefix / ".bootstrap.lock"):
            config_path = prefix / "server.json"
            if config_path.exists():
                config = read(config_path)
                if config["execution"]["environment_id"] != DATA_PROTOCOL + "-" + code_identity()[:16]:
                    raise ProtocolError("Controller/adapter version changed; use a new prefix")
                if args.model and config["agent"]["model"] != args.model:
                    raise ProtocolError("Model changed; use a new prefix")
                if str(args.gpu) != config["execution"]["gpu_device"] or args.isolation != config["execution"]["isolation"]:
                    raise ProtocolError("Execution settings changed; use original settings or a new prefix")
                if args.cpu and any(t["builtin_fml"]["device"] != "cpu" for t in config["tasks"].values()):
                    raise ProtocolError("Cannot change an existing CUDA study to CPU")
                for task, card in config["tasks"].items():
                    python = card["validation_argv"][0]
                    command([python, "-m", "staged_research.fml.data", "--task", task,
                        "--raw", args.data_dir or prefix / "raw-cache", "--output", prefix / "data" / task],
                        log=prefix / "logs" / (task + "-data.log"))
            elif args.stage == "data":
                pythons = {task: prefix / "envs" / task / "bin" / "python" for task in TASKS}
                if not all(p.is_file() for p in pythons.values()):
                    raise ProtocolError("Run setup_fml.sh first to create task environments")
                for task, python in pythons.items():
                    command([python, "-m", "staged_research.fml.data", "--task", task, "--raw", args.data_dir or prefix / "raw-cache",
                        "--output", prefix / "data" / task], log=prefix / "logs" / (task + "-data.log"))
                return 0
            else:
                if not args.model or args.model in ("opus", "sonnet", "haiku", "latest"):
                    raise ProtocolError("Supply --model with an explicit versioned model ID")
                mamba = ensure_mamba(prefix)
                pythons = {task: ensure_environment(prefix, task, mamba, args.cpu) for task in TASKS}
                fml = checkout("https://github.com/qrzou/FML-bench.git", FML_COMMIT, prefix / "upstream" / "FML-bench")
                domainbed = checkout("https://github.com/facebookresearch/DomainBed.git", DOMAINBED_COMMIT, prefix / "upstream" / "DomainBed")
                public = public_sources(prefix, fml, domainbed)
                for task, python in pythons.items():
                    print("Preparing data: " + task, flush=True)
                    command([python, "-m", "staged_research.fml.data", "--task", task, "--raw", args.data_dir or prefix / "raw-cache",
                        "--output", prefix / "data" / task], log=prefix / "logs" / (task + "-data.log"))
                binary, version, files, roots = ensure_agent(prefix, args.agent_binary)
                config = build_config(prefix, args.model, binary, version, files, roots, pythons, public, args.gpu, args.cpu, args.isolation)
                write(config_path, config, exclusive=True)
            if args.stage in ("setup", "data"):
                print("Environment, source and data ready. No training/model calls performed. Config: " + str(config_path))
                return 0
            if args.stage == "run" and not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProtocolError("Set ANTHROPIC_API_KEY before baseline training/search; setup does not need a key")
            # Check actual device availability before reserving an expensive
            # baseline call; setup may have run on a CPU-only login node.
            for card in config["tasks"].values():
                if card["builtin_fml"]["device"] == "cuda":
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=config["execution"]["gpu_device"])
                    available = command([card["validation_argv"][0], "-c", "import torch; print(torch.cuda.is_available())"], env=env)
                    if available != "True":
                        raise ProtocolError("Registered GPU is unavailable; no baseline training was launched")
            if config["execution"]["isolation"] == "bubblewrap":
                from ..local import bubblewrap_command
                inner = bubblewrap_command(["/usr/bin/true"], [], cwd="/tmp", network=False, roots=[], files=[])
                outer = bubblewrap_command(inner, [], cwd="/tmp", network=False, roots=[], files=[])
                command(outer)  # both worker and scorer namespaces must work
            smoke_receipt = prefix / "smoke-passed.json"
            if args.stage == "smoke" or not smoke_receipt.exists():
                from .smoke import run_smoke
                run_smoke(config, prefix)
            if args.stage == "smoke":
                print("Development-only real-model smoke checks passed; no hidden test/model API used")
                return 0
            if (prefix / "study" / "registration.json").exists():
                from ..study import load_registration
                registration = load_registration(prefix / "study")
                if registration["config"] != config:
                    raise ProtocolError("Server config changed after freeze")
            else:
                config = calibrate(prefix, config)
                write(config_path, config)
            errors = check_config(config, prefix / "study", check_images=True)
            if errors:
                raise ProtocolError("Server readiness failed:\n- " + "\n- ".join(errors))
            if args.stage == "calibrate":
                print("Measured baselines and config ready; no agent search started. " + str(config_path))
                return 0
            result = evaluate_study(prefix / "study", prefix / "report", config_path=config_path,
                keep_going=args.keep_going)
            print(json.dumps(result, indent=2))
            return 0 if result.get("completed_trajectories") == 60 else 1
    except (ProtocolError, OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
