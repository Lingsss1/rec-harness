"""Prepare and run all four staged methods on the existing Movies task."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .codex import CodexRuntime
from .engine import Trajectory
from .local import clean_environment, gpu_lease, invoke_process
from .report import report
from .storage import ProtocolError, inventory, locked, now, read, write
from .study import MOVIES_PROFILE, check_config, freeze, load_registration, matrix, template

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def prepare(args):
    sys.path.insert(0, str(WORKSPACE))
    from mle_star_codex.run import baseline_source, sha
    prefix = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    data_source = args.data.resolve()
    source_repo = args.repo.resolve()
    devices = [s.strip() for s in args.gpus.split(",")]
    if len(devices) != 4 or len(set(devices)) != 4 or any(not s.isdigit() for s in devices):
        raise ProtocolError("Exactly four distinct physical GPU ordinals required")
    inputs = {"data": str(data_source), "repo": str(source_repo), "python": str(args.python.resolve()),
              "agent": str(args.agent), "model": args.model, "effort": args.effort, "gpus": devices, "trial_budget": 44, "replicates": 1}
    recipe = prefix / "bootstrap.json"
    if recipe.exists() and read(recipe) != inputs:
        raise ProtocolError("Movies setup differs from this prefix; use a new prefix")
    write(recipe, inputs)
    public = prefix / "public"
    public.mkdir(exist_ok=True)
    source = baseline_source(source_repo / "gagc/templates/nextitnet_perdataset/train.py")
    model = public / "model.py"
    if model.exists() and model.read_text() != source:
        raise ProtocolError("Public baseline source changed")
    model.write_text(source)
    for split in ("validation", "test"):
        target = prefix / "data" / split
        names = ["trainval/Movies_and_TV_train.txt", "trainval/Movies_and_TV_valid.txt", "trainval/dataset_stats.json"]
        if split == "test":
            names.append("test/Movies_and_TV_test.txt")
        for name in names:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha(destination) != sha(data_source / name):
                    raise ProtocolError("Prepared Movies input changed")
            else:
                shutil.copyfile(data_source / name, destination)
    envlock = prefix / "environment.json"
    if not envlock.exists():
        lock = {"python_version": subprocess.check_output([str(args.python), "--version"], text=True).strip(),
            "pip_freeze": subprocess.check_output([str(args.python), "-m", "pip", "freeze"], text=True).splitlines(),
            "codex_version": subprocess.check_output([str(args.agent), "--version"], text=True).strip(),
            "model": args.model, "effort": args.effort,
            "preparation": "One fresh baseline training and one baseline test inference, separately accounted"}
        write(envlock, lock)
    evaluator = HERE / "movies_evaluator.py"
    base_args = [str(args.python), str(evaluator), "--candidate", str(public)]
    validation = prefix / "calibration" / "validation"
    test = prefix / "calibration" / "test"
    for split, output in (("validation", validation), ("test", test)):
        if (output / "metrics.json").exists():
            continue
        reservation = prefix / "calibration" / (split + "-reservation.json")
        if reservation.exists():
            raise ProtocolError("Baseline outcome unknown; retain calibration reservation and inspect before retry")
        output.mkdir(parents=True, exist_ok=True)
        write(reservation, {"phase": split, "started_at": now(), "gpu": devices[0]}, exclusive=True)
        argv = base_args + ["--data", str(prefix / "data" / split), "--result", str(output), "--phase", split]
        if split == "test":
            argv += ["--checkpoint", str(validation / "checkpoint")]
        env = clean_environment(output, gpu=devices[0])
        env["PATH"] = str(args.python.parent) + ":" + env["PATH"]
        print("Preparing fresh Movies baseline: " + split, flush=True)
        result = invoke_process(argv, 1800, cwd=output, env=env,
            receipt=prefix / "calibration" / (split + "-process.json"))
        write(output / "transport.json", result, exclusive=True)
        if result["returncode"] != 0 or result["timed_out"]:
            raise ProtocolError("Movies calibration failed; inspect " + str(output / "transport.json"))
    c = template("local", MOVIES_PROFILE)
    c["protocol"].update(trial_budget=44, replicates=1)
    c["sources"].update(rec_harness_commit="74a927fefaf7d120c52b6e3a0809958bd33984f0",
                        fml_repository="not used by the Movies profile")
    c["agent"].update(provider="codex", binary=str(args.agent), model=args.model,
        effort=args.effort, timeout_seconds=1200, max_budget_usd_per_call=None,
        version=read(envlock)["codex_version"])
    c["execution"].update(environment_id="movies-nextitnet-seed2024-codex-native-v1",
        gpu_device=devices[0], gpu_devices=devices, agent_env_passthrough=[])
    pins = [str(args.python), str(args.agent), str(evaluator), str(envlock),
        str(WORKSPACE / "mle_star_codex/worker.py"), str(WORKSPACE / "mle_star_codex/run.py")]
    node = Path(args.agent).parent / "node"
    if node.exists():
        pins.append(str(node))
    # The npm shim delegates to a platform binary. Freeze that executable too.
    npm = Path(args.agent).resolve().parents[1]
    for candidate in npm.parent.glob("codex-*/vendor/*/bin/codex"):
        if candidate.is_file():
            pins.append(str(candidate))
    for candidate in (npm / "node_modules/@openai").glob("codex-*/vendor/*/bin/codex"):
        if candidate.is_file():
            pins.append(str(candidate))
    c["execution"]["pinned_files"] = list(dict.fromkeys(pins))
    task = c["tasks"]["movies_nextitnet"]
    task.update(public_source=str(public), validation_data=str(prefix / "data/validation"),
        test_data=str(prefix / "data/test"), baseline_checkpoint=str(validation / "checkpoint"),
        baseline_validation=read(validation / "metrics.json")["metric"],
        baseline_test=read(test / "metrics.json")["metric"], timeout_seconds=1800,
        attestation={"test_labels_absent_from_public_and_validation": True,
            "evaluator_outside_candidate_process": True,
            "protocol": "Identical frozen RecHarness training; final scalar computed in external parent from shuffled unlabeled prediction requests",
            "evidence": str(prefix / "calibration")})
    for split in ("validation", "test"):
        task[split + "_argv"] = [str(args.python), str(evaluator), "--candidate", "{candidate}",
            "--data", "{data}", "--result", "{result}", "--phase", split]
        if split == "test":
            task[split + "_argv"] += ["--checkpoint", "{checkpoint}"]
    config_path = prefix / "study/study.draft.json"
    if config_path.exists():
        existing = read(config_path)
        # created_at belongs to the first draft; never change registered identity.
        c["created_at"] = existing["created_at"]
        if existing != c:
            raise ProtocolError("Existing Movies draft differs from generated configuration")
    else:
        write(config_path, c, exclusive=True)
    errors = check_config(c, prefix / "study", check_images=True)
    if errors:
        raise ProtocolError("Movies preflight failed: " + "; ".join(errors))
    print(json.dumps({"prepared": True, "config": str(config_path), "trajectories": len(matrix(c)),
        "validation_calls": sum(cell["trial_budget"] for cell in matrix(c)), "gpus": devices, "baseline_validation": task["baseline_validation"]}), flush=True)


def run(prefix):
    root = prefix / "study"
    with locked(root / ".movies-controller.lock"):
        registration = load_registration(root) if (root / "registration.json").exists() else freeze(root)
        c = registration["config"]
        if c.get("study_profile") != MOVIES_PROFILE:
            raise ProtocolError("Expected Movies study profile")
        devices = c["execution"]["gpu_devices"]
        cells = matrix(c)
        # Assign shuffled matrix cells to GPUs before launch. Assignments remain
        # fixed on restart, preventing changes in device/cell attribution.
        queues = {gpu: [cell for i, cell in enumerate(cells) if i % len(devices) == j]
                  for j, gpu in enumerate(devices)}
        schedule = prefix / "schedule.json"
        if schedule.exists() and read(schedule) != queues:
            raise ProtocolError("Frozen matrix GPU schedule changed")
        write(schedule, queues)

        def worker(gpu):
            outcomes = []
            runtime = CodexRuntime(c["agent"], c["execution"], root / "workers" / ("gpu-" + gpu), gpu)
            with gpu_lease(c, gpu):
                for cell in queues[gpu]:
                    label = "%s/%s/%d" % (cell["task"], cell["arm"], cell["replicate"])
                    print(json.dumps({"event": "trajectory_started", "gpu": gpu, "cell": label, "time": now()}), flush=True)
                    try:
                        state = Trajectory(root, registration, cell["task"], cell["arm"], cell["replicate"], runtime).run()
                        result = {"cell": label, "gpu": gpu, "status": state["status"], "trials": len(state["records"]),
                                  "best_validation": state["best_metric"], "test_status": (state.get("test") or {}).get("status")}
                    except Exception as exc:
                        result = {"cell": label, "gpu": gpu, "status": "failed", "error": str(exc), "time": now()}
                        write(root / "execution_failures" / (uuid.uuid4().hex + ".json"), result, exclusive=True)
                    outcomes.append(result)
                    print(json.dumps({"event": "trajectory_finished", **result}), flush=True)
            return outcomes

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in devices]
            outcomes = [item for future in futures for item in future.result()]
        write(prefix / "execution-outcomes.json", outcomes)
        summary = report(root, registration, prefix / "report")
        print(json.dumps({"event": "study_finished", "completed": summary["completed_trajectories"],
                          "expected": summary["expected_trajectories"]}), flush=True)


def status(prefix):
    states = []
    for path in sorted((prefix / "study/trajectories").glob("*/*/*/state.json")):
        state = read(path)
        states.append({"task": state["task"], "arm": state["arm"], "replicate": state["replicate"],
            "trials": len(state["records"]), "status": state["status"], "best_validation": state["best_metric"],
            "last_validations": [{k: row.get(k) for k in ("trial", "metric", "status", "diagnostic")} for row in state["records"][-3:]],
            "agent_calls": state["agent_calls"], "pending_trial": (state.get("pending") or {}).get("trial"),
            "plans": len(state["plans"]), "pre_reviews": len(state["pre_reviews"]), "final_reviews": len(state["final_reviews"])})
    failures = list((prefix / "study/execution_failures").glob("*.json"))
    config = read(prefix / "study/study.draft.json")
    cells = matrix(config)
    print(json.dumps({"expected_trajectories": len(cells), "expected_training_calls": sum(c["trial_budget"] for c in cells),
        "accounted_training_calls": sum(x["trials"] for x in states),
        "completed_trajectories": sum(x["status"] == "completed" for x in states),
        "failures": [read(p) for p in failures], "trajectories": states}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "run", "launch", "status"])
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--data", type=Path, default=WORKSPACE / "artifacts/mle-star-movies-2024/data")
    p.add_argument("--repo", type=Path, default=WORKSPACE / "artifacts/RecHarness-6lyc")
    p.add_argument("--python", type=Path, default=WORKSPACE / "artifacts/RecHarness-6lyc/.venv/bin/python")
    p.add_argument("--agent", type=Path, default=Path(shutil.which("codex") or "/missing/codex"))
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="high")
    p.add_argument("--gpus", default="4,5,6,7")
    args = p.parse_args()
    prefix = args.prefix.resolve()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(prefix)
    elif args.command == "status":
        status(prefix)
    else:
        prefix.mkdir(parents=True, exist_ok=True)
        previous = prefix / "launcher.json"
        if previous.exists() and Path("/proc/" + str(read(previous)["pid"])).exists():
            raise ProtocolError("Movies controller already exists; inspect before relaunch")
        argv = [sys.executable, "-u", "-m", "staged_research.movies", "run", "--prefix", str(prefix)]
        with (prefix / "controller.log").open("a") as log:
            proc = subprocess.Popen(argv, cwd=HERE.parent, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(previous, {"pid": proc.pid, "command": argv, "started_at": now()})
        time.sleep(2)
        if proc.poll() is not None:
            raise ProtocolError("Movies controller exited immediately; inspect controller.log")
        print(json.dumps(read(previous)))


if __name__ == "__main__":
    main()
