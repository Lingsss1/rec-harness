from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from .engine import Trajectory
from .report import report
from .local import gpu_lease, make_runtime
from .evaluate import evaluate_study
from .prepare import prepare_fml
from .storage import ProtocolError, read
from .study import check_config, freeze, initialize, load_registration, matrix


def main(argv=None):
    return interruptible(dispatch, argv)


def interruptible(function, argv=None):
    received = []
    def terminate(signum, frame):
        received.append(signum)
        raise KeyboardInterrupt()
    install_handler = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, terminate) if install_handler else None
    try:
        return function(argv)
    except KeyboardInterrupt:
        print("Interrupted; workers stopped where possible. Rerun the same command to resume; unknown hidden tests are never repeated.", file=sys.stderr)
        return 128 + (received[-1] if received else signal.SIGINT)
    finally:
        if install_handler:
            signal.signal(signal.SIGTERM, previous)


def dispatch(argv=None):
    parser = argparse.ArgumentParser(description="Native-agent stage planning study. init/matrix/preflight never train or call a model.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "prepare-fml", "matrix", "preflight", "freeze", "run", "run-matrix", "report", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "init":
            command.add_argument("--backend", choices=["docker", "local"], default="docker")
        if name == "preflight":
            command.add_argument("--check-images", "--check-environment", dest="check_images", action="store_true")
        if name == "prepare-fml":
            command.add_argument("--checkout", type=Path, required=True)
            command.add_argument("--assets", type=Path, required=True)
        if name == "run":
            command.add_argument("--task", choices=["colored_mnist", "openood", "opacus"], required=True)
            command.add_argument("--arm", choices=list("ABCD"), required=True)
            command.add_argument("--replicate", type=int, choices=range(1, 6), required=True)
        if name in ("report", "evaluate"):
            command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--config", type=Path)
            command.add_argument("--check-only", action="store_true", help="Check readiness without freezing, model calls or training")
            command.add_argument("--keep-going", action="store_true", help="Continue other matrix cells after a failure; preserve missing/failed cells")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(initialize(args.root, args.backend))
        elif args.command == "evaluate":
            result = evaluate_study(args.root, args.output, config_path=args.config,
                                   check_only=args.check_only, keep_going=args.keep_going)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] and (args.check_only or result.get("completed_trajectories") == 60) else 1
        elif args.command == "prepare-fml":
            prepare_fml(args.root, args.checkout, args.assets)
            print("Exported public task source; evaluators/environment/data still need binding. No training performed.")
        elif args.command == "matrix":
            path = args.root / "registration.json"
            config = read(path)["config"] if path.exists() else read(args.root / "study.draft.json")
            rows = matrix(config)
            print(json.dumps({"trajectories": len(rows), "validation_calls": sum(r["trial_budget"] for r in rows), "matrix": rows}, indent=2))
        elif args.command == "preflight":
            path = args.root / "registration.json"
            config = load_registration(args.root)["config"] if path.exists() else read(args.root / "study.draft.json")
            errors = check_config(config, args.root, check_images=args.check_images)
            print(json.dumps({"ready": not errors, "errors": errors}, ensure_ascii=False, indent=2))
            return 1 if errors else 0
        elif args.command == "freeze":
            print(freeze(args.root)["registration_id"])
        else:
            registration = load_registration(args.root)
            if args.command == "report":
                result = report(args.root, registration, args.output)
                print("Completed %d / 60; report: %s" % (result["completed_trajectories"], args.output))
            else:
                errors = check_config(registration["config"], args.root, check_images=True)
                if errors:
                    raise ProtocolError("Preflight failed:\n- " + "\n- ".join(errors))
                runtime = make_runtime(registration["config"], args.root)
                cells = matrix(registration["config"]) if args.command == "run-matrix" else [{"task": args.task, "arm": args.arm, "replicate": args.replicate}]
                with gpu_lease(registration["config"]):
                    for cell in cells:
                        state = Trajectory(args.root, registration, cell["task"], cell["arm"], cell["replicate"], runtime).run()
                        print(json.dumps({k: state[k] for k in ("task", "arm", "replicate", "status", "best_metric")}))
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
