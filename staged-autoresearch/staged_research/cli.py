from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Trajectory
from .report import report
from .runtime import DockerRuntime
from .prepare import prepare_fml
from .storage import ProtocolError, read
from .study import check_config, freeze, initialize, load_registration, matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native-agent stage planning study. init/matrix/preflight never train or call a model.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "prepare-fml", "matrix", "preflight", "freeze", "run", "run-matrix", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--check-images", action="store_true")
        if name == "prepare-fml":
            command.add_argument("--checkout", type=Path, required=True)
            command.add_argument("--assets", type=Path, required=True)
        if name == "run":
            command.add_argument("--task", choices=["colored_mnist", "openood", "opacus"], required=True)
            command.add_argument("--arm", choices=list("ABCD"), required=True)
            command.add_argument("--replicate", type=int, choices=range(1, 6), required=True)
        if name == "report":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(initialize(args.root))
        elif args.command == "prepare-fml":
            prepare_fml(args.root, args.checkout, args.assets)
            print("Exported public task source; evaluation images/data still need binding. No training performed.")
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
                runtime = DockerRuntime(registration["config"]["agent"], registration["config"]["protocol"]["gpu_device"])
                cells = matrix(registration["config"]) if args.command == "run-matrix" else [{"task": args.task, "arm": args.arm, "replicate": args.replicate}]
                for cell in cells:
                    state = Trajectory(args.root, registration, cell["task"], cell["arm"], cell["replicate"], runtime).run()
                    print(json.dumps({k: state[k] for k in ("task", "arm", "replicate", "status", "best_metric")}))
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
