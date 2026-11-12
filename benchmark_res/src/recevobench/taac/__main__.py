"""python -m recevobench.taac --help (torch is loaded only for execution)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .common import read, verify_data


def main(argv=None):
    parser = argparse.ArgumentParser(description="TencentGR local autoresearch task adapter")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Prepare native local Parquet and freeze local protocol")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = subs.add_parser("fixture", help="Generate explicitly synthetic native-format test data")
    p.add_argument("--output", required=True)
    p = subs.add_parser("verify")
    p.add_argument("--data", required=True)
    p = subs.add_parser("init")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    for name in [
        "baseline",
        "context",
        "research-context",
        "status",
        "recover",
        "finalize",
        "consolidate",
    ]:
        p = subs.add_parser(name)
        p.add_argument("--study", required=True)
    p = subs.add_parser("_retrieve")
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    p = subs.add_parser("trial")
    p.add_argument("--study", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--design", required=True)
    p = subs.add_parser("run")
    p.add_argument("--study", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p = subs.add_parser("preflight")
    p.add_argument("--config", required=True)
    for name in ["train", "_fit"]:
        p = subs.add_parser(name)
        for option in ["public", "candidate", "output", "protocol"]:
            p.add_argument("--" + option, required=True)
        p.add_argument("--device", required=True)
        p.add_argument("--seed", required=True, type=int)
    p = subs.add_parser("predict")
    for option in ["public", "candidate", "checkpoint", "queries", "output", "device"]:
        p.add_argument("--" + option, required=True)
    p = subs.add_parser("score")
    for option in ["predictions", "answers", "candidates"]:
        p.add_argument("--" + option, required=True)
    p.add_argument("--output")
    p = subs.add_parser("diagnose")
    for option in ["public", "answers", "inference", "output"]:
        p.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .data import prepare

            result = prepare(args.config, args.output)
        elif args.command == "fixture":
            from .fixture import make_fixture

            result = {"config": make_fixture(args.output), "fixture": True}
        elif args.command == "verify":
            result = verify_data(args.data)
        elif args.command == "init":
            from .loop import initialize

            result = initialize(args.config, args.output)
        elif args.command in {
            "baseline",
            "context",
            "research-context",
            "status",
            "recover",
            "finalize",
            "consolidate",
        }:
            from . import loop

            functions = {
                "baseline": loop.baseline,
                "context": loop.context,
                "research-context": loop.research_context,
                "status": lambda r: loop.load(r)[2],
                "recover": loop.recover,
                "finalize": loop.finish,
                "consolidate": loop.stage_memory,
            }
            result = functions[args.command](args.study)
        elif args.command == "_retrieve":
            from .retrieval import worker

            worker(args.request, args.output)
            result = {"output": args.output}
        elif args.command == "trial":
            from .loop import trial

            result = trial(args.study, args.candidate, args.design)
        elif args.command == "run":
            from .loop import run

            result = run(args.study, args.rounds)
        elif args.command == "preflight":
            config = read(args.config)
            base = Path(args.config).resolve().parent
            deps = {
                k: importlib.util.find_spec(k) is not None
                for k in ["torch", "numpy", "pyarrow", "tqdm"]
            }
            device_available = None
            if deps["torch"]:
                import torch

                device_available = config["device"] == "cpu" or (
                    config["device"].startswith("cuda") and torch.cuda.is_available()
                )
            result = {
                "dependencies": deps,
                "device_available": device_available,
                "data_exists": (base / config["data"]).exists(),
                "candidate_exists": (base / config["candidate"]).is_file(),
            }
        elif args.command in {"train", "_fit"}:
            from .engine import predict, train

            output = Path(args.output)
            target = output / "train" if args.command == "_fit" else output
            result = train(
                args.public, args.candidate, target, read(args.protocol), args.device, args.seed
            )
            if args.command == "_fit":
                predict(
                    args.public,
                    args.candidate,
                    target / "checkpoint.pt",
                    Path(args.public) / "feedback.jsonl",
                    output / "feedback",
                    args.device,
                )
        elif args.command == "predict":
            from .engine import predict

            result = predict(
                args.public, args.candidate, args.checkpoint, args.queries, args.output, args.device
            )
        elif args.command == "score":
            from .evaluate import score

            result = score(args.predictions, args.answers, read(args.candidates), args.output)
        else:
            from .evaluate import diagnose

            result = diagnose(args.public, args.answers, args.inference, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and result.get("status") in {"crash", "timeout", "interrupted"}:
            return 1
        return 0
    except (ValueError, FileNotFoundError, KeyError, ImportError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
