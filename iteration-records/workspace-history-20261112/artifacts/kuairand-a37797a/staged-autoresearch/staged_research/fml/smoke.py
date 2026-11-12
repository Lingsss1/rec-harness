"""Small real-model contract checks using DEVELOPMENT data only, no hidden test."""
from __future__ import annotations

import copy
import uuid
from pathlib import Path

from ..contracts import score_result
from ..local import LocalRuntime, gpu_lease
from ..storage import ProtocolError, now, read, write
from .data import manifest, save_arrays


def prepare_smoke_data(task, development, output):
    import numpy as np
    development, output = Path(development), Path(output)
    validation, test = output / "validation", output / "test"
    validation.mkdir(parents=True)
    test.mkdir()
    counts = {}
    names = ("train_env0", "train_env1") if task == "colored_mnist" else ("train",)
    for name in names:
        values = [np.load(development / (name + "_" + suffix + ".npy"), allow_pickle=False, mmap_mode="r")[:128].copy()
                  for suffix in ("x", "y", "ids")]
        if len(values[0]) < 64:
            raise ProtocolError("Smoke training needs at least 64 development examples")
        save_arrays(validation, name, values)
        counts[name] = len(values[0])
    if task == "colored_mnist":
        import shutil
        shutil.copyfile(development / "rng_state.npy", validation / "rng_state.npy")
    for name, start, end in (("val", 0, 16), ("test", 16, 32)):
        values = [np.load(development / ("val_" + suffix + ".npy"), allow_pickle=False, mmap_mode="r")[start:end].copy()
                  for suffix in ("x", "y", "ids")]
        if len(values[0]) != 16:
            raise ProtocolError("Smoke validation needs at least 32 development examples")
        save_arrays(validation if name == "val" else test, name, values)
        counts[name] = 16
    if task == "openood":
        for directory, start in ((validation, 0), (test, 16)):
            values = [np.load(development / ("ood_" + suffix + ".npy"), allow_pickle=False, mmap_mode="r")[start:start + 16].copy()
                      for suffix in ("x", "y", "ids")]
            if len(values[0]) != 16:
                raise ProtocolError("Smoke OOD check needs 32 development OOD examples")
            save_arrays(directory, "ood", values)
        counts.update(ood_val=16, ood_test=16)
    manifest(validation, task, "validation", "smoke", {k: v for k, v in counts.items() if k not in ("test", "ood_test")})
    manifest(test, task, "test", "smoke", {k: v for k, v in counts.items() if k in ("test", "ood_test")})


def run_smoke(config, prefix):
    from .bootstrap import command
    prefix = Path(prefix)
    result_root = prefix / "smoke-runs" / uuid.uuid4().hex
    runtime = LocalRuntime(config["agent"], config["execution"], result_root / "backend", config["execution"]["gpu_device"])
    receipts = []
    with gpu_lease(config):
        for task, card in config["tasks"].items():
            print("Real-model development-only smoke check: " + task, flush=True)
            output = result_root / task
            # The controller requires only stdlib. Array preparation must use
            # the task interpreter, where NumPy is actually installed.
            command([card["validation_argv"][0], "-m", "staged_research.fml.smoke",
                "--task", task, "--development", card["validation_data"],
                "--output", output / "data"], log=output / "preparation.log")
            local = copy.deepcopy(card)
            local.update(validation_data=str(output / "data/validation"), test_data=str(output / "data/test"))
            for phase in ("validation", "test"):
                local[phase + "_argv"] += ["--smoke"]
                checkpoint = output / "validation/checkpoint" if phase == "test" else None
                transport = runtime.evaluate(local, Path(card["public_source"]), output / phase, phase, checkpoint=checkpoint)
                if transport["returncode"] != 0 or transport["timed_out"]:
                    raise ProtocolError("Real-model smoke check failed: " + str(output / phase))
                score_result(read(output / phase / "metrics.json"), card)
            receipts.append({"task": task, "passed": True, "output": str(output)})
    value = {"passed": True, "at": now(), "profile": "smoke", "hidden_test_used": False,
             "environment_id": config["execution"]["environment_id"], "tasks": receipts,
             "purpose": "real-model interface checks, not full-budget FML scores"}
    write(prefix / "smoke-passed.json", value)
    return value


def main():
    import argparse
    from .specs import DEFAULTS
    parser = argparse.ArgumentParser(description="Prepare development-only model smoke arrays")
    parser.add_argument("--task", choices=list(DEFAULTS), required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_smoke_data(args.task, args.development, args.output)


if __name__ == "__main__":
    main()
