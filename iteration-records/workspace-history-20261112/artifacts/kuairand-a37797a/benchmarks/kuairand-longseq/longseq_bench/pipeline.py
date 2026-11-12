"""Download, prepare, run four A800 models and summarize one verified protocol."""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from .download import MD5, atomic_json, download_archive, extract_logs
from .registry import MODEL_NAMES

ROOT = Path(__file__).resolve().parents[1]


def report(message):
    print("[%s] %s" % (datetime.now(timezone.utc).isoformat(timespec="seconds"), message), flush=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline owns " + str(path)) from None
        yield


def gpu_preflight(nproc, precision):
    import torch
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Training requires Linux and CUDA; use --prepare-only for data preparation")
    count = torch.cuda.device_count()
    nproc = nproc or min(count, 2)
    if count < nproc:
        raise RuntimeError("Requested %d GPUs, only %d visible" % (nproc, count))
    devices = []
    for index in range(nproc):
        with torch.cuda.device(index):
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 is unsupported on visible GPU %d" % index)
            # Check driver/wheel compatibility before downloading ten GB.
            (torch.ones(1, device="cuda") + 1).cpu()
            devices.append({"name": torch.cuda.get_device_name(index),
                            "memory_bytes": torch.cuda.get_device_properties(index).total_memory})
    return nproc, {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices}


def prepare_data(raw, processed, cache, history, manifest):
    """Publish complete directories only; validate completed stages on restart."""
    import pandas as pd
    from .data import make_cache, sha256, verify_cache
    from .prepare import PROFILE, preprocess_author_code
    processed = Path(processed)
    source = processed / "sasrec_format.csv"
    if processed.exists():
        provenance = json.loads(source.with_suffix(".provenance.json").read_text())
        if (provenance["profile"] != PROFILE or provenance["raw_files"] != manifest["files"]
                or provenance["pandas_version"] != pd.__version__
                or provenance["output_sha256"] != sha256(source)):
            raise ValueError("Existing preprocessing differs from raw logs/profile/runtime")
        report("Verified existing processed sequences")
    else:
        stage = processed.with_name(processed.name + ".preparing")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        try:
            report("Preprocessing four logs with the released FuXi code profile (RAM intensive)")
            provenance = preprocess_author_code(raw, stage / source.name)
            if provenance["raw_files"] != manifest["files"]:
                raise ValueError("Raw logs changed during preprocessing")
            stage.rename(processed)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    cache = Path(cache)
    if cache.exists():
        meta = verify_cache(cache)
        if (meta["source_csv_sha256"] != provenance["output_sha256"]
                or meta["max_history"] != history or meta["preprocessing_manifest"] != provenance):
            raise ValueError("Existing cache has different data/history/provenance")
        report("Verified existing model cache")
        return meta
    stage = cache.with_name(cache.name + ".preparing")
    if stage.exists():
        shutil.rmtree(stage)
    try:
        report("Building history-%d memory-mapped cache" % history)
        meta = make_cache(source, stage, history, provenance)
        stage.rename(cache)
        return meta
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run_models(cache, output, nproc, precision, allow_variant):
    command = ["bash", str(ROOT / "scripts/run_a800.sh"), str(cache), str(output), str(nproc), precision]
    if allow_variant:
        command.append("--allow-data-variant")
    subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHON_BIN": sys.executable}, check=True)


def gpu_smoke(output, nproc, precision):
    parent = Path(output) / "preflight-smoke"
    parent.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="cuda-", dir=parent))
    for model in MODEL_NAMES:
        report("CUDA/DDP synthetic preflight: " + model)
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", "--nproc-per-node=" + str(nproc),
                        "-m", "longseq_bench.train", "--model", model, "--smoke", "--device", "cuda",
                        "--precision", precision, "--output", str(directory / model)], cwd=ROOT, check=True)
    return str(directory)


def execute(args):
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "stage": "preflight", "data_root": str(args.data_root),
             "output": str(args.output), "precision": args.precision,
             "preprocessing_profile": "author-code-350-unfiltered-df1",
             "started_at": datetime.now(timezone.utc).isoformat()}

    def stage(name):
        state.update(stage=name, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / "pipeline.json", state)
        report("Stage: " + name)

    # Locks cover both shared datasets and outputs, including all child training.
    with ExitStack() as stack:
        for path in sorted({args.data_root / ".pipeline.lock", args.output / ".pipeline.lock"}):
            stack.enter_context(lock(path))
        try:
            stage("preflight")
            if not args.prepare_only:
                args.nproc, state["gpu_runtime"] = gpu_preflight(args.nproc, args.precision)
                state["nproc"] = args.nproc
                stage("cuda_synthetic_preflight")
                state["cuda_smoke_output"] = gpu_smoke(args.output, args.nproc, args.precision)
            archive = (args.archive or args.data_root / "downloads/KuaiRand-27K.tar.gz").resolve()
            raw, processed, cache = (args.data_root / name for name in ("raw", "processed", "cache-l1024"))
            # A completed pipeline needs only checkpoint/result headroom. The
            # initial download/extraction/preprocess needs substantially more.
            if not cache.exists():
                free = shutil.disk_usage(args.data_root).free / 2 ** 30
                if free < args.min_free_gib:
                    raise RuntimeError("Only %.1f GiB free; need --min-free-gib %.1f before preparation" %
                                       (free, args.min_free_gib))
            stage("download")
            digest = download_archive(archive, report=report)
            if digest["md5"] != MD5:
                raise ValueError("Archive is not the pinned official KuaiRand-27K release")
            state["archive"] = {"path": str(archive), **digest}
            stage("extract")
            manifest = extract_logs(archive, raw, digest, report=report)
            stage("preprocess_and_cache")
            meta = prepare_data(raw, processed, cache, 1024, manifest)
            state["dataset"] = {key: meta[key] for key in
                                ("statistics", "paper_statistics_match", "source_csv_sha256", "max_history")}
            atomic_json(args.output / "dataset.json", {**meta, "archive": state["archive"]})
            if args.require_paper_statistics and not meta["paper_statistics_match"]:
                raise ValueError("Released-code data differs from paper statistics; see dataset.json")
            if args.prepare_only:
                state["status"] = "prepared"
                stage("complete")
                return state
            if not meta["paper_statistics_match"]:
                report("Released-code statistics differ from the paper; all runs are explicitly labeled data variants")
            stage("train_four_models")
            run_models(cache, args.output, args.nproc, args.precision, not meta["paper_statistics_match"])
            stage("summarize")
            subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), str(args.output),
                            "--require-complete"], check=True)
            state["status"] = "completed"
            stage("complete")
            report("Finished. Results: " + str(args.output / "summary.md"))
            return state
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=str(error), updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(args.output / "pipeline.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kuairand-27k")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a800")
    parser.add_argument("--archive", type=Path, help="Reuse an existing official tar.gz (MD5 still checked)")
    parser.add_argument("--nproc", type=int, choices=(1, 2), help="Default: up to two visible GPUs")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-paper-statistics", action="store_true",
                        help="Fail if released-code data does not match paper counts")
    parser.add_argument("--min-free-gib", type=float, default=50,
                        help="Initial free-space floor, excluding future checkpoint growth (default 50 GiB)")
    args = parser.parse_args()
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must be nonnegative")
    execute(args)


if __name__ == "__main__":
    main()
