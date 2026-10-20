"""Offline fixtures for download integrity, restart safety and result reporting."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from longseq_bench import download, pipeline
from longseq_bench.checkpoints import prepare_run, recover_metrics
from longseq_bench.data import sha256

ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}

    def getcode(self):
        return self.status


def fixture_archive(path, missing=False, unsafe=False):
    with tarfile.open(path, "w:gz") as bundle:
        names = download.LOGS[:-1] if missing else download.LOGS
        for index, name in enumerate(names):
            # Each item meets the author >=350 threshold; all users pass >=20.
            payload = "user_id,video_id,time_ms,play_time_ms,is_hate\n" + "".join(
                "%d,%d,%d,0,1\n" % (user, user * 10, index * 1000 + i)
                for user in range(2) for i in range(90))
            info = tarfile.TarInfo(("../" if unsafe else "KuaiRand-27K/data/") + name)
            info.size = len(payload.encode())
            bundle.addfile(info, io.BytesIO(payload.encode()))
        info = tarfile.TarInfo("KuaiRand-27K/data/unneeded_features.csv")
        info.size = 4
        bundle.addfile(info, io.BytesIO(b"skip"))


def write_results(output, cache, kind="kuairand27k_benchmark"):
    meta = json.loads((cache / "metadata.json").read_text())
    for name in ("hstu", "ultra_hstu", "dare", "ccformer"):
        directory = output / name
        directory.mkdir(exist_ok=True)
        run = {"fingerprint": name, "run_kind": kind, "source_csv_sha256": meta["source_csv_sha256"],
               "cache_files": meta["files"], "paper_statistics_match": meta["paper_statistics_match"],
               "preprocessing_manifest": meta["preprocessing_manifest"], "data_statistics": meta["statistics"],
               "model": {"name": name, "max_history": 1024, "num_items": 131091, "dim": 128},
               "precision": "fp32", "training": {"epochs": 1}, "world_size": 1,
               "gradient_accumulation": 2, "code_files": {"test": "fixture"}, "parameters": 100,
               "evaluation": {"batch_size": 16, "candidate_chunk": 256}}
        run["adaptations"] = json.loads((ROOT / "configs/benchmark.json").read_text())["adaptations"]
        final = {"fingerprint": name, "epoch": 0, "training_complete": True,
                 "metrics": {"ndcg@10": .1, "ndcg@50": .2, "hr@10": .3, "hr@50": .4,
                             "mrr": .05, "mrr_exact": .04},
                 "rank_training_costs": [{"seconds": 2, "peak_allocated_bytes": 4096}],
                 "evaluation_seconds": 3}
        (directory / "run.json").write_text(json.dumps(run))
        (directory / "final.json").write_text(json.dumps(final))
        # A duplicate logged epoch must not double counted in costs.
        (directory / "metrics.jsonl").write_text((json.dumps(final) + "\n") * 2)


class DownloadTests(unittest.TestCase):
    def test_range_resume_and_verified_file_reuse(self):
        payload = b"verified KuaiRand fixture"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive.tar.gz"
            path.with_name(path.name + ".part").write_bytes(payload[:8])
            response = Response(payload[8:], 206, {"Content-Length": str(len(payload) - 8),
                "Content-Range": "bytes 8-%d/%d" % (len(payload) - 1, len(payload))})
            with patch.object(download, "urlopen", return_value=response) as request:
                digest = download.download_archive(path, expected_md5=hashlib.md5(payload).hexdigest())
                self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=8-")
                download.download_archive(path, expected_md5=digest["md5"])
                self.assertEqual(request.call_count, 1)
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_ignored_range_restarts_instead_of_appending(self):
        payload = b"full archive"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive"
            path.with_name(path.name + ".part").write_bytes(b"old prefix")
            with patch.object(download, "urlopen", return_value=Response(payload)):
                download.download_archive(path, expected_md5=hashlib.md5(payload).hexdigest())
            self.assertEqual(path.read_bytes(), payload)

    def test_incomplete_response_resumes_next_attempt(self):
        payload = b"complete archive"
        responses = [Response(payload[:5], headers={"Content-Length": str(len(payload))}),
                     Response(payload[5:], 206, {"Content-Length": str(len(payload) - 5),
                         "Content-Range": "bytes 5-%d/%d" % (len(payload) - 1, len(payload))})]
        with tempfile.TemporaryDirectory() as folder, patch.object(download, "time") as timer:
            path = Path(folder) / "archive"
            timer.monotonic.return_value = 0
            with patch.object(download, "urlopen", side_effect=responses) as request:
                download.download_archive(path, expected_md5=hashlib.md5(payload).hexdigest())
                self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=5-")
            self.assertEqual(path.read_bytes(), payload)

    def test_wrong_checksum_never_publishes_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive"
            with patch.object(download, "urlopen", return_value=Response(b"wrong")):
                with self.assertRaisesRegex(ValueError, "MD5"):
                    download.download_archive(path)
            self.assertFalse(path.exists())

    def test_selective_extraction_and_raw_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_archive(root / "archive")
            digest = download.hashes(root / "archive")
            download.extract_logs(root / "archive", root / "raw", digest)
            self.assertEqual(set(p.name for p in (root / "raw").iterdir()), set(download.LOGS) | {"download.json"})
            download.extract_logs(root / "archive", root / "raw", digest)
            (root / "raw" / download.LOGS[0]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "Changed raw log"):
                download.extract_logs(root / "archive", root / "raw", digest)

    def test_missing_or_unsafe_members_leave_no_completed_raw_directory(self):
        for options in ({"missing": True}, {"unsafe": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture_archive(root / "archive", **options)
                with self.assertRaises(ValueError):
                    download.extract_logs(root / "archive", root / "raw", download.hashes(root / "archive"))
                self.assertFalse((root / "raw").exists())
                self.assertFalse((root / "raw.extracting").exists())


class CheckpointRecoveryTests(unittest.TestCase):
    def test_distributed_smoke_only_rank_zero_builds_shared_cache(self):
        from longseq_bench.train import prepare_smoke_cache
        root = Path("fixture-output")
        with patch("longseq_bench.train.smoke_input") as write, patch("longseq_bench.train.dist.barrier") as barrier:
            self.assertEqual(prepare_smoke_cache(root, False, 0, 2), root / "synthetic-cache")
            write.assert_called_once_with(root)
            prepare_smoke_cache(root, False, 1, 2)
            prepare_smoke_cache(root, True, 0, 2)
            self.assertEqual(write.call_count, 1)
            self.assertEqual(barrier.call_count, 3)

    def test_first_epoch_failure_is_preserved_before_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "hstu"
            directory.mkdir()
            (directory / "run.json").write_text(json.dumps({"model": {"name": "hstu"},
                                                          "run_kind": "kuairand27k_benchmark"}))
            (directory / "last.pt.tmp").write_bytes(b"incomplete checkpoint")
            backup = prepare_run(root, "hstu")
            self.assertFalse(directory.exists())
            self.assertEqual((backup / "last.pt.tmp").read_bytes(), b"incomplete checkpoint")
            self.assertIsNone(prepare_run(root, "hstu"))

    def test_does_not_restart_completed_or_unrecognized_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "hstu"
            directory.mkdir()
            (directory / "other.txt").write_text("unrelated")
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                prepare_run(root, "hstu")
            (directory / "final.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing its last.pt"):
                prepare_run(root, "hstu")
            (directory / "last.pt").write_bytes(b"checkpoint")
            self.assertIsNone(prepare_run(root, "hstu"))

    def test_metrics_recover_from_checkpoint_after_interrupted_append(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            epoch0 = {"epoch": 0, "fingerprint": "same"}
            epoch1 = {"epoch": 1, "fingerprint": "same"}
            epoch2 = {"epoch": 2, "fingerprint": "same"}
            metrics = root / "metrics.jsonl"
            # Duplicate and uncommitted records, then a torn final line.
            metrics.write_text("\n".join(map(json.dumps, (epoch0, epoch0, epoch2))) + '\n{"epoch":')
            recover_metrics(root, "same", 2, epoch1)
            self.assertEqual([json.loads(line) for line in metrics.read_text().splitlines()], [epoch0, epoch1])
            self.assertEqual(json.loads((root / "latest.json").read_text()), epoch1)
            recover_metrics(root, "same", 2, epoch1)
            self.assertEqual(len(metrics.read_text().splitlines()), 2)

    def test_metrics_mismatch_cannot_be_recovered_as_another_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "metrics.jsonl").write_text(json.dumps({"epoch": 0, "fingerprint": "foreign"}))
            with self.assertRaisesRegex(ValueError, "different run"):
                recover_metrics(root, "expected", 1, {"epoch": 0, "fingerprint": "expected"})


@unittest.skipUnless(importlib.util.find_spec("pandas"), "Requires pandas")
class WorkflowTests(unittest.TestCase):
    def test_download_to_preprocessing_training_dispatch_and_summary_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "archive.tar.gz"
            fixture_archive(archive)
            digest = download.hashes(archive)
            args = argparse.Namespace(data_root=root / "data", output=root / "results", archive=archive,
                nproc=1, precision="fp32", prepare_only=False, require_paper_statistics=False, min_free_gib=0)

            def train(cache, output, nproc, precision, allow_variant):
                self.assertEqual((nproc, precision, allow_variant), (1, "fp32", True))
                write_results(output, cache)

            with patch.object(pipeline, "MD5", digest["md5"]), \
                 patch.object(pipeline, "download_archive", return_value=digest), \
                 patch.object(pipeline, "gpu_preflight", return_value=(1, {"test": "GPU mocked"})), \
                 patch.object(pipeline, "gpu_smoke", return_value="mocked-cuda-smoke"), \
                 patch.object(pipeline, "run_models", side_effect=train) as dispatch:
                self.assertEqual(pipeline.execute(args)["status"], "completed")
                source = args.data_root / "processed/sasrec_format.csv"
                before = (sha256(source), source.stat().st_mtime_ns)
                self.assertEqual(pipeline.execute(args)["status"], "completed")
                self.assertEqual(before, (sha256(source), source.stat().st_mtime_ns))
                self.assertEqual(dispatch.call_count, 2)
            state = json.loads((args.output / "pipeline.json").read_text())
            self.assertFalse(state["dataset"]["paper_statistics_match"])
            self.assertEqual(state["dataset"]["statistics"], {"users": 2, "observed_items": 2, "interactions": 720})
            summary = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(len(summary), 4)
            self.assertEqual(summary[0]["costs"]["training_seconds"], 2)
            self.assertEqual(summary[0]["preprocessing_profile"], "author-code-350-unfiltered-df1")
            with (args.output / "summary.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            self.assertIn("paper statistics match=False", (args.output / "summary.md").read_text())
            # Strict paper counts fail before any model dispatch.
            args.require_paper_statistics = True
            with patch.object(pipeline, "MD5", digest["md5"]), \
                 patch.object(pipeline, "download_archive", return_value=digest), \
                 patch.object(pipeline, "gpu_preflight", return_value=(1, {})), \
                 patch.object(pipeline, "gpu_smoke", return_value="mocked-cuda-smoke"), \
                 patch.object(pipeline, "run_models") as dispatch:
                with self.assertRaisesRegex(ValueError, "paper statistics"):
                    pipeline.execute(args)
                dispatch.assert_not_called()
            self.assertEqual(json.loads((args.output / "pipeline.json").read_text())["status"], "failed")

    def test_failed_preprocessing_is_not_published_and_retry_succeeds(self):
        from longseq_bench.prepare import preprocess_author_code
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_archive(root / "archive")
            manifest = download.extract_logs(root / "archive", root / "raw", download.hashes(root / "archive"))

            def fail(raw, output):
                output.write_text("incomplete")
                raise RuntimeError("interrupted preprocessing")

            with patch("longseq_bench.prepare.preprocess_author_code", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    pipeline.prepare_data(root / "raw", root / "processed", root / "cache", 4, manifest)
            self.assertFalse((root / "processed").exists())
            self.assertFalse((root / "processed.preparing").exists())
            meta = pipeline.prepare_data(root / "raw", root / "processed", root / "cache", 4, manifest)
            self.assertEqual(meta["max_history"], 4)
            (root / "processed/sasrec_format.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Existing preprocessing"):
                pipeline.prepare_data(root / "raw", root / "processed", root / "cache", 4, manifest)

    def test_summary_rejects_incomplete_and_synthetic_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_archive(root / "archive")
            manifest = download.extract_logs(root / "archive", root / "raw", download.hashes(root / "archive"))
            pipeline.prepare_data(root / "raw", root / "processed", root / "cache", 4, manifest)
            output = root / "results"
            output.mkdir()
            command = [sys.executable, str(ROOT / "scripts/summarize.py"), str(output), "--require-complete"]
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("All four models", failed.stderr)
            write_results(output, root / "cache", kind="synthetic_smoke")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Synthetic smoke output", failed.stderr)

    def test_lock_blocks_concurrent_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".lock"
            with pipeline.lock(path):
                with self.assertRaisesRegex(RuntimeError, "Another pipeline"):
                    with pipeline.lock(path):
                        pass

    def test_summary_requires_ccformer_and_matching_four_model_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_archive(root / "archive")
            manifest = download.extract_logs(root / "archive", root / "raw", download.hashes(root / "archive"))
            pipeline.prepare_data(root / "raw", root / "processed", root / "cache", 4, manifest)
            output = root / "results"
            output.mkdir()
            write_results(output, root / "cache")
            command = [sys.executable, str(ROOT / "scripts/summarize.py"), str(output), "--require-complete"]
            final_path = output / "ccformer/final.json"
            final = final_path.read_text()
            final_path.unlink()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("All four models", failed.stderr)
            final_path.write_text(final)
            run_path = output / "ccformer/run.json"
            run = json.loads(run_path.read_text())
            run["model"]["max_history"] = 512
            run_path.write_text(json.dumps(run))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("different data, training protocol", failed.stderr)
            run["model"]["max_history"] = 1024
            run_path.write_text(json.dumps(run))
            subprocess.run(command, capture_output=True, text=True, check=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary[-1]["model"], "ccformer")
            self.assertIn("CCFormer-adapted", summary[-1]["model_label"])
            self.assertIn("Triangular", summary[-1]["adaptation"])
            self.assertIn("CCFormer-adapted", (output / "summary.md").read_text())

    def test_cuda_preflight_dispatches_all_models_with_requested_ddp_and_precision(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline.subprocess, "run") as run:
            directory = pipeline.gpu_smoke(Path(folder), 2, "bf16")
            self.assertTrue(Path(directory).is_dir())
            self.assertEqual(run.call_count, 4)
            names = []
            for call in run.call_args_list:
                command = call.args[0]
                self.assertIn("--nproc-per-node=2", command)
                self.assertIn("--smoke", command)
                self.assertEqual(command[command.index("--precision") + 1], "bf16")
                names.append(command[command.index("--model") + 1])
            self.assertEqual(names, ["hstu", "ultra_hstu", "dare", "ccformer"])


if __name__ == "__main__":
    unittest.main()
