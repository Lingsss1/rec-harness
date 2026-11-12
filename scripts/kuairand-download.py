"""Download the pinned official archive without requiring the ML environment."""
import argparse
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("benchmark", type=Path)
p.add_argument("data_root", type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.benchmark.resolve()))
from longseq_bench.download import download_archive

download_archive(a.data_root.resolve() / "downloads/KuaiRand-27K.tar.gz",
                 report=lambda line: print(line, flush=True))
