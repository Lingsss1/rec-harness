"""Use isolated SDK additions without modifying historical environments."""
from pathlib import Path
import sys


def bootstrap():
    workspace = Path(__file__).resolve().parents[1]
    for path in [workspace / 'staged-autoresearch',
                 workspace / 'artifacts/amazon-aligned-20261022/runtime-packages']:
        sys.path.insert(0, str(path))
