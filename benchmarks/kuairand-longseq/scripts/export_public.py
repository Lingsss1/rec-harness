#!/usr/bin/env python3
"""Build an allowlisted source archive; excludes datasets, runs and vendor trees."""
import argparse
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {"README.md", "LICENSE", "NOTICE", ".gitignore", "pyproject.toml",
         "requirements-linux.txt", "Dockerfile"}
DIRS = {"longseq_bench", "configs", "scripts", "tests", ".github"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.is_symlink() or "__pycache__" in relative.parts:
                continue
            if relative.parts[0] not in DIRS and str(relative) not in FILES:
                continue
            if path.suffix not in {".py", ".json", ".gin", ".sh", ".md", ".yml", ".yaml"} and str(relative) not in FILES:
                continue
            archive.add(path, arcname=str(Path("longseq-kuairand-bench") / relative), recursive=False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
