#!/usr/bin/env bash
# Complete Linux/A800 workflow, including an isolated pinned Python environment.
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL=0
ARGS=(-m longseq_bench.pipeline)
for ARG in "$@"; do
  case "$ARG" in
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help)
      echo "Usage: bash scripts/oneclick_a800.sh [--skip-install] [pipeline options]"
      echo "Default: create .venv-a800, install CUDA 12.1 PyTorch, download and run all four models."
      echo "PYTHON_BIN selects Python 3.9-3.11; VENV_DIR overrides .venv-a800."
      PYTHONPATH="$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m longseq_bench.pipeline --help
      exit 0 ;;
    *) ARGS+=("$ARG") ;;
  esac
done
if [[ "$(uname -s)" != Linux ]]; then
  echo "This launcher targets Linux/A800. For local checks, use the CPU smoke instructions." >&2
  exit 2
fi
if [[ "$SKIP_INSTALL" == 0 ]]; then
  "$PYTHON_BIN" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 11), "Use Python 3.9-3.11 (recommended: Python 3.10)"'
  VENV_DIR="${VENV_DIR:-$BENCH_ROOT/.venv-a800}"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  "$PYTHON_BIN" -m pip install 'pip<26' 'setuptools>=68' wheel
  "$PYTHON_BIN" -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
  "$PYTHON_BIN" -m pip install -r "$BENCH_ROOT/requirements-linux.txt"
  "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$BENCH_ROOT"
fi
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
export PYTHON_BIN
export PYTHONPATH="$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" "${ARGS[@]}"
