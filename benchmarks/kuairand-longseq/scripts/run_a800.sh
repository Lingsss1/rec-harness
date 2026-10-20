#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_a800.sh CACHE OUTPUT [NPROC=2] [PRECISION=fp32] [extra train args...]" >&2
  exit 2
fi
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CACHE="$1"
OUTPUT="$2"
NPROC="${3:-2}"
PRECISION="${4:-fp32}"
if [[ "$NPROC" != 1 && "$NPROC" != 2 ]]; then
  echo "The frozen global batch supports 1 or 2 processes with local batch 64." >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs"
for MODEL in hstu ultra_hstu dare ccformer; do
  "$PYTHON_BIN" -m longseq_bench.checkpoints "$OUTPUT" "$MODEL"
  COMMAND=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC"
    -m longseq_bench.train --model "$MODEL" --config "$BENCH_ROOT/configs/benchmark.json"
    --cache "$CACHE" --output "$OUTPUT/$MODEL" --precision "$PRECISION")
  if [[ -f "$OUTPUT/$MODEL/last.pt" ]]; then
    COMMAND+=(--resume "$OUTPUT/$MODEL/last.pt")
  fi
  if [[ $# -gt 4 ]]; then COMMAND+=("${@:5}"); fi
  "${COMMAND[@]}" 2>&1 | tee -a "$OUTPUT/logs/$MODEL.log"
done
"$PYTHON_BIN" "$BENCH_ROOT/scripts/summarize.py" "$OUTPUT" --require-complete
