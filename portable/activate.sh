#!/usr/bin/env bash
REC_HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REC_HARNESS_ROOT
export PATH="$REC_HARNESS_ROOT/portable/bin:$PATH"
export REC_HARNESS_PORTABLE=1
printf 'RecHarness portable root: %s\n' "$REC_HARNESS_ROOT"
