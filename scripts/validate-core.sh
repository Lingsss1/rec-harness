#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_CACHE="${PYTHONPYCACHEPREFIX:-/tmp/rec-harness-pycache}"

PYTHONPYCACHEPREFIX="${PYTHON_CACHE}" "${PYTHON_BIN}" -m py_compile \
  "${REPO_ROOT}/model-diagnostic-toolkit/scripts/analyze_hook_arrays.py" \
  "${REPO_ROOT}/model-diagnostic-toolkit/scripts/analyze_perturbations.py" \
  "${REPO_ROOT}/model-diagnostic-toolkit/scripts/diagnose_collapse.py" \
  "${REPO_ROOT}/model-diagnostic-tool-evolver/scripts/manage_candidate.py" \
  "${REPO_ROOT}/scripts/check_public_tree.py"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_public_tree.py" "${REPO_ROOT}"

echo "rec-harness public core validation passed"
