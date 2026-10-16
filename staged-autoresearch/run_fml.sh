#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
exec "${STAGED_RESEARCH_PYTHON:-python3}" -m staged_research.fml.bootstrap "$@"
