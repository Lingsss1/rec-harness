#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .runtime/bin/node ]; then export PATH="$PWD/.runtime/bin:$PATH"; fi
exec node server/index.mjs
