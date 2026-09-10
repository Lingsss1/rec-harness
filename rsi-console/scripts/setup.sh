#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .runtime/bin/node ]; then export PATH="$PWD/.runtime/bin:$PATH"; fi
node -e 'if(Number(process.versions.node.split(".")[0])<22)throw Error("Node.js 22.13+ required")'
npm ci
if [ ! -d vendor/self_evolverec/.git ]; then git clone https://github.com/Sein-Kim/self_evolverec.git vendor/self_evolverec; fi
git -C vendor/self_evolverec checkout --detach edeff0126f237483e2d30651b5179eaea0b0ad22
python3 -m venv .venv
.venv/bin/pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy==2.2.6 tqdm==4.67.1
npm run build
printf '%s\n' '请在本机执行 codex login，然后 bash scripts/start.sh'
