#!/usr/bin/env bash
# LongWaitBench v1 full run (resume-safe; parallel sessions inside main.py).
set -euo pipefail
cd "$(dirname "$0")/.."
export TAU2_DATA_DIR="$PWD/data/tau2bench/data"
export LC1_MANIFEST="$PWD/data/manifest.json"
export PYTHONPATH="$PWD/data/pylibs"
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "FATAL: DEEPSEEK_API_KEY is not set" >&2
  exit 1
fi
mkdir -p results
exec /data3/huangwenrui/wenrui_workdir/AutoResearchClaw/.venv/bin/python3 main.py "$@" 2>&1 | tee results/run.log
