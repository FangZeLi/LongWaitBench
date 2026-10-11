#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TAU2_DATA_DIR="$PWD/data/tau2bench/data"
export LC1_MANIFEST="$PWD/data/manifest.json"
exec /data3/huangwenrui/wenrui_workdir/AutoResearchClaw/.venv/bin/python3 analysis/analyze.py
