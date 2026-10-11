#!/usr/bin/env python3
"""Smoke test: one session per path into /tmp trace (does not touch results/).

  smoke.py synthetic   — ExponentialBackoff on f3-control-0 + tau2-retail-0
  smoke.py base        — real-LLM base condition on f3-control-0 + tau2-retail-0
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT / "data" / "tau2bench" / "data"))
os.environ.setdefault("LC1_MANIFEST", str(ROOT / "data" / "manifest.json"))

from longwaitbench.manifest_loader import load_manifest  # noqa: E402
from longwaitbench.recording_client import RecordingClient  # noqa: E402
from longwaitbench.session import run_session  # noqa: E402
from longwaitbench.trace import TraceWriter  # noqa: E402

TRACE = Path("/tmp/lwb_smoke_trace.jsonl")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if TRACE.exists():
        TRACE.unlink()
    trace = TraceWriter(TRACE)
    manifest = load_manifest()
    by_id = {i["id"]: i for i in manifest["instances"]}

    client = RecordingClient()
    for inst_id in ("f3-control-0", "tau2-retail-0"):
        inst = by_id[inst_id]
        if mode == "synthetic":
            task = run_session("ExponentialBackoff", inst, 0, trace)
            print(inst_id, "ExponentialBackoff ->", task["outcome"], task["error"])
        else:
            task = run_session("LongCall-v1", inst, 0, trace, client=client)
            print(inst_id, "LongCall-v1 ->", task["outcome"], task["error"])

    print("--- trace entries ---")
    for line in open(TRACE, encoding="utf-8"):
        e = json.loads(line)
        t = e["type"]
        if t == "task":
            print(
                f"task {e['task_id']} outcome={e['outcome']} decisions={e['decision_count']} "
                f"tc={e['true_completion']}"
            )
        elif t == "decision":
            print(
                f"  decision {e['action']:13s} pred={e['time_prediction']} "
                f"effect={ {k: v for k, v in e['effect'].items()} }"
            )
        elif t == "llm":
            print(f"  llm {e['usage']} cache={e['cache_hit']}")


if __name__ == "__main__":
    main()
