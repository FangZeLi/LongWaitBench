#!/usr/bin/env python3
"""Backfill task.first_slow_submit_at from the first slow submission decision.

T (true completion) = first slow submit time + job duration. Traces written
before the fix lack the field; derive it from decision entries.
Usage: python scripts/backfill_t_anchor.py <trace.jsonl> [more traces...]
"""
import json
import sys
from pathlib import Path


def backfill(trace_path: str):
    p = Path(trace_path)
    if not p.exists():
        print(f"{trace_path}: missing")
        return
    entries = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    submits = {}
    for e in entries:
        if e.get("type") == "decision" and e.get("action") == "work":
            eff = e.get("effect") or {}
            sub = eff.get("submit") or {}
            if sub.get("job_handle"):
                sid = e["session_id"]
                if sid not in submits:
                    submits[sid] = e["t"]
    n = 0
    for e in entries:
        if e.get("type") == "task":
            sid = e["session_id"]
            if e.get("true_completion") and sid in submits:
                if "first_slow_submit_at" not in e:
                    e["first_slow_submit_at"] = submits[sid]
                    n += 1
    p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
    print(f"{trace_path}: set first_slow_submit_at on {n} task entries")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        backfill(path)
