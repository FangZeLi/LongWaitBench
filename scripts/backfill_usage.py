#!/usr/bin/env python3
"""Backfill llm usage entries into harness_trace.jsonl from persisted DSH
sessions (repairs runs where the driver's usage injection was skipped)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from longwaitbench.dsh_session_usage import extract  # noqa: E402

TRACE = ROOT / "results" / "harness_trace.jsonl"
HOME = ROOT / ".dsh-home"


def main():
    entries = []
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))

    have_usage = {
        l["session_id"] for l in entries
        if l.get("type") == "llm" and l.get("usage", {}).get("prompt_tokens", 0) > 0
    }
    tasks = [e for e in entries if e.get("type") == "task"]
    missing = [t["session_id"] for t in tasks if t["session_id"] not in have_usage]
    print(f"{len(tasks)} tasks, {len(missing)} missing usage entries")

    session_files = {
        p.parent.parent.name: p
        for p in HOME.glob("sessions/*/session-*/session.jsonl.zstd")
    }
    # index by workspace-encoded dir name containing the session id
    index = {}
    for p in HOME.glob("sessions/*/session-*/session.jsonl.zstd"):
        key = str(p.parent.parent.name)
        index[key] = p

    appended = 0
    for sid in missing:
        # find session dir whose name contains the session id
        hit = None
        for key, p in index.items():
            if sid in key:
                hit = p
                break
        if hit is None:
            print(f"  {sid}: no DSH session dir found")
            continue
        u = extract(hit.parent)
        task = next(t for t in tasks if t["session_id"] == sid)
        entries.append(
            {
                "type": "llm",
                "llm_request_id": None,
                "session_id": sid,
                "task_id": task["task_id"],
                "condition": task["condition"],
                "usage": {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                },
                "reasoning_tokens": u.get("reasoning_tokens", 0),
                "cache_read_tokens": u.get("cache_read_tokens", 0),
                "n_model_calls": u.get("n_model_calls", 0),
                "model": "deepseek-v4-pro",
                "cache_hit": bool(u.get("cache_read_tokens", 0)) or None,
            }
        )
        appended += 1
        print(f"  {sid}: {u.get('n_model_calls')} calls, "
              f"{u.get('input_tokens')} in / {u.get('output_tokens')} out")

    with open(TRACE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"appended {appended} usage entries")


if __name__ == "__main__":
    main()
