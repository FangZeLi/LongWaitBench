#!/usr/bin/env python3
"""Backfill llm_timing (think_seconds) entries into existing v2 traces.

v2 was run before the time-decomposition metric existed; the raw harness
session records (dsh transcripts, opencode dbs) still contain the generation
timestamps, so think time can be recovered WITHOUT re-running sessions.

Usage:
  python scripts/backfill_llm_timing.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))

from longwaitbench.llm_timing import extract_dsh_think, extract_opencode_think  # noqa: E402

DSH_HOME = ROOT / ".dsh-home"
SESSIONS = ROOT / "results" / "sessions"
BACKUP = ROOT / "results" / "sessions_backup_v2"


def dsh_transcript_dir(session_id: str) -> Path:
    workspace = (SESSIONS / session_id).resolve()
    # dsh mangles the absolute workspace path into a dir name:
    # "/a/b" -> "--a-b--" (leading slash dropped, separators become dashes)
    mangled = "--" + str(workspace)[1:].replace("/", "-") + "--"
    d = DSH_HOME / "sessions" / mangled
    return d if d.exists() else None


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traces", nargs="*", default=None,
        help="harness:path pairs, e.g. "
             "dsh-headless:results/v3_dsh_trace.jsonl "
             "opencode:results/v3_opencode_trace.jsonl",
    )
    args = ap.parse_args()
    if args.traces:
        pairs = [(t.split(":", 1)[0], ROOT / t.split(":", 1)[1])
                 for t in args.traces]
    else:
        pairs = [
            ("dsh-headless", ROOT / "results" / "v2_dsh_trace.jsonl"),
            ("opencode", ROOT / "results" / "v2_opencode_trace.jsonl"),
        ]
    for harness, trace_path in pairs:
        if not trace_path.exists():
            print(f"missing {trace_path}", file=sys.stderr)
            continue
        sids = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "session":
                sids.append(e["session_id"])
        # strip any previously backfilled llm_timing lines (redo-safe)
        kept = [
            line
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if '"type": "llm_timing"' not in line
        ]
        trace_path.write_text("\n".join(kept) + ("\n" if kept else ""),
                              encoding="utf-8")
        # per-session judged window [started_at, ended_at] in epoch ms
        windows = {}
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "session":
                windows.setdefault(e["session_id"], {})["start"] = e.get("started_at")
            if e.get("type") == "task":
                windows.setdefault(e["session_id"], {})["end"] = e.get("ended_at")
        appended = 0
        with trace_path.open("a", encoding="utf-8") as f:
            for sid in sids:
                w = windows.get(sid, {})
                start = w.get("start")
                end = w.get("end")
                if start is None:
                    print(f"SKIP {sid}: no session window")
                    continue
                clip = (start * 1000, end * 1000 if end else None)
                think = None
                src = None
                if harness == "dsh-headless":
                    d = dsh_transcript_dir(sid)
                    if d is not None:
                        think = extract_dsh_think(d, *clip)
                        src = "dsh-transcript-chunks"
                else:
                    db = SESSIONS / sid / ".xdg-data" / "opencode" / "opencode.db"
                    if not db.exists():
                        db = BACKUP / sid / ".xdg-data" / "opencode" / "opencode.db"
                    if db.exists():
                        think = extract_opencode_think(db, *clip)
                        src = "opencode-db-parts"
                if think is None:
                    print(f"SKIP {sid}: no session records found")
                    continue
                f.write(
                    json.dumps(
                        {
                            "type": "llm_timing",
                            "session_id": sid,
                            "task_id": sid.split("-seed")[0].split("-", 1)[1],
                            "condition": harness,
                            "think_seconds": round(think, 3),
                            "source": src,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
                appended += 1
                print(f"{sid}: think={think:.1f}s ({src})")
        print(f"{harness}: appended {appended} llm_timing entries")


if __name__ == "__main__":
    main()
