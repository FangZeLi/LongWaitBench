#!/usr/bin/env python3
"""Deduplicate cheat-session entries in canonical_trace.jsonl.

The first full run crashed after writing its 6 cheat sessions; the second
run re-appended them (the old runner keyed resume on session entries whose
ids lacked the policy suffix). Keep the FIRST occurrence of every
(session_id, policy) pair and drop the later duplicates.
"""
import json
import sys
from pathlib import Path

TRACE = Path(__file__).resolve().parent.parent / "results" / "canonical_trace.jsonl"


def main():
    entries = []
    with open(TRACE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    seen = set()
    keep = []
    dropped = 0
    current_key = None
    current_keep = True
    for e in entries:
        if e.get("type") == "session":
            current_key = (e.get("session_id"), e.get("policy"))
            if current_key in seen:
                current_keep = False
                dropped += 1
            else:
                seen.add(current_key)
                current_keep = True
        if current_keep:
            keep.append(e)
        else:
            dropped += 1

    with open(TRACE, "w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"kept {len(keep)} entries, dropped {dropped} duplicate cheat entries")
    print("cheat sessions now:", len([k for k in seen if k[1] and k[1].startswith('cheat')]))


if __name__ == "__main__":
    sys.exit(main() or 0)
