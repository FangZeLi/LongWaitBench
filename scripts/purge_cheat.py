#!/usr/bin/env python3
"""Purge all cheat-detection entries from canonical_trace.jsonl so they can
be re-run with correctly suffixed session ids."""
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
    keep = []
    dropped = 0
    for e in entries:
        is_cheat = (
            e.get("cheat_detection")
            or e.get("condition") == "certificate"
            or (e.get("policy") or "").startswith("cheat")
        )
        if is_cheat:
            dropped += 1
        else:
            keep.append(e)
    with open(TRACE, "w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"kept {len(keep)}, dropped {dropped} cheat entries")


if __name__ == "__main__":
    sys.exit(main() or 0)
