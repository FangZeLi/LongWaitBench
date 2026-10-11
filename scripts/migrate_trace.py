#!/usr/bin/env python3
"""One-shot migration + dedupe for canonical_trace.jsonl.

1. Add budget.decision_budget to task entries written before the field
   existed (value = MAX_DECISIONS_PER_TASK).
2. Deduplicate cheat sessions by (policy, instance_id, seed) — earlier
   runner versions keyed resume on ids without the policy suffix, so the
   same cheat session was appended more than once.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from longwaitbench import constants as C  # noqa: E402

TRACE = ROOT / "results" / "canonical_trace.jsonl"


def main():
    entries = []
    with open(TRACE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    seen_cheat = set()
    keep = []
    dropped = 0
    current_key = None
    current_keep = True
    for e in entries:
        if e.get("type") == "task":
            e.setdefault("budget", {}).setdefault(
                "decision_budget", C.MAX_DECISIONS_PER_TASK
            )
        if e.get("type") == "session":
            policy = e.get("policy") or ""
            if policy.startswith("cheat"):
                current_key = (policy, e.get("instance_id"), e.get("seed"))
                if current_key in seen_cheat:
                    current_keep = False
                else:
                    seen_cheat.add(current_key)
                    current_keep = True
            else:
                current_key = (e.get("session_id"),)
                current_keep = True
        if current_keep:
            keep.append(e)
        else:
            dropped += 1

    with open(TRACE, "w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_sessions = sum(1 for e in keep if e.get("type") == "session")
    n_cheat = sum(1 for e in keep if e.get("type") == "session"
                  and (e.get("policy") or "").startswith("cheat"))
    n_tasks = sum(1 for e in keep if e.get("type") == "task" and not e.get("cheat_detection"))
    print(f"kept {len(keep)} entries, dropped {dropped}")
    print(f"sessions={n_sessions} (cheat={n_cheat}), regular tasks={n_tasks}")


if __name__ == "__main__":
    main()
