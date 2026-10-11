#!/usr/bin/env python3
"""Screen the full banking task pool for v4 F2 replacement candidates.

Selection criteria (v4 task spec, checklist C.3 + B.3):
  - short gold path (content difficulty must not drown the wait effect);
  - gold-path core-tool repetition: query tools >=3 (CD2) or
    modify/submit tools >=2 (CD5 single-slot) -> binding is guaranteed on
    the SUCCESS path.
Excludes the 12 banking tasks already used in v2/v3.
Output: table sorted by gold-path action count.
"""
import json
from collections import Counter
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tau2bench" / \
    "data" / "tau2" / "domains" / "banking_knowledge" / "tasks"

# already used in v2 (050,052,084,086,096) or v3 (080,048,097,081,091,041,082,039)
USED = {"050", "052", "084", "086", "096", "080", "048", "097",
        "081", "091", "041", "082", "039"}

QUERY_PREFIXES = ("get_", "list_")
MODIFY_PREFIXES = ("file_", "submit_", "order_", "close_", "update_", "schedule_",
                   "transfer_", "cancel_", "pay_", "link_", "redeem_")


def gold_tools(task: dict) -> Counter:
    c = Counter()
    for a in task.get("evaluation_criteria", {}).get("actions", []):
        name = a.get("name", "")
        if name == "call_discoverable_agent_tool":
            name = (a.get("arguments") or {}).get("agent_tool_name") or "?"
        c[name] += 1
    return c


def main():
    rows = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        tid = f.stem.replace("task_", "")
        if tid in USED:
            continue
        task = json.loads(f.read_text(encoding="utf-8"))
        c = gold_tools(task)
        rows.append((tid, sum(c.values()), c))

    rows.sort(key=lambda r: r[1])
    print(f"{'task':10s} {'acts':>4s}  mount-candidates (query>=3 / modify>=2)")
    print("-" * 90)
    for tid, n, c in rows:
        q = {t: k for t, k in c.items() if t.startswith(QUERY_PREFIXES) and k >= 3}
        m = {t: k for t, k in c.items() if t.startswith(MODIFY_PREFIXES) and k >= 2}
        cand = []
        for t, k in sorted(q.items(), key=lambda x: -x[1])[:3]:
            cand.append(f"{t}×{k}")
        for t, k in sorted(m.items(), key=lambda x: -x[1])[:3]:
            cand.append(f"{t}×{k}")
        top = ", ".join(f"{t}×{k}" for t, k in c.most_common(3))
        flag = "  <<<" if (q or m) else ""
        print(f"task_{tid:5s} {n:4d}  [{top}]  BIND: {', '.join(cand) or '-'}{flag}")


if __name__ == "__main__":
    main()
