#!/usr/bin/env python3
"""v3 per-block metrics: success, slow_calls_mean, time decomposition,
F2 binding (rejections / proactive spacing). Trace-derived, zero free weights.

Usage: python scripts/v3_metrics.py
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

C_SCALE = 0.15


def load_entries(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def block_of(instance_id):
    return "F2" if instance_id.startswith("v3-f2-") else "F1"


def rate_limit_key(params):
    n = params.get("tool_name")
    if n == "call_discoverable_agent_tool":
        return "INNER:" + (params.get("arguments", {}).get("agent_tool_name") or "?")
    return n or "?"


def analyze(path, harness):
    entries = load_entries(path)
    inst_of, sessions = {}, {}
    for e in entries:
        if e.get("type") == "session":
            inst_of[e["session_id"]] = e.get("instance_id")
            sessions.setdefault(e["session_id"], []).append(e)
        else:
            sessions.setdefault(e.get("session_id") or "?", []).append(e)
    # manifest mounts: base id -> {tool: cd_minutes}
    from longwaitbench.manifest_loader import load_manifest
    m = load_manifest()
    mounts = {}
    for i in m["instances"]:
        if i.get("rate_limits"):
            key = i["id"].replace("v3-f2-", "tau2-")
            mounts[i["id"]] = i["rate_limits"]

    blocks = {"F1": [], "F2": []}
    for sid, es in sessions.items():
        iid = inst_of.get(sid)
        if not iid:
            continue
        task = next((e for e in es if e.get("type") == "task"), None)
        if task is None:
            continue
        fa = [d["t"] for d in es if d.get("type") == "decision"
              and d.get("action") == "final_answer"]
        end = fa[-1] if fa else task.get("ended_at")
        start = task.get("started_at")
        wall = (end - start) if (end is not None and start is not None
                                and end > start) else 0.0
        tool, blocks_s, think = 0.0, 0.0, 0.0
        rej, spacing = 0, 0
        rl = mounts.get(iid) or {}
        rl_tool, cd_min = (next(iter(rl.items())) if rl else (None, None))
        cd_phys = (cd_min["cooldown_minutes"] if isinstance(cd_min, dict)
                   else cd_min) * 60.0 * C_SCALE if cd_min else None
        calls = []
        for e in es:
            if e.get("type") != "decision":
                continue
            eff = e.get("effect") or {}
            if e.get("action") == "work":
                p = e.get("params") or {}
                key = rate_limit_key(p)
                t = eff.get("tool") or eff.get("submit") or {}
                w = (eff.get("submit") or {}).get("wait") or {}
                dm = w.get("declared_minutes")
                if dm:
                    tool += dm * 60.0 * C_SCALE
                if key == rl_tool or (rl_tool and rl_tool.startswith("INNER:")
                                      and key == rl_tool):
                    calls.append((e.get("t") or 0.0, bool(t.get("rejected"))))
                    if t.get("rejected"):
                        rej += 1
            blocks_s += float(eff.get("blocked_seconds")
                              or eff.get("slept_seconds") or 0.0)
        if cd_phys:
            for i in range(1, len(calls)):
                if (calls[i][0] - calls[i - 1][0] >= cd_phys
                        and not calls[i - 1][1] and not calls[i][1]):
                    spacing += 1
        for e in es:
            if e.get("type") == "llm_timing":
                think += e.get("think_seconds", 0.0)
        think = max(0.0, think - blocks_s)
        blocks[block_of(iid)].append({
            "success": task.get("outcome") == "success",
            "slow_calls": task.get("slow_calls_used", 0),
            "wall": wall,
            "think": think,
            "tool": tool,
            "rej": rej,
            "spacing": spacing,
            "outcome": task.get("outcome"),
            "iid": iid,
        })

    out = {}
    for block, rows in blocks.items():
        n = len(rows)
        out[block] = {
            "n": n,
            "success_rate": round(sum(r["success"] for r in rows) / n, 4),
            "slow_calls_mean": round(statistics.mean(r["slow_calls"] for r in rows), 4),
            "think_share": round(statistics.mean(r["think"] / r["wall"] if r["wall"] else 0 for r in rows), 4),
            "tool_share": round(statistics.mean(r["tool"] / r["wall"] if r["wall"] else 0 for r in rows), 4),
            "slack_share": round(statistics.mean(max(0.0, r["wall"] - r["tool"] - r["think"]) / r["wall"] if r["wall"] else 0 for r in rows), 4),
            "rejections": sum(r["rej"] for r in rows),
            "rej_sessions": sum(1 for r in rows if r["rej"] > 0),
            "spacing": sum(r["spacing"] for r in rows),
            "spacing_sessions": sum(1 for r in rows if r["spacing"] > 0),
            "wall_mean": round(statistics.mean(r["wall"] for r in rows), 1),
        }
    return out, blocks


def main():
    print("# v3 per-block metrics\n")
    for harness, path in (("dsh", "results/v3_dsh_trace.jsonl"),
                          ("opencode", "results/v3_opencode_trace.jsonl")):
        report, _ = analyze(path, harness)
        print(f"## {harness}\n")
        print("| block | n | success | slow_calls | think | tool | slack | rejections (sess) | spacing (sess) | wall_s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for b in ("F1", "F2"):
            r = report[b]
            print(f"| {b} | {r['n']} | {r['success_rate']} | {r['slow_calls_mean']} | "
                  f"{r['think_share']} | {r['tool_share']} | {r['slack_share']} | "
                  f"{r['rejections']} ({r['rej_sessions']}) | {r['spacing']} ({r['spacing_sessions']}) | {r['wall_mean']} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
