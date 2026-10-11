#!/usr/bin/env python3
"""v4 analysis: F1 / F2 / baseline blocks + within-class diffs.

Usage:
  python scripts/v4_analyze.py --dsh results/v4_dsh_trace.jsonl \
      --opencode results/v4_opencode_trace.jsonl

Blocks:
  F1        : base instances not mounted with rate limits (24 tasks)
  F2        : v3-f2-* instances (24 tasks, CD on)
  BASELINE  : v4-base-* instances (48 tasks, all waits removed)

Within-class diffs (no same-task cross-class comparison):
  F1 - BASELINE(f1) : net effect of execution latency
  F2 - BASELINE(f2) : net effect of cooldown

Metrics per block per harness (trace-derived, zero free weights):
  success_rate, slow_calls_mean, rejection events (raw case evidence),
  proactive spacing, wall time mean.

Requires LC1_MANIFEST pointing at data/manifest_v4.json.
"""
import collections
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

C_SCALE = 0.15


def load_manifest() -> dict:
    from longwaitbench.manifest_loader import load_manifest as _lm
    return _lm()


def load_entries(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rate_limit_key(entry: dict) -> str:
    p = entry.get("params") or {}
    name = p.get("tool_name")
    if name == "call_discoverable_agent_tool":
        return "INNER:" + (p.get("arguments", {}).get("agent_tool_name") or "?")
    return name or "?"


def classify(instance_id: str) -> str:
    if instance_id.startswith("v3-f2-"):
        return "F2"
    if instance_id.startswith("v4-base-"):
        return "BASELINE"
    return "F1"


def analyze(path: str, harness: str):
    entries = load_entries(path)
    sessions = collections.defaultdict(list)
    for e in entries:
        if e.get("type") == "session":
            sessions[e["session_id"]] = []
        sessions[e.get("session_id") or "?"].append(e)

    inst_of = {}
    for sid, es in sessions.items():
        for e in es:
            if e.get("type") == "session":
                inst_of[sid] = e.get("instance_id")
                break

    manifest = load_manifest()
    by_id = {i["id"]: i for i in manifest["instances"]}

    blocks = collections.defaultdict(list)
    for sid, es in sessions.items():
        inst_id = inst_of.get(sid)
        if inst_id is None:
            continue
        block = classify(inst_id)
        task = next((e for e in es if e.get("type") == "task"), None)
        if task is None:
            continue
        out = {
            "session_id": sid,
            "instance_id": inst_id,
            "pair_class": (by_id.get(inst_id) or {}).get("pair_class"),
            "success": task.get("outcome") == "success",
            "outcome": task.get("outcome"),
            "decisions": task.get("decision_count", 0),
            "slow_calls": task.get("slow_calls_used", 0),
            "wall_s": (task.get("ended_at") or 0) - (task.get("started_at") or 0),
            "rejections": 0,
            "spacing": 0,
        }
        rl_tool, cd_min = None, None
        inst = by_id.get(inst_id)
        if inst and inst.get("rate_limits"):
            rl_tool, spec = next(iter(inst["rate_limits"].items()))
            cd_min = float(spec["cooldown_minutes"])
        cd_phys = cd_min * 60.0 * C_SCALE if cd_min else None

        calls = []
        for e in es:
            if e.get("type") != "decision" or e.get("action") != "work":
                continue
            key = rate_limit_key(e)
            eff = e.get("effect") or {}
            t = eff.get("tool") or eff.get("submit") or {}
            if rl_tool and key != rl_tool and not rl_tool.startswith("INNER:"):
                continue
            if rl_tool and rl_tool.startswith("INNER:") and key != rl_tool:
                continue
            if rl_tool is None:
                continue
            rejected = bool(t.get("rejected"))
            calls.append((e.get("t") or 0.0, rejected))
            if rejected:
                out["rejections"] += 1
        if cd_phys:
            for i in range(1, len(calls)):
                t0, r0 = calls[i - 1]
                t1, r1 = calls[i]
                if t1 - t0 >= cd_phys and not r0 and not r1:
                    out["spacing"] += 1
        blocks[block].append(out)

    report = {}
    for block, sess in blocks.items():
        n = len(sess)
        succ = sum(1 for s in sess if s["success"])
        report[block] = {
            "n": n,
            "success_rate": round(succ / n, 4) if n else None,
            "slow_calls_mean": (round(statistics.mean(s["slow_calls"] for s in sess), 4)
                                if n else None),
            "rejections_total": sum(s["rejections"] for s in sess),
            "rejections_sessions": sum(1 for s in sess if s["rejections"] > 0),
            "spacing_total": sum(s["spacing"] for s in sess),
            "spacing_sessions": sum(1 for s in sess if s["spacing"] > 0),
            "wall_s_mean": round(statistics.mean(s["wall_s"] for s in sess), 1) if n else None,
        }
    return report, blocks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsh", required=True)
    ap.add_argument("--opencode", required=True)
    args = ap.parse_args()

    print("# v4 blocks\n")
    for harness, path in (("dsh", args.dsh), ("opencode", args.opencode)):
        if not os.path.exists(path):
            print(f"## {harness}: trace missing ({path})")
            continue
        report, blocks = analyze(path, harness)
        print(f"## {harness}\n")
        hdr = ("| block | n | success | slow_calls | rejections (sessions) | "
               "spacing (sessions) | wall_s |")
        print(hdr)
        print("|---|---|---|---|---|---|---|")
        for block in ("F1", "BASELINE", "F2"):
            r = report.get(block)
            if not r:
                continue
            print(f"| {block} | {r['n']} | {r['success_rate']} | "
                  f"{r['slow_calls_mean']} | {r['rejections_total']} "
                  f"({r['rejections_sessions']}) | {r['spacing_total']} "
                  f"({r['spacing_sessions']}) | {r['wall_s_mean']} |")
        # within-class diffs
        print()
        for cls in ("f1", "f2"):
            a = report.get("F1" if cls == "f1" else "F2")
            b = [s for s in blocks.get("BASELINE", [])
                 if s["pair_class"] == cls]
            if not a or not b:
                continue
            n = len(b)
            succ_b = sum(1 for s in b if s["success"])
            rate_b = round(succ_b / n, 4) if n else None
            diff = None
            if a["success_rate"] is not None and rate_b is not None:
                diff = round((a["success_rate"] - rate_b) * 100, 1)
            label = "F1" if cls == "f1" else "F2"
            print(f"  {label} - baseline({cls}) success-rate diff: {diff} pp "
                  f"({label} {a['success_rate']} vs baseline {rate_b}, n={n})")
        print()
        # per-task F2 evidence
        f2 = blocks.get("F2", [])
        if f2:
            print("| F2 task | outcome | rejections | spacing |")
            print("|---|---|---|---|")
            for s in sorted(f2, key=lambda x: x["instance_id"]):
                print(f"| {s['instance_id']} | {s['outcome']} | "
                      f"{s['rejections']} | {s['spacing']} |")
            print()
        # baseline per-task outcomes
        base = blocks.get("BASELINE", [])
        if base:
            fails = sorted((s for s in base if not s["success"]),
                           key=lambda x: x["instance_id"])
            print(f"baseline failures ({len(fails)}/{len(base)}): "
                  + ", ".join(s["instance_id"] for s in fails))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
