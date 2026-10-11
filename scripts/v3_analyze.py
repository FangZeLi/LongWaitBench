#!/usr/bin/env python3
"""v3 analysis: F1 block + F2 (cooldown) block.

Usage:
  python scripts/v3_analyze.py --dsh results/v3_dsh_trace.jsonl \
      --opencode results/v3_opencode_trace.jsonl

Blocks:
  F1           : base instances NOT mounted with rate limits (24 tasks)
  F2           : v3-f2-* instances (24 tasks, CD on; no control variant)

Metrics per block per harness (all trace-derived, zero free weights):
  success_rate, slow_calls_mean, rejection_count (raw case evidence, not a
  formal metric), proactive_spacing_count (adjacent rate-limited-tool calls
  with gap >= cooldown and no rejection between), wall time mean.

Requires LC1_MANIFEST pointing at data/manifest_v3_draft.json and
DATA dir data/f2_mounts.json for the mount/CD definitions.
"""
import collections
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

C_SCALE = 0.15

MOUNTS_PATH = os.path.join(ROOT, "data", "f2_mounts.json")


def load_mounts() -> dict:
    with open(MOUNTS_PATH, encoding="utf-8") as f:
        mounts = json.load(f)
    mounts.pop("_comment", None)
    return mounts


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
    """(session-scope) resolve the rate-limited tool name for a work decision."""
    p = entry.get("params") or {}
    name = p.get("tool_name")
    if name == "call_discoverable_agent_tool":
        return "INNER:" + (p.get("arguments", {}).get("agent_tool_name") or "?")
    return name or "?"


def classify(instance_id: str, mounts: dict) -> str:
    if instance_id.startswith("v3-f2-"):
        return "F2"
    return "F1"


def analyze(path: str, mounts: dict, harness: str) -> dict:
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

    blocks = collections.defaultdict(list)  # block -> list of per-session dicts
    for sid, es in sessions.items():
        inst_id = inst_of.get(sid)
        if inst_id is None:
            continue
        block = classify(inst_id, mounts)
        task = next((e for e in es if e.get("type") == "task"), None)
        if task is None:
            continue
        out = {
            "session_id": sid,
            "instance_id": inst_id,
            "base": inst_id.replace("v3-f2-", ""),
            "success": task.get("outcome") == "success",
            "outcome": task.get("outcome"),
            "decisions": task.get("decision_count", 0),
            "slow_calls": task.get("slow_calls_used", 0),
            "wall_s": (task.get("ended_at") or 0) - (task.get("started_at") or 0),
            "rejections": 0,
            "spacing": 0,
            "rejected_events": [],
        }
        # rate-limited tool + cooldown for this instance
        rl_tool, cd_min = None, None
        inst = next((i for i in load_manifest()["instances"] if i["id"] == inst_id), None)
        if inst and inst.get("rate_limits"):
            rl_tool, spec = next(iter(inst["rate_limits"].items()))
            cd_min = float(spec["cooldown_minutes"])
        cd_phys = cd_min * 60.0 * C_SCALE if cd_min else None

        calls = []  # (t, rejected)
        for e in es:
            if e.get("type") != "decision" or e.get("action") != "work":
                continue
            key = rate_limit_key(e)
            eff = e.get("effect") or {}
            t = eff.get("tool") or eff.get("submit") or {}
            is_target = key == rl_tool or (
                rl_tool and rl_tool.startswith("INNER:") and key == rl_tool
            )
            if not is_target:
                continue
            rejected = bool(t.get("rejected"))
            calls.append((e.get("t") or 0.0, rejected))
            if rejected:
                out["rejections"] += 1
                out["rejected_events"].append(round(e.get("t") or 0.0, 1))
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
            "slow_calls_mean": round(statistics.mean(s["slow_calls"] for s in sess), 4) if n else None,
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

    mounts = load_mounts()
    print("# v3 blocks\n")
    for harness, path in (("dsh", args.dsh), ("opencode", args.opencode)):
        if not os.path.exists(path):
            print(f"## {harness}: trace missing ({path})")
            continue
        report, blocks = analyze(path, mounts, harness)
        print(f"## {harness}\n")
        print("| block | n | success | slow_calls | rejections (sessions) | spacing (sessions) | wall_s |")
        print("|---|---|---|---|---|---|---|")
        for block in ("F1", "F2"):
            r = report.get(block)
            if not r:
                continue
            print(f"| {block} | {r['n']} | {r['success_rate']} | {r['slow_calls_mean']} | "
                  f"{r['rejections_total']} ({r['rejections_sessions']}) | "
                  f"{r['spacing_total']} ({r['spacing_sessions']}) | {r['wall_s_mean']} |")
        print()
        # per-task F2 evidence (rejections / proactive spacing)
        f2 = [s for s in blocks.get("F2", [])]
        if f2:
            print("| F2 task | outcome | rejections | spacing |")
            print("|---|---|---|---|")
            for s in sorted(f2, key=lambda x: x["instance_id"]):
                print(f"| {s['instance_id']} | {s['outcome']} | {s['rejections']} | {s['spacing']} |")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
