#!/usr/bin/env python3
"""Harness-mode report (parameterized): metrics for one harness run.

Usage:
  python longwaitbench/harness_report.py --harness dsh-headless \
      --trace results/harness_trace.jsonl --out results/harness_results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from longwaitbench import constants as C  # noqa: E402
from longwaitbench.metrics import ConditionMetrics, load_entries  # noqa: E402

HARNESS_PASS_K = (4, 8, 12, 16, 20, 24, 28, 32)


def trace_coverage(entries, expected: set) -> dict:
    """Benchmark-own integrity metric: what fraction of expected sessions
    produced complete, parseable records."""
    sessions = {e["session_id"] for e in entries if e.get("type") == "session"}
    tasks = {e["session_id"] for e in entries if e.get("type") == "task"}
    llms = {e["session_id"] for e in entries if e.get("type") == "llm"
            and e.get("usage", {}).get("prompt_tokens", 0) > 0}
    n = len(expected)
    return {
        "sessions_expected": n,
        "session_entries": len(sessions & expected),
        "task_entries": len(tasks & expected),
        "usage_entries": len(llms & expected),
        "trace_coverage": round(len(tasks & expected) / n, 4) if n else None,
    }


def seed_consistency(tasks: list) -> dict:
    """Reproducibility as a benchmark-own metric: how consistent outcomes
    are across seeds, per task and overall."""
    per_task = {}
    for t in tasks:
        per_task.setdefault(t["task_id"], {})[t["seed"]] = t["outcome"]
    agree = []
    for tid, by_seed in per_task.items():
        vals = [by_seed.get(s) for s in C.SEEDS]
        agree.append(1.0 if all(v is not None and v == vals[0] for v in vals) else 0.0)
    return {
        "agreement_rate": round(sum(agree) / len(agree), 4) if agree else None,
        "n_tasks": len(agree),
        "per_seed_success": {
            str(s): round(
                sum(1 for t in tasks if t["seed"] == s and t["outcome"] == "success")
                / max(1, sum(1 for t in tasks if t["seed"] == s)),
                4,
            )
            for s in C.SEEDS
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="dsh-headless")
    ap.add_argument("--trace", default=str(ROOT / "results" / "harness_trace.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "harness_results.json"))
    args = ap.parse_args()

    entries = load_entries(Path(args.trace))
    tasks = [e for e in entries if e.get("type") == "task"]
    manifest_path = Path(os.environ.get("LC1_MANIFEST",
                                        str(ROOT / "data" / "manifest.json")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f"{args.harness}-{i['id']}-seed{s}" for i in manifest["instances"] for s in C.SEEDS}
    cm = ConditionMetrics(entries, args.harness, len(tasks))
    res = cm.compute(pass_k_ks=HARNESS_PASS_K)
    vec, diag = res["vector"], res["diagnostics"]

    llms = [e for e in entries if e.get("type") == "llm" and e.get("condition") == args.harness]
    total_in = sum(l["usage"]["prompt_tokens"] for l in llms)
    total_out = sum(l["usage"]["completion_tokens"] for l in llms)
    reasoning = sum(l.get("reasoning_tokens", 0) for l in llms)
    cache_read = sum(l.get("cache_read_tokens", 0) for l in llms)
    n_calls = sum(l.get("n_model_calls") or 0 for l in llms)

    out = {
        "benchmark": "LongWaitBench — harness stage",
        "condition": args.harness,
        "model": "deepseek-v4-pro",
        "reasoning": "enabled (model native mode)",
        "pass_k_ks": list(HARNESS_PASS_K),
        "vector": vec,
        "diagnostics": diag,
        "action_counts": res["action_counts"],
        "pass_k_curve": res["pass_k_curve"],
        "usage": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "reasoning_tokens": reasoning,
            "cache_read_tokens": cache_read,
            "n_model_calls": n_calls,
            "sessions": len(llms),
        },
        "integrity": trace_coverage(entries, expected),
        "run_conditions": {
            "clock_fidelity": cm.clock_fidelity(),
            "rate_limit_events": cm.rate_limit_events(),
        },
        "seed_consistency": seed_consistency(tasks),
        "per_task": [
            {
                "task_id": t["task_id"],
                "seed": t["seed"],
                "outcome": t["outcome"],
                "decisions": t["decision_count"],
                "elapsed_sec": round(t["ended_at"] - t["started_at"], 1),
                "slow_calls_used": t.get("slow_calls_used"),
            }
            for t in sorted(tasks, key=lambda x: (x["task_id"], x["seed"]))
        ],
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"condition={args.harness} success_rate: {vec['success_rate']}")
    print(f"condition={args.harness} call_efficiency: {vec['call_efficiency']}")
    print(f"condition={args.harness} pass_k_curve: "
          + ",".join(f"{x:.6f}" for x in res["pass_k_curve"]))
    print(f"condition={args.harness} time_efficiency: {vec['time_efficiency']}")
    print(f"condition={args.harness} abandon_rate: {diag['abandon_rate']}")
    print(f"condition={args.harness} slow_calls_mean: {diag['slow_calls_mean']}")
    print(f"condition={args.harness} think_share: {diag['think_share']}")
    print(f"condition={args.harness} tool_share: {diag['tool_share']}")
    print(f"condition={args.harness} slack_share: {diag['slack_share']}")
    print(f"condition={args.harness} total_tokens: {total_in + total_out}")
    print(f"condition={args.harness} n_model_calls: {n_calls}")
    print(f"condition={args.harness} reasoning_tokens: {reasoning}")
    print(f"condition={args.harness} cache_read_tokens: {cache_read}")
    print(f"condition={args.harness} action_counts: {json.dumps(res['action_counts'])}")
    print(f"condition={args.harness} trace_coverage: {out['integrity']['trace_coverage']}")
    print(f"condition={args.harness} clock_fidelity: "
          f"{json.dumps(out['run_conditions']['clock_fidelity'])}")
    print(f"condition={args.harness} rate_limit_events: "
          f"{out['run_conditions']['rate_limit_events']}")
    print(f"condition={args.harness} seed_consistency: "
          f"{out['seed_consistency']['agreement_rate']}")
    print()
    print("per-task outcomes:")
    for t in sorted(tasks, key=lambda x: (x["task_id"], x["seed"])):
        print(
            f"  {t['task_id']:20s} s{t['seed']} {t['outcome']:10s} "
            f"decisions={t['decision_count']:3d} elapsed={t['ended_at'] - t['started_at']:7.1f}s"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
