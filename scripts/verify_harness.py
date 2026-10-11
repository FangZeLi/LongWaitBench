#!/usr/bin/env python3
"""Harness-mode contract verifier (parameterized: dsh-headless | opencode).

Usage:
  python scripts/verify_harness.py --harness dsh-headless \
      --trace results/harness_trace.jsonl --report results/harness_results.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))

from longwaitbench import constants as C  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def static_oracle_check(harness: str):
    if harness == "dsh-headless":
        patch = (ROOT / ".dsh-home" / "profiles" / "lwb" / "cordis.patch.yml").read_text(
            encoding="utf-8"
        )
        check("harness profile has no bash/editor tool plugins",
              not any(x in patch for x in ("dsh-tool-bash", "str_replace_editor", "dsh-fs")))
        check("harness profile mounts only the lwb MCP client",
              "dsh-mcp-client" in patch and patch.count("- id: ") == 1)
    elif harness == "opencode":
        configs = list((ROOT / "results" / "sessions").glob(
            "opencode-*/opencode.json"))
        if not configs:
            check("opencode session configs exist", False)
            return
        cfg = json.loads(configs[0].read_text(encoding="utf-8"))
        tools = cfg.get("tools", {})
        native_on = [k for k, v in tools.items() if v]
        check("opencode config disables all native tools", not native_on,
              f"enabled: {native_on}")
        mcp = cfg.get("mcp", {})
        check("opencode config mounts the lwb MCP server",
              "lwb" in mcp and mcp["lwb"].get("type") == "local"
              and mcp["lwb"].get("enabled"))
        api_key_val = json.dumps(cfg.get("provider", {}))
        check("no API key literal in opencode config (env template only)",
              "sk-" not in api_key_val)
        check("model is deepseek-v4-pro",
              cfg.get("model") == "lwbdeepseek/deepseek-v4-pro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="dsh-headless")
    ap.add_argument("--trace", default=str(ROOT / "results" / "harness_trace.jsonl"))
    ap.add_argument("--report", default=str(ROOT / "results" / "harness_results.json"))
    args = ap.parse_args()

    trace = Path(args.trace)
    if not trace.exists():
        check("trace jsonl exists", False)
        sys.exit(1)
    entries = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    check("trace parses as JSONL", True)

    sessions = [e for e in entries if e.get("type") == "session"]
    tasks = [e for e in entries if e.get("type") == "task"]
    decisions = [e for e in entries if e.get("type") == "decision"]
    llms = [e for e in entries if e.get("type") == "llm"]

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    expected = {f"{args.harness}-{i['id']}-seed{s}" for i in manifest["instances"] for s in C.SEEDS}
    seen = {s["session_id"] for s in sessions}
    check("24 sessions present", expected <= seen,
          f"missing {len(expected - seen)}: {sorted(expected - seen)[:5]}")
    check("task entry per session", {t["session_id"] for t in tasks} >= expected)
    check("task entries carry budget/outcome/decision_count/true_completion",
          all("budget" in t and "outcome" in t and "decision_count" in t
              and "true_completion" in t for t in tasks))
    check("task entries carry first_slow_submit_at (T anchor)",
          all("first_slow_submit_at" in t for t in tasks
              if t.get("true_completion")))

    ok_dec = all(
        d.get("action") in C.ACTIONS and "effect" in d and "params" in d
        for d in decisions
    )
    check("every decision: action space (incl. wait_for) + params + effect", ok_dec)
    check("harness mode: no elicited time_prediction (documented deviation)",
          all(d.get("time_prediction") is None for d in decisions))
    check("exit-code contract: decisions recorded", len(decisions) > 0,
          f"{len(decisions)} decisions")

    check("usage entry per session", {l["session_id"] for l in llms} >= expected)
    check("usage entries carry real token counts",
          all(l["usage"]["prompt_tokens"] > 0 and l["usage"]["completion_tokens"] > 0
              for l in llms))
    reasoning = sum(l.get("reasoning_tokens", 0) for l in llms)
    check("reasoning tokens present (reasoning enabled, model native mode)",
          reasoning > 0, f"{reasoning} reasoning tokens")

    static_oracle_check(args.harness)

    # benchmark-own integrity metrics
    n = len(expected)
    task_ids = {t["session_id"] for t in tasks}
    check("trace_coverage == 1.0 (task entry for every expected session)",
          task_ids >= expected,
          f"coverage={len(task_ids & expected)}/{n}")

    # independent recomputation of headline numbers from the trace
    n_tasks = len(tasks)
    sr = sum(1 for t in tasks if t["outcome"] == "success") / n_tasks
    times = sorted(t["ended_at"] - t["started_at"] for t in tasks
                   if t["outcome"] == "success")
    auc = 0.0
    if times:
        tmax = times[-1]
        tot, cur, prev = 0.0, 0.0, 0.0
        for t in times:
            tot += cur * (t - prev)
            cur += 1.0 / n_tasks
            prev = t
        tot += cur * (tmax - prev)
        auc = tot / tmax

    report_path = Path(args.report)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        vec = report["vector"]
        check("success_rate consistent with trace recompute",
              abs(vec["success_rate"] - sr) < 1e-6)
        check("time_efficiency consistent with trace recompute",
              abs(vec["time_efficiency"] - auc) < 1e-6)
        for key in ("success_rate", "call_efficiency", "time_efficiency",
                    "total_tokens"):
            check(f"vector key '{key}' present", key in vec)
        check("action_counts present at report level",
              "action_counts" in report and report["action_counts"].get("total", 0) > 0)
        pk = report.get("pass_k_curve", [])
        check("pass_k_curve (appendix) 8 values monotone",
              len(pk) == 8 and all(pk[i] <= pk[i + 1] + 1e-9 for i in range(7)))
        check("usage section present", "usage" in report)
        check("integrity section present (trace_coverage)",
              "integrity" in report and "trace_coverage" in report["integrity"])
        check("run_conditions section present (g1/g2)",
              "run_conditions" in report
              and "clock_fidelity" in report["run_conditions"]
              and "rate_limit_events" in report["run_conditions"])
        check("seed_consistency section present",
              "seed_consistency" in report
              and "agreement_rate" in report["seed_consistency"])
    else:
        check("harness report json exists", False)

    print()
    if FAILURES:
        print(f"VERIFY FAIL: {len(FAILURES)} violations")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("VERIFY PASS: harness-mode checks green")


if __name__ == "__main__":
    main()
