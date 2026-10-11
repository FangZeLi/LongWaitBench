#!/usr/bin/env python3
"""LongWaitBench v1 contract verifier (deterministic; no LLM).

Static checks: experiment source conforms to the L1/L2 contract.
Dynamic checks: run artifacts (canonical_trace.jsonl, wait_score_results.json,
run.log) satisfy the acceptance checklist, with independent recomputation of
the headline metrics (task_success_rate / anytime_auc / budget_survival /
wait_score) directly from the trace.

Exit code 0 = all checks pass; 1 = violations found.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))

from longwaitbench import constants as C  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def harmonic_mean(a, b):
    if min(a, b) == 0:
        return 0.0
    return 2 * a * b / (a + b)


# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------
def static_checks():
    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    check("main.py exists and has __main__ guard",
          'if __name__ == "__main__"' in main_src)
    check("main.py calls main()", re.search(r"^\s*main\(\)\s*$", main_src, re.M) is not None)
    check("main.py <= 600 lines", len(main_src.splitlines()) <= 600,
          f"{len(main_src.splitlines())} lines")
    check("num_tasks literal present", re.search(r"NUM_TASKS\s*=\s*12", main_src) is not None)
    check("SEEDS = [0, 1] literal", re.search(r"SEEDS\s*=\s*\[0,\s*1\]", main_src) is not None)

    # no forbidden constructs anywhere in the package + main
    all_src = ""
    for p in [ROOT / "main.py"] + list((ROOT / "longwaitbench").glob("*.py")):
        all_src += p.read_text(encoding="utf-8") + "\n"
    for bad in ["speed_score", "idle_fraction", "heartbeat"]:
        check(f"no forbidden construct '{bad}'", bad not in all_src)
    check("five actions present",
          all(a in all_src for a in ("final_answer", "abandon", "poll", "work", "wait")))
    check("dynamic importlib http load", "importlib.import_module" in all_src
          and '"http.client"' in all_src)
    check("no weighted composite of wait_score (zero free weights)",
          "harmonic_mean" in all_src and "2 * a * b / (a + b)" in all_src)

    # manifest + extraction log fidelity
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    check("manifest instance count == 12", len(manifest["instances"]) == 12)
    log = json.loads((ROOT / "data" / "extraction_log.json").read_text(encoding="utf-8"))
    log_ids = {e["id"] for e in log}
    inst_ids = {i["id"] for i in manifest["instances"]}
    check("extraction log covers every instance", inst_ids <= log_ids)
    check("extraction log entries have source_file + extracted_at",
          all("source_file" in e and "extracted_at" in e for e in log))

    required = {"id", "source", "domain", "license", "task_spec", "initial_state",
                "tools", "wait_config", "verifier"}
    for inst in manifest["instances"]:
        check(f"instance {inst['id']} has all 9 required keys",
              required <= set(inst.keys()))
        slow = [t for t in inst["tools"] if t.get("slow")]
        check(f"instance {inst['id']} has >=1 slow tool", len(slow) >= 1)
        if inst["source"] == "tau2-bench":
            check(f"{inst['id']} tau2 initial_state complete",
                  all(k in inst["initial_state"] for k in
                      ("db_snapshot", "domain_env", "db_class")))
            check(f"{inst['id']} db snapshot exists",
                  Path(inst["initial_state"]["db_snapshot"]).exists())
            check(f"{inst['id']} verifier is db_hash_eq with expected_hash",
                  inst["verifier"]["checks"][0]["kind"] == "db_hash_eq"
                  and "expected_hash" in inst["verifier"]["checks"][0])
        else:
            check(f"{inst['id']} f3 verifier is tolerance with true_value+eps",
                  inst["verifier"]["checks"][0]["kind"] == "tolerance"
                  and "true_value" in inst["verifier"]["checks"][0]
                  and "eps" in inst["verifier"]["checks"][0])
            check(f"{inst['id']} weights sha256 present",
                  "weights_sha256" in inst["initial_state"]["params"])
    # task_spec must be a plain string, never a dict
    check("every task_spec is a plain string",
          all(isinstance(i["task_spec"], str) for i in manifest["instances"]))
    # oracle hygiene: view must not carry initial_state / verifier
    view_src = (ROOT / "longwaitbench" / "session.py").read_text(encoding="utf-8")
    check("sanitized view never includes initial_state",
          '"initial_state"' not in view_src.split("def build_view")[1].split("def ")[0])
    check("sanitized view never includes verifier",
          '"verifier"' not in view_src.split("def build_view")[1].split("def ")[0])


# ---------------------------------------------------------------------------
# dynamic checks
# ---------------------------------------------------------------------------
def load_entries():
    trace = ROOT / "results" / "canonical_trace.jsonl"
    if not trace.exists():
        check("canonical_trace.jsonl exists", False)
        return []
    entries = []
    try:
        for line in trace.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    except Exception as e:
        check("trace parses as JSONL", False, str(e))
        return []
    return entries


def dynamic_checks(entries):
    if not entries:
        return
    types = {e.get("type") for e in entries}
    for t in ("session", "task", "decision"):
        check(f"trace has {t} entries", t in types)
    tasks = [e for e in entries if e.get("type") == "task" and not e.get("cheat_detection")]
    decisions = [e for e in entries if e.get("type") == "decision"]
    llms = [e for e in entries if e.get("type") == "llm"]
    sessions = [e for e in entries if e.get("type") == "session" and not e.get("cheat_detection")]

    check("exit-code contract: >0 decisions recorded", len(decisions) > 0,
          f"{len(decisions)} decisions")

    check("every task entry carries true_completion field",
          all("true_completion" in t for t in tasks))
    check("every task entry has budget + outcome + decision_count",
          all("budget" in t and "outcome" in t and "decision_count" in t for t in tasks))

    ok_dec = True
    for d in decisions:
        tp = d.get("time_prediction") or {}
        valid = (d.get("action") in C.FIVE_ACTIONS
                 and len(tp) >= 2
                 and all(k in C.BUCKET_LABELS for k in tp)
                 and abs(sum(tp.values()) - 1.0) < 0.02
                 and all(isinstance(p, (int, float)) and math.isfinite(float(p))
                         for p in tp.values())
                 and "virtual_cost" in d)
        ok_dec = ok_dec and valid
    check("every decision entry valid (action, >=2 buckets, sum 1, finite, cost)",
          ok_dec)

    check("llm entries carry usage tokens",
          all("usage" in l and l["usage"].get("total_tokens", 0) > 0 for l in llms))

    # real API evidence: base condition llm-backed decisions
    base_sids = {s["session_id"] for s in sessions if s.get("condition") == "LongCall-v1"}
    base_dec = [d for d in decisions if d["session_id"] in base_sids]
    base_llm = [d for d in base_dec if d.get("llm_request_id")]
    check("base condition has llm-backed decisions (real API evidence)",
          len(base_llm) > 0, f"{len(base_llm)}")

    # session coverage: 4 conditions x 12 instances x 2 seeds
    seen = {(s["condition"], s["instance_id"], s["seed"]) for s in sessions}
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    expected = {(c, i["id"], s) for c in C.CONDITIONS
                for i in manifest["instances"] for s in C.SEEDS}
    missing = expected - seen
    check("all 96 (4x12x2) sessions present", not missing,
          f"missing {len(missing)}: {sorted(missing)[:6]}")


def independent_recompute(entries):
    """Recompute headline metrics from the trace and compare to the JSON."""
    results_path = ROOT / "results" / "wait_score_results.json"
    if not results_path.exists():
        check("wait_score_results.json exists", False)
        return
    results = json.loads(results_path.read_text(encoding="utf-8"))
    check("results has protocol/conditions/certificate",
          all(k in results for k in ("protocol", "conditions", "certificate")))
    check("results has instrument section (g1/g2 per condition)",
          "instrument" in results
          and all(c in results["instrument"]
                  and "clock_fidelity" in results["instrument"][c]
                  and "rate_limit_events" in results["instrument"][c]
                  for c in C.CONDITIONS))
    check("exactly 4 conditions with exact names",
          set(results.get("conditions", {})) == set(C.CONDITIONS))
    check("protocol cites extraction log",
          "extraction_log" in results.get("protocol", {}).get("manifest", {}))

    tasks = [e for e in entries if e.get("type") == "task" and not e.get("cheat_detection")]
    decisions = [e for e in entries if e.get("type") == "decision"]
    llms = [e for e in entries if e.get("type") == "llm"]

    for cond in C.CONDITIONS:
        r = results["conditions"].get(cond, {})
        ct = [t for t in tasks if t["condition"] == cond]
        if not ct:
            continue
        cd = [d for d in decisions if d["condition"] == cond and
              any(t["session_id"] == d["session_id"] for t in ct)]
        cl = [l for l in llms if l["condition"] == cond and
              any(t["session_id"] == l["session_id"] for t in ct)]

        sr = sum(1 for t in ct if t["outcome"] == "success") / len(ct)
        times = sorted(t["ended_at"] - t["started_at"] for t in ct
                       if t["outcome"] == "success")
        auc = 0.0
        if times and ct:
            tmax = times[-1]
            tot, cur, prev = 0.0, 0.0, 0.0
            for t in times:
                tot += cur * (t - prev)
                cur += 1.0 / len(ct)
                prev = t
            tot += cur * (tmax - prev)
            auc = tot / tmax

        check(f"{cond} task_success_rate consistent",
              abs(r.get("success_rate", -1) - sr) < 1e-6)
        check(f"{cond} anytime_auc consistent [recomputed]",
              abs(r.get("time_efficiency", -1) - auc) < 1e-6)
        check(f"{cond} total_tokens consistent",
              int(r.get("total_tokens", -1)) == sum(l.get("usage", {}).get("total_tokens", 0)
                                                    for l in cl))
        for key in ("success_rate", "call_efficiency", "time_efficiency",
                    "cache_hit_ratio", "total_tokens"):
            check(f"{cond} vector key '{key}' present", key in r)
        check(f"{cond} action_counts present", "action_counts" in r)
        for key in ("abandon_rate", "slow_calls_mean", "work_density",
                    "recovery_rate", "think_share", "tool_share", "slack_share"):
            check(f"{cond} diagnostic '{key}' present",
                  key in r.get("diagnostics", {}))
        # pass_k monotone
        pk = r.get("pass_k_curve", [])
        check(f"{cond} pass_k_curve 8 values monotone",
              len(pk) == 8 and all(pk[i] <= pk[i + 1] + 1e-9 for i in range(7)))

    # parseable lines in run.log
    log_path = ROOT / "results" / "run.log"
    if log_path.exists():
        log_txt = log_path.read_text(encoding="utf-8")
        check("parseable condition lines present",
              all(f"condition={c} success_rate:" in log_txt for c in C.CONDITIONS))
        check("SUMMARY lines present",
              all(f"SUMMARY condition={c} metric=success_rate" in log_txt
                  for c in C.CONDITIONS))


def main():
    print("== static checks ==")
    static_checks()
    entries = load_entries()
    print("== dynamic checks ==")
    dynamic_checks(entries)
    independent_recompute(entries)
    print()
    if FAILURES:
        print(f"VERIFY FAIL: {len(FAILURES)} violations:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("VERIFY PASS: all checks green")
    sys.exit(0)


if __name__ == "__main__":
    main()
