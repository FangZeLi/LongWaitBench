#!/usr/bin/env python3
"""LongWaitBench v1 — experiment entry point.

Contract anchors:
- task content loads ONLY from the vendored manifest (env LC1_MANIFEST or
  the default path); no hardcoded task content, no runtime generation;
- num_tasks literal (12) must match the manifest instance count;
- SEEDS = [0, 1]; exactly four conditions;
- writes ONE canonical_trace.jsonl and ONE wait_score_results.json;
- prints parseable condition lines; exit code 1 if zero decisions recorded.

Usage:  python main.py [--workers N]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data" / "pylibs"))
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT / "data" / "tau2bench" / "data"))
os.environ.setdefault("LC1_MANIFEST", str(ROOT / "data" / "manifest.json"))

from longwaitbench import constants as C  # noqa: E402
from longwaitbench.manifest_loader import load_manifest  # noqa: E402
from longwaitbench.metrics import (  # noqa: E402
    ConditionMetrics,
    certificate,
    load_entries,
    protocol_section,
)
from longwaitbench.policies import CHEAT_POLICIES  # noqa: E402
from longwaitbench.recording_client import RecordingClient  # noqa: E402
from longwaitbench.session import run_session  # noqa: E402
from longwaitbench.trace import TraceWriter  # noqa: E402

NUM_TASKS = 12
SEEDS = [0, 1]
CONDITIONS = ["LongCall-v1", "ExponentialBackoff", "AggressivePoll", "WaitAndAnswer"]
CHEAT_INSTANCE = "f3-control-0"
DEFAULT_WORKERS = 4


def _parse_args(argv):
    workers = DEFAULT_WORKERS
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--workers" and args:
            workers = int(args.pop(0))
        elif a == "--workers=1":
            workers = 1
    return workers


def _run_one(condition, inst, seed, trace, client):
    session_id = f"{condition}-{inst['id']}-seed{seed}"
    print(f"[run] {session_id} start", flush=True)
    t0 = time.time()
    try:
        run_session(condition, inst, seed, trace, client=client)
    except Exception as e:
        import traceback

        trace.write(
            {
                "type": "task",
                "task_id": inst["id"],
                "session_id": session_id,
                "condition": condition,
                "seed": seed,
                "started_at": t0,
                "ended_at": time.time(),
                "true_completion": None,
                "budget": {
                    "time_physical_seconds": round(
                        inst["wait_config"]["budget"]["wall_minutes"]
                        * 60.0
                        * C.COMPRESSION_SCALE,
                        4,
                    ),
                    "max_calls": inst["wait_config"]["budget"]["slow_calls"],
                    "decision_budget": C.MAX_DECISIONS_PER_TASK,
                },
                "outcome": "failed",
                "decision_count": 0,
                "error": f"session exception: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:],
            }
        )
        print(f"[run] {session_id} EXCEPTION {type(e).__name__}: {e}", flush=True)
    print(f"[run] {session_id} done {time.time() - t0:.1f}s", flush=True)


def _run_cheat(cheat_cls, inst, seed, trace):
    cond = "certificate"
    session_id = f"{cond}-{inst['id']}-seed{seed}-{cheat_cls.__name__}"
    print(f"[cheat] {session_id} start", flush=True)
    try:
        run_session(cond, inst, seed, trace, cheat_policy=cheat_cls,
                    session_id=session_id)
    except Exception as e:
        print(f"[cheat] {session_id} EXCEPTION {type(e).__name__}: {e}", flush=True)
    print(f"[cheat] {session_id} done", flush=True)


def _print_parseable_lines(cond_results, entries):
    for name, r in cond_results.items():
        vec = r["vector"]
        print(f"condition={name} success_rate: {vec['success_rate']}")
        print(f"condition={name} call_efficiency: {vec['call_efficiency']}")
        print(
            f"condition={name} pass_k_curve: "
            + ",".join(f"{x:.6f}" for x in r["pass_k_curve"])
        )
        print(f"condition={name} time_efficiency: {vec['time_efficiency']}")
        print(
            f"condition={name} cache_hit_ratio: "
            + (f"{vec['cache_hit_ratio']}" if vec["cache_hit_ratio"] is not None else "None")
        )
        print(f"condition={name} total_tokens: {vec['total_tokens']}")
        print(f"condition={name} abandon_rate: {r['diagnostics'].get('abandon_rate')}")
        print(
            f"SUMMARY condition={name} metric=success_rate "
            f"mean={vec['success_rate']} std=0.0"
        )


def _existing_keys(trace):
    """Resume keys: session_id for regular sessions; (policy, instance, seed)
    for cheat-detection sessions (their recorded ids predate the policy suffix)."""
    keys = set()
    if not C.CANONICAL_TRACE_PATH.exists():
        return keys
    with open(C.CANONICAL_TRACE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "session":
                continue
            policy = e.get("policy") or ""
            if policy.startswith("cheat"):
                keys.add(("cheat", policy, e.get("instance_id"), e.get("seed")))
            else:
                keys.add(e.get("session_id"))
    return keys


def main() -> None:
    workers = _parse_args(sys.argv[1:])
    t_start = time.time()
    manifest = load_manifest()
    assert len(manifest["instances"]) == NUM_TASKS, (
        f"manifest instance count {len(manifest['instances'])} != NUM_TASKS {NUM_TASKS}"
    )
    assert C.NUM_TASKS == NUM_TASKS and C.SEEDS == SEEDS and C.CONDITIONS == CONDITIONS

    trace = TraceWriter(C.CANONICAL_TRACE_PATH)
    existing = _existing_keys(trace)
    print(f"[run] {len(existing)} sessions already recorded (resume-safe)")

    jobs = []
    clients = {cond: RecordingClient() for cond in CONDITIONS}
    for condition in CONDITIONS:
        client = clients[condition]
        for seed in SEEDS:
            for inst in manifest["instances"]:
                sid = f"{condition}-{inst['id']}-seed{seed}"
                if sid in existing:
                    continue
                jobs.append(("session", condition, inst, seed, client))
    cheat_inst = next(
        i for i in manifest["instances"] if i["id"] == CHEAT_INSTANCE
    )
    for cheat_cls in CHEAT_POLICIES:
        for seed in SEEDS:
            key = ("cheat", cheat_cls.name, cheat_inst["id"], seed)
            if key in existing:
                continue
            jobs.append(("cheat", cheat_cls, cheat_inst, seed, None))

    print(f"[run] {len(jobs)} jobs to execute with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for job in jobs:
            if job[0] == "session":
                _, condition, inst, seed, client = job
                futures.append(pool.submit(_run_one, condition, inst, seed, trace, client))
            else:
                _, cheat_cls, inst, seed, _ = job
                futures.append(pool.submit(_run_cheat, cheat_cls, inst, seed, trace))
        for fut in futures:
            fut.result()

    entries = load_entries(C.CANONICAL_TRACE_PATH)
    cond_results = {}
    cm_objs = {}
    for condition in CONDITIONS:
        cm = ConditionMetrics(entries, condition, NUM_TASKS)
        # legacy v1 stage uses the decision-count grid 1..8
        cond_results[condition] = cm.compute(pass_k_ks=(1, 2, 3, 4, 5, 6, 7, 8))
        cm_objs[condition] = cm

    results = {
        "benchmark": "LongWaitBench v1",
        "seeds": SEEDS,
        "num_tasks": NUM_TASKS,
        "sessions_per_condition": NUM_TASKS * len(SEEDS),
        "protocol": protocol_section(manifest),
        "conditions": {
            name: {
                **r["vector"],
                "diagnostics": r["diagnostics"],
                "action_counts": r["action_counts"],
                "pass_k_curve": r["pass_k_curve"],
            }
            for name, r in cond_results.items()
        },
        "instrument": {
            name: {
                "clock_fidelity": cm_objs[name].clock_fidelity(),
                "rate_limit_events": cm_objs[name].rate_limit_events(),
            }
            for name in CONDITIONS
        },
        "certificate": certificate(entries, NUM_TASKS),
    }
    with open(C.RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    _print_parseable_lines(cond_results, entries)

    n_decisions = sum(1 for e in entries if e.get("type") == "decision")
    elapsed = time.time() - t_start
    print(f"[run] DONE decisions={n_decisions} elapsed={elapsed:.1f}s")
    if n_decisions == 0:
        print("FATAL: zero decisions recorded — experiment did not run")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
