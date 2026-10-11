#!/usr/bin/env python3
"""LongWaitBench v2 contract verifier.

Phase 1 (manifest integrity, no trace needed):
  --manifest-only
  - 24 instances across 4 domain types (8 retail / 6 airline / 5 banking / 5 telecom)
  - every db_hash_eq expected hash is re-derived by replaying the task's
    golden actions through a fresh environment built from the stored snapshot
  - every telecom env_assertions set passes on the golden replay
  - the 8 carried-over v1 instances' hashes equal the v1 manifest hashes

Phase 2 (trace contracts):
  --harness dsh-headless|opencode --trace results/v2_<h>.jsonl
  - one session+task+llm entry per instance x seed
  - outcomes in {success, failed, abandoned}
  - decisions carry action/params/effect
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))
logging.disable(logging.CRITICAL)

from longwaitbench import constants as C  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def rederive(manifest):
    from tau2.data_model.message import ToolCall
    from tau2.data_model.tasks import EnvAssertion

    bad = []
    for inst in manifest["instances"]:
        dom = inst["domain"]
        checks = inst["verifier"]["checks"]
        if dom == "telecom":
            from tau2.data_model.tasks import EnvFunctionCall
            from tau2.domains.telecom.environment import get_environment
            env = get_environment(solo_mode=True)
            init_calls = [
                EnvFunctionCall(env_type=a["env_type"], func_name=a["func_name"],
                                arguments=a.get("arguments") or {})
                for a in inst["initial_state"]["initialization_actions"]
            ]
            env.set_state(initialization_data=None,
                          initialization_actions=init_calls,
                          message_history=[])
            from tau2.domains.telecom.environment import get_tasks_small
            task = {t.id: t for t in get_tasks_small()}[inst["task_id"]]
            for a in task.evaluation_criteria.actions:
                env.get_response(ToolCall(id="g", name=a.name, arguments=a.arguments,
                                          requestor="assistant"))
            all_ok = True
            for a in checks[0]["assertions"]:
                ea = EnvAssertion(env_type=a.get("env_type", "user"),
                                  func_name=a["func_name"],
                                  arguments=a.get("arguments") or {},
                                  assert_value=a.get("assert_value", True))
                all_ok = env.run_env_assertion(ea, raise_assertion_error=False) and all_ok
            if not all_ok:
                bad.append(inst["id"])
            continue
        # db_hash_eq domains
        if dom == "banking":
            from tau2.domains.banking_knowledge.data_model import TransactionalDB
            from tau2.domains.banking_knowledge.environment import get_environment
            db = TransactionalDB.model_validate(
                json.loads(Path(inst["initial_state"]["db_snapshot"]).read_text()))
            env = get_environment(db=db, retrieval_variant="bm25_grep")
            task_file = Path(inst["task_file"])
        else:
            from tau2.domains.retail.data_model import RetailDB
            from tau2.domains.airline.data_model import FlightDB
            db_class = RetailDB if dom == "retail" else FlightDB
            db = db_class.model_validate(
                json.loads(Path(inst["initial_state"]["db_snapshot"]).read_text()))
            from tau2.domains.retail.environment import get_environment as gre
            from tau2.domains.airline.environment import get_environment as gae
            env = (gre if dom == "retail" else gae)(db=db)
            from tau2.domains.retail.environment import get_tasks as grt
            from tau2.domains.airline.environment import get_tasks as gat
            task_key = inst.get("task_id") or inst["id"].split("-")[-1]
            task = {t.id: t for t in (grt if dom == "retail" else gat)()}[task_key]
            actions = task.evaluation_criteria.actions
            for a in actions:
                env.get_response(ToolCall(id="g", name=a.name, arguments=a.arguments,
                                          requestor="assistant"))
            h = env.get_db_hash()
            expected = [c["expected_hash"] for c in checks if c["kind"] == "db_hash_eq"][0]
            if h != expected:
                bad.append(inst["id"])
            continue
        # banking: replay golden from the task file
        import tau2.domains.banking_knowledge.environment as benv_mod
        tasks = {t.id: t for t in benv_mod.get_tasks()}
        task = tasks[inst["task_id"]]
        for a in task.evaluation_criteria.actions:
            env.get_response(ToolCall(id="g", name=a.name, arguments=a.arguments,
                                      requestor="assistant"))
        h = env.get_db_hash()
        expected = [c["expected_hash"] for c in checks if c["kind"] == "db_hash_eq"][0]
        if h != expected:
            bad.append(inst["id"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(
        (ROOT / "data" / "manifest_v2.json").read_text(encoding="utf-8"))
    counts = {}
    for i in manifest["instances"]:
        counts[i["domain"]] = counts.get(i["domain"], 0) + 1
    check("24 instances total", manifest["num_tasks"] == 24, str(manifest["num_tasks"]))
    check("8 retail / 6 airline / 5 banking / 5 telecom",
          counts == {"retail": 8, "airline": 6, "banking": 5, "telecom": 5},
          str(counts))
    ids = [i["id"] for i in manifest["instances"]]
    check("instance ids unique", len(ids) == len(set(ids)))

    v1 = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    v1_by_id = {i["id"]: i for i in v1["instances"]}
    carried = [i for i in manifest["instances"] if i["id"] in v1_by_id]
    same_hash = all(
        [c["expected_hash"] for c in i["verifier"]["checks"] if c["kind"] == "db_hash_eq"][0]
        == [c["expected_hash"] for c in v1_by_id[i["id"]]["verifier"]["checks"]
            if c["kind"] == "db_hash_eq"][0]
        for i in carried
    )
    check("8 carried-over v1 instances keep v1 expected hashes", same_hash)

    bad = rederive(manifest)
    check("expected hashes / assertions re-derived from source content (golden replay)",
          not bad, f"mismatch: {bad}")

    if args.manifest_only:
        print()
        if FAILURES:
            print(f"VERIFY FAIL: {len(FAILURES)} violations")
            sys.exit(1)
        print("VERIFY PASS: v2 manifest integrity green")
        return

    trace = Path(args.trace)
    entries = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
    sessions = [e for e in entries if e.get("type") == "session"]
    tasks = [e for e in entries if e.get("type") == "task"]
    llms = [e for e in entries if e.get("type") == "llm"]
    decisions = [e for e in entries if e.get("type") == "decision"]
    h = args.harness
    expected = {f"{h}-{i['id']}-seed{s}" for i in manifest["instances"] for s in C.SEEDS}
    seen = {s["session_id"] for s in sessions}
    check(f"{h}: 48 sessions present", expected <= seen,
          f"missing {len(expected - seen)}: {sorted(expected - seen)[:5]}")
    check(f"{h}: task entry per session", {t["session_id"] for t in tasks} >= expected)
    check(f"{h}: task outcomes valid",
          all(t["outcome"] in ("success", "failed", "abandoned") for t in tasks))
    check(f"{h}: usage entry per session", {l["session_id"] for l in llms} >= expected)
    check(f"{h}: decisions well-formed",
          all(d.get("action") in C.ACTIONS and "effect" in d and "params" in d
              for d in decisions))
    ok_success = sum(1 for t in tasks if t["outcome"] == "success")
    check(f"{h}: success_rate", ok_success / len(tasks), f"{ok_success}/{len(tasks)}")

    print()
    if FAILURES:
        print(f"VERIFY FAIL: {len(FAILURES)} violations")
        sys.exit(1)
    print(f"VERIFY PASS: v2 {h} trace contracts green")


if __name__ == "__main__":
    main()
