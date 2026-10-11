#!/usr/bin/env python3
"""v4 manifest mechanism test (no LLM): every runnable instance constructs,
arm invariants hold, and the baseline/F2 functional paths behave.

Checks:
  A. all 96 runnable instances build a UnifiedEnv without error;
  B. baseline arm: no slow waits, no rate-limit text, previously-slow tools
     resolve to the synchronous path;
  C. F2 arm: rate-limit text present; a new banking CD5 mount is busy-rejected
     on the second submission; a new banking CD2 mount is rejected on the
     immediate second call;
  D. F1 arm: policy text has no rate-limit section.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data", "pylibs"))
os.environ.setdefault("TAU2_DATA_DIR",
                      os.path.join(ROOT, "data", "tau2bench", "data"))
os.environ["LC1_MANIFEST"] = os.path.join(ROOT, "data", "manifest_v4.json")

from longwaitbench.manifest_loader import load_manifest  # noqa: E402
from longwaitbench.unified_env import UnifiedEnv  # noqa: E402

manifest = load_manifest()
insts = {i["id"]: i for i in manifest["instances"]}
runnable = [i for i in manifest["instances"]
            if i.get("benchmark", True) and not i.get("f2_base")
            and i.get("arm") in ("f1", "f2", "baseline")]
assert len(runnable) == 96, f"expected 96 runnable, got {len(runnable)}"

failures = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        failures.append(name)


# --- A/B/D: construct all, verify arm invariants ---------------------------
baseline_n = f2_n = f1_n = 0
for inst in runnable:
    try:
        env = UnifiedEnv(inst, 0)
    except Exception as e:  # noqa: BLE001
        check(f"construct:{inst['id']}", False, repr(e)[:200])
        continue
    arm = inst.get("arm")
    if arm == "baseline":
        baseline_n += 1
        check(f"baseline-no-slow:{inst['id']}", env.slow_waits == {},
              str(env.slow_waits)[:80])
        check(f"baseline-no-rl:{inst['id']}",
              "Rate limits" not in (env.policy_text or ""))
    elif arm == "f2":
        f2_n += 1
        check(f"f2-has-rl-text:{inst['id']}",
              "Rate limits" in (env.policy_text or ""))
    else:
        f1_n += 1
        check(f"f1-no-rl-text:{inst['id']}",
              "Rate limits" not in (env.policy_text or ""))
    env.shutdown()
check("arm-counts", baseline_n == 48 and f2_n == 24 and f1_n == 24,
      f"baseline={baseline_n} f2={f2_n} f1={f1_n}")

# --- C: functional behavior on the new banking mounts ----------------------
# arguments taken from the task's own gold path
TASKS = os.path.join(ROOT, "data", "tau2bench", "data", "tau2", "domains",
                     "banking_knowledge", "tasks")


def first_call_args(task_id: str, tool: str) -> dict:
    task = json.load(open(os.path.join(TASKS, f"task_{task_id}.json"),
                          encoding="utf-8"))
    for a in task["evaluation_criteria"]["actions"]:
        name = a.get("name")
        if name == "call_discoverable_agent_tool":
            name = (a.get("arguments") or {}).get("agent_tool_name")
        if name == tool:
            return (a.get("arguments") or {}).get("arguments") or {}
    return {}


# C1: v3-f2-banking-037 (dispute CD5 single-slot): 2nd submit busy-rejected
env = UnifiedEnv(insts["v3-f2-banking-037"], 0)
args = first_call_args("037", "file_credit_card_transaction_dispute_4829")
s1 = env.submit_slow("file_credit_card_transaction_dispute_4829", args)
check("banking-037-first-submit-ok", s1.get("ok") is True, str(s1)[:160])
s2 = env.submit_slow("file_credit_card_transaction_dispute_4829", args)
check("banking-037-second-busy-rejected",
      s2.get("rejected") is True and s2.get("busy") is True, str(s2)[:160])
env.shutdown()

# C2: v3-f2-banking-073 (query CD2): immediate 2nd call rejected
env = UnifiedEnv(insts["v3-f2-banking-073"], 0)
qargs = first_call_args("073", "get_bank_account_transactions_9173")
r1 = env.call_fast("call_discoverable_agent_tool",
                   {"agent_tool_name": "get_bank_account_transactions_9173",
                    "arguments": qargs})
check("banking-073-first-query-ok", r1.get("ok") is True, str(r1)[:160])
r2 = env.call_fast("call_discoverable_agent_tool",
                   {"agent_tool_name": "get_bank_account_transactions_9173",
                    "arguments": qargs})
check("banking-073-second-query-rejected",
      r2.get("rejected") is True and r2.get("retry_after", 0) > 0, str(r2)[:160])
env.shutdown()

# C3: v4-base-banking-037: the dispute tool is now synchronous
env = UnifiedEnv(insts["v4-base-banking-037"], 0)
res = env.resolve_call("call_discoverable_agent_tool",
                       {"agent_tool_name": "file_credit_card_transaction_dispute_4829"})
check("baseline-037-resolves-fast", res.get("kind") == "fast", str(res)[:120])
r3 = env.call_fast("call_discoverable_agent_tool",
                   {"agent_tool_name": "file_credit_card_transaction_dispute_4829",
                    "arguments": args})
check("baseline-037-call-ok", r3.get("ok") is True, str(r3)[:200])
env.shutdown()

# C4: v4-base-retail-0: previously-slow exchange tool resolves fast
env = UnifiedEnv(insts["v4-base-retail-0"], 0)
res = env.resolve_call("exchange_delivered_order_items", {})
check("baseline-retail-0-resolves-fast", res.get("kind") == "fast", str(res)[:120])
env.shutdown()

# C5: v4-base-telecom-data-refuel: refuel_data resolves fast
env = UnifiedEnv(insts["v4-base-telecom-data-refuel"], 0)
res = env.resolve_call("refuel_data", {})
check("baseline-telecom-resolves-fast", res.get("kind") == "fast", str(res)[:120])
env.shutdown()

if failures:
    print(f"\n{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("\nall v4 manifest mechanism checks passed")
