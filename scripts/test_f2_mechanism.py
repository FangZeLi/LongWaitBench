#!/usr/bin/env python3
"""Unit test for the F2 rate-limit (cooldown) mechanism.

Covers:
  1. fast tool: first call ok; immediate second call rejected with retry_after.
  2. cooldown expiry: after the window, calls succeed again.
  3. slow tool: submission accepted; after job completion a follow-up
     submission is rejected (cooldown from run END).
  4. save_state/restore preserves cooldown state and rejection counter.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data", "pylibs"))
os.environ.setdefault("TAU2_DATA_DIR", os.path.join(ROOT, "data", "tau2bench", "data"))
os.environ["LC1_MANIFEST"] = os.path.join(ROOT, "data", "manifest_v3_draft.json")

from longwaitbench.manifest_loader import load_manifest  # noqa: E402
from longwaitbench.unified_env import UnifiedEnv  # noqa: E402

manifest = load_manifest()
inst = next(i for i in manifest["instances"] if i["id"] == "f2-mech-retail")

# a REAL product id from the retail DB (the mechanism is under test, not the
# domain content: any existing product works)
from tau2.domains.retail.data_model import RetailDB  # noqa: E402
db = RetailDB.load(os.path.join(ROOT, "data", "tau2bench", "data",
                                "tau2", "domains", "retail", "db.json"))
_products = getattr(db, "products", {})
if isinstance(_products, dict):
    VALID_PID = next(iter(_products))
else:
    VALID_PID = getattr(_products[0], "id", str(_products[0]))
print("using product_id:", VALID_PID)

failures = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        failures.append(name)


env = UnifiedEnv(inst, 0)
SCALE = 0.15

# policy text carries the rate-limit rules
check("policy-text-has-rules", "Rate limits" in env.policy_text)

# 1. fast tool
r1 = env.call_fast("get_product_details", {"product_id": VALID_PID})
check("fast-first-ok", r1.get("ok") is True, str(r1)[:120])
r2 = env.call_fast("get_product_details", {"product_id": VALID_PID})
check("fast-second-rejected", r2.get("rejected") is True, str(r2)[:160])
check("fast-retry_after>0", r2.get("retry_after", 0) > 0, str(r2)[:160])
check("rejection-count-1", env.rejection_count == 1, f"count={env.rejection_count}")

# 2. cooldown expiry (0.2 declared min = 1.8s)
time.sleep(0.2 * 60 * SCALE + 0.5)
r3 = env.call_fast("get_product_details", {"product_id": VALID_PID})
check("fast-after-expiry-ok", r3.get("ok") is True, str(r3)[:120])
check("rejection-count-still-1", env.rejection_count == 1)

# 3. slow tool with cooldown from run END
s1 = env.submit_slow("return_delivered_order_items",
                     {"order_id": "0", "item_ids": []})
check("slow-submit-ok", s1.get("ok") is True, str(s1)[:160])
# 3b. single-slot: a second submission while the first is RUNNING is busy-rejected
sb = env.submit_slow("return_delivered_order_items",
                     {"order_id": "0", "item_ids": []})
check("slow-busy-rejected", sb.get("rejected") is True and sb.get("busy") is True,
      str(sb)[:160])
check("busy-retry_after>0", sb.get("retry_after", 0) > 0, str(sb)[:160])
res = env.wait_for(s1["job_handle"], timeout=30)
check("slow-job-settled", res.get("status") in ("completed", "failed"),
      str(res)[:160])
s2 = env.submit_slow("return_delivered_order_items",
                     {"order_id": "0", "item_ids": []})
check("slow-during-cd-rejected", s2.get("rejected") is True, str(s2)[:160])
check("rejection-count-3", env.rejection_count == 3,
      f"count={env.rejection_count} (1 fast + 1 busy + 1 cd)")

# 4. save/restore preserves cooldown + counters
state = env.save_state()
env2 = UnifiedEnv.restore(inst, 0, state)
s3 = env2.submit_slow("return_delivered_order_items",
                      {"order_id": "0", "item_ids": []})
check("restore-during-cd-rejected", s3.get("rejected") is True, str(s3)[:160])
check("restore-rejection-count", env2.rejection_count == 4,
      f"count={env2.rejection_count} (3 preserved + 1 new rejection)")
f2 = env2.call_fast("get_product_details", {"product_id": "1547678989"})
check("restore-fast-cd-state", f2.get("rejected") is True
      or f2.get("ok") is True, str(f2)[:160])

# 5. expiry after restore
time.sleep(0.2 * 60 * SCALE + 0.5)
s4 = env2.submit_slow("return_delivered_order_items",
                      {"order_id": "0", "item_ids": []})
check("restore-after-expiry-ok", s4.get("ok") is True, str(s4)[:160])

env.shutdown()
env2.shutdown()

if failures:
    print(f"\n{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("\nall F2 mechanism checks passed")
