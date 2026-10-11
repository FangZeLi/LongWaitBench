#!/usr/bin/env python3
"""Extract the 7 replacement banking F2 tasks for v4 (banking F2 换题).

Selection = v4 task spec (analysis/v4_task_spec.md), data-driven:
  - short gold path (content difficulty must not drown the wait effect);
  - binding guaranteed on the SUCCESS path: modify/submit >=2 (CD5 single-slot)
    or query >=3 (CD2).
  modify CD5: task_037 (dispute x2, 8 acts) / task_038 (dispute x3, 9 acts)
              / task_083 (dispute x4, 10 acts) / task_085 (dispute x3, 13 acts)
  query  CD2: task_073 (transactions x3, 11 acts) / task_089 (debit cards x3,
              13 acts) / task_074 (transactions x4, 13 acts)
Slow map follows the v3 banking convention: external-async dispute submits are
the slow tools (10 declared minutes); queries stay instant.

Outputs data/new_banking_v4.json + prints the f1_wait / f2_mount entries.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))

logging.disable(logging.CRITICAL)

from extract_v3 import extract_banking  # noqa: E402

DATA = ROOT / "data"

# task_id -> slow map (v3 convention: dispute submits are the slow tools)
BANKING_V4 = {
    "037": {"file_credit_card_transaction_dispute_4829": 10.0},
    "038": {"file_credit_card_transaction_dispute_4829": 10.0},
    "083": {"file_debit_card_transaction_dispute_6281": 10.0},
    "085": {"file_debit_card_transaction_dispute_6281": 10.0},
    "073": {},
    "089": {},
    "074": {},
}

DISCOVERABLE_NOTE = (
    "Banking operations are exposed as DISCOVERABLE tools: call "
    "list_discoverable_agent_tools to see them, then "
    "unlock_discoverable_agent_tool and call_discoverable_agent_tool "
    "(its 'arguments' field is a JSON string)."
)


def fix_prompt_notes(out: list) -> None:
    """Tasks with no slow tools must not carry the async job_handle note."""
    for inst in out:
        slow = [t for t in inst.get("tools", []) if t.get("slow")]
        if not slow:
            inst["prompt_note"] = DISCOVERABLE_NOTE


def main():
    out = []
    extract_banking(list(BANKING_V4.items()), out)
    fix_prompt_notes(out)
    (DATA / "new_banking_v4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote data/new_banking_v4.json with {len(out)} instances")
    for inst in out:
        slow = [t["name"] for t in inst.get("tools", []) if t.get("slow")]
        print(f"  {inst['id']}: tools={len(inst.get('tools', []))} slow={slow}")
    f1 = {f"tau2-banking-{t}": sm for t, sm in BANKING_V4.items() if sm}
    f2 = {
        "tau2-banking-037": {"file_credit_card_transaction_dispute_4829": 5.0},
        "tau2-banking-038": {"file_credit_card_transaction_dispute_4829": 5.0},
        "tau2-banking-083": {"file_debit_card_transaction_dispute_6281": 5.0},
        "tau2-banking-085": {"file_debit_card_transaction_dispute_6281": 5.0},
        "tau2-banking-073": {"get_bank_account_transactions_9173": 2.0},
        "tau2-banking-089": {"get_debit_cards_by_account_id_7823": 2.0},
        "tau2-banking-074": {"get_bank_account_transactions_9173": 2.0},
    }
    print("\nF1_WAIT_ENTRIES = " + json.dumps(f1, ensure_ascii=False, indent=1))
    print("\nF2_MOUNT_ENTRIES = " + json.dumps(f2, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
