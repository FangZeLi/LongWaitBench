#!/usr/bin/env python3
"""Extract the 24 NEW v3 tasks (F1 +9, F2 +15) from vendored tau2 content.

Selection was data-driven (gold-path action counts + per-tool repetition):
  - F1 +9  : retail 41,55 · airline 32,33 · banking 080,048,097 · telecom combo x2
  - F2 +15 : retail 30,32 · airline 44,42,41,4 · banking 081,091,041,082 · telecom combo x5
Every expected value is computed by deterministic golden replay — no LLM.

Outputs:
  - data/new_v3_tasks.json (24 instance dicts, v2-shaped)
  - prints the f1_wait_config.json and f2_mounts.json entries to append.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))

logging.disable(logging.CRITICAL)

from extract_v2 import (  # noqa: E402
    DBS_V2, TAU2_DATA, make_tau2_instance, replay, task_spec_of,
)

DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# selection tables (task id -> slow map)
# ---------------------------------------------------------------------------
F1_NEW = {
    "retail": {
        "55": {"return_delivered_order_items": 5},
        "41": {"modify_pending_order_address": 5},
    },
    "airline": {
        "32": {"update_reservation_flights": 5},
        "33": {"book_reservation": 5},
    },
    "banking": {
        "080": {"file_credit_card_transaction_dispute_4829": 10},
        "048": {"close_credit_card_account_7834": 8},
        "097": {"submit_interest_discrepancy_report_7294": 10},
    },
    "telecom": [
        ("[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]", None),
        ("[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]", None),
    ],
}

# (task id, slow map, mount map) — mount = {tool: cooldown_minutes}
# telecom 不在此列（组合题黄金路径无重复工具，绑定不可靠；telecom F2 仅
# data-refuel run_speed_test CD8 一题，烟雾已验证绑定）。
F2_NEW = {
    "retail": [
        ("30", {"return_delivered_order_items": 5}, {"get_order_details": 2.0}),
        ("32", {"cancel_pending_order": 5}, {"get_order_details": 2.0}),
        ("23", {"exchange_delivered_order_items": 5}, {"get_order_details": 2.0}),
        ("28", {"return_delivered_order_items": 5}, {"get_order_details": 2.0}),
        ("20", {"return_delivered_order_items": 5}, {"get_product_details": 2.0}),
    ],
    "airline": [
        ("44", {"update_reservation_flights": 5}, {"search_direct_flight": 2.0}),
        ("42", {"cancel_reservation": 5}, {"get_reservation_details": 2.0}),
        ("41", {"cancel_reservation": 5}, {"get_reservation_details": 2.0}),
        ("23", {"book_reservation": 5}, {"book_reservation": 5.0}),
        ("12", {"book_reservation": 5}, {"search_direct_flight": 2.0}),
    ],
    "banking": [
        ("081", {"file_credit_card_transaction_dispute_4829": 10},
         {"get_debit_cards_by_account_id_7823": 2.0}),
        ("091", {"file_debit_card_transaction_dispute_6281": 10},
         {"get_bank_account_transactions_9173": 2.0}),
        ("041", {"file_credit_card_transaction_dispute_4829": 10},
         {"file_credit_card_transaction_dispute_4829": 5.0}),
        ("082", {"file_debit_card_transaction_dispute_6281": 10},
         {"file_debit_card_transaction_dispute_6281": 5.0}),
        ("039", {"file_credit_card_transaction_dispute_4829": 10},
         {"file_credit_card_transaction_dispute_4829": 5.0}),
    ],
    "telecom": [],
}

# telecom standard slow durations per provisioning tool
TELECOM_SLOW_STD = {
    "enable_roaming": 5, "toggle_roaming": 2, "refuel_data": 5,
    "reset_apn_settings": 2, "reboot_device": 2,
    "set_network_mode_preference": 3, "send_payment_request": 8,
    "resume_line": 5,
}


def telecom_slow_map_for(task) -> dict:
    """Slow = the provisioning tools the gold path actually uses."""
    used = {a.name for a in task.evaluation_criteria.actions}
    return {t: m for t, m in TELECOM_SLOW_STD.items() if t in used}


def set_budget_from_gold_path(inst: dict, task) -> None:
    """Budget must cover the gold path: slow_calls = max(8, gold_path_slow + 4).

    v2 tasks needed 1-8 slow calls; the new hard banking tasks need up to 16
    (banking-041 files 16 disputes). A hardcoded 8 made those instances
    structurally infeasible (observed: slow-call budget exhausted mid-task).
    """
    slow = {t["name"] for t in inst.get("tools", []) if t.get("slow")}
    n = 0
    for a in task.evaluation_criteria.actions:
        name = a.name
        if name == "call_discoverable_agent_tool":
            name = a.arguments.get("agent_tool_name") or ""
        if name in slow:
            n += 1
    budget = max(8, n + 4)
    inst["wait_config"]["budget"]["slow_calls"] = budget


def extract_retail_airline(domain, ids_slow, out):
    from tau2.domains.retail.environment import get_tasks as retail_tasks
    from tau2.domains.retail.environment import get_environment as retail_env
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.airline.environment import get_tasks as airline_tasks
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.airline.data_model import FlightDB

    if domain == "retail":
        get_tasks, get_env, db_class = retail_tasks, retail_env, RetailDB
    else:
        get_tasks, get_env, db_class = airline_tasks, airline_env, FlightDB
    base_db = TAU2_DATA / "tau2" / "domains" / domain / "db.json"
    tasks = {t.id: t for t in get_tasks()}
    for tid, slow_map in ids_slow:
        task = tasks[tid]
        db = db_class.load(str(base_db))
        env = get_env(db=db)
        if task.initial_state and task.initial_state.initialization_actions:
            env.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=[],
            )
        snap = DBS_V2 / f"{domain}_{tid}_v3_init.json"
        live_db = getattr(env.tools, "db", None) or db
        snap.write_text(json.dumps(live_db.model_dump(mode="json"),
                                   ensure_ascii=False), encoding="utf-8")
        inst = make_tau2_instance(domain, task, env, slow_map,
                                  f"tau2-{domain}-{tid}")
        inst["task_id"] = f"{domain}-{tid}"  # raw task ids collide across domains
        set_budget_from_gold_path(inst, task)
        inst["initial_state"] = {
            "db_snapshot": str(snap),
            "domain_env": f"tau2.domains.{domain}.environment",
            "db_class": f"tau2.domains.{domain}.data_model.{db_class.__name__}",
        }
        out.append(inst)
        print(f"[{domain}] {tid}: acts={len(task.evaluation_criteria.actions)} "
              f"hash={inst['verifier']['checks'][0]['expected_hash'][:16]}")


def extract_banking(ids_slow, out):
    from tau2.domains.banking_knowledge.environment import (
        get_environment, get_tasks, get_db,
    )
    tasks = {t.id: t for t in get_tasks()}
    for tid, slow_map in ids_slow:
        task = tasks[f"task_{tid}"]
        db = get_db()
        env = get_environment(db=db, retrieval_variant="bm25_grep")
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=[],
        )
        snap = DBS_V2 / f"banking_{tid}_v3_init.json"
        live_db = getattr(env.tools, "db", None) or db
        snap.write_text(json.dumps(live_db.model_dump(mode="json"),
                                   ensure_ascii=False), encoding="utf-8")
        inst = make_tau2_instance("banking", task, env, slow_map,
                                  f"tau2-banking-{tid}",
                                  policy_text=env.get_policy())
        inst["task_id"] = f"banking-{tid}"
        set_budget_from_gold_path(inst, task)
        inst["task_file"] = str(TAU2_DATA / "tau2" / "domains"
                                / "banking_knowledge" / "tasks"
                                / f"task_{tid}.json")
        inst["retrieval_variant"] = "bm25_grep"
        inst["scenario_type"] = "external-async"
        inst["initial_state"] = {
            "db_snapshot": str(snap),
            "domain_env": "tau2.domains.banking_knowledge.environment",
            "db_class": "tau2.domains.banking_knowledge.data_model.TransactionalDB",
            "retrieval_variant": "bm25_grep",
        }
        slow_note = "; ".join(
            f"{n} (SLOW, ~{dm} declared minutes)"
            for n, dm in slow_map.items()
        )
        inst["prompt_note"] = (
            "Banking operations are exposed as DISCOVERABLE tools: call "
            "list_discoverable_agent_tools to see them, then "
            "unlock_discoverable_agent_tool and call_discoverable_agent_tool "
            "(its 'arguments' field is a JSON string). The slow submit tools here: "
            f"{slow_note}. They are asynchronous: calling one returns a job_handle "
            "and the record appears in the system (visible through the domain's "
            "status/history tools) only after the job completes — poll the handle "
            "to learn completion."
        )
        out.append(inst)
        print(f"[banking] task_{tid}: acts="
              f"{len(task.evaluation_criteria.actions)} "
              f"hash={inst['verifier']['checks'][0]['expected_hash'][:16]}")


def extract_telecom(full_ids, out, slug_prefix):
    from tau2.domains.telecom.environment import get_environment, get_tasks
    tasks = {t.id: t for t in get_tasks()}
    for i, full_id in enumerate(full_ids):
        task = tasks[full_id]
        env = get_environment(solo_mode=True)
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=[],
        )
        errs = replay(env, task)
        assertions = [
            {"env_type": a.env_type, "func_name": a.func_name,
             "arguments": a.arguments, "assert_value": a.assert_value}
            for a in (task.evaluation_criteria.env_assertions or [])
        ]
        assert_ok = all(
            env.run_env_assertion(a, raise_assertion_error=False)
            for a in (task.evaluation_criteria.env_assertions or [])
        )
        assert assert_ok, f"golden replay failed assertions for {full_id}"
        init_actions = [
            {"env_type": a.env_type, "func_name": a.func_name,
             "arguments": a.arguments}
            for a in (task.initial_state.initialization_actions or [])
        ]
        slow_map = telecom_slow_map_for(task)
        slug = f"{slug_prefix}-{i + 1}"
        inst = {
            "id": f"tau2-telecom-{slug}",
            "source": "tau2-bench",
            "domain": "telecom",
            "license": "MIT",
            "task_id": full_id,
            "task_spec": task_spec_of("telecom", task),
            "prompt_policy": env.get_policy(),
            "scenario_type": "network-async",
            "initial_state": {"initialization_actions": init_actions},
            "tools": [
                {"name": n, "slow": True,
                 "wait": {"type": "policy", "declared_minutes": dm}}
                for n, dm in slow_map.items()
            ] + [{"surface": "full_domain"}],
            "wait_config": {
                "mode": "per_tool",
                "budget": {"slow_calls": 8, "wall_minutes": 60},
            },
            "verifier": {"checks": [{"kind": "env_assertions",
                                     "assertions": assertions}]},
            "difficulty": {"source_label": f"tau2:telecom:{full_id}",
                           "calibrated": False},
            "replay_errors": errs,
        }
        slow_note = "; ".join(
            f"{n} (~{dm} declared minutes)" for n, dm in slow_map.items()
        )
        set_budget_from_gold_path(inst, task)
        inst["prompt_note"] = (
            "This is a tech-support session: you can also act on the customer's "
            "device via the user-side tools (check_status_bar, toggle_roaming, "
            "reboot_device, run_speed_test, ...) — in this environment you perform "
            "those device actions yourself after the customer consents. Network and "
            "device operations marked SLOW are asynchronous: calling one returns a "
            f"job_handle and the device state only reflects the change after the "
            f"job completes (slow tools: {slow_note}). Device status tools always "
            "show the CURRENT state, so re-check status after a slow operation "
            "completes."
        )
        out.append(inst)
        print(f"[telecom] {slug}: acts={len(task.evaluation_criteria.actions)} "
              f"slow={sorted(slow_map)} assertions={len(assertions)}")


def main():
    out = []
    for domain, ids_slow in F1_NEW.items():
        if domain == "telecom":
            extract_telecom([t for t, _ in ids_slow], out, "combo-f1")
        elif domain == "banking":
            extract_banking(list(ids_slow.items()), out)
        else:
            extract_retail_airline(domain, list(ids_slow.items()), out)
    for domain, rows in F2_NEW.items():
        if domain == "telecom":
            extract_telecom([t for t, _ in rows], out, "combo-f2")
        elif domain == "banking":
            extract_banking([(t, s) for t, s, _ in rows], out)
        else:
            extract_retail_airline(domain, [(t, s) for t, s, _ in rows], out)

    assert len({i["id"] for i in out}) == len(out), "duplicate instance ids!"
    (DATA / "new_v3_tasks.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote data/new_v3_tasks.json with {len(out)} instances")

    # print config entries to append
    f1 = {}
    f2 = {}
    for domain, ids_slow in F1_NEW.items():
        if domain == "telecom":
            for i, (_, _) in enumerate(ids_slow):
                f1[f"tau2-telecom-combo-f1-{i + 1}"] = "AUTO"
        elif domain == "banking":
            for tid, sm in ids_slow.items():
                f1[f"tau2-banking-{tid}"] = sm
        else:
            for tid, sm in ids_slow.items():
                f1[f"tau2-{domain}-{tid}"] = sm
    for domain, rows in F2_NEW.items():
        if domain == "telecom":
            for i, (_, _) in enumerate(rows):
                f2[f"tau2-telecom-combo-f2-{i + 1}"] = {"check_status_bar": 2.0}
        else:
            for tid, sm, mount in rows:
                f2[f"tau2-{domain}-{tid}"] = mount
    # fill telecom F1 slow maps from the extracted instances
    for inst in out:
        if inst["id"] in f1 and f1[inst["id"]] == "AUTO":
            f1[inst["id"]] = {
                t["name"]: t["wait"]["declared_minutes"]
                for t in inst["tools"] if t.get("slow")
            }
    print("\nF1_WAIT_ENTRIES = " + json.dumps(f1, ensure_ascii=False, indent=1))
    print("\nF2_MOUNT_ENTRIES = " + json.dumps(f2, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
