#!/usr/bin/env python3
"""Extract LongWaitBench v2 manifest: 24 instances across 4 tau2 domain types.

- retail: keep the 5 validated v1 instances verbatim + extract 3 new (ids 2,3,4)
- airline: keep the 3 validated v1 instances verbatim + extract 3 new (ids 8,22,39)
- banking (NEW TYPE: external-async): 5 tasks from banking_knowledge
  (submit -> external team -> status poll), bm25_grep retrieval variant
- telecom (NEW TYPE: network/user-side async): 5 tasks from telecom tasks_small,
  solo mode (agent performs user-side device actions), env_assertions verdict

Every expected value is computed by deterministically replaying the task's
reference actions through the vendored tau2 environment — no LLM involved.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))

logging.disable(logging.CRITICAL)

from tau2.data_model.message import ToolCall  # noqa: E402

DATA = ROOT / "data"
TAU2_DATA = DATA / "tau2bench" / "data"
DBS_V2 = DATA / "dbs_v2"
DBS_V2.mkdir(parents=True, exist_ok=True)

SCALE = 0.15  # must match longwaitbench.constants.COMPRESSION_SCALE


def load_v1_instances():
    m = json.load(open(DATA / "manifest.json", encoding="utf-8"))
    keep = [i for i in m["instances"] if i["id"].startswith(("tau2-retail", "tau2-airline"))]
    return keep


def replay(env, task, requestor="assistant"):
    errs = []
    for a in task.evaluation_criteria.actions:
        r = env.get_response(
            ToolCall(id=str(a.action_id), name=a.name,
                     arguments=a.arguments, requestor=requestor)
        )
        if r.error:
            errs.append(a.name)
    return errs


def make_tau2_instance(domain, task, env, slow_tools, instance_id, policy_text=None):
    """Shared shape for retail/airline/banking db-hash instances."""
    from tau2.data_model.tasks import Task
    spec = task_spec_of(domain, task)
    errs = replay(env, task)
    expected = env.get_db_hash()
    inst = {
        "id": instance_id,
        "source": "tau2-bench",
        "domain": domain,
        "license": "MIT",
        "task_id": task.id,
        "task_spec": spec,
        "tools": [
            {"name": n, "slow": True,
             "wait": {"type": "policy", "declared_minutes": dm}}
            for n, dm in slow_tools.items()
        ] + [{"surface": "full_domain"}],
        "wait_config": {
            "mode": "per_tool",
            "budget": {"slow_calls": 8, "wall_minutes": 60},
        },
        "verifier": {
            "checks": [{"kind": "db_hash_eq", "expected_hash": expected}]
        },
        "difficulty": {"source_label": f"tau2:{domain}:{task.id}",
                       "calibrated": False},
        "replay_errors": errs,
    }
    if policy_text:
        inst["prompt_policy"] = policy_text
    return inst


def task_spec_of(domain, task):
    us = task.user_scenario
    instr = getattr(us, "instructions", None) or str(us)
    if not isinstance(instr, str):
        # telecom packs structured user instructions (StructuredUserInstructions)
        parts = []
        for key in ("reason_for_call", "known_info", "unknown_info",
                    "task_instructions"):
            val = getattr(instr, key, None)
            if val:
                parts.append(f"{key}: {val}")
        instr = "\n".join(parts) or str(instr)
    ticket = getattr(task, "ticket", None)
    if domain == "telecom" and ticket:
        return f"{ticket}\n\n{instr}"
    return instr


# ---------------------------------------------------------------------------
# retail / airline expansion (db snapshot + golden-replay hash)
# ---------------------------------------------------------------------------
def extract_retail_airline():
    from tau2.domains.retail.environment import get_tasks as retail_tasks
    from tau2.domains.retail.environment import get_environment as retail_env
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.airline.environment import get_tasks as airline_tasks
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.airline.data_model import FlightDB

    out = []
    for domain, ids, slow_map in [
        ("retail", ["2", "3", "4"],
         {"2": {"return_delivered_order_items": 5},
          "3": {"modify_pending_order_items": 5},
          "4": {"modify_pending_order_items": 5}}),
        ("airline", ["8", "22", "39"],
         {"8": {"book_reservation": 5},
          "22": {"update_reservation_flights": 5},
          "39": {"cancel_reservation": 5}}),
    ]:
        get_tasks = retail_tasks if domain == "retail" else airline_tasks
        get_env = retail_env if domain == "retail" else airline_env
        db_class = RetailDB if domain == "retail" else FlightDB
        base_db = TAU2_DATA / "tau2" / "domains" / domain / "db.json"
        tasks = {t.id: t for t in get_tasks()}
        for tid in ids:
            task = tasks[tid]
            db = db_class.load(str(base_db))
            env = get_env(db=db)
            if task.initial_state and task.initial_state.initialization_actions:
                env.set_state(
                    initialization_data=task.initial_state.initialization_data,
                    initialization_actions=task.initial_state.initialization_actions,
                    message_history=[],
                )
            snap = DBS_V2 / f"{domain}_{tid}_init.json"
            live_db = getattr(env.tools, "db", None) or db
            snap.write_text(json.dumps(live_db.model_dump(mode="json"), ensure_ascii=False),
                            encoding="utf-8")
            inst = make_tau2_instance(domain, task, env, slow_map[tid],
                                      f"tau2-{domain}-{tid}")
            inst["initial_state"] = {
                "db_snapshot": str(snap),
                "domain_env": f"tau2.domains.{domain}.environment",
                "db_class": f"tau2.domains.{domain}.data_model.{db_class.__name__}",
            }
            out.append(inst)
            print(f"[{domain}] {tid}: actions={len(task.evaluation_criteria.actions)} "
                  f"hash={inst['verifier']['checks'][0]['expected_hash'][:16]}")
    return out


# ---------------------------------------------------------------------------
# banking_knowledge (external-async type)
# ---------------------------------------------------------------------------
BANKING_TASKS = {
    "050": {"slow": {"submit_credit_limit_increase_request_7392": 8},
            "gating": {
                "approve_credit_limit_increase_5847": {
                    "submit_tool": "submit_credit_limit_increase_request_7392",
                    "resource_key": "credit_card_account_id",
                },
                "deny_credit_limit_increase_5848": {
                    "submit_tool": "submit_credit_limit_increase_request_7392",
                    "resource_key": "credit_card_account_id",
                },
            }},
    "052": {"slow": {"submit_credit_limit_increase_request_7392": 8},
            "gating": {
                "approve_credit_limit_increase_5847": {
                    "submit_tool": "submit_credit_limit_increase_request_7392",
                    "resource_key": "credit_card_account_id",
                },
                "deny_credit_limit_increase_5848": {
                    "submit_tool": "submit_credit_limit_increase_request_7392",
                    "resource_key": "credit_card_account_id",
                },
            }},
    "084": {"slow": {"file_debit_card_transaction_dispute_6281": 10}},
    "086": {"slow": {"file_debit_card_transaction_dispute_6281": 10,
                     "order_debit_card_5739": 15}},
    "096": {"slow": {"submit_interest_discrepancy_report_7294": 10}},
}


def extract_banking():
    from tau2.domains.banking_knowledge.environment import (
        get_environment, get_tasks, get_db,
    )
    tasks = {t.id: t for t in get_tasks()}
    out = []
    for tid, cfg in BANKING_TASKS.items():
        task = tasks[f"task_{tid}"]
        db = get_db()
        env = get_environment(db=db, retrieval_variant="bm25_grep")
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=[],
        )
        snap = DBS_V2 / f"banking_{tid}_init.json"
        # set_state may rebuild the toolkit db instance (update_db creates a new
        # instance), so dump the environment's live db, not the pre-init handle
        live_db = getattr(env.tools, "db", None) or db
        snap.write_text(json.dumps(live_db.model_dump(mode="json"), ensure_ascii=False),
                        encoding="utf-8")
        slow_tools = {n: dm for n, dm in cfg["slow"].items()}
        inst = make_tau2_instance("banking", task, env, slow_tools,
                                  f"tau2-banking-{tid}",
                                  policy_text=env.get_policy())
        inst["task_file"] = str(TAU2_DATA / "tau2" / "domains" / "banking_knowledge"
                                / "tasks" / f"task_{tid}.json")
        inst["retrieval_variant"] = "bm25_grep"
        inst["scenario_type"] = "external-async"
        inst["initial_state"] = {
            "db_snapshot": str(snap),
            "domain_env": "tau2.domains.banking_knowledge.environment",
            "db_class": "tau2.domains.banking_knowledge.data_model.TransactionalDB",
            "retrieval_variant": "bm25_grep",
        }
        if cfg.get("gating"):
            inst["gating"] = cfg["gating"]
        slow_note = "; ".join(
            f"{n} (SLOW, ~{dm} declared minutes)" for n, dm in slow_tools.items()
        )
        gated_note = ""
        if cfg.get("gating"):
            gated = ", ".join(cfg["gating"].keys())
            gated_note = (f" The tools {gated} can only take effect after the "
                          "corresponding slow submit job for the same account has "
                          "completed.")
        inst["prompt_note"] = (
            "Banking operations are exposed as DISCOVERABLE tools: call "
            "list_discoverable_agent_tools to see them, then "
            "unlock_discoverable_agent_tool and call_discoverable_agent_tool "
            "(its 'arguments' field is a JSON string). The slow submit tools here: "
            f"{slow_note}. They are asynchronous: calling one returns a job_handle "
            "and the record appears in the system (visible through the domain's "
            "status/history tools) only after the job completes — poll the handle "
            "to learn completion." + gated_note
        )
        out.append(inst)
        print(f"[banking] task_{tid}: actions="
              f"{len(task.evaluation_criteria.actions)} hash="
              f"{inst['verifier']['checks'][0]['expected_hash'][:16]}")
    return out


# ---------------------------------------------------------------------------
# telecom (network/user-side async type)
# ---------------------------------------------------------------------------
TELECOM_TASKS = {
    "roaming-abroad-hard": (
        "[mobile_data_issue]user_abroad_roaming_disabled_off[PERSONA:Hard]",
        {"enable_roaming": 5, "toggle_roaming": 2},
    ),
    "data-refuel": (
        "[mobile_data_issue]data_usage_exceeded[PERSONA:Easy]",
        {"refuel_data": 5},
    ),
    "network-preference": (
        "[mobile_data_issue]bad_network_preference[PERSONA:Hard]",
        {"set_network_mode_preference": 3},
    ),
    "overdue-payment": (
        "[service_issue]overdue_bill_suspension[PERSONA:Easy]",
        {"send_payment_request": 8, "resume_line": 5, "reboot_device": 2},
    ),
    "apn-reset": (
        "[service_issue]break_apn_settings[PERSONA:None]",
        {"reset_apn_settings": 2, "reboot_device": 2},
    ),
}


def extract_telecom():
    from tau2.domains.telecom.environment import get_environment, get_tasks_small
    tasks = {t.id: t for t in get_tasks_small()}
    out = []
    for slug, (full_id, slow_map) in TELECOM_TASKS.items():
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
        print(f"[telecom] {slug}: actions={len(task.evaluation_criteria.actions)} "
              f"assertions={len(assertions)}")
    return out


def main():
    v1_keep = load_v1_instances()
    print(f"keeping {len(v1_keep)} validated v1 instances verbatim")
    new_ra = extract_retail_airline()
    banking = extract_banking()
    telecom = extract_telecom()
    instances = v1_keep + new_ra + banking + telecom
    manifest = {
        "manifest_version": "2",
        "num_tasks": len(instances),
        "instances": instances,
    }
    out_path = DATA / "manifest_v2.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    log = {
        "created_from": "scripts/extract_v2.py",
        "method": ("deterministic golden-replay through vendored tau2 "
                   "environments; no LLM"),
        "retrieval_variant": "bm25_grep",
        "compression_scale": SCALE,
        "domains": {
            "retail": len([i for i in instances if i["domain"] == "retail"]),
            "airline": len([i for i in instances if i["domain"] == "airline"]),
            "banking": len([i for i in instances if i["domain"] == "banking"]),
            "telecom": len([i for i in instances if i["domain"] == "telecom"]),
        },
    }
    (DATA / "extraction_log_v2.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_path} with {len(instances)} instances")


if __name__ == "__main__":
    main()
