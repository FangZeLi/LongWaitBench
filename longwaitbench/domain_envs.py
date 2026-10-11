"""Domain adapters for LongWaitBench v2 task types.

- banking (external-async): tau2 banking_knowledge environment, bm25_grep
  retrieval; slow tools are the submit/order tools exposed as *discoverable*
  tools (reached through call_discoverable_agent_tool); completion only
  observable through the domain's own status/history tools.
- telecom (network-async): tau2 telecom environment in solo mode — the agent
  performs user-side device actions too; slow tools are network/device
  operations whose effects appear in device state only after completion.

We consume the vendored environments (tools + db + policy); the tau2 agent
loop / evaluator / runner are never run.
"""
import json
import sys

from . import constants as C


def _ensure_paths():
    pylib = str(C.PYLIB_DIR)
    if pylib not in sys.path:
        sys.path.insert(0, pylib)
    import os

    if not os.environ.get("TAU2_DATA_DIR"):
        os.environ["TAU2_DATA_DIR"] = str(C.DEFAULT_TAU2_DATA_DIR)


def call_domain_tool(env, tool_name, arguments, call_id="t", requestor="assistant"):
    """Execute one tau2 tool call through env.get_response (which also runs
    the domain's sync_tools hook)."""
    from tau2.data_model.message import ToolCall

    tc = ToolCall(id=call_id, name=tool_name, arguments=arguments,
                  requestor=requestor)
    try:
        resp = env.get_response(tc)
        return {"ok": not resp.error, "result": resp.content, "error": resp.error}
    except Exception as e:  # env-level crash: surface as tool error, never kill session
        return {"ok": False, "result": f"Error: {e}", "error": True}


def _parse_slow_waits(instance):
    waits = {}
    for t in instance.get("tools", []):
        if t.get("slow") and t.get("wait"):
            waits[t["name"]] = dict(t["wait"])
    return waits


def _tool_docs(env_tools, slow_waits):
    docs = []
    for t in sorted(env_tools, key=lambda x: x.name):
        slow = t.name in slow_waits
        doc = {
            "name": t.name,
            "description": (t._get_description() or t.short_desc or "").strip(),
            "parameters": t.openai_schema["function"].get("parameters", {}),
            "slow": slow,
        }
        if slow:
            doc["wait"] = slow_waits[t.name]
        docs.append(doc)
    return docs


def build_banking_environment(instance):
    """(env, tool_docs, slow_waits, policy_text) for a banking instance."""
    _ensure_paths()
    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.domains.banking_knowledge.environment import get_environment

    snap = instance["initial_state"]["db_snapshot"]
    with open(snap, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    db = TransactionalDB.model_validate(snapshot)
    variant = instance["initial_state"].get("retrieval_variant", "bm25_grep")
    env = get_environment(db=db, retrieval_variant=variant)
    slow_waits = _parse_slow_waits(instance)
    docs = _tool_docs(env.get_tools(), {})  # slow inner tools are discoverable-only
    return env, docs, slow_waits, env.get_policy()


def build_telecom_environment(instance):
    """(env, tool_docs, slow_waits, policy_text) for a telecom instance."""
    _ensure_paths()
    from tau2.data_model.tasks import EnvFunctionCall
    from tau2.domains.telecom.environment import get_environment

    env = get_environment(solo_mode=True)
    init_actions = instance["initial_state"].get("initialization_actions") or []
    init_calls = [
        EnvFunctionCall(env_type=a["env_type"], func_name=a["func_name"],
                        arguments=a.get("arguments") or {})
        for a in init_actions
    ]
    env.set_state(initialization_data=None, initialization_actions=init_calls,
                  message_history=[])
    slow_waits = _parse_slow_waits(instance)
    tools = list(env.get_tools()) + list(env.get_user_tools())
    docs = _tool_docs(tools, slow_waits)
    return env, docs, slow_waits, env.get_policy()


def run_telecom_assertions(env, assertions):
    """True iff every env_assertion holds on the live environment."""
    from tau2.data_model.tasks import EnvAssertion

    ok = True
    for a in assertions:
        ea = EnvAssertion(env_type=a.get("env_type", "user"),
                          func_name=a["func_name"],
                          arguments=a.get("arguments") or {},
                          assert_value=a.get("assert_value", True))
        ok = env.run_env_assertion(ea, raise_assertion_error=False) and ok
    return ok
