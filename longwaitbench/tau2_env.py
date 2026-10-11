"""tau2-bench environment wiring.

Builds a live tau2 domain environment from the vendored pruned `tau2`
package (data/pylibs) and a db snapshot. We consume the environment
(tools + db) but never the tau2 agent loop / evaluator / runner.

The pruned package loads its domain data (policy.md etc.) from the
directory given by the TAU2_DATA_DIR environment variable.
"""
import importlib
import json
import sys

from tau2.data_model.message import ToolCall

from . import constants as C


def _ensure_paths() -> None:
    pylib = str(C.PYLIB_DIR)
    if pylib not in sys.path:
        sys.path.insert(0, pylib)
    # make sure TAU2_DATA_DIR is visible to tau2's config module
    import os

    if not os.environ.get("TAU2_DATA_DIR"):
        os.environ["TAU2_DATA_DIR"] = str(C.DEFAULT_TAU2_DATA_DIR)


def build_tau2_environment(instance: dict):
    """Return (env, tool_docs, slow_tool_waits).

    tool_docs: agent-visible tool surface (name/description/parameters
               plus slow annotations and declared waits).
    slow_tool_waits: {tool_name: {"type": "policy", "declared_minutes": float}}
    """
    _ensure_paths()
    init = instance["initial_state"]
    db_class = _import_attr(init["db_class"])

    with open(init["db_snapshot"], "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    db = db_class.model_validate(snapshot)
    return build_tau2_environment_from_db(instance, db)


def build_tau2_environment_from_db(instance: dict, db):
    """Same as build_tau2_environment, but with a caller-supplied db
    (used by state restore: the current db is replayed through the same
    construction path instead of the initial snapshot)."""
    _ensure_paths()
    init = instance["initial_state"]
    env_module = _import_module(init["domain_env"])
    env = env_module.get_environment(db=db)

    tools = env.get_tools()
    slow_waits = _parse_slow_waits(instance)

    tool_docs = []
    for t in sorted(tools, key=lambda x: x.name):
        slow = t.name in slow_waits
        doc = {
            "name": t.name,
            "description": (t._get_description() or t.short_desc or "").strip(),
            "parameters": t.openai_schema["function"].get("parameters", {}),
            "slow": slow,
        }
        if slow:
            doc["wait"] = slow_waits[t.name]
        tool_docs.append(doc)

    return env, tool_docs, slow_waits


def _parse_slow_waits(instance: dict) -> dict:
    """Slow-tool wait policies come from the manifest tools list."""
    waits = {}
    for t in instance.get("tools", []):
        if t.get("slow") and t.get("wait"):
            waits[t["name"]] = dict(t["wait"])
    return waits


def call_tau2_tool(env, tool_name: str, arguments: dict, call_id: str = "t") -> dict:
    """Execute one tau2 tool call synchronously.

    Returns {"ok": bool, "result": str, "error": bool} where result is the
    JSON-serialized tool response (or the error message when the tool failed).
    """
    tc = ToolCall(id=call_id, name=tool_name, arguments=arguments, requestor="assistant")
    try:
        resp = env.get_response(tc)
        return {"ok": not resp.error, "result": resp.content, "error": resp.error}
    except Exception as e:  # env-level crash: surface as tool error, never kill session
        return {"ok": False, "result": f"Error: {e}", "error": True}


def env_policy_text(env) -> str:
    try:
        return env.get_policy()
    except Exception:
        return ""


def _import_module(path: str):
    return importlib.import_module(path)


def _import_attr(path: str):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
