"""Session loop: one instance x seed x condition = one session.

Contract anchors implemented here:
- five-action decision loop; every decision carries a validated
  time_prediction (>=2 buckets, masses > 0, sum to 1, finite);
- the policy sees ONLY the sanitized view (task_spec, tool docs, service
  policy, budgets, open-job summary, its own history) — never the private
  initial state, never T, never the verifier;
- poll-only completion, no push; poll returns status + integer counter
  (no progress ratio);
- slow-call budget is a hard cap; wall deadline yields outcome "timeout";
- every instance x condition runs under try/except and never kills the run.
"""
import json
import math
import time

from . import constants as C
from .policies import CONDITION_POLICIES, CHEAT_POLICIES, naive_prediction
from .unified_env import UnifiedEnv


def _normalize_bucket_label(label) -> str:
    """Accept the canonical labels and common interval-string variants."""
    if label in C.BUCKET_LABELS:
        return label
    if not isinstance(label, str):
        return None
    import re

    m = re.match(r"^\s*([0-9.]+)\s*<=\s*t\s*<\s*([0-9.]+)\s*$", label)
    if m:
        lo = float(m.group(1))
        return C.BUCKET_LABELS[
            next(i for i, (_, blo, bhi) in enumerate(C.TIME_BUCKETS) if blo <= lo < bhi)
        ]
    m = re.match(r"^\s*t\s*>=\s*([0-9.]+)\s*$", label)
    if m and float(m.group(1)) >= 60.0:
        return "60s+"
    return None


def validate_decision(d) -> dict:
    """Normalize and validate one decision dict. Returns None if invalid.

    NaN guard: any non-finite float anywhere makes the decision invalid.
    """
    if not isinstance(d, dict):
        return None
    action = d.get("action")
    if action not in C.FIVE_ACTIONS:
        return None
    params = d.get("params")
    if not isinstance(params, dict):
        params = {}
    tp = d.get("time_prediction")
    if not isinstance(tp, dict):
        return None
    if len(tp) < 2:
        return None
    probs = []
    for raw_label, p in tp.items():
        label = _normalize_bucket_label(raw_label)
        if label is None:
            return None
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(p) or p < 0:
            return None
        probs.append((label, p))
    # merge duplicate labels after normalization
    merged = {}
    for label, p in probs:
        merged[label] = merged.get(label, 0.0) + p
    total = sum(merged.values())
    if total <= 0:
        return None
    normalized = {label: p / total for label, p in merged.items()}
    # >=2 buckets must be present; zero-mass buckets are legitimate (one-hot
    # forecasts) and are kept as-is
    if len(normalized) < 2:
        return None
    return {"action": action, "params": params, "time_prediction": normalized}


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:1500]
    except Exception:
        return str(obj)[:1500]


def build_view(uenv, instance, seconds_left: float, history: list) -> dict:
    return {
        "task_spec": instance["task_spec"],
        "tool_docs": uenv.tool_docs,
        "policy_text": uenv.policy_text,
        "slow_calls_left": uenv.slow_calls_left,
        "seconds_left": max(0.0, seconds_left),
        "deadline": time.time() + seconds_left,
        "open_jobs": uenv.open_jobs(),
        "history": history[-40:],
    }


def _apply(uenv, action, params):
    """Apply one decision. Returns (effect_dict, observation_string).

    No behavior limits (user principle): wait is uncapped (positive only,
    input sanity), no deadline clamp, no decision cap.
    """
    if action == "wait":
        try:
            secs = float(params.get("seconds", 1.0))
        except (TypeError, ValueError):
            secs = 1.0
        secs = max(0.0, secs)
        t0 = time.time()
        time.sleep(secs)
        slept = time.time() - t0
        effect = {"slept_seconds": round(slept, 4)}
        obs = f"wait({secs:.2f}s) -> slept {slept:.3f}s"
        return effect, obs

    if action == "poll":
        handle = params.get("job_handle")
        res = uenv.poll(handle)
        effect = {"poll": res}
        obs = f"poll({handle}) -> {_safe_json(res)}"
        return effect, obs

    if action == "work":
        tool_name = params.get("tool_name")
        args = params.get("arguments")
        if not isinstance(args, dict):
            args = {}
        if not isinstance(tool_name, str) or not tool_name:
            return {"error": "work requires tool_name"}, "work -> invalid tool_name"
        if uenv.is_slow(tool_name):
            res = uenv.submit_slow(tool_name, args)
            effect = {"submit": {k: v for k, v in res.items() if k != "result"}}
            if res.get("ok"):
                obs = f"work {tool_name} submitted -> job_handle={res['job_handle']}"
            else:
                obs = f"work {tool_name} rejected -> {res.get('error')}"
            return effect, obs
        res = uenv.call_fast(tool_name, args)
        effect = {"tool": {"ok": res.get("ok"), "error": res.get("error")}}
        obs = f"work {tool_name}({_safe_json(args)}) -> {_safe_json(res.get('result'))}"
        return effect, obs

    if action == "final_answer":
        return {"final_answer": True}, f"final_answer({_safe_json(params)[:200]})"

    if action == "abandon":
        reason = str(params.get("reason", ""))
        return {"abandon": True}, f"abandon({reason})"


def run_session(condition, instance, seed, trace, cheat_policy=None, client=None,
                session_id=None):
    session_id = session_id or f"{condition}-{instance['id']}-seed{seed}"
    started = time.time()
    uenv = UnifiedEnv(instance, seed)
    deadline = started + uenv.wall_minutes * 60.0 * C.COMPRESSION_SCALE

    if cheat_policy is not None:
        policy = cheat_policy(instance, seed)
        policy_name = policy.name
    else:
        cls = CONDITION_POLICIES[condition]
        if cls.__name__ == "BasePolicy":
            policy = cls(client if client is not None else _CLIENT, instance, seed)
        else:
            policy = cls(instance, seed)
        policy_name = policy.name

    trace.write(
        {
            "type": "session",
            "session_id": session_id,
            "condition": condition,
            "policy": policy_name,
            "seed": seed,
            "instance_id": instance["id"],
            "source": instance["source"],
            "domain": instance["domain"],
            "started_at": started,
            "cheat_detection": cheat_policy is not None,
        }
    )

    history = []
    decision_count = 0
    n_invalid = 0
    outcome = "failed"
    final_error = None
    try:
        while True:
            now = time.time()
            seconds_left = max(0.0, deadline - now)  # informational only
            view = build_view(uenv, instance, seconds_left, history)
            out = policy.decide(view)
            if isinstance(out, dict) and "action" in out:
                # synthetic policies return a bare decision dict
                decision = validate_decision(out)
                llm_record = None
                llm_id = None
                invalid_fallback = out.get("invalid_fallback", False)
            else:
                decision = validate_decision(out.get("decision") if isinstance(out, dict) else None)
                llm_record = out.get("llm_record") if isinstance(out, dict) else None
                llm_id = llm_record.get("llm_request_id") if llm_record else None
                invalid_fallback = bool(out.get("invalid_fallback")) if isinstance(out, dict) else False
            if decision is None:
                n_invalid += 1
                decision = {
                    "action": "wait",
                    "params": {"seconds": 1.0},
                    "time_prediction": naive_prediction(seconds_left),
                }
            effect, obs = _apply(uenv, decision["action"], decision["params"])
            trace.write(
                {
                    "type": "decision",
                    "task_id": instance["id"],
                    "session_id": session_id,
                    "condition": condition,
                    "seed": seed,
                    "t": time.time(),
                    "action": decision["action"],
                    "params": decision["params"],
                    "time_prediction": decision["time_prediction"],
                    "virtual_cost": C.ACTION_VIRTUAL_COST[decision["action"]],
                    "llm_request_id": llm_id,
                    "effect": effect,
                    "invalid_fallback": invalid_fallback,
                }
            )
            decision_count += 1
            if llm_record:
                trace.write(
                    {
                        "type": "llm",
                        "llm_request_id": llm_id,
                        "session_id": session_id,
                        "task_id": instance["id"],
                        "condition": condition,
                        "request": llm_record.get("request", {}),
                        "response": llm_record.get("response", {}),
                        "usage": llm_record.get("usage", {}),
                        "latency_sec": llm_record.get("latency_sec"),
                        "cache_hit": llm_record.get("cache_hit"),
                        "rate_limited": llm_record.get("rate_limited", False),
                    }
                )
            if policy_name != "LongCall-v1" and hasattr(policy, "observe"):
                policy.observe(effect)
            history.append(f"[{decision_count}] action={decision['action']} -> {obs}")
            if decision["action"] == "final_answer":
                est = decision["params"].get("estimate")
                ok = uenv.verify(est)
                outcome = "success" if ok else "failed"
                break
            if decision["action"] == "abandon":
                outcome = "abandoned"
                break
    except Exception as e:
        final_error = f"{type(e).__name__}: {e}"
        outcome = "failed"

    tc = uenv.true_completion()
    uenv.shutdown()
    ended = time.time()

    task_entry = {
        "type": "task",
        "task_id": instance["id"],
        "session_id": session_id,
        "condition": condition,
        "seed": seed,
        "started_at": started,
        "ended_at": ended,
        "true_completion": tc,
        "budget": {
            "time_physical_seconds": round(uenv.wall_minutes * 60.0 * C.COMPRESSION_SCALE, 4),
            "max_calls": uenv.budget_slow_calls,
            "decision_budget": C.MAX_DECISIONS_PER_TASK,
        },
        "outcome": outcome,
        "decision_count": decision_count,
        "invalid_decisions": n_invalid,
        "slow_calls_used": uenv.budget_slow_calls - uenv.slow_calls_left,
        "error": final_error,
        "cheat_detection": cheat_policy is not None,
    }
    trace.write(task_entry)
    return task_entry


_CLIENT = None


def set_client(client):
    global _CLIENT
    _CLIENT = client
