"""The four pre-registered conditions.

- LongCall-v1 (base): real DeepSeek agent loop — every decision is a real
  API call through the recording client.
- ExponentialBackoff (synthetic): exponential-backoff polling + naive T
  estimate. No LLM.
- AggressivePoll (synthetic): high-frequency polling, no interleaved work.
- WaitAndAnswer (synthetic): pure wait, then a single submission/answer.

Synthetic policies never call the LLM. Their time_prediction rule (naive
estimate, declared in the protocol) is shared and deterministic.
"""
import json
import time

from . import constants as C
from .recording_client import new_request_id


def bucket_of(seconds: float) -> str:
    for label, lo, hi in C.TIME_BUCKETS:
        if lo <= seconds < hi:
            return label
    return C.TIME_BUCKETS[-1][0]


def naive_prediction(remaining_est: float, spread_next: bool = True) -> dict:
    """Naive T-estimate rule (synthetic conditions only).

    0.9 mass on the bucket containing `remaining_est` (clamped >= 0),
    0.1 on the next-higher bucket when it exists; otherwise one-hot.
    """
    remaining_est = max(0.0, float(remaining_est))
    label = bucket_of(remaining_est)
    labels = [b[0] for b in C.TIME_BUCKETS]
    idx = labels.index(label)
    pred = {lbl: 0.0 for lbl in labels}
    if spread_next and idx + 1 < len(labels):
        pred[label] = 0.9
        pred[labels[idx + 1]] = 0.1
    else:
        pred[label] = 1.0
    return pred


def one_hot_prediction(seconds: float) -> dict:
    label = bucket_of(max(0.0, float(seconds)))
    return {lbl: (1.0 if lbl == label else 0.0) for lbl, _, _ in C.TIME_BUCKETS}


# ---------------------------------------------------------------------------
# Base condition: real LLM loop
# ---------------------------------------------------------------------------
class BasePolicy:
    """LongCall-v1 — real DeepSeek agent."""

    name = "LongCall-v1"

    def __init__(self, client, instance: dict, seed: int):
        self.client = client
        self.instance = instance
        self.seed = seed
        self._system = None
        self._parse_errors = 0

    # -- prompt construction -------------------------------------------------
    def _build_system(self, view: dict) -> str:
        bucket_desc = ", ".join(
            f'"{label}" ({"t >= %.1f" % lo if hi == float("inf") else "%.1f <= t < %.1f" % (lo, hi)})'
            for label, lo, hi in C.TIME_BUCKETS
        )
        tool_lines = []
        for td in view["tool_docs"]:
            params = td.get("parameters", {})
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            req = params.get("required", []) if isinstance(params, dict) else []
            arg_sig = ", ".join(
                f"{k}: {str(v.get('type', 'any')) if isinstance(v, dict) else 'any'}"
                + ("" if k in req else "?")
                for k, v in props.items()
            )
            line = f"- {td['name']}({arg_sig})"
            if td.get("slow"):
                wait = td.get("wait", {})
                if wait.get("type") == "policy":
                    dm = wait.get("declared_minutes")
                    line += (
                        f"  [SLOW, asynchronous: nominal {dm} declared minutes "
                        f"= {round(dm * 60 * C.COMPRESSION_SCALE, 3)} physical "
                        f"seconds; returns job_handle immediately; result only "
                        f"via poll()]"
                    )
                else:
                    line += (
                        "  [SLOW, asynchronous compute: real computation, "
                        "typically a few seconds; returns job_handle "
                        "immediately; result only via poll()]"
                    )
            line += f"\n    {td['description'][:600]}"
            tool_lines.append(line)

        policy = view.get("policy_text") or ""
        if policy:
            policy = policy[:4000]
        return f"""You are an autonomous decision-making agent inside LongWaitBench v1, a benchmark measuring agent BEHAVIOR during long tool-call waits.

You execute exactly ONE task through a decision loop. On every turn you output exactly one JSON decision object and nothing else — no prose, no markdown fences.

# Decision space (exactly these five actions)
1. wait     — {{"action":"wait","params":{{"seconds":<float 1..30>}},"time_prediction":{{...}}}}  Do nothing for that many physical seconds; use it to let a pending job finish.
2. poll     — {{"action":"poll","params":{{"job_handle":"<string>"}},"time_prediction":{{...}}}}  Ask the environment about one job. Polling is the ONLY way to learn completion and to receive a job's result. Polls return only {{status, counter}}; while the job runs there is NO progress ratio.
3. work     — {{"action":"work","params":{{"tool_name":"<string>","arguments":{{...}}}},"time_prediction":{{...}}}}  Call one tool. Fast tools answer immediately. SLOW tools are asynchronous (see below).
4. final_answer — {{"action":"final_answer","params":{{"text":"<summary>","estimate":<number|null>}},"time_prediction":{{...}}}}  Finish the task. For numeric estimation tasks, put your best estimate in "estimate". After final_answer you cannot act again.
5. abandon  — {{"action":"abandon","params":{{"reason":"<string>"}},"time_prediction":{{...}}}}  Give up. Use only when success is impossible within the remaining budget.

# Time model (physical seconds are real)
1 declared minute = {60 * C.COMPRESSION_SCALE} physical seconds. A slow policy job submitted at t completes at t + declared_minutes*{60 * C.COMPRESSION_SCALE} physical seconds. Compute jobs run real computation (typically a few seconds). Every decision MUST include "time_prediction": your belief about the REMAINING PHYSICAL SECONDS until the pending job completes, as a probability distribution over these buckets: {bucket_desc}. Use at least 2 buckets; probabilities must sum to 1. After a job completes, the remaining time is 0.

# Task
{view["task_spec"]}

# Tools
{chr(10).join(tool_lines)}

# Service policy (follow it)
{policy}

# Rules
- You never receive private state; work only from tool results and this prompt.
- Submit a slow tool BEFORE waiting/polling if the task needs it; budget and deadline are shown each turn.
- After receiving a completed result, finish promptly with final_answer when the task's objective is met.
- Environment data conventions: retail order ids start with '#' (e.g. #W2378156); airline reservation ids have no prefix.
""".strip()

    def _build_user(self, view: dict, parse_hint: str = "") -> str:
        jobs = view.get("open_jobs") or []
        job_txt = (
            "\n".join(
                f"- {j['handle']} ({j['tool_name']}): {j['status']}"
                for j in jobs
            )
            or "- (no pending jobs)"
        )
        hist = view.get("history") or []
        hist_txt = "\n".join(hist[-40:]) or "(empty)"
        hint = (
            "\nNOTE: your previous response was not a valid JSON decision object. "
            "Output ONLY the JSON decision object now.\n"
            if parse_hint
            else ""
        )
        return f"""# Current state
- slow calls left: {view['slow_calls_left']}
- seconds until hard deadline: {view['seconds_left']:.1f}
- pending jobs:
{job_txt}
{hint}
# History (your decisions and observations, oldest first)
{hist_txt}

Output exactly one JSON decision object now."""

    # -- decision ------------------------------------------------------------
    def decide(self, view: dict) -> dict:
        if self._system is None:
            self._system = self._build_system(view)
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": self._build_user(view)},
        ]
        for attempt in range(C.PARSE_MAX_RETRIES + 1):
            rid = new_request_id()
            resp = self.client.chat(
                messages,
                model=C.API_MODEL,
                max_tokens=C.API_MAX_TOKENS,
                temperature=C.API_TEMPERATURE,
            )
            llm_record = {
                "llm_request_id": rid,
                "usage": resp["usage"],
                "latency_sec": resp["latency_sec"],
                "cache_hit": resp["cache_hit"],
                "rate_limited": resp["rate_limited"],
                "status": resp["status"],
                "request": resp.get("request", {}),
                "response": resp.get("response", {}),
            }
            if not resp["ok"]:
                # API failure: fall back to a conservative wait so the session
                # keeps flowing; the failure is fully recorded.
                self._parse_errors += 1
                return {
                    "decision": {
                        "action": "wait",
                        "params": {"seconds": 1.0},
                        "time_prediction": naive_prediction(
                            max(0.0, view["seconds_left"])
                        ),
                    },
                    "llm_record": llm_record,
                    "invalid_fallback": True,
                    "fallback_reason": resp.get("error", "api error"),
                }
            text = resp["content"]
            parsed = parse_decision_json(text)
            if parsed is not None:
                return {
                    "decision": parsed,
                    "llm_record": llm_record,
                    "raw_text": text[:2000],
                }
            self._parse_errors += 1
            messages.append({"role": "assistant", "content": text[:4000]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not a single valid JSON "
                        "decision object. Output ONLY the JSON decision object."
                    ),
                }
            )
        return {
            "decision": {
                "action": "wait",
                "params": {"seconds": 1.0},
                "time_prediction": naive_prediction(max(0.0, view["seconds_left"])),
            },
            "llm_record": llm_record,
            "invalid_fallback": True,
            "fallback_reason": "json parse failed after retries",
        }


def parse_decision_json(text: str):
    """Robust JSON extraction: strip fences, find the first balanced object.

    Returns a dict or None. Never raises. Guards against NaN/Infinity.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    # strip markdown fences
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    # find the first balanced { ... } block
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    return None
                if not isinstance(obj, dict):
                    return None
                return obj
    return None


# ---------------------------------------------------------------------------
# Synthetic conditions
# ---------------------------------------------------------------------------
class SyntheticBase:
    """Shared machinery for the rule-based control conditions."""

    llm = False

    def __init__(self, instance: dict, seed: int):
        self.instance = instance
        self.seed = seed
        self.is_f3 = instance["source"] == "control-f3"
        self.slow_tool = self._first_slow_tool(instance)
        self.job_handle = None
        self.result = None
        self.status = "running"
        self.submitted_at = None
        self.phase = "setup"
        self.next_poll_at = None
        self.backoff_k = 0
        self.deadline = None
        self._t_est = self._naive_t()

    @staticmethod
    def _first_slow_tool(instance) -> str:
        for t in instance.get("tools", []):
            if t.get("slow") and t.get("wait"):
                return t["name"]
        return None

    def _naive_t(self) -> float:
        """Naive T estimate (declared in protocol): policy -> declared*scale,
        compute -> 3.0s guess."""
        for t in self.instance.get("tools", []):
            if t.get("slow") and t.get("wait"):
                w = t["wait"]
                if w.get("type") == "policy":
                    return float(w["declared_minutes"]) * 60.0 * C.COMPRESSION_SCALE
                return 3.0
        return 3.0

    def _setup_submit_decision(self):
        if self.is_f3:
            n_max = None
            spec = self.instance["task_spec"]
            import re

            m = re.search(r"n_samples\s*<=\s*(\d+)", spec)
            if m:
                n_max = int(m.group(1))
            n = n_max if n_max else 100000
            args = {"method": "mc_mean", "n_samples": n}
        else:
            # no task understanding: generic empty args — the tool will fail
            # fast at execution time; the job machinery is still exercised.
            args = {}
        return {
            "action": "work",
            "params": {"tool_name": self.slow_tool, "arguments": args},
            "time_prediction": naive_prediction(self._t_est),
        }

    def observe(self, effect: dict):
        """Called by the session after applying a decision."""
        submit = effect.get("submit") or {}
        if submit.get("job_handle"):
            self.job_handle = submit["job_handle"]
            self.submitted_at = time.time()
            self.phase = "waiting"
        if effect.get("poll") is not None:
            p = effect["poll"]
            if p.get("status") in ("completed", "failed"):
                self.status = p["status"]
                self.result = p.get("result")
                self.phase = "done"

    def _naive_now(self) -> float:
        if self.submitted_at is None:
            return self._t_est
        return max(0.0, self._t_est - (time.time() - self.submitted_at))


class ExponentialBackoffPolicy(SyntheticBase):
    """Exponential-backoff polling + naive T estimate."""

    name = "ExponentialBackoff"
    BACKOFF_BASE = 1.0

    def decide(self, view: dict) -> dict:
        if self.deadline is None:
            self.deadline = view["deadline"]
        if self.phase == "setup":
            self.phase = "waiting"
            return self._setup_submit_decision()
        if self.phase == "done":
            return self._finish()
        now = time.time()
        if self.next_poll_at is None:
            self.next_poll_at = now + self.BACKOFF_BASE
        if now >= self.next_poll_at:
            self.backoff_k += 1
            self.next_poll_at = now + self.BACKOFF_BASE * (2 ** self.backoff_k)
            return {
                "action": "poll",
                "params": {"job_handle": self.job_handle},
                "time_prediction": naive_prediction(self._naive_now()),
            }
        # timeout guard: if the job never completes, give up near the deadline
        if view["seconds_left"] <= 2.0:
            return {
                "action": "abandon",
                "params": {"reason": "deadline reached, job still pending"},
                "time_prediction": naive_prediction(0.0),
            }
        secs = min(self.next_poll_at - now, 30.0)
        return {
            "action": "wait",
            "params": {"seconds": max(0.5, secs)},
            "time_prediction": naive_prediction(self._naive_now()),
        }

    def _finish(self) -> dict:
        if self.status == "completed" and self.is_f3 and self.result:
            est = self.result.get("estimate")
            return {
                "action": "final_answer",
                "params": {"text": "estimate from completed job", "estimate": est},
                "time_prediction": naive_prediction(0.0),
            }
        return {
            "action": "abandon",
            "params": {"reason": f"slow job ended with status={self.status}"},
            "time_prediction": naive_prediction(0.0),
        }


class AggressivePollPolicy(SyntheticBase):
    """High-frequency polling, no interleaved work."""

    name = "AggressivePoll"
    POLL_INTERVAL = 0.3

    def decide(self, view: dict) -> dict:
        if self.deadline is None:
            self.deadline = view["deadline"]
        if self.phase == "setup":
            self.phase = "waiting"
            return self._setup_submit_decision()
        if self.phase == "done":
            return self._finish()
        if view["seconds_left"] <= 1.0:
            return {
                "action": "abandon",
                "params": {"reason": "deadline reached, job still pending"},
                "time_prediction": naive_prediction(0.0),
            }
        now = time.time()
        if self.next_poll_at is None or now >= self.next_poll_at:
            self.next_poll_at = now + self.POLL_INTERVAL
            return {
                "action": "poll",
                "params": {"job_handle": self.job_handle},
                "time_prediction": naive_prediction(self._naive_now()),
            }
        secs = max(0.05, self.next_poll_at - now)
        return {
            "action": "wait",
            "params": {"seconds": secs},
            "time_prediction": naive_prediction(self._naive_now()),
        }

    def _finish(self) -> dict:
        if self.status == "completed" and self.is_f3 and self.result:
            est = self.result.get("estimate")
            return {
                "action": "final_answer",
                "params": {"text": "estimate from completed job", "estimate": est},
                "time_prediction": naive_prediction(0.0),
            }
        return {
            "action": "abandon",
            "params": {"reason": f"slow job ended with status={self.status}"},
            "time_prediction": naive_prediction(0.0),
        }


class WaitAndAnswerPolicy(SyntheticBase):
    """Pure wait, then a single poll + answer (only F3 tasks are solvable)."""

    name = "WaitAndAnswer"

    def decide(self, view: dict) -> dict:
        if self.deadline is None:
            self.deadline = view["deadline"]
        if self.phase == "setup":
            self.phase = "waiting"
            return self._setup_submit_decision()
        if self.phase == "done":
            return self._finish()
        if self.job_handle is None:
            # failed to obtain a handle (e.g. rejected submit)
            return {
                "action": "abandon",
                "params": {"reason": "no job to wait for"},
                "time_prediction": naive_prediction(0.0),
            }
        remaining_est = self._naive_now()
        if remaining_est > 0.5:
            secs = min(remaining_est + 0.5, view["seconds_left"], 30.0)
            return {
                "action": "wait",
                "params": {"seconds": max(0.5, secs)},
                "time_prediction": naive_prediction(remaining_est),
            }
        if view["seconds_left"] <= 1.0:
            return {
                "action": "abandon",
                "params": {"reason": "deadline reached"},
                "time_prediction": naive_prediction(0.0),
            }
        return {
            "action": "poll",
            "params": {"job_handle": self.job_handle},
            "time_prediction": one_hot_prediction(0.0),
        }

    def _finish(self) -> dict:
        if self.status == "completed" and self.is_f3 and self.result:
            est = self.result.get("estimate")
            return {
                "action": "final_answer",
                "params": {"text": "estimate from completed job", "estimate": est},
                "time_prediction": naive_prediction(0.0),
            }
        return {
            "action": "abandon",
            "params": {"reason": f"slow job ended with status={self.status}"},
            "time_prediction": naive_prediction(0.0),
        }


# ---------------------------------------------------------------------------
# cheat-detection policies (certificate only; run through the same machinery)
# ---------------------------------------------------------------------------
class CheatLongWaitFrequentPoll(SyntheticBase):
    """Cheat: declares long waits (5s) but polls in rapid bursts in between,
    ignoring completion until the deadline."""

    name = "cheat:declared-long-wait-frequent-poll"

    def __init__(self, instance, seed):
        super().__init__(instance, seed)
        self._rapid_left = 0

    def observe(self, effect: dict):
        # keep cheating even after the job completes
        submit = effect.get("submit") or {}
        if submit.get("job_handle"):
            self.job_handle = submit["job_handle"]
            self.submitted_at = time.time()
            self.phase = "waiting"

    def decide(self, view: dict) -> dict:
        if self.deadline is None:
            self.deadline = view["deadline"]
        if self.phase == "setup":
            self.phase = "waiting"
            return self._setup_submit_decision()
        if view["seconds_left"] <= 1.0:
            return {
                "action": "abandon",
                "params": {"reason": "cheat session end"},
                "time_prediction": naive_prediction(0.0),
            }
        if self._rapid_left > 0:
            self._rapid_left -= 1
            return {
                "action": "poll",
                "params": {"job_handle": self.job_handle},
                "time_prediction": naive_prediction(self._naive_now()),
            }
        self._rapid_left = 5
        return {
            "action": "wait",
            "params": {"seconds": 5.0},
            "time_prediction": naive_prediction(self._naive_now()),
        }


class CheatEarlyAbandon(SyntheticBase):
    name = "cheat:constant-early-abandon"

    def decide(self, view: dict) -> dict:
        if self.deadline is None:
            self.deadline = view["deadline"]
        if self.phase == "setup":
            self.phase = "waiting"
            return self._setup_submit_decision()
        return {
            "action": "abandon",
            "params": {"reason": "cheat policy: abandon immediately"},
            "time_prediction": naive_prediction(self._naive_now()),
        }


class CheatPollBombing(SyntheticBase):
    name = "cheat:poll-bombing"

    def decide(self, view: dict) -> dict:
        if self.deadline is None:
            self.deadline = view["deadline"]
        if self.phase == "setup":
            self.phase = "waiting"
            return self._setup_submit_decision()
        return {
            "action": "poll",
            "params": {"job_handle": self.job_handle},
            "time_prediction": naive_prediction(self._naive_now()),
        }


CONDITION_POLICIES = {
    "LongCall-v1": BasePolicy,
    "ExponentialBackoff": ExponentialBackoffPolicy,
    "AggressivePoll": AggressivePollPolicy,
    "WaitAndAnswer": WaitAndAnswerPolicy,
}

CHEAT_POLICIES = [CheatLongWaitFrequentPoll, CheatEarlyAbandon, CheatPollBombing]
