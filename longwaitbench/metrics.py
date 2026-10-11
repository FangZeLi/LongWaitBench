"""All headline metrics, diagnostics, and the certificate.

Every number here is mechanically recomputable from canonical_trace.jsonl
plus the hidden verifier outcomes alone. No LLM judge, no unmeasured state.
wait_score = H(anytime_auc, budget_survival) with ZERO free weights.
"""
import json
import math
import statistics
from pathlib import Path

from . import constants as C


def load_entries(trace_path) -> list:
    entries = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def harmonic_mean(a, b):
    if min(a, b) == 0:
        return 0.0
    return 2 * a * b / (a + b)


def _bucket_of(seconds: float) -> str:
    for label, lo, hi in C.TIME_BUCKETS:
        if lo <= seconds < hi:
            return label
    return C.TIME_BUCKETS[-1][0]


class ConditionMetrics:
    def __init__(self, entries, condition, num_tasks):
        self.entries = entries
        self.condition = condition
        self.num_tasks = num_tasks
        self.sessions = [e for e in entries if e.get("type") == "session"
                         and e.get("condition") == condition
                         and not e.get("cheat_detection")]
        self.session_ids = {e["session_id"] for e in self.sessions}
        self.tasks = [e for e in entries if e.get("type") == "task"
                      and e.get("condition") == condition
                      and not e.get("cheat_detection")]
        self.task_map = {t["session_id"]: t for t in self.tasks}
        self.decisions = [e for e in entries if e.get("type") == "decision"
                          and e.get("condition") == condition
                          and e["session_id"] in self.session_ids]
        self.llms = [e for e in entries if e.get("type") == "llm"
                     and e.get("condition") == condition
                     and e["session_id"] in self.session_ids]
        self.llm_timings = [e for e in entries if e.get("type") == "llm_timing"
                            and e.get("condition") == condition
                            and e["session_id"] in self.session_ids]

    # -- helpers -------------------------------------------------------------
    def _t_of(self, task) -> float:
        tc = task.get("true_completion")
        if not tc or not tc.get("physical_seconds"):
            return None
        # T = completion of the first slow job = its submit time + duration
        submit_at = task.get("first_slow_submit_at")
        if submit_at is not None:
            return submit_at + tc["physical_seconds"]
        # legacy fallback (v1 pilot traces): assume submit at task start
        return task["started_at"] + tc["physical_seconds"]

    def _deadline_of(self, task) -> float:
        return task["started_at"] + task["budget"]["time_physical_seconds"]

    # -- vector metrics ------------------------------------------------------
    def success_rate(self):
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t["outcome"] == "success") / len(self.tasks)

    def pass_k_curve(self, ks=(1, 2, 3, 4, 5, 6, 7, 8)):
        curve = []
        for k in ks:
            n = sum(
                1
                for t in self.tasks
                if t["outcome"] == "success" and t["decision_count"] <= k
            )
            curve.append(n / len(self.tasks) if self.tasks else 0.0)
        return curve

    def time_efficiency(self):
        times = sorted(
            t["ended_at"] - t["started_at"]
            for t in self.tasks
            if t["outcome"] == "success"
        )
        if not times or not self.tasks:
            return 0.0
        tmax = times[-1]
        total = 0.0
        cur = 0.0
        prev_t = 0.0
        n_total = len(self.tasks)
        for t in times:
            total += cur * (t - prev_t)
            cur += 1.0 / n_total
            prev_t = t
        total += cur * (tmax - prev_t)
        return total / tmax

    def budget_survival_parts(self):
        n = 0
        n_time = 0
        n_calls = 0
        n_both = 0
        for t in self.tasks:
            if t["outcome"] != "success":
                continue
            n += 1
            elapsed = t["ended_at"] - t["started_at"]
            time_budget = t["budget"]["time_physical_seconds"]
            # call budget = slow-call budget + decision-loop budget (each
            # decision turn costs one call slot; declared in the protocol)
            call_budget = t["budget"]["max_calls"] + t["budget"]["decision_budget"]
            sid = t["session_id"]
            polls = sum(
                1 for d in self.decisions if d["session_id"] == sid and d["action"] == "poll"
            )
            works = sum(
                1 for d in self.decisions if d["session_id"] == sid and d["action"] == "work"
            )
            llmcalls = sum(
                (l.get("n_model_calls") or 1) for l in self.llms if l["session_id"] == sid
            )
            call_count = polls + works + llmcalls
            ok_time = elapsed <= C.DEPLOYMENT_BUDGET_SCALE * time_budget
            ok_calls = call_count <= C.DEPLOYMENT_BUDGET_SCALE * call_budget
            if ok_time:
                n_time += 1
            if ok_calls:
                n_calls += 1
            if ok_time and ok_calls:
                n_both += 1
        if n == 0:
            return 0.0, 0.0, 0.0
        return n_time / n, n_calls / n, n_both / n

    def brier(self):
        scores = []
        for d in self.decisions:
            task = self.task_map.get(d["session_id"])
            if task is None:
                continue
            if not d.get("time_prediction"):
                continue  # harness-mode decisions carry no elicited prediction
            T = self._t_of(task)
            if T is None:
                continue
            remaining = max(0.0, T - d["t"])
            truth = _bucket_of(remaining)
            tp = d.get("time_prediction") or {}
            s = 0.0
            for label in C.BUCKET_LABELS:
                p = tp.get(label, 0.0)
                o = 1.0 if label == truth else 0.0
                s += (p - o) ** 2
            scores.append(s)
        return (sum(scores) / len(scores)) if scores else None

    def brier_decomposition(self):
        """Murphy decomposition over binary (decision, bucket) forecast events."""
        events = []  # (p, o)
        for d in self.decisions:
            task = self.task_map.get(d["session_id"])
            if task is None:
                continue
            if not d.get("time_prediction"):
                continue  # harness-mode decisions carry no elicited prediction
            T = self._t_of(task)
            if T is None:
                continue
            remaining = max(0.0, T - d["t"])
            truth = _bucket_of(remaining)
            tp = d.get("time_prediction") or {}
            for label in C.BUCKET_LABELS:
                p = tp.get(label, 0.0)
                o = 1.0 if label == truth else 0.0
                events.append((p, o))
        if not events:
            return None
        o_bar = statistics.mean(o for _, o in events)
        # group by forecast bin (0.1 width)
        groups = {}
        for p, o in events:
            g = round(p, 1)
            groups.setdefault(g, []).append(o)
        reliability = 0.0
        resolution = 0.0
        n = len(events)
        for g, outs in groups.items():
            n_g = len(outs)
            o_g = statistics.mean(outs)
            reliability += (n_g / n) * (g - o_g) ** 2
            resolution += (n_g / n) * (o_g - o_bar) ** 2
        uncertainty = o_bar * (1 - o_bar)
        return {
            "reliability": round(reliability, 6),
            "resolution": round(resolution, 6),
            "uncertainty": round(uncertainty, 6),
        }

    def costs(self):
        total_tokens = sum(l.get("usage", {}).get("total_tokens", 0) or 0 for l in self.llms)
        prompt = sum(l.get("usage", {}).get("prompt_tokens", 0) or 0 for l in self.llms)
        cache_read = sum(l.get("cache_read_tokens", 0) or 0 for l in self.llms)
        if cache_read > 0:
            # token-level cache ratio: how much of the prompt came from cache
            denom = cache_read + prompt
            ratio = cache_read / denom if denom else None
        else:
            hits = sum(1 for l in self.llms if l.get("cache_hit") is True)
            misses = sum(1 for l in self.llms if l.get("cache_hit") is False)
            ratio = (hits / (hits + misses)) if (hits + misses) else None
        counts = {a: 0 for a in C.ACTIONS}
        for d in self.decisions:
            if d.get("action") in counts:
                counts[d["action"]] += 1
        counts["total"] = len(self.decisions)
        return total_tokens, ratio, counts

    # -- diagnostics ---------------------------------------------------------
    def abandon_rate(self):
        """Fraction of sessions that end in abandon (premature give-up
        frequency; replaces false_give_up)."""
        if not self.tasks:
            return None
        return sum(1 for t in self.tasks if t["outcome"] == "abandoned") / len(self.tasks)

    def slow_calls_mean(self):
        """Mean number of long (slow) tool submissions per task, across ALL
        sessions (success and failure alike). A low value on a task whose
        golden solution uses slow tools is the trace signature of fast-path
        avoidance; a high value flags over-submission."""
        vals = [t.get("slow_calls_used", 0) for t in self.tasks]
        return statistics.mean(vals) if vals else None

    def time_decomposition(self):
        """Second-layer time budget decomposition (mean of per-session
        shares): wall = think + tool + slack.

        wall  = final_answer.t (or task ended_at) - task started_at
        tool  = sum of physical durations of all slow submissions
                (declared_minutes * 60 * COMPRESSION_SCALE each). Overlapping
                jobs are counted twice on purpose: a parallel agent's tool
                share can exceed 1.0.
        think = llm_timing.think_seconds minus blocking tool windows. Both
                harnesses timestamp their generation chunks such that the
                step containing a wait_for/wait call spans the block, so the
                recorded blocked/slept seconds are subtracted back out.
                NOTE: think and tool are NOT disjoint by construction — an
                agent that works (thinks/uses fast tools) while a slow job
                runs scores think time that overlaps the tool window. The
                share sum can therefore exceed 1.0; the excess is exactly
                the measured overlap, and is a desirable signal (same family
                as work_density).
        slack = wall - tool - think (fast tool calls, harness overhead,
                wake-up latency, idle)
        """
        think_shares, tool_shares, slack_shares = [], [], []
        timing_map = {}
        for l in self.llm_timings:
            timing_map.setdefault(l["session_id"], 0.0)
            timing_map[l["session_id"]] += l.get("think_seconds", 0.0)
        for t in self.tasks:
            sid = t["session_id"]
            fa = [d["t"] for d in self.decisions
                  if d["session_id"] == sid and d["action"] == "final_answer"]
            end = fa[-1] if fa else t.get("ended_at")
            start = t.get("started_at")
            if end is None or start is None or end <= start:
                continue
            wall = end - start
            tool = 0.0
            blocks = 0.0
            for d in self.decisions:
                if d["session_id"] != sid:
                    continue
                eff = d.get("effect") or {}
                w = (eff.get("submit") or {}).get("wait") or {}
                dm = w.get("declared_minutes")
                if dm:
                    tool += dm * 60.0 * C.COMPRESSION_SCALE
                blocks += float(eff.get("blocked_seconds")
                                or eff.get("slept_seconds") or 0.0)
            think = max(0.0, timing_map.get(sid, 0.0) - blocks)
            think_shares.append(think / wall)
            tool_shares.append(tool / wall)
            slack_shares.append(max(0.0, wall - tool - think) / wall)
        return {
            "think_share": round(statistics.mean(think_shares), 4)
            if think_shares else None,
            "tool_share": round(statistics.mean(tool_shares), 4)
            if tool_shares else None,
            "slack_share": round(statistics.mean(slack_shares), 4)
            if slack_shares else None,
        }

    def clock_fidelity(self):
        ratios = []
        mismatch = 0
        for t in self.tasks:
            tc = t.get("true_completion") or {}
            if tc.get("kind") != "policy":
                continue
            declared = tc.get("declared_minutes")
            physical = tc.get("physical_seconds")
            if not declared:
                continue
            expected = declared * 60.0 * C.COMPRESSION_SCALE
            ratios.append(physical / declared)
            if abs(physical - expected) > 0.02 * expected:
                mismatch += 1
        if not ratios:
            return None
        return {
            "median_physical_per_declared_minute": round(statistics.median(ratios), 4),
            "mismatch_count": mismatch,
        }

    def rate_limit_events(self):
        return sum(1 for l in self.llms if l.get("rate_limited"))

    def work_density(self):
        """Fraction of wait/work decisions that are work: how much the agent
        used the waiting time for useful tool work instead of pure sleep."""
        waits = sum(1 for d in self.decisions if d["action"] == "wait")
        works = sum(1 for d in self.decisions if d["action"] == "work")
        return works / (waits + works) if (waits + works) else 0.0

    def recovery_rate(self):
        """Among sessions that hit an API rate limit, the fraction that still
        ended in success (1.0 when no session was rate-limited)."""
        rl_tasks = {l["session_id"] for l in self.llms if l.get("rate_limited")}
        if rl_tasks:
            return sum(
                1 for sid in rl_tasks
                if self.task_map.get(sid, {}).get("outcome") == "success"
            ) / len(rl_tasks)
        return 1.0

    def per_seed(self):
        out = {}
        for seed in C.SEEDS:
            ts = [t for t in self.tasks if t["seed"] == seed]
            if not ts:
                out[str(seed)] = {"success_rate": None}
                continue
            out[str(seed)] = {
                "success_rate": sum(
                    1 for t in ts if t["outcome"] == "success"
                ) / len(ts)
            }
        return out

    # -- aggregate -----------------------------------------------------------
    def call_efficiency(self, ks=(4, 8, 12, 16, 20, 24, 28, 32)):
        """AUC-shaped version of pass_k: rewards solving with fewer tool
        calls, exactly as anytime_auc rewards earlier completion.
        C(k) = successes with <= k calls / N (step curve over the k grid);
        call_auc = integral / k_max."""
        curve = self.pass_k_curve(ks)
        kmax = ks[-1]
        auc = 0.0
        prev_k, prev_c = 0, 0.0
        for k, c in zip(ks, curve):
            auc += prev_c * (k - prev_k)
            prev_k, prev_c = k, c
        auc += prev_c * (kmax - prev_k)
        return auc / kmax if kmax else 0.0

    def compute(self, pass_k_ks=(4, 8, 12, 16, 20, 24, 28, 32)) -> dict:
        sr = self.success_rate()
        te = self.time_efficiency()
        ce = self.call_efficiency(pass_k_ks)
        total_tokens, cache_ratio, counts = self.costs()
        vec = {
            "success_rate": round(sr, 6),
            "time_efficiency": round(te, 6),
            "call_efficiency": round(ce, 6),
            "cache_hit_ratio": round(cache_ratio, 6) if cache_ratio is not None else None,
            "total_tokens": int(total_tokens),
        }
        diag = {
            "abandon_rate": round(self.abandon_rate(), 4)
            if self.abandon_rate() is not None else None,
            "slow_calls_mean": round(self.slow_calls_mean(), 4)
            if self.slow_calls_mean() is not None else None,
            "work_density": round(self.work_density(), 4),
            "recovery_rate": round(self.recovery_rate(), 4),
        }
        diag.update(self.time_decomposition())
        return {
            "vector": vec,
            "diagnostics": diag,
            "action_counts": counts,
            "pass_k_curve": [round(x, 6) for x in self.pass_k_curve(pass_k_ks)],
        }


def certificate(entries, num_tasks) -> dict:
    sessions = [e for e in entries if e.get("type") == "session"]
    decisions = [e for e in entries if e.get("type") == "decision"]
    llms = [e for e in entries if e.get("type") == "llm"]
    llm_ids = {l["llm_request_id"] for l in llms}

    # g3 trace integrity
    llm_backed = [d for d in decisions if d.get("llm_request_id")]
    resolved = sum(1 for d in llm_backed if d["llm_request_id"] in llm_ids)
    llm_resolution = (resolved / len(llm_backed)) if llm_backed else 1.0
    valid_tp = 0
    for d in decisions:
        tp = d.get("time_prediction") or {}
        try:
            ok = (
                len(tp) >= 2
                and all(k in C.BUCKET_LABELS for k in tp)
                and abs(sum(tp.values()) - 1.0) < 0.02
                and all(math.isfinite(float(p)) for p in tp.values())
            )
        except Exception:
            ok = False
        if ok:
            valid_tp += 1
    tp_fraction = (valid_tp / len(decisions)) if decisions else 1.0
    g3 = {
        "llm_backed_decisions": len(llm_backed),
        "llm_request_resolution": round(llm_resolution, 6),
        "valid_time_prediction_fraction": round(tp_fraction, 6),
        "trace_parseable": True,
    }

    # cheat detection
    cheat_sessions = [
        s for s in sessions if s.get("cheat_detection") and s.get("policy", "").startswith("cheat")
    ]
    detections = {}
    for s in cheat_sessions:
        sid = s["session_id"]
        ds = [d for d in decisions if d["session_id"] == sid]
        ts = [t for t in entries if t.get("type") == "task" and t.get("session_id") == sid]
        policy = s["policy"]
        if policy == "cheat:declared-long-wait-frequent-poll":
            waits = [d for d in ds if d["action"] == "wait"]
            polls = [d for d in ds if d["action"] == "poll"]
            mean_wait = statistics.mean(d["effect"]["slept_seconds"] for d in waits) if waits else 0
            intervals = [polls[i + 1]["t"] - polls[i]["t"] for i in range(len(polls) - 1)]
            mean_interval = statistics.mean(intervals) if intervals else float("inf")
            # invariant: polls much faster than the declared wait
            detections.setdefault(policy, []).append(
                mean_wait >= 4.0 and mean_interval <= max(0.2, mean_wait / 3.0)
            )
        elif policy == "cheat:constant-early-abandon":
            task = ts[0] if ts else None
            flagged = False
            if task:
                abandons = [d for d in ds if d["action"] == "abandon"]
                if abandons:
                    elapsed = abandons[0]["t"] - task["started_at"]
                    budget = task["budget"]["time_physical_seconds"]
                    flagged = elapsed <= 0.1 * budget
            detections.setdefault(policy, []).append(flagged)
        elif policy == "cheat:poll-bombing":
            polls = [d for d in ds if d["action"] == "poll"]
            if len(polls) >= 2:
                span = polls[-1]["t"] - polls[0]["t"]
                rate = (len(polls) - 1) / span if span > 0 else float("inf")
            else:
                rate = 0.0
            detections.setdefault(policy, []).append(rate >= 5.0)
    cheat_summary = {
        name: {
            "sessions": len(flags),
            "flagged": sum(flags),
            "detection_power": round(sum(flags) / len(flags), 4) if flags else None,
        }
        for name, flags in detections.items()
    }

    # IRT
    cond_tasks = {}
    for t in entries:
        if t.get("type") == "task" and not t.get("cheat_detection"):
            cond_tasks.setdefault(t["condition"], {}).setdefault(t["task_id"], []).append(t)
    task_ids = sorted({t["task_id"] for t in entries if t.get("type") == "task" and not t.get("cheat_detection")})
    irt_rows = []
    for tid in task_ids:
        successes = {}
        for c, per_task in cond_tasks.items():
            ts = per_task.get(tid, [])
            if ts:
                successes[c] = statistics.mean(
                    [1.0 if t["outcome"] == "success" else 0.0 for t in ts]
                )
        mean_s = statistics.mean(successes.values()) if successes else 0.0
        disc = (max(successes.values()) - min(successes.values())) if successes else 0.0
        irt_rows.append(
            {"task_id": tid, "difficulty": round(1.0 - mean_s, 4), "discrimination": round(disc, 4)}
        )
    irt = {"per_task": irt_rows}

    # reproducibility
    repro = {}
    for cond in C.CONDITIONS:
        per_task = {}
        for seed in C.SEEDS:
            sids = {
                s["session_id"] for s in sessions
                if s.get("condition") == cond and s.get("seed") == seed and not s.get("cheat_detection")
            }
            for s in sessions:
                if s["session_id"] in sids:
                    per_task.setdefault(s["instance_id"], {})[seed] = [
                        d["action"] for d in decisions if d["session_id"] == s["session_id"]
                    ]
        n_match = sum(
            1 for tid, by_seed in per_task.items()
            if by_seed.get(0) == by_seed.get(1)
        )
        repro[cond] = {
            "exact_action_sequence_match_across_seeds": n_match,
            "comparable_tasks": len(per_task),
        }
    return {"g3_trace_integrity": g3, "cheat_detection": cheat_summary, "irt": irt, "reproducibility": repro}


def protocol_section(manifest) -> dict:
    return {
        "benchmark": "LongWaitBench v1",
        "manifest": {
            "path": str(C.DEFAULT_MANIFEST_PATH),
            "manifest_version": manifest.get("manifest_version"),
            "num_instances": len(manifest["instances"]),
            "sources": sorted({i["source"] for i in manifest["instances"]}),
            "extraction_log": str(C.EXTRACTION_LOG_PATH),
        },
        "conditions": [
            {
                "name": "LongCall-v1",
                "kind": "llm",
                "description": (
                    "Real DeepSeek agent loop (deepseek-v4-pro via "
                    "https://api.deepseek.com/v1, recording client). Every "
                    "decision is a real API call; the LLM sees task_spec + "
                    "tool docs + service policy + its own history."
                ),
            },
            {
                "name": "ExponentialBackoff",
                "kind": "synthetic",
                "description": (
                    "Submits one setup slow job (f3: submit_job(n_max); tau2: "
                    "first slow tool with empty args, fails fast at execution) "
                    "then polls with exponential backoff (base 1.0s). "
                    "Time_prediction: 0.9 on bucket of naive remaining "
                    "T_est - elapsed (T_est = declared*scale for policy, 3s "
                    "for compute), 0.1 on the next bucket."
                ),
            },
            {
                "name": "AggressivePoll",
                "kind": "synthetic",
                "description": (
                    "Same setup submission, then polls every 0.3s with no "
                    "interleaved work. Same naive time_prediction rule."
                ),
            },
            {
                "name": "WaitAndAnswer",
                "kind": "synthetic",
                "description": (
                    "Same setup submission, then pure wait until the naive "
                    "remaining time elapses, one poll, then final_answer. "
                    "Only f3 tasks are solvable this way (tau2 jobs fail "
                    "without task understanding)."
                ),
            },
        ],
        "wait_model": {
            "compression_scale": C.COMPRESSION_SCALE,
            "policy_wait_physical": "declared_minutes * 60 * COMPRESSION_SCALE seconds",
            "compute_wait": "real in-process numpy; declared_minutes = physical / (COMPRESSION_SCALE * 60)",
            "poll_semantics": "poll returns {status, counter}; result included only once completed; no push, no progress ratio",
            "true_completion": "completion of the FIRST submitted slow job (T anchor)",
        },
        "budget": {
            "per_instance": "manifest wait_config.budget {slow_calls, wall_minutes}",
            "physical_wall": "wall_minutes * 60 * COMPRESSION_SCALE seconds",
            "deployment_budget_scale": C.DEPLOYMENT_BUDGET_SCALE,
            "call_budget_per_task": (
                "slow_calls + MAX_DECISIONS_PER_TASK (slow-call budget plus the "
                "decision-loop budget: every decision turn costs one call slot; "
                "budget_survival_calls counts poll+work+LLM calls against "
                "0.5 * call_budget)"
            ),
        },
        "time_prediction_buckets": [f"{lo}-{hi}" if hi != float("inf") else f"{lo}+" for _, lo, hi in C.TIME_BUCKETS],
        "action_counts": "raw per-action tool-call counts (zero weights; the former action_cost virtual weights are removed)",
        "api": {"base": C.API_BASE, "model": C.API_MODEL,
                "thinking": "disabled (reasoning tokens off; decision-loop latency control)"},
        "verifier": {
            "tau2": "final DB hash == manifest expected_hash (db_hash_eq)",
            "f3": "|final_answer.estimate - true_value| <= eps (tolerance)",
            "hidden": "checks never shown to the agent",
        },
        "behavior_metrics": {
            "work_density": "work / (wait + work) decision counts",
            "recovery_rate": "success rate of rate-limited sessions (1.0 if none)",
        },
        "limitations": [
            "v1 waits are declared policy latencies with compressed physical sleep (trace records both)",
            "compute-type waits (F3) are real in-process numpy",
            "no stall/dead event mechanics in v1",
            "SWE-bench-Lite/AppWorld sources deferred to v2",
            "synthetic controls cannot construct tau2 tool arguments; their tau2 sessions end in failed-job/abandon and are expected to score 0 on tau2 tasks (design: WaitAndAnswer solvable only on F3)",
            "airline-7 has two slow tools; T anchors the first submitted job",
            "anytime_auc uses the metrics_spec formula on physical seconds (per-instance normalized variant reported in analysis)",
        ],
    }
