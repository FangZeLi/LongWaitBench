"""Unified environment facade: one wait layer over tau2 (policy waits)
and control-f3 (compute waits). The agent never sees this object — only
the sanitized view built by session.py.

v2 adds two task types:
- banking (external-async): slow tools are *discoverable* tools reached
  through call_discoverable_agent_tool; submit/order jobs complete in the
  background and their records appear in the domain DB only at completion.
- telecom (network-async): solo environment; slow network/device operations
  take effect in device state only at job completion; judged by the task's
  env_assertions at final_answer time.
"""
import json
import time

from . import constants as C
from .domain_envs import (
    build_banking_environment,
    build_telecom_environment,
    call_domain_tool,
    run_telecom_assertions,
)
from .f3_env import F3Environment
from .job_registry import JobRegistry
from .tau2_env import (
    build_tau2_environment,
    build_tau2_environment_from_db,
    call_tau2_tool,
    env_policy_text,
)


class UnifiedEnv:
    def __init__(self, instance: dict, seed: int):
        self.instance = instance
        self.seed = seed
        self.source = instance["source"]
        self.domain = instance["domain"]
        self.budget_slow_calls = int(instance["wait_config"]["budget"]["slow_calls"])
        self.wall_minutes = float(instance["wait_config"]["budget"]["wall_minutes"])
        self.slow_calls_left = self.budget_slow_calls
        self.registry = JobRegistry()
        self.submit_index = 0
        # handle -> (tool_name, arguments) for gating checks
        self.submit_args = {}
        self.submit_raw_args = {}
        self.submit_outer = {}
        self.gating = instance.get("gating") or {}
        # F2 rate limits: tool_name -> {"cooldown_minutes": X}; cooldown starts
        # when each run ENDS (fast call: call end; slow call: job completion).
        self.rate_limits = instance.get("rate_limits") or {}
        self._cd_until = {}  # tool_name -> epoch sec when cooldown expires
        self._cd_minutes = {
            name: float(rl["cooldown_minutes"])
            for name, rl in self.rate_limits.items()
        }
        self.rejection_count = 0
        self._tau2_env = None

        if self.source == "control-f3":
            self.f3 = F3Environment(instance, seed)
            self.slow_waits = {"submit_job": {"type": "compute"}}
            self.policy_text = ""
            self.tool_docs = [
                {
                    "name": "submit_job",
                    "description": (
                        "Submit an estimation job. Draws n_samples independent "
                        "standard-normal vectors and returns the Monte Carlo mean "
                        "of the private function plus the real compute time. "
                        "SLOW (compute-type): the job runs real computation; "
                        "results are available only through poll()."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["mc_mean"],
                                "description": "estimation method",
                            },
                            "n_samples": {
                                "type": "integer",
                                "description": f"number of samples, <= {self.f3.n_max}",
                            },
                        },
                        "required": ["method", "n_samples"],
                    },
                    "slow": True,
                    "wait": {"type": "compute"},
                }
            ]
        elif self.domain == "banking":
            (
                self._tau2_env,
                self.tool_docs,
                self.slow_waits,
                self.policy_text,
            ) = build_banking_environment(instance)
            self.f3 = None
        elif self.domain == "telecom":
            (
                self._tau2_env,
                self.tool_docs,
                self.slow_waits,
                self.policy_text,
            ) = build_telecom_environment(instance)
            self.f3 = None
        else:
            self._tau2_env, self.tool_docs, self.slow_waits = build_tau2_environment(
                instance
            )
            self.policy_text = env_policy_text(self._tau2_env)
            self.f3 = None

        if self.rate_limits:
            lines = []
            for name, rl in self.rate_limits.items():
                note = rl.get("note")
                head = f"- {name}: {note} " if note else f"- {name}: "
                lines.append(
                    head + f"cooldown {rl['cooldown_minutes']} declared minutes "
                    "after each run ends; calling it during the cooldown returns "
                    "{rejected: true, retry_after: <remaining declared minutes>}."
                )
            rl_text = "Rate limits:\n" + "\n".join(lines)
            self.policy_text = (
                (self.policy_text + "\n\n" if self.policy_text else "") + rl_text
            )

    # --- F2 rate limits -----------------------------------------------------
    def _rate_limit_key(self, tool_name: str, arguments: dict):
        """Resolve the rate-limit key for an incoming call.

        Banking's fast tools live inside call_discoverable_agent_tool; the
        manifest keys banking rate limits by the INNER tool name. Everything
        else is keyed by the outer tool name.
        """
        if (self.domain == "banking"
                and tool_name == "call_discoverable_agent_tool"):
            inner = arguments.get("agent_tool_name") or ""
            if inner in self.rate_limits:
                return inner
        if tool_name in self.rate_limits:
            return tool_name
        return None

    def _cooldown_check(self, tool_name: str):
        """Return (blocked, retry_after_declared_minutes)."""
        until = self._cd_until.get(tool_name, 0.0)
        if time.time() < until:
            rem = (until - time.time()) / (60.0 * C.COMPRESSION_SCALE)
            return True, rem
        return False, 0.0

    def _start_cooldown(self, tool_name: str, from_ts: float = None):
        cd = self._cd_minutes.get(tool_name)
        if cd is None:
            return
        ts = from_ts if from_ts is not None else time.time()
        self._cd_until[tool_name] = ts + cd * 60.0 * C.COMPRESSION_SCALE

    # --- queries -----------------------------------------------------------
    def is_slow(self, tool_name: str) -> bool:
        return tool_name in self.slow_waits

    def resolve_call(self, tool_name: str, arguments: dict) -> dict:
        """Classify an incoming domain-tool call into slow/fast.

        For banking, slow tools are discoverable inner tools: a call to
        call_discoverable_agent_tool whose inner agent_tool_name is a slow
        tool is routed to the async submit path.
        """
        if self.domain == "banking" and tool_name == "call_discoverable_agent_tool":
            inner = arguments.get("agent_tool_name") or ""
            if inner in self.slow_waits:
                return {"kind": "slow", "slow_name": inner,
                        "outer_name": tool_name}
        if tool_name in self.slow_waits:
            return {"kind": "slow", "slow_name": tool_name, "outer_name": None}
        return {"kind": "fast"}

    def is_compute(self, tool_name: str) -> bool:
        return self.slow_waits.get(tool_name, {}).get("type") == "compute"

    def declared_minutes(self, tool_name: str):
        w = self.slow_waits.get(tool_name, {})
        if w.get("type") == "policy":
            return float(w.get("declared_minutes"))
        return None

    def open_jobs(self) -> list:
        return self.registry.open_jobs()

    def get_db_hash(self):
        if self._tau2_env is None:
            return None
        return self._tau2_env.get_db_hash()

    def verify(self, estimate) -> bool:
        """Deterministic hidden verifier from the manifest checks."""
        if self.source == "control-f3":
            return self.f3.check_answer(estimate)
        for check in self.instance["verifier"]["checks"]:
            if check["kind"] == "db_hash_eq":
                return self.get_db_hash() == check["expected_hash"]
            if check["kind"] == "env_assertions":
                return run_telecom_assertions(
                    self._tau2_env, check.get("assertions") or []
                )
        return False

    # --- actions -----------------------------------------------------------
    def call_fast(self, tool_name: str, arguments: dict) -> dict:
        rl_key = self._rate_limit_key(tool_name, arguments)
        if rl_key is not None:
            blocked, rem = self._cooldown_check(rl_key)
            if blocked:
                self.rejection_count += 1
                return {
                    "ok": False,
                    "error": True,
                    "rejected": True,
                    "retry_after": round(rem, 3),
                    "result": (
                        f"Error: {rl_key} is rate-limited (cooldown active, "
                        f"{rem:.1f} declared minutes remaining). Wait for the "
                        f"cooldown to expire, then retry."
                    ),
                }
        if self.source == "control-f3":
            return {"ok": False, "result": f"Error: unknown tool {tool_name}", "error": True}

        # banking: gate approve/deny on the corresponding slow submit job
        if self.domain == "banking" and tool_name == "call_discoverable_agent_tool":
            inner = arguments.get("agent_tool_name") or ""
            if inner in self.gating:
                gate = self.gating[inner]
                submit_tool = gate["submit_tool"]
                resource_key = gate.get("resource_key")
                try:
                    inner_args = json.loads(arguments.get("arguments") or "{}")
                except json.JSONDecodeError:
                    inner_args = {}
                resource = inner_args.get(resource_key)
                with self.registry._lock:
                    handles = list(self.registry._jobs.keys())
                for h in handles:
                    job = self.registry._jobs[h]
                    with job._lock:
                        running = job.status == "running"
                    tname, targs = self.submit_args.get(h, (None, None))
                    if running and tname == submit_tool and resource is not None \
                            and targs.get(resource_key) == resource:
                        return {
                            "ok": False,
                            "error": True,
                            "result": (
                                f"Error: {inner} cannot be applied yet — the "
                                f"{submit_tool} request for {resource_key} "
                                f"'{resource}' is still being processed by the "
                                f"review team (job {h}). Wait for that job to "
                                f"complete, then retry."
                            ),
                        }

        if self.domain in ("banking", "telecom"):
            res = call_domain_tool(self._tau2_env, tool_name, arguments)
        else:
            res = call_tau2_tool(self._tau2_env, tool_name, arguments)
        if rl_key is not None:
            self._start_cooldown(rl_key)
        return res

    def submit_slow(self, tool_name: str, arguments: dict, outer_name: str = None) -> dict:
        if tool_name not in self.slow_waits:
            return {"ok": False, "error": f"Error: {tool_name} is not a slow tool"}
        if tool_name in self.rate_limits:
            blocked, rem = self._cooldown_check(tool_name)
            if blocked:
                self.rejection_count += 1
                return {
                    "ok": False,
                    "error": True,
                    "rejected": True,
                    "retry_after": round(rem, 3),
                    "result": (
                        f"Error: {tool_name} is rate-limited (cooldown active, "
                        f"{rem:.1f} declared minutes remaining). Wait for the "
                        f"cooldown to expire, then retry."
                    ),
                }
            # single-slot semantics: one active run at a time for a
            # rate-limited tool (frequency limit = concurrency 1). A second
            # submission while a run is active gets a structured busy
            # rejection with retry_after = remaining run + cooldown.
            with self.registry._lock:
                jobs = list(self.registry._jobs.values())
            for job in jobs:
                with job._lock:
                    running = job.status == "running"
                    submitted_at = job.submitted_at
                    wait_seconds = getattr(job, "wait_seconds", 0.0)
                if running and job.tool_name == tool_name:
                    self.rejection_count += 1
                    rem_run = max(
                        0.0, submitted_at + wait_seconds - time.time()
                    ) / (60.0 * C.COMPRESSION_SCALE)
                    cd = self._cd_minutes.get(tool_name, 0.0)
                    retry = rem_run + cd
                    return {
                        "ok": False,
                        "error": True,
                        "rejected": True,
                        "busy": True,
                        "retry_after": round(retry, 3),
                        "result": (
                            f"Error: {tool_name} is rate-limited (busy: one "
                            f"request at a time; the current request finishes "
                            f"in ~{rem_run:.1f} declared minutes, then a "
                            f"{cd:g}-minute cooldown applies). Wait, then retry."
                        ),
                    }
        # no submission cap (user decision: no artificial limits; judgment
        # only at final_answer) — slow_calls_left is counted for reporting
        # only and can go negative if the agent exceeds the declared figure.
        self.slow_calls_left -= 1
        self.submit_index += 1

        if self.source == "control-f3":
            n_samples = arguments.get("n_samples")
            err = self.f3.validate_submit(n_samples)
            if err:
                return {"ok": False, "error": f"Error: {err}", "rejected": True}
            seed, idx = self.seed, self.submit_index

            def execute():
                out = self.f3.run_job(n_samples)
                return {
                    "ok": True,
                    "result": out,
                    "compute_seconds": out["compute_seconds"],
                }

            handle = self.registry.submit(
                tool_name=tool_name,
                kind="compute",
                declared_minutes=None,
                execute_fn=execute,
                wait_seconds=0.0,
            )
            return {"ok": True, "job_handle": handle, "slow": True, "wait": {"type": "compute"}}

        # tau2 policy job: sleep declared*scale then execute the real tool call
        declared = float(self.slow_waits[tool_name]["declared_minutes"])
        wait_seconds = declared * 60.0 * C.COMPRESSION_SCALE
        env = self._tau2_env
        domain = self.domain
        effective_name = outer_name or tool_name

        if outer_name is not None:
            # banking discoverable slow tool: execute through the outer call
            def execute():
                return call_domain_tool(env, outer_name, arguments)
        elif domain in ("banking", "telecom"):
            def execute():
                return call_domain_tool(env, tool_name, arguments)
        else:
            def execute():
                return call_tau2_tool(env, tool_name, arguments)

        # gating bookkeeping uses the inner arguments when routed through the
        # discoverable wrapper
        inner_args = arguments
        if outer_name is not None:
            try:
                inner_args = json.loads(arguments.get("arguments") or "{}")
            except json.JSONDecodeError:
                inner_args = {}

        def on_complete(job, tn=tool_name):
            if tn in self.rate_limits:
                self._start_cooldown(tn, from_ts=job.completed_at)

        handle = self.registry.submit(
            tool_name=tool_name,
            kind="policy",
            declared_minutes=declared,
            execute_fn=execute,
            wait_seconds=wait_seconds,
            on_complete=on_complete,
        )
        self.submit_args[handle] = (tool_name, inner_args)
        self.submit_raw_args[handle] = arguments
        self.submit_outer[handle] = outer_name
        return {
            "ok": True,
            "job_handle": handle,
            "slow": True,
            "wait": {"type": "policy", "declared_minutes": declared},
            "effective_tool": effective_name,
        }

    def poll(self, handle) -> dict:
        return self.registry.poll(handle)

    def wait_for(self, handle, timeout=None) -> dict:
        """Block until the job completes and return its state/result."""
        with self.registry._lock:
            job = self.registry._jobs.get(handle)
        if job is None:
            return {"status": "unknown_handle", "counter": 0, "changed": False,
                    "error": f"no job with handle {handle}"}
        return job.wait_for(timeout)

    def true_completion(self) -> dict:
        """Exogenous T of the task = completion of the first submitted job."""
        job = self.registry.first_job()
        if job is None:
            return None
        with job._lock:
            if job.completed_at is None:
                return None
            declared = job.declared_minutes
            physical = job.completed_at - job.submitted_at
        if job.kind == "compute" and declared is None:
            declared = physical / (C.COMPRESSION_SCALE * 60.0)
        return {
            "declared_minutes": round(declared, 6),
            "physical_seconds": round(physical, 6),
            "kind": job.kind,
            "status": job.status,
        }

    def shutdown(self):
        self.registry.shutdown()

    # -- state persistence (harness continuation across processes) -----------
    def save_state(self) -> dict:
        """Full resumable snapshot: domain DB(s), counters, open jobs."""
        state = {
            "domain": self.domain,
            "slow_calls_left": self.slow_calls_left,
            "submit_index": self.submit_index,
            "submit_args": {
                h: [a[0], a[1]] for h, a in self.submit_args.items()
            },
            "submit_raw_args": dict(self.submit_raw_args),
            "submit_outer": dict(self.submit_outer),
            "cd_until": dict(self._cd_until),
            "rejection_count": self.rejection_count,
            "jobs": [],
        }
        if self._tau2_env is not None and hasattr(self._tau2_env, "tools"):
            state["tau2_db"] = self._tau2_env.tools.db.model_dump(mode="json")
            if (getattr(self._tau2_env, "user_tools", None) is not None
                    and hasattr(self._tau2_env.user_tools, "db")):
                state["user_db"] = (
                    self._tau2_env.user_tools.db.model_dump(mode="json")
                )
        with self.registry._lock:
            for job in self.registry._jobs.values():
                with job._lock:
                    if job.status != "running":
                        continue
                    state["jobs"].append(
                        {
                            "handle": job.handle,
                            "tool_name": job.tool_name,
                            "kind": job.kind,
                            "declared_minutes": job.declared_minutes,
                            "submitted_at": job.submitted_at,
                            "arguments": self.submit_raw_args.get(job.handle),
                            "outer_name": self.submit_outer.get(job.handle),
                        }
                    )
        return state

    @classmethod
    def restore(cls, instance: dict, seed: int, state: dict):
        """Rebuild a UnifiedEnv from a save_state() snapshot and re-materialize
        open jobs with their original handles and remaining wait time."""
        self = cls.__new__(cls)
        self.instance = instance
        self.seed = seed
        self.source = instance["source"]
        self.domain = instance["domain"]
        self.budget_slow_calls = int(instance["wait_config"]["budget"]["slow_calls"])
        self.wall_minutes = float(instance["wait_config"]["budget"]["wall_minutes"])
        self.slow_calls_left = state.get("slow_calls_left", self.budget_slow_calls)
        self.submit_index = state.get("submit_index", 0)
        self.registry = JobRegistry()
        self.submit_args = {
            h: tuple(a) for h, a in (state.get("submit_args") or {}).items()
        }
        self.submit_raw_args = dict(state.get("submit_raw_args") or {})
        self.submit_outer = dict(state.get("submit_outer") or {})
        self.gating = instance.get("gating") or {}
        self.rate_limits = instance.get("rate_limits") or {}
        self._cd_until = dict(state.get("cd_until") or {})
        self._cd_minutes = {
            name: float(rl["cooldown_minutes"])
            for name, rl in self.rate_limits.items()
        }
        self.rejection_count = int(state.get("rejection_count", 0))
        self.f3 = None

        from .domain_envs import _parse_slow_waits, _tool_docs

        if self.domain == "banking":
            from tau2.domains.banking_knowledge.data_model import TransactionalDB
            from tau2.domains.banking_knowledge.environment import get_environment

            db = TransactionalDB.model_validate(state["tau2_db"])
            variant = instance["initial_state"].get("retrieval_variant", "bm25_grep")
            self._tau2_env = get_environment(db=db, retrieval_variant=variant)
            self.slow_waits = _parse_slow_waits(instance)
            self.tool_docs = _tool_docs(self._tau2_env.get_tools(), {})
            self.policy_text = self._tau2_env.get_policy()
        elif self.domain == "telecom":
            from tau2.domains.telecom.environment import get_environment

            env = get_environment(solo_mode=True)
            env.tools.db = type(env.tools.db).model_validate(state["tau2_db"])
            if state.get("user_db") is not None:
                env.user_tools.db = type(env.user_tools.db).model_validate(
                    state["user_db"]
                )
            self._tau2_env = env
            self.slow_waits = _parse_slow_waits(instance)
            tools = list(env.get_tools()) + list(env.get_user_tools())
            self.tool_docs = _tool_docs(tools, self.slow_waits)
            self.policy_text = env.get_policy()
        else:
            init = instance["initial_state"]
            from .tau2_env import _import_attr

            db_class = _import_attr(init["db_class"])
            db = db_class.model_validate(state["tau2_db"])
            self._tau2_env, self.tool_docs, self.slow_waits = (
                build_tau2_environment_from_db(instance, db)
            )
            self.policy_text = env_policy_text(self._tau2_env)

        # re-materialize open jobs with their ORIGINAL handles
        now = time.time()
        for j in state.get("jobs") or []:
            wait = 0.0
            if j["kind"] == "policy" and j.get("declared_minutes"):
                wait = j["declared_minutes"] * 60.0 * C.COMPRESSION_SCALE
            remaining = max(0.0, wait - (now - j["submitted_at"]))
            env = self._tau2_env
            domain = self.domain
            tool_name = j["tool_name"]
            outer_name = j.get("outer_name")
            args = j.get("arguments") or {}

            if outer_name is not None:
                def execute(o=outer_name, a=args):
                    return call_domain_tool(env, o, a)
            elif domain in ("banking", "telecom"):
                def execute(t=tool_name, a=args):
                    return call_domain_tool(env, t, a)
            else:
                def execute(t=tool_name, a=args):
                    return call_tau2_tool(env, t, a)

            def on_complete(job, tn=tool_name):
                if tn in self.rate_limits:
                    self._start_cooldown(tn, from_ts=job.completed_at)

            self.registry.submit(
                tool_name=tool_name,
                kind=j["kind"],
                declared_minutes=j.get("declared_minutes"),
                execute_fn=execute,
                wait_seconds=remaining,
                handle=j["handle"],
                on_complete=on_complete,
            )
        return self
