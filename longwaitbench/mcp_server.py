#!/usr/bin/env python3
"""LongWaitBench MCP server (stdio transport, or unix-socket daemon) — one
process per session.

Exposes the benchmark tool surface to any MCP-capable agent harness:
  - every domain tool (tau2 retail/airline or f3 submit_job) as its own tool,
  - plus the four control tools: wait / poll / final_answer / abandon.

Every tool call is recorded into canonical_trace.jsonl as a decision entry
(domain tools map to action "work"; control tools map 1:1). The server owns
the session: budgets, hard deadline, hidden verifier, and the task entry.

Environment (set by the session driver):
  LWB_INSTANCE_ID   manifest instance id (e.g. tau2-retail-0)
  LWB_SEED          int
  LWB_SESSION_ID    trace session id
  LWB_TRACE_PATH    trace jsonl path
  LWB_HARNESS       harness name (recorded in trace)
  LWB_DAEMON_SOCKET optional unix-socket path: daemon mode (multiple MCP
                    clients may connect sequentially, sharing one env state —
                    used for harness continuation loops)
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "pylibs"))
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT / "data" / "tau2bench" / "data"))
os.environ.setdefault("LC1_MANIFEST", str(ROOT / "data" / "manifest.json"))

from longwaitbench import constants as C  # noqa: E402
from longwaitbench.manifest_loader import load_manifest  # noqa: E402
from longwaitbench.trace import TraceWriter  # noqa: E402
from longwaitbench.unified_env import UnifiedEnv  # noqa: E402

SUPPORTED_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
LATEST = "2025-11-25"


class LwbMcpServer:
    def __init__(self):
        self.instance_id = os.environ["LWB_INSTANCE_ID"]
        self.seed = int(os.environ.get("LWB_SEED", "0"))
        self.session_id = os.environ.get(
            "LWB_SESSION_ID", f"mcp-{self.instance_id}-seed{self.seed}"
        )
        self.harness = os.environ.get("LWB_HARNESS", "mcp-harness")
        self.model = os.environ.get("LWB_MODEL", "unknown")
        trace_path = os.environ.get(
            "LWB_TRACE_PATH", str(ROOT / "results" / "canonical_trace.jsonl")
        )
        self.trace = TraceWriter(trace_path)
        manifest = load_manifest()
        inst = next(i for i in manifest["instances"] if i["id"] == self.instance_id)
        self.instance = inst
        self.state_path = os.environ.get("LWB_STATE_PATH")
        self.state = None
        if (os.environ.get("LWB_RESUME") == "1" and self.state_path
                and Path(self.state_path).exists()):
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        if self.state is not None:
            self.uenv = UnifiedEnv.restore(inst, self.seed, self.state)
            self.started = float(self.state.get("started") or time.time())
            self.decisions = int(self.state.get("decisions", 0))
            self.first_slow_submit_at = self.state.get("first_slow_submit_at")
            self._log(f"session {self.session_id} RESUMED from {self.state_path} "
                      f"(decisions={self.decisions})")
        else:
            self.uenv = UnifiedEnv(inst, self.seed)
            self.started = time.time()
            self.decisions = 0
            self.first_slow_submit_at = None
        # declared wall budget is informational only: no hard deadline, no
        # decision cap — sessions end via final_answer/abandon/EOF
        self.wall_minutes = self.uenv.wall_minutes
        self.finished = False
        self.outcome = None
        self.final_estimate = None
        self.final_error = None
        # stdout sink for MCP responses: sys.stdout in stdio mode, the active
        # client connection's stream in daemon mode
        self._out = sys.stdout
        if self.state is None:
            self.trace.write(
                {
                    "type": "session",
                    "session_id": self.session_id,
                    "condition": self.harness,
                    "policy": self.harness,
                    "model": self.model,
                    "seed": self.seed,
                    "instance_id": inst["id"],
                    "source": inst["source"],
                    "domain": inst["domain"],
                    "started_at": self.started,
                    "cheat_detection": False,
                }
            )
        self._log(f"session {self.session_id} started; no hard deadline; "
                  f"slow_calls_budget={self.uenv.budget_slow_calls} (declared)")
        self._log(f"declared wall budget {self.wall_minutes} min "
                  f"(informational, {self.wall_minutes * 60.0 * C.COMPRESSION_SCALE:.0f}s physical)")

    # -- helpers -------------------------------------------------------------
    def _log(self, msg: str):
        print(f"[lwb-mcp] {msg}", file=sys.stderr, flush=True)

    def _save_state(self):
        """Persist the resumable snapshot (env DBs, counters, open jobs)."""
        if not self.state_path:
            return
        try:
            state = self.uenv.save_state()
            state["started"] = self.started
            state["decisions"] = self.decisions
            state["first_slow_submit_at"] = self.first_slow_submit_at
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            os.replace(tmp, self.state_path)
        except Exception as e:
            self._log(f"state save failed: {e}")

    def _respond(self, req_id, result=None, error=None):
        out = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            out["error"] = {"code": -32603, "message": str(error)}
        else:
            out["result"] = result
        self._out.write(json.dumps(out, ensure_ascii=False) + "\n")
        self._out.flush()

    def _tool_result(self, text: str, is_error: bool = False) -> dict:
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    def _seconds_left(self) -> float:
        # informational: time since start vs declared wall budget
        return max(0.0, self.wall_minutes * 60.0 * C.COMPRESSION_SCALE
                   - (time.time() - self.started))

    def _decision(self, action: str, params: dict, effect: dict):
        self.decisions += 1
        self.trace.write(
            {
                "type": "decision",
                "task_id": self.instance_id,
                "session_id": self.session_id,
                "condition": self.harness,
                "seed": self.seed,
                "t": time.time(),
                "action": action,
                "params": params,
                "time_prediction": None,
                "virtual_cost": C.ACTION_VIRTUAL_COST.get(action, 0.0),
                "llm_request_id": None,
                "effect": effect,
                "invalid_fallback": False,
            }
        )
        self._save_state()

    def _finalize(self, outcome: str):
        if self.finished:
            return
        self.finished = True
        if outcome is None:
            # EOF without a final_answer: the task is judged ONLY at
            # final_answer time, so an unsubmitted session is simply failed
            outcome = "failed"
            self.final_error = "harness exited without submitting final_answer"
        self.outcome = outcome
        tc = self.uenv.true_completion()
        self.trace.write(
            {
                "type": "task",
                "task_id": self.instance_id,
                "session_id": self.session_id,
                "condition": self.harness,
                "seed": self.seed,
                "started_at": self.started,
                "ended_at": time.time(),
                "true_completion": tc,
                "first_slow_submit_at": self.first_slow_submit_at,
                "budget": {
                    # declared only; not enforced (no hard deadline in
                    # harness mode)
                    "time_physical_seconds": round(
                        self.uenv.wall_minutes * 60.0 * C.COMPRESSION_SCALE, 4
                    ),
                    "max_calls": self.uenv.budget_slow_calls,
                    "decision_budget": None,
                },
                "outcome": outcome,
                "decision_count": self.decisions,
                "invalid_decisions": 0,
                "slow_calls_used": (
                    self.uenv.budget_slow_calls - self.uenv.slow_calls_left
                ),
                "error": self.final_error,
                "cheat_detection": False,
            }
        )
        self._log(f"session finalized outcome={outcome} decisions={self.decisions}")

    # -- MCP -----------------------------------------------------------------
    def handle(self, req: dict):
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}
        if os.environ.get("LWB_DAEMON_DEBUG"):
            self._log(f"<- {method} id={req_id}")

        if method == "initialize":
            requested = params.get("protocolVersion", LATEST)
            version = requested if requested in SUPPORTED_VERSIONS else LATEST
            self._respond(
                req_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "longwaitbench", "version": "1.0.0"},
                },
            )
            return
        if method == "notifications/initialized":
            return  # notification: no response
        if method == "ping":
            self._respond(req_id, {})
            return
        if method == "tools/list":
            self._respond(req_id, {"tools": self._tool_schemas()})
            return
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            text, is_error = self._call_tool(name, args)
            self._respond(req_id, self._tool_result(text, is_error))
            return
        self._respond(req_id, error=f"unknown method {method}")

    # -- tool surface ---------------------------------------------------------
    def _tool_schemas(self) -> list:
        tools = []
        for td in self.uenv.tool_docs:
            desc = td.get("description", "")
            params = td.get("parameters") or {"type": "object", "properties": {}}
            if td.get("slow"):
                wait = td.get("wait", {})
                if wait.get("type") == "policy":
                    dm = wait.get("declared_minutes")
                    phys = round(dm * 60.0 * C.COMPRESSION_SCALE, 1)
                    desc = (
                        f"[SLOW, asynchronous] {desc}\nCalling this tool returns a "
                        f"job_handle immediately. The operation takes about {phys} "
                        f"physical seconds (declared {dm} minutes; 1 declared minute "
                        f"= {60 * C.COMPRESSION_SCALE} physical seconds). You can only "
                        f"learn completion by calling mcp__lwb__poll."
                    )
                else:
                    desc = (
                        f"[SLOW, asynchronous compute] {desc}\nCalling this tool "
                        f"returns a job_handle immediately; real computation runs in "
                        f"the background (typically a few seconds). You can only learn "
                        f"completion by calling mcp__lwb__poll."
                    )
            rl = self.uenv.rate_limits.get(td["name"])
            if rl:
                note = rl.get("note")
                if note:
                    desc += f"\n[RATE-LIMITED] {note}"
                desc += (
                    f"\n[RATE-LIMITED] cooldown {rl['cooldown_minutes']} declared "
                    "minutes after each run ends. Calling it during the cooldown "
                    "returns rejected:true with retry_after (remaining declared "
                    "minutes) — wait for the cooldown to expire, then retry."
                )
            tools.append(
                {
                    "name": td["name"],
                    "description": desc,
                    "inputSchema": params,
                }
            )
        tools.append(
            {
                "name": "wait",
                "description": (
                    "Do nothing for the given number of physical seconds "
                    "(any positive duration; sleeps for exactly that long). "
                    "Use it to let a pending slow job finish."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"],
                },
            }
        )
        tools.append(
            {
                "name": "wait_for",
                "description": (
                    "Block until the given job leaves the running state, then "
                    "return its result directly (same shape as poll: status, "
                    "counter, and the result once completed). No polling loop "
                    "needed — prefer this when you simply want the result."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_handle": {"type": "string"}},
                    "required": ["job_handle"],
                },
            }
        )
        tools.append(
            {
                "name": "poll",
                "description": (
                    "Check the status of one submitted job. Returns "
                    "{status: running|completed|failed, counter: int}; when "
                    "completed, includes the result. This is the ONLY way to learn "
                    "that a job finished and to receive its result."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_handle": {"type": "string"}},
                    "required": ["job_handle"],
                },
            }
        )
        tools.append(
            {
                "name": "final_answer",
                "description": (
                    "Finish the session by submitting your final answer. For "
                    "numeric estimation tasks, put the estimate in the estimate "
                    "field; otherwise leave it null. After this call the session "
                    "ends and no further tool calls have any effect."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "estimate": {"type": ["number", "null"]},
                    },
                    "required": ["text"],
                },
            }
        )
        tools.append(
            {
                "name": "abandon",
                "description": (
                    "Give up on the task. The session ends with outcome abandoned."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            }
        )
        return tools

    def _call_tool(self, name: str, args: dict):
        if self.finished:
            return "session already finished; no further tool calls have effect", True

        if name == "wait":
            try:
                secs = float(args.get("seconds", 1.0))
            except (TypeError, ValueError):
                return "wait requires a numeric 'seconds'", True
            if not 0 < secs <= 86400 * 7:
                return "seconds must be > 0 and <= 7 days", True
            t0 = time.time()
            time.sleep(secs)
            slept = time.time() - t0
            self._decision("wait", {"seconds": secs}, {"slept_seconds": round(slept, 4)})
            return (
                json.dumps({"slept_seconds": round(slept, 4)}),
                False,
            )

        if name == "wait_for":
            handle = args.get("job_handle")
            if not isinstance(handle, str) or not handle:
                return "wait_for requires a job_handle string", True
            t0 = time.time()
            res = self.uenv.wait_for(handle)
            blocked = time.time() - t0
            self._decision(
                "wait_for",
                {"job_handle": handle},
                {"wait_for": res, "blocked_seconds": round(blocked, 4)},
            )
            return json.dumps(res, ensure_ascii=False), res.get("status") == "unknown_handle"

        if name == "poll":
            handle = args.get("job_handle")
            if not isinstance(handle, str) or not handle:
                return "poll requires a job_handle string", True
            res = self.uenv.poll(handle)
            self._decision("poll", {"job_handle": handle}, {"poll": res})
            return json.dumps(res, ensure_ascii=False), res.get("status") == "unknown_handle"

        if name == "final_answer":
            estimate = args.get("estimate")
            text = str(args.get("text", ""))
            self.final_estimate = estimate
            self._decision(
                "final_answer",
                {"text": text, "estimate": estimate},
                {"final_answer": True},
            )
            try:
                ok = self.uenv.verify(estimate)
                self._finalize("success" if ok else "failed")
            except Exception as e:
                self._log(f"verify error: {e}")
                self._finalize("failed")
            return "answer recorded; session complete", False

        if name == "abandon":
            reason = str(args.get("reason", ""))
            self._decision("abandon", {"reason": reason}, {"abandon": True})
            self._finalize("abandoned")
            return "session abandoned", False

        # domain tool -> action "work"
        resolved = self.uenv.resolve_call(name, args)
        if resolved.get("kind") == "slow":
            slow_name = resolved["slow_name"]
            res = self.uenv.submit_slow(slow_name, args, outer_name=resolved.get("outer_name"))
            if not res.get("ok"):
                effect = {"submit": res}
                if res.get("rejected"):
                    effect["rejected"] = True
                    effect["retry_after"] = res.get("retry_after")
                self._decision("work", {"tool_name": name, "arguments": args}, effect)
                return res.get("result") or res.get("error", "tool rejected"), True
            if self.first_slow_submit_at is None:
                self.first_slow_submit_at = time.time()
            self._decision("work", {"tool_name": name, "arguments": args}, {"submit": res})
            return json.dumps({"job_handle": res["job_handle"]}), False
        res = self.uenv.call_fast(name, args)
        effect = {"tool": {"ok": res.get("ok"), "error": res.get("error")}}
        if res.get("rejected"):
            effect["tool"]["rejected"] = True
            effect["tool"]["retry_after"] = res.get("retry_after")
        self._decision(
            "work",
            {"tool_name": name, "arguments": args},
            effect,
        )
        return res.get("result", ""), res.get("error", False)

    # -- main loop -------------------------------------------------------------
    def run(self):
        sock_path = os.environ.get("LWB_DAEMON_SOCKET")
        if sock_path:
            self._run_daemon(sock_path)
        else:
            self._run_stdio()

    def _run_stdio(self):
        # read exactly one line at a time: iterating sys.stdin buffers 8KB
        # chunks and deadlocks while the client keeps the pipe open
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                self.handle(req)
            except Exception as e:
                self._log(f"request error: {e}")
        if not self.finished:
            if os.environ.get("LWB_EOF_FINALIZE") == "0":
                # harness-continuation mode: this process ends but the session
                # may continue in another process — park state, no task entry
                self._save_state()
                self._log("session parked (EOF, LWB_EOF_FINALIZE=0)")
            else:
                self._finalize(None)

    def _run_daemon(self, sock_path: str):
        import signal
        import socket

        path = Path(sock_path)
        if path.exists():
            path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(4)
        self._log(f"daemon listening on {sock_path}")

        def shutdown():
            if not self.finished:
                self._finalize(None)
            try:
                srv.close()
            except Exception:
                pass
            if path.exists():
                path.unlink()
            os._exit(0)

        signal.signal(signal.SIGTERM, lambda *_: shutdown())
        while True:
            conn, _ = srv.accept()
            self._log("daemon connection accepted")
            f = conn.makefile("rb")
            saved_out = self._out
            # text-mode writer: _respond writes str payloads (stdio mode uses
            # a text stdout, so the daemon's sink must be text too)
            self._out = conn.makefile("w", encoding="utf-8", newline="")
            try:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        req = json.loads(line.decode("utf-8"))
                    except Exception as e:
                        self._log(f"unparseable request: {e}")
                        continue
                    if req.get("method") == "lwb/shutdown":
                        shutdown()
                    try:
                        self.handle(req)
                    except Exception as e:
                        self._log(f"request error: {e}")
            except Exception as e:
                self._log(f"connection error: {e}")
            finally:
                self._out = saved_out
                try:
                    conn.close()
                except Exception:
                    pass


def main():
    server = LwbMcpServer()
    server.run()


if __name__ == "__main__":
    main()
