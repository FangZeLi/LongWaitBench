#!/usr/bin/env python3
"""Run ONE LongWaitBench session with opencode (non-interactive `opencode run`).

The opencode agent (harness + model) is the unit under test. It interacts
with the benchmark environment through the MCP tools served by
longwaitbench/mcp_server.py. Native opencode tools (bash/edit/read/...) are
DISABLED in the per-session config so the agent cannot read the manifest,
snapshots, or the trace (oracle hygiene), mirroring the DSH profile.

Usage:
  python longwaitbench/run_opencode_session.py --instance tau2-retail-0 --seed 0 \
      --trace results/opencode_trace.jsonl
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
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

HARNESS = "opencode"
MODEL = "deepseek-v4-pro"
PY = "/data3/huangwenrui/wenrui_workdir/AutoResearchClaw/.venv/bin/python3"
MCP_SERVER = str(ROOT / "longwaitbench" / "mcp_server.py")

NATIVE_TOOLS_DISABLED = {
    "bash": False, "edit": False, "write": False, "patch": False, "read": False,
    "glob": False, "grep": False, "list": False, "ls": False, "webfetch": False,
    "todowrite": False, "task": False, "question": False, "plan": False,
    "snapshot": False,
}


def build_config(instance_id: str, seed: int, session_id: str, trace_path: str,
                 state_path: str, resume: bool, eof_finalize: bool) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"lwbdeepseek/{MODEL}",
        "provider": {
            "lwbdeepseek": {
                "options": {
                    "baseURL": "https://api.deepseek.com/v1",
                    # key comes from the process env at runtime; never written to a file
                    "apiKey": "{env:DEEPSEEK_API_KEY}",
                },
                "models": {
                    MODEL: {
                        "name": "DeepSeek V4 Pro",
                        # opencode REQUIRES a numeric limit.output; we set it
                        # far above any API-side cap so the benchmark itself
                        # never truncates agent output (v2: 3 sessions died
                        # when reasoning hit the old 16384 cap)
                        "limit": {"context": 128000, "output": 131072},
                    }
                },
            }
        },
        "mcp": {
            "lwb": {
                "type": "local",
                # fresh server per opencode process; continuation runs
                # (--session) spawn one with LWB_RESUME=1, which reloads the
                # environment state written by the previous server process
                "command": [PY, MCP_SERVER],
                "environment": {
                    "LWB_INSTANCE_ID": instance_id,
                    "LWB_SEED": str(seed),
                    "LWB_SESSION_ID": session_id,
                    "LWB_TRACE_PATH": trace_path,
                    "LWB_HARNESS": HARNESS,
                    "LWB_MODEL": MODEL,
                    "LWB_STATE_PATH": state_path,
                    "LWB_RESUME": "1" if resume else "0",
                    "LWB_EOF_FINALIZE": "1" if eof_finalize else "0",
                    "TAU2_DATA_DIR": os.environ["TAU2_DATA_DIR"],
                    "LC1_MANIFEST": os.environ["LC1_MANIFEST"],
                },
                "enabled": True,
                # MCP tool-call timeout, ms. Default 30min is far above the
                # longest blocking lwb_wait_for in the current short-wait pool
                # (max ~135s physical); raise via env for long-wait tiers.
                "timeout": int(os.environ.get(
                    "LWB_OPENCODE_MCP_TIMEOUT_MS", "1800000")),
            }
        },
        "tools": NATIVE_TOOLS_DISABLED,
        "permission": {
            "edit": "deny",
            "bash": "deny",
            # benchmark tasks legitimately repeat tool calls (e.g. submit_job
            # several times and average); a real user would approve them —
            # allow instead of opencode's default ask/auto-reject in run mode
            "doom_loop": "allow",
        },
        "instructions": [],
    }


def build_prompt(instance: dict, deadline_sec: float) -> str:
    spec = instance["task_spec"]
    sections = [
        "You are an agent solving ONE task inside the LongWaitBench benchmark environment.",
        "",
        "# Task",
        spec,
        "",
        "# Tool access",
        "You interact with the environment ONLY through the MCP tools named lwb_* "
        "(mcp__lwb__*): domain tools plus lwb_wait / lwb_wait_for / lwb_poll / "
        "lwb_final_answer / lwb_abandon. Read their descriptions before acting. "
        "You have NO other tools.",
    ]
    if instance.get("prompt_policy"):
        sections += ["", "# Policy", str(instance["prompt_policy"])]
    rate_limits = instance.get("rate_limits") or {}
    if rate_limits:
        lines = ["Some tools are rate-limited (cooldown):"]
        for name, rl in rate_limits.items():
            note = rl.get("note")
            head = f"- {name}: {note} " if note else f"- {name}: "
            lines.append(
                head + f"cooldown {rl['cooldown_minutes']} declared minutes "
                "after each run ends; calling it during the cooldown returns "
                "{rejected: true, retry_after: <remaining declared minutes>}. "
                "Wait for the cooldown to expire before retrying."
            )
        sections += ["", "# Rate limits", "\n".join(lines)]
    if instance.get("prompt_note"):
        sections += ["", "# Environment notes", str(instance["prompt_note"])]
    has_slow = any(t.get("slow") for t in instance.get("tools", []))
    if has_slow:
        sections += [
            "",
            "# Wait model",
            "Slow tools are asynchronous: calling one returns a job_handle immediately and the "
            f"operation finishes later (1 declared minute = {60 * C.COMPRESSION_SCALE:.0f} physical "
            "seconds; typical slow operations take tens of physical seconds). You learn completion "
            "ONLY by calling lwb_poll on the handle; poll returns a status and an integer counter "
            "(no progress percentage). lwb_wait_for blocks until the job finishes and returns its "
            "result.",
        ]
    sections += [
        "",
        "# Budget",
        f"Declared budget (informational, not enforced): {deadline_sec:.0f} physical seconds "
        f"from session start. Slow-call budget: "
        f"{instance['wait_config']['budget']['slow_calls']} (informational, "
        "not enforced — submissions are never rejected for count). Fast tools are free.",
        "",
        "# Finishing",
        "When the task's objective is met, call lwb_final_answer with your answer text (for "
        'numeric estimation tasks, put the estimate in the "estimate" field). If success is '
        "impossible within the remaining budget, call lwb_abandon.",
    ]
    return "\n".join(sections)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", default=str(ROOT / "results" / "opencode_trace.jsonl"))
    ap.add_argument("--opencode", default=shutil.which("opencode") or "opencode")
    args = ap.parse_args()

    manifest = load_manifest()
    inst = next(i for i in manifest["instances"] if i["id"] == args.instance)
    session_id = f"{HARNESS}-{inst['id']}-seed{args.seed}"
    workspace = ROOT / "results" / "sessions" / session_id
    state_path = str(workspace / "lwb_state.json")
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    deadline_sec = (
        inst["wait_config"]["budget"]["wall_minutes"] * 60.0 * C.COMPRESSION_SCALE
    )
    prompt = build_prompt(inst, deadline_sec)
    (workspace / "task_prompt.txt").write_text(prompt, encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "OPENCODE_CONFIG": str(workspace / "opencode.json"),
            "XDG_DATA_HOME": str(workspace / ".xdg-data"),
            "XDG_CONFIG_HOME": str(workspace / ".xdg-config"),
            "XDG_CACHE_HOME": str(workspace / ".xdg-cache"),
        }
    )

    t0 = time.time()

    def session_finished() -> bool:
        tp = Path(args.trace)
        if not tp.exists():
            return False
        for line in tp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if (
                e.get("type") == "decision"
                and e.get("session_id") == session_id
                and e.get("action") in ("final_answer", "abandon")
            ):
                return True
        return False

    def latest_session_id() -> str | None:
        db_path = workspace / ".xdg-data" / "opencode" / "opencode.db"
        if not db_path.exists():
            return None
        import sqlite3

        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT id FROM session ORDER BY time_created DESC LIMIT 1"
        ).fetchone()
        con.close()
        return row[0] if row else None

    continuation_limit = int(os.environ.get("LWB_CONTINUATIONS", "3"))
    continuation_text = (
        "Continue. Your previous message was not delivered to anyone: there is "
        "NO live customer in this environment. Every fact you need about the "
        "customer is in the task description — use it as the source for any "
        "answer the customer 'would give', and answer your own questions from "
        "it. Keep working on the task until its objective is met, then call "
        "lwb_final_answer."
    )
    msg = prompt
    for attempt in range(continuation_limit + 1):
        cmd = [args.opencode, "run"]
        if attempt > 0:
            sid = latest_session_id()
            if not sid:
                print("[driver] WARNING: no opencode session id found; "
                      "aborting continuation loop", flush=True)
                break
            cmd += ["--session", sid]
            msg = continuation_text
        # each opencode process spawns a fresh MCP server; continuation runs
        # resume the environment state written by the previous one. Only the
        # LAST attempt may finalize the session at EOF (no task entry spam).
        config = build_config(inst["id"], args.seed, session_id, args.trace,
                              state_path, resume=(attempt > 0),
                              eof_finalize=(attempt == continuation_limit))
        (workspace / "opencode.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[driver] {session_id} opencode run #{attempt} "
              f"(resume={attempt > 0}, msg={msg[:60]!r}...)", flush=True)
        proc = subprocess.run(
            cmd + [msg],
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
        )
        stdout_tail = (proc.stdout or "").strip().splitlines()
        last_msg = stdout_tail[-1][:2000] if stdout_tail else ""
        print(f"[driver] {session_id} opencode run #{attempt} "
              f"exit={proc.returncode}", flush=True)
        print(f"[driver] last agent message: {last_msg}", flush=True)
        if proc.stderr:
            err_tail = (proc.stderr or "").strip().splitlines()[-10:]
            for line in err_tail:
                print(f"    {line[:300]}", flush=True)
        if session_finished():
            break
    else:
        print(f"[driver] {session_id} continuation limit "
              f"({continuation_limit}) reached", flush=True)
    elapsed = time.time() - t0
    print(f"[driver] {session_id} done (elapsed={elapsed:.0f}s)", flush=True)

    trace = TraceWriter(args.trace)
    # inject LLM generation (thinking) time from the opencode session db
    try:
        from longwaitbench.llm_timing import extract_opencode_think

        db_path = workspace / ".xdg-data" / "opencode" / "opencode.db"
        # judged session window (epoch s -> ms) for clipping db part times
        s_start, s_end = None, None
        tp = Path(args.trace)
        if tp.exists():
            for line in tp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if (e.get("type") == "session"
                        and e.get("session_id") == session_id):
                    s_start = e.get("started_at")
                if (e.get("type") == "task"
                        and e.get("session_id") == session_id):
                    s_end = e.get("ended_at")
        clip = (
            s_start * 1000 if s_start is not None else None,
            s_end * 1000 if s_end is not None else None,
        )
        think = extract_opencode_think(db_path, *clip)
        trace.write(
            {
                "type": "llm_timing",
                "session_id": session_id,
                "task_id": inst["id"],
                "condition": HARNESS,
                "think_seconds": round(think, 3),
                "source": "opencode-db-parts",
            }
        )
        print(f"[driver] llm think time: {think:.1f}s", flush=True)
    except Exception as e:
        print(f"[driver] think extraction failed: {e}", flush=True)

    # inject real LLM usage from the opencode session sqlite
    try:
        import sqlite3

        db_path = workspace / ".xdg-data" / "opencode" / "opencode.db"
        if db_path.exists():
            con = sqlite3.connect(str(db_path))
            row = con.execute(
                "SELECT model, tokens_input, tokens_output, tokens_reasoning, "
                "tokens_cache_read, tokens_cache_write FROM session ORDER BY id DESC LIMIT 1"
            ).fetchone()
            con.close()
            if row:
                model_info, t_in, t_out, t_reason, t_cache_r, t_cache_w = row
                trace.write(
                    {
                        "type": "llm",
                        "llm_request_id": None,
                        "session_id": session_id,
                        "task_id": inst["id"],
                        "condition": HARNESS,
                        "usage": {
                            "prompt_tokens": int(t_in or 0),
                            "completion_tokens": int(t_out or 0),
                            "total_tokens": int(t_in or 0) + int(t_out or 0),
                        },
                        "reasoning_tokens": int(t_reason or 0),
                        "cache_read_tokens": int(t_cache_r or 0),
                        "cache_write_tokens": int(t_cache_w or 0),
                        "n_model_calls": None,
                        "model": MODEL,
                        "cache_hit": bool(t_cache_r) or None,
                    }
                )
                print(
                    f"[driver] usage: in={t_in} out={t_out} reasoning={t_reason} "
                    f"cache_read={t_cache_r}",
                    flush=True,
                )
    except Exception as e:
        print(f"[driver] usage extraction failed: {e}", flush=True)

    task_seen = False
    trace_path = Path(args.trace)
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "task" and e.get("session_id") == session_id:
                task_seen = True
                print(
                    f"[driver] task entry: outcome={e.get('outcome')} "
                    f"decisions={e.get('decision_count')} tc={e.get('true_completion')}",
                    flush=True,
                )
    if not task_seen:
        print("[driver] WARNING: no task entry from MCP server; writing fallback failed entry", flush=True)
        trace.write(
            {
                "type": "task",
                "task_id": inst["id"],
                "session_id": session_id,
                "condition": HARNESS,
                "seed": args.seed,
                "started_at": t0,
                "ended_at": time.time(),
                "true_completion": None,
                "budget": {
                    "time_physical_seconds": round(deadline_sec, 4),
                    "max_calls": inst["wait_config"]["budget"]["slow_calls"],
                    "decision_budget": C.MAX_DECISIONS_PER_TASK,
                },
                "outcome": "failed",
                "decision_count": 0,
                "error": f"opencode exit={proc.returncode}; no session finalization from MCP server",
            }
        )
    return 0 if (task_seen or proc.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
