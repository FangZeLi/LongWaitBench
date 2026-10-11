#!/usr/bin/env python3
"""Run ONE LongWaitBench session with the DeepSeek Harness headless agent.

The harness process is the unit under test (harness + model). It interacts
with the benchmark environment exclusively through the MCP tools served by
longwaitbench/mcp_server.py (spawned by the harness's own MCP client).

Usage:
  python longwaitbench/run_dsh_session.py --instance tau2-retail-0 --seed 0 \
      --profile lwb [--trace results/harness_trace.jsonl] [--dsh /path/to/dsh]
"""
import argparse
import json
import os
import shutil
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

HARNESS = "dsh-headless"
MODEL = "deepseek-v4-pro"


def build_prompt(instance: dict, slow_budget: int, deadline_sec: float) -> str:
    spec = instance["task_spec"]
    tools_hint = (
        "You interact with the environment ONLY through tools named mcp__lwb__* "
        "(domain tools plus mcp__lwb__wait / mcp__lwb__wait_for / mcp__lwb__poll / "
        "mcp__lwb__final_answer / mcp__lwb__abandon — read their descriptions)."
    )
    sections = [
        "You are an agent solving ONE task inside the LongWaitBench benchmark environment.",
        "",
        "# Task",
        spec,
        "",
        "# Tool access",
        tools_hint,
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
            "Slow tools are asynchronous: calling one returns a job_handle immediately and "
            f"the operation finishes later (1 declared minute = {60 * C.COMPRESSION_SCALE:.0f} "
            "physical seconds; typical slow operations take tens of physical seconds). "
            "You can learn completion ONLY by calling mcp__lwb__poll on the handle; poll "
            "returns a status and an integer counter (no progress percentage). "
            "mcp__lwb__wait_for blocks until the job finishes and returns its result.",
        ]
    sections += [
        "",
        "# Budget",
        f"Declared budget (informational, not enforced): {deadline_sec:.0f} physical seconds "
        f"from session start. Slow-call budget: {slow_budget} (informational, "
        "not enforced — submissions are never rejected for count). Fast tools are free.",
        "",
        "# Finishing",
        "When the task's objective is met, call mcp__lwb__final_answer with your "
        "answer text (for numeric estimation tasks, put the estimate in the "
        '"estimate" field). If success is impossible within the remaining budget, '
        "call mcp__lwb__abandon.",
    ]
    return "\n".join(sections)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="lwb")
    ap.add_argument("--dsh", default=shutil.which("dsh") or "dsh")
    ap.add_argument("--trace", default=str(ROOT / "results" / "canonical_trace.jsonl"))
    args = ap.parse_args()

    manifest = load_manifest()
    inst = next(i for i in manifest["instances"] if i["id"] == args.instance)
    session_id = f"{HARNESS}-{inst['id']}-seed{args.seed}"
    workspace = ROOT / "results" / "sessions" / session_id
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    deadline_sec = (
        inst["wait_config"]["budget"]["wall_minutes"] * 60.0 * C.COMPRESSION_SCALE
    )
    prompt = build_prompt(
        inst, inst["wait_config"]["budget"]["slow_calls"], deadline_sec
    )
    (workspace / "task_prompt.txt").write_text(prompt, encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "DSH_HOME": str(ROOT / ".dsh-home"),
            "DSH_TOOLS_MODE": "native",
            "LWB_INSTANCE_ID": inst["id"],
            "LWB_SEED": str(args.seed),
            "LWB_SESSION_ID": session_id,
            "LWB_TRACE_PATH": args.trace,
            "LWB_HARNESS": HARNESS,
            "LWB_MODEL": MODEL,
            # continuation-on-text-stop (lwb-runner): how many follow-up
            # turns and what to say when the model stops with text only
            "LWB_CONTINUATIONS": os.environ.get("LWB_CONTINUATIONS", "3"),
            "LWB_CONTINUATION_TEXT": os.environ.get(
                "LWB_CONTINUATION_TEXT",
                "Continue. Your previous message was not delivered to anyone: "
                "there is NO live customer in this environment. Every fact you "
                "need about the customer is in the task description — use it as "
                "the source for any answer the customer 'would give', and "
                "answer your own questions from it. Keep working on the task "
                "until its objective is met, then call mcp__lwb__final_answer.",
            ),
        }
    )

    t0 = time.time()
    print(f"[driver] {session_id} launching: {args.dsh} --profile {args.profile}", flush=True)
    proc = subprocess.run(
        [args.dsh, "--profile", args.profile, prompt],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    stdout_tail = (proc.stdout or "").strip().splitlines()
    last_msg = stdout_tail[-1][:2000] if stdout_tail else ""
    print(f"[driver] {session_id} dsh exit={proc.returncode} elapsed={elapsed:.0f}s", flush=True)
    print(f"[driver] last agent message: {last_msg}", flush=True)
    if proc.stderr:
        err_tail = (proc.stderr or "").strip().splitlines()[-15:]
        print("[driver] dsh stderr tail:", flush=True)
        for line in err_tail:
            print(f"    {line[:300]}", flush=True)

    # inject real LLM usage from the persisted DSH session into the trace
    trace = TraceWriter(args.trace)
    try:
        from longwaitbench.dsh_session_usage import extract

        home = Path(env["DSH_HOME"])
        cand = sorted(
            home.glob("sessions/*/session-*/session.jsonl.zstd"),
            key=lambda p: p.stat().st_mtime,
        )
        fresh = [p for p in cand if p.stat().st_mtime >= t0]
        if fresh:
            usage = extract(fresh[-1].parent)
            trace.write(
                {
                    "type": "llm",
                    "llm_request_id": None,
                    "session_id": session_id,
                    "task_id": inst["id"],
                    "condition": HARNESS,
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0),
                    },
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                    "cache_read_tokens": usage.get("cache_read_tokens", 0),
                    "n_model_calls": usage.get("n_model_calls", 0),
                    "model": MODEL,
                    "cache_hit": (
                        bool(usage.get("cache_read_tokens", 0)) or None
                    ),
                }
            )
            print(
                f"[driver] usage: {usage.get('n_model_calls')} calls, "
                f"{usage.get('input_tokens')} in / {usage.get('output_tokens')} out / "
                f"{usage.get('reasoning_tokens')} reasoning tokens",
                flush=True,
            )
    except Exception as e:
        print(f"[driver] usage extraction failed: {e}", flush=True)

    # inject LLM generation (thinking) time from the persisted DSH session
    try:
        from longwaitbench.llm_timing import extract_dsh_think

        home = Path(env["DSH_HOME"])
        cand = sorted(
            home.glob("sessions/*/session-*/session.jsonl.zstd"),
            key=lambda p: p.stat().st_mtime,
        )
        fresh = [p for p in cand if p.stat().st_mtime >= t0]
        if fresh:
            # judged session window (epoch s -> ms) for clipping transcript
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
            think = extract_dsh_think(fresh[-1].parent, *clip)
            trace.write(
                {
                    "type": "llm_timing",
                    "session_id": session_id,
                    "task_id": inst["id"],
                    "condition": HARNESS,
                    "think_seconds": round(think, 3),
                    "source": "dsh-transcript-chunks",
                }
            )
            print(f"[driver] llm think time: {think:.1f}s", flush=True)
    except Exception as e:
        print(f"[driver] think extraction failed: {e}", flush=True)
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
                "error": f"dsh exit={proc.returncode}; no session finalization from MCP server",
            }
        )
    return 0 if (task_seen or proc.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
