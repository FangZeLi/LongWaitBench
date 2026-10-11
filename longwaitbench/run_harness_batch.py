#!/usr/bin/env python3
"""Batch runner: all manifest instances x seeds with the DSH headless harness.

Usage:
  python longwaitbench/run_harness_batch.py [--workers 2] \
      [--trace results/harness_trace.jsonl] [--only tau2-retail-0]
Resume-safe: sessions already recorded in the trace are skipped.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from longwaitbench import constants as C  # noqa: E402
from longwaitbench.manifest_loader import load_manifest  # noqa: E402

DRIVER = ROOT / "longwaitbench" / "run_dsh_session.py"
PY = "/data3/huangwenrui/wenrui_workdir/AutoResearchClaw/.venv/bin/python3"


def existing_sessions(trace_path: Path) -> set:
    if not trace_path.exists():
        return set()
    ids = set()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "session":
            ids.add(e.get("session_id"))
    return ids


def run_one(instance_id: str, seed: int, trace_path: str):
    sid = f"dsh-headless-{instance_id}-seed{seed}"
    print(f"[batch] {sid} start", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            PY,
            str(DRIVER),
            "--instance",
            instance_id,
            "--seed",
            str(seed),
            "--profile",
            "lwb",
            "--trace",
            trace_path,
        ],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip().splitlines()
    tail = "\n".join(out[-6:])
    print(f"[batch] {sid} done {time.time() - t0:.0f}s exit={proc.returncode}\n{tail}", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--trace", default=str(ROOT / "results" / "harness_trace.jsonl"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    manifest = load_manifest()
    jobs = []
    for inst in manifest["instances"]:
        if inst.get("benchmark") is False:
            continue  # mechanism-test fixtures are not benchmark content
        if inst.get("f2_base"):
            continue  # F2-mounted base tasks are not run (F2 = v3-f2-* only)
        if args.only and inst["id"] != args.only:
            continue
        for seed in C.SEEDS:
            sid = f"dsh-headless-{inst['id']}-seed{seed}"
            jobs.append((inst["id"], seed, sid))
    done = existing_sessions(Path(args.trace))
    jobs = [j for j in jobs if j[2] not in done]
    print(f"[batch] {len(jobs)} sessions to run with {args.workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, i, s, args.trace) for i, s, _ in jobs]
        codes = [f.result() for f in futs]
    n_ok = sum(1 for c in codes if c == 0)
    print(f"[batch] DONE {n_ok}/{len(codes)} sessions exited 0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
