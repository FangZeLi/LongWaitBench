#!/usr/bin/env python3
"""Extract LLM usage from a persisted DSH headless session (session.jsonl.zstd).

Usage: python longwaitbench/dsh_session_usage.py <session-dir>
Prints JSON: {input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,
n_model_calls, model, duration_sec}.
"""
import json
import sys
from pathlib import Path

try:
    from compression import zstd  # python 3.14 stdlib
except ImportError:
    zstd = None


def extract(session_dir: Path) -> dict:
    files = list(Path(session_dir).glob("session-*/session.jsonl.zstd"))
    if not files:
        files = list(Path(session_dir).glob("**/session.jsonl.zstd"))
    if not files:
        raise FileNotFoundError(f"no session.jsonl.zstd under {session_dir}")
    path = files[0]
    raw = path.read_bytes()
    if zstd is not None:
        text = zstd.decompress(raw).decode("utf-8", errors="replace")
    else:
        import subprocess

        text = subprocess.run(
            ["zstd", "-dc", str(path)], capture_output=True, text=True
        ).stdout

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "n_model_calls": 0,
    }
    model = None
    first_ts = None
    last_ts = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "assistant/chunk":
            data = e.get("data") or {}
            chunk = data.get("chunk") or {}
            if chunk.get("type") == "usage":
                u = chunk.get("usage") or {}
                totals["input_tokens"] += int(u.get("inputTokens", 0) or 0)
                totals["output_tokens"] += int(u.get("outputTokens", 0) or 0)
                totals["reasoning_tokens"] += int(u.get("reasoningTokens", 0) or 0)
                totals["cache_read_tokens"] += int(u.get("cacheReadTokens", 0) or 0)
                totals["n_model_calls"] += 1
        if e.get("type") == "request/header":
            model = (e.get("data") or {}).get("model") or model
        ts = e.get("time")
        if ts:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
    totals["model"] = model
    if first_ts and last_ts:
        totals["duration_sec"] = round((last_ts - first_ts) / 1000.0, 1)
    return totals


def main():
    print(json.dumps(extract(Path(sys.argv[1])), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
