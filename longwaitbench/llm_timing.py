#!/usr/bin/env python3
"""Extract LLM generation (thinking) time from harness session records.

dsh:      session.jsonl.zstd transcript — chunk entries carry millisecond
          timestamps; per model step the generation span is
          max(chunk time) - min(chunk time) over text/reasoning chunks.
opencode: opencode.db `part` table — per step, generation span is
          step-finish.time_created - step-start.time_created (tool execution
          happens between steps, so it is naturally excluded).
"""
import io
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "pylibs"))


def _decompress_zstd(raw: bytes) -> str:
    """Decompress a zstd blob across interpreter variants:
    py3.14 stdlib `compression.zstd`, then vendored `zstandard`, then the
    `zstd` CLI; fall back to empty text."""
    try:
        from compression import zstd  # python 3.14 stdlib

        return zstd.decompress(raw).decode("utf-8", errors="replace")
    except Exception:
        pass
    try:
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        return dctx.stream_reader(io.BytesIO(raw)).read().decode(
            "utf-8", errors="replace"
        )
    except Exception:
        pass
    try:
        import subprocess

        proc = subprocess.run(
            ["zstd", "-dc"], input=raw, capture_output=True
        )
        if proc.returncode == 0:
            return proc.stdout.decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


def extract_dsh_think(session_dir: Path, clip_start_ms=None,
                      clip_end_ms=None) -> float:
    files = list(Path(session_dir).glob("session-*/session.jsonl.zstd"))
    if not files:
        files = list(Path(session_dir).glob("**/session.jsonl.zstd"))
    if not files:
        return 0.0
    text = _decompress_zstd(files[0].read_bytes())
    if not text:
        return 0.0

    spans = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        typ = e.get("type")
        if typ not in ("text-chunks", "reasoning-chunks", "assistant/chunk"):
            continue
        t = e.get("time") or e.get("time0")
        data = e.get("data") or {}
        turn, step = data.get("turn"), data.get("step")
        if t is None or turn is None or step is None:
            continue
        key = (turn, step)
        if key not in spans:
            spans[key] = [t, t]
        else:
            if t < spans[key][0]:
                spans[key][0] = t
            if t > spans[key][1]:
                spans[key][1] = t
    total = 0.0
    for lo, hi in spans.values():
        # clip to the benchmark session window: transcripts can keep
        # recording after the judged session has ended
        if clip_start_ms is not None:
            lo = max(lo, clip_start_ms)
        if clip_end_ms is not None:
            hi = min(hi, clip_end_ms)
        if hi > lo:
            total += (hi - lo) / 1000.0
    return total


def extract_opencode_think(db_path: Path, clip_start_ms=None,
                           clip_end_ms=None) -> float:
    if not Path(db_path).exists():
        return 0.0
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT message_id, time_created, data FROM part ORDER BY time_created"
    ).fetchall()
    con.close()
    total = 0.0
    for mid, tc, data in rows:
        try:
            d = json.loads(data)
        except Exception:
            continue
        if d.get("type") != "step-finish":
            continue
        # pair with the most recent step-start of the same message
        start = None
        for mid2, tc2, data2 in rows:
            if mid2 != mid or tc2 >= tc:
                continue
            try:
                d2 = json.loads(data2)
            except Exception:
                continue
            if d2.get("type") == "step-start":
                start = tc2  # keep the latest one before this finish
        if start is None or tc <= start:
            continue
        lo, hi = start, tc
        if clip_start_ms is not None:
            lo = max(lo, clip_start_ms)
        if clip_end_ms is not None:
            hi = min(hi, clip_end_ms)
        if hi > lo:
            total += (hi - lo) / 1000.0
    return total


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1])
    if path.is_dir():
        print(f"dsh think_seconds={extract_dsh_think(path):.3f}")
    else:
        print(f"opencode think_seconds={extract_opencode_think(path):.3f}")
