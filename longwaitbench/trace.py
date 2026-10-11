"""canonical_trace.jsonl writer — one JSON per line, JSON-safe by construction."""
import json
import math
import threading
from pathlib import Path


def sanitize(obj):
    """Replace non-finite floats (NaN/Inf) with None; dict/list recursion."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


class TraceWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0

    def write(self, entry: dict):
        line = json.dumps(sanitize(entry), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._count += 1

    def existing_session_ids(self) -> set:
        if not self.path.exists():
            return set()
        ids = set()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("type") == "session":
                    ids.add(e.get("session_id"))
        return ids
