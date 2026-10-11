"""Async job registry: slow calls become jobs with poll-only completion.

Semantics (L1 anchor):
- submitting a slow tool returns a job_handle immediately;
- the job completes asynchronously (policy waits: sleep declared*scale
  physical seconds then execute the real tool call; compute waits: real
  numpy computation running immediately);
- the agent learns about completion ONLY via poll();
- poll returns {"status", "counter"} and, once completed, the result;
- no push notifications, no progress ratio.
"""
import threading
import time
import uuid

from . import constants as C


class Job:
    def __init__(self, handle, tool_name, kind, declared_minutes, execute_fn,
                 on_complete=None):
        self.handle = handle
        self.tool_name = tool_name
        self.kind = kind  # "policy" | "compute"
        self.declared_minutes = declared_minutes
        self.execute_fn = execute_fn  # () -> {"ok", "result", "error"}
        self.submitted_at = time.time()
        self.status = "running"
        self.result = None
        self.error = None
        self.completed_at = None
        self.poll_count = 0
        self.last_seen = None  # (status, counter) tuple for change detection
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self.on_complete = on_complete  # called once, after run() settles

    def wait_for(self, timeout: float = None) -> dict:
        """Block until the job leaves the running state, then return its
        state dict (same shape as poll(), without consuming a poll counter)."""
        self._done.wait(timeout)
        return self.poll()

    def run(self, wait_seconds: float = 0.0):
        try:
            if wait_seconds > 0:
                self._cancelled.wait(wait_seconds)
                if self._cancelled.is_set():
                    with self._lock:
                        self.status = "cancelled"
                    return
            with self._lock:
                if self._cancelled.is_set():
                    self.status = "cancelled"
                    return
            outcome = self.execute_fn()
            with self._lock:
                self.completed_at = time.time()
                if outcome.get("compute_seconds"):
                    self.declared_minutes = float(outcome["compute_seconds"]) / (
                        C.COMPRESSION_SCALE * 60.0
                    )
                if outcome.get("ok"):
                    self.status = "completed"
                    self.result = outcome.get("result")
                else:
                    self.status = "failed"
                    self.error = outcome.get("result") or outcome.get("error")
        except Exception as e:  # thread safety net
            with self._lock:
                self.completed_at = time.time()
                self.status = "failed"
                self.error = f"Error: {e}"
        finally:
            if self.on_complete is not None:
                try:
                    self.on_complete(self)
                except Exception:
                    pass
            self._done.set()

    def poll(self) -> dict:
        with self._lock:
            self.poll_count += 1
            delivered = self.status in ("completed", "failed")
            state = (self.status, delivered)
            changed = state != self.last_seen
            self.last_seen = state
            out = {
                "status": self.status,
                "counter": self.poll_count,
                "changed": changed,
            }
            if self.status == "completed":
                out["result"] = self.result
            if self.status == "failed":
                out["error"] = self.error
            return out


class JobRegistry:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._threads = []

    def submit(self, tool_name, kind, declared_minutes, execute_fn,
               wait_seconds=0.0, handle=None, on_complete=None):
        handle = handle or ("job-" + uuid.uuid4().hex[:8])
        job = Job(handle, tool_name, kind, declared_minutes, execute_fn,
                  on_complete=on_complete)
        job.wait_seconds = wait_seconds  # for F2 busy/retry_after computation
        with self._lock:
            self._jobs[handle] = job
        t = threading.Thread(
            target=job.run, kwargs={"wait_seconds": wait_seconds}, daemon=True
        )
        t.start()
        self._threads.append(t)
        return handle

    def poll(self, handle) -> dict:
        with self._lock:
            job = self._jobs.get(handle)
        if job is None:
            return {"status": "unknown_handle", "counter": 0, "changed": False,
                    "error": f"no job with handle {handle}"}
        return job.poll()

    def open_jobs(self) -> list:
        with self._lock:
            jobs = list(self._jobs.values())
        out = []
        for j in jobs:
            with j._lock:
                out.append(
                    {
                        "handle": j.handle,
                        "tool_name": j.tool_name,
                        "status": j.status,
                        "counter": j.poll_count,
                    }
                )
        return out

    def first_job(self) -> Job:
        """The first submitted job anchors the task's true_completion (T)."""
        with self._lock:
            if not self._jobs:
                return None
            return min(self._jobs.values(), key=lambda j: j.submitted_at)

    def shutdown(self):
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            j._cancelled.set()
