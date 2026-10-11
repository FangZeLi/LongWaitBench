"""F3 control environment: private weighted quadratic MC estimation.

Task content: estimate E[f(X)] where X ~ N(0, I_d) and f is the private
function f(X) = sum_i w_i * X_i^2. The weights are the private parameters;
E[f] = sum_i w_i (never revealed). submit_job(n_samples) draws n_samples
Gaussian vectors with a deterministic RNG and returns the sample estimate
plus the real compute time. Compute happens in-process with numpy.
"""
import hashlib
import json
import re

import numpy as np


class F3Environment:
    def __init__(self, instance: dict, seed: int):
        self.instance = instance
        params = instance["initial_state"]["params"]
        self.dim = int(params["dim"])
        self.weights = np.asarray(params["weights"], dtype=np.float64)
        self.weights_sha256 = params["weights_sha256"]
        self.true_value = instance["verifier"]["checks"][0]["true_value"]
        self.eps = instance["verifier"]["checks"][0]["eps"]
        self.n_max = self._parse_n_max(instance["task_spec"])
        self.seed = seed
        self._job_counter = 0

        # integrity guards (deterministic, no LLM involved)
        assert self.weights.shape == (self.dim,), "weights dim mismatch"
        assert abs(float(self.weights.sum()) - float(self.true_value)) < 1e-9, (
            "true_value != sum(weights): manifest corrupted"
        )
        digest = hashlib.sha256(self.weights.tobytes()).hexdigest()
        assert digest == self.weights_sha256, "weights sha256 mismatch"

    @staticmethod
    def _parse_n_max(task_spec: str) -> int:
        m = re.search(r"n_samples\s*<=\s*(\d+)", task_spec)
        if not m:
            raise ValueError("f3 task_spec lacks 'n_samples <= N'")
        return int(m.group(1))

    def validate_submit(self, n_samples) -> str:
        """Return an error message or '' when valid."""
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            return "n_samples must be an integer"
        if n_samples <= 0:
            return "n_samples must be > 0"
        if n_samples > self.n_max:
            return f"n_samples exceeds allowed maximum {self.n_max}"
        return ""

    def run_job(self, n_samples: int) -> dict:
        """Real MC computation. Deterministic per (seed, submission index).

        Returns {"estimate": float, "compute_seconds": float}.
        """
        self._job_counter += 1
        rng = np.random.default_rng(
            self.seed * 1000003 + self._job_counter * 1000033
        )
        start = __import__("time").perf_counter()
        total = 0.0
        # chunked to bound peak memory
        chunk = 65536
        done = 0
        while done < n_samples:
            n = min(chunk, n_samples - done)
            X = rng.standard_normal((n, self.dim))
            total += float(np.sum(self.weights * (X * X)))
            done += n
        estimate = total / n_samples
        compute_seconds = __import__("time").perf_counter() - start
        return {"estimate": float(estimate), "compute_seconds": float(compute_seconds)}

    def check_answer(self, estimate) -> bool:
        if not isinstance(estimate, (int, float)) or isinstance(estimate, bool):
            return False
        if not float(estimate) == float(estimate):  # NaN guard
            return False
        return abs(float(estimate) - float(self.true_value)) <= float(self.eps)
