#!/usr/bin/env python3
"""REJECTED (2026-10-11): MC numerical-family F3 build.

User rejected this plan — their standing F3 position is RE-Bench only
("F3 只有 RE-Bench（长等待档）", heartbeat/DISCUSSION_STATE). Kept on disk
for the record only; NOT mounted (INCLUDE_F3=False in make_v4_manifest.py).

(Original docstring follows.)
"""

F3 = self-chosen scale latency: the agent picks n_samples per job; real
in-process numpy compute is the wait; reward scales with samples
(accuracy vs the hidden tolerance). Zero-limit protocol: no job-count
budget (the v1 sentence "Budget: at most 6 jobs." is removed); n_max is a
per-call mechanism bound (memory guard), not a budget.

Tasks:
  v4-f3-0..3 : revive the four v1 control tasks verbatim (params/hashes from
               data/manifest.json), minus the budget sentence.
  v4-f3-4..7 : new draws (deterministic RNG), each calibrated so that the
               statistically sufficient sample count n_req (3-sigma margin)
               runs the target real compute time (tier A/B/C ~45/90/150s).

Outputs data/new_f3_v4.json + a calibration table. No manifest changes here.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys_path = __import__("sys")
sys_path.path.insert(0, str(ROOT))
sys_path.path.insert(0, str(ROOT / "data" / "pylibs"))

V1_MANIFEST = DATA / "manifest.json"

TIERS = [(45.0, 3.0), (90.0, 3.0), (150.0, 3.0), (90.0, 3.0)]  # (target_s, sigma_margin)
NEW_DIMS = [32, 64, 16, 8]

BUDGET_META = {"slow_calls": 1000, "wall_minutes": 240}  # trace metadata only


def mc_work(weights: np.ndarray, n: int, seed: int, job_idx: int) -> float:
    """Same computation as f3_env.F3Environment.run_job (chunked)."""
    rng = np.random.default_rng(seed * 1000003 + job_idx * 1000033)
    chunk = 65536
    done = 0
    total = 0.0
    while done < n:
        m = min(chunk, n - done)
        X = rng.standard_normal((m, weights.shape[0]))
        total += float(np.sum(weights * (X * X)))
        done += m
    return total / n


def throughput(weights: np.ndarray) -> float:
    """samples/second measured on 2 chunks of the real computation."""
    n = 2 * 65536
    t0 = time.perf_counter()
    mc_work(weights, n, 1, 1)
    return n / max(time.perf_counter() - t0, 1e-9)


def make_task(idx: int, dim: int, weights: np.ndarray, eps: float,
             n_max: int, note: str) -> dict:
    true = float(np.asarray(weights).sum())
    digest = hashlib.sha256(np.asarray(weights, dtype=np.float64).tobytes()).hexdigest()
    return {
        "id": f"v4-f3-{idx}",
        "source": "control-f3",
        "domain": "f3",
        "license": "self",
        "arm": "f3",
        "f3": True,
        "task_spec": (
            f"Estimate E[f(X)] where X is a standard normal vector in R^{dim} "
            f"and f is a fixed private function defined by the environment. "
            f"Report the estimate with absolute error <= {eps}. You may submit "
            f"estimation jobs (each job draws n_samples <= {n_max} and returns "
            "the sample estimate plus its real compute time)."
        ),
        "initial_state": {
            "params": {
                "dim": dim,
                "weights": [float(w) for w in weights],
                "weights_sha256": digest,
            }
        },
        "tools": [
            {
                "name": "submit_job",
                "slow": True,
                "wait": {"type": "compute"},
                "params_schema": {"method": "mc_mean", "n_samples": f"int <= {n_max}"},
            }
        ],
        "wait_config": {"mode": "compute", "budget": dict(BUDGET_META)},
        "verifier": {"checks": [{"kind": "tolerance",
                                 "true_value": true, "eps": eps}]},
        "difficulty": {"source_label": f"control-f3-v4:{note}", "calibrated": True},
        "calibration": {
            "target_compute_s": None,
            "sigma_margin": None,
            "n_req": None,
            "n_max": n_max,
            "eps": eps,
        },
    }


def main():
    out = []
    v1 = json.load(open(V1_MANIFEST, encoding="utf-8"))
    v1_f3 = [i for i in v1["instances"] if i.get("source") == "control-f3"]
    assert len(v1_f3) == 4
    for idx, src in enumerate(v1_f3):
        t = make_task(idx, src["initial_state"]["params"]["dim"],
                      src["initial_state"]["params"]["weights"],
                      src["verifier"]["checks"][0]["eps"],
                      1000000, "v1-revived")
        out.append(t)

    rng = np.random.default_rng(20261011)
    print("new-task calibration:")
    print(f"{'id':10s} {'dim':>4s} {'samples/s':>10s} {'n_req':>10s} "
          f"{'est_sec':>8s} {'eps':>12s} {'n_max':>10s}")
    for j, (dim, (target_s, margin)) in enumerate(zip(NEW_DIMS, TIERS)):
        idx = 4 + j
        weights = rng.uniform(0.5, 2.0, size=dim)
        sd = float(np.sqrt(2.0 * float(np.sum(weights * weights))))
        rate = throughput(weights)
        n_req = int(target_s * rate)
        eps = float(f"{margin * sd / np.sqrt(n_req):.4g}")
        n_max = max(int(4 * n_req), 1_000_000)
        t = make_task(idx, dim, weights, eps, n_max, f"new-{j}")
        t["calibration"].update({
            "target_compute_s": target_s,
            "sigma_margin": margin,
            "n_req": n_req,
            "samples_per_s": round(rate, 1),
        })
        out.append(t)
        print(f"{t['id']:10s} {dim:4d} {rate:10.1f} {n_req:10d} "
              f"{target_s:8.1f} {eps:12.6f} {n_max:10d}")

    (DATA / "new_f3_v4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote data/new_f3_v4.json with {len(out)} F3 instances "
          "(draft; not mounted into manifest_v4.json)")


if __name__ == "__main__":
    main()
