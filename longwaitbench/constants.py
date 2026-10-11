"""LongWaitBench v1 — global constants.

All tunable protocol values live here so the protocol section of
wait_score_results.json can be generated mechanically from them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PACKAGE_DIR = ROOT / "longwaitbench"
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"

PYLIB_DIR = DATA_DIR / "pylibs"
DEFAULT_TAU2_DATA_DIR = DATA_DIR / "tau2bench" / "data"
DEFAULT_MANIFEST_PATH = DATA_DIR / "manifest.json"
EXTRACTION_LOG_PATH = DATA_DIR / "extraction_log.json"

CANONICAL_TRACE_PATH = RESULTS_DIR / "canonical_trace.jsonl"
RESULTS_JSON_PATH = RESULTS_DIR / "wait_score_results.json"
PROTOCOL_PATH = RESULTS_DIR / "protocol.json"
RUN_LOG_PATH = RESULTS_DIR / "run.log"

# --- wait model -------------------------------------------------------------
# 1 declared minute == COMPRESSION_SCALE * 60 physical seconds (policy waits).
# Chosen so that real LLM decision latency (~15s/call for deepseek-v4-pro,
# measured) fits inside the per-instance wall budget; experiment_design_v4
# delegates this value to the experiment code ("建议 0.02").
COMPRESSION_SCALE = 0.15
# deployment budget scale: budget_survival thresholds use 0.5 * per-task budget
DEPLOYMENT_BUDGET_SCALE = 0.5

# --- decision space ---------------------------------------------------------
FIVE_ACTIONS = ("wait", "poll", "work", "final_answer", "abandon")
# harness stage adds the blocking wait primitive
ACTIONS = ("wait", "poll", "wait_for", "work", "final_answer", "abandon")
ACTION_VIRTUAL_COST = {
    "wait": 0.1,
    "wait_for": 0.5,
    "poll": 0.5,
    "work": 1.0,
    "final_answer": 1.0,
    "abandon": 1.0,
}
# 零限制原则（用户拍板，2026-09-27）：agent 行为不受任何预算/上限约束——
# 无慢调用提交上限、无决策数上限、无 deadline 超时、无输出截断、无 wait 钳制。
# 以下两个数值仅作为 trace 元数据（declared figures）与遗留迁移脚本使用，
# 不对任何会话强制执行。
MAX_DECISIONS_PER_TASK = 40  # trace metadata only (declared); never enforced
MAX_WAIT_SECONDS = 30.0  # legacy v1 value kept for trace-migration compatibility

# fixed shared time-prediction buckets (physical seconds), label -> [lo, hi)
TIME_BUCKETS = [
    ("0-2s", 0.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5-15s", 5.0, 15.0),
    ("15-60s", 15.0, 60.0),
    ("60s+", 60.0, float("inf")),
]
BUCKET_LABELS = [b[0] for b in TIME_BUCKETS]

# --- exact condition registry (authoritative; do not rename) ----------------
CONDITIONS = [
    "LongCall-v1",
    "ExponentialBackoff",
    "AggressivePoll",
    "WaitAndAnswer",
]

# --- real API ---------------------------------------------------------------
API_BASE = "https://api.deepseek.com/v1/chat/completions"
API_MODEL = "deepseek-v4-pro"
API_MAX_TOKENS = 4096
API_TEMPERATURE = 0.2
API_MAX_RETRIES = 3  # network/5xx retries per request
PARSE_MAX_RETRIES = 2  # JSON parse retries per decision

# 测试阶段永远只跑一个种子（用户拍板 2026-10-11）。已完成的多种子场次留在
# trace 中作历史数据；后续跑批只补 seed 0。
SEEDS = [0]
NUM_TASKS = 12  # literal; must match manifest instance count
