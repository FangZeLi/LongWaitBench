#!/usr/bin/env python3
"""Build data/manifest_v4.json — v4 benchmark manifest (96 runnable instances).

Pools (see analysis/v4_plan.md):
- 题库 = v2 24 + new_v3 24 + new_banking_v4 7 = 55 base tasks.
- F1 池 = 24（与 v3 完全一致：v2 遗留 15 + 新抽 9）。
- F2 池 = 24（v3 的 17 个挂载原样保留 + 7 个新 banking 挂载，替换 v3 的 7 个
  banking 弱绑定挂载；被替换的 7 题退出所有池，benchmark=False + dropped 标记）。
- 基线臂 = 48：F1 24 + F2 24 的基题各出一个无等待版本 v4-base-*——
  慢工具降为同步、rate_limits 移除、异步说明从 prompt_note 剥离；内容与判题不变。
  类内差值 F1−基线 / F2−基线（无同题跨类对照）。

场次 = (24 F1 + 24 F2 + 48 基线) × 2 种子 × 2 harness = 384。

时长编辑点（改完重跑本脚本生效）：
- data/f1_wait_config_v4.json（F1 慢工具声明分钟）
- data/f2_mounts_v4.json（F2 冷却分钟）
"""
import copy
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "data", "manifest_v2.json")
NEW3 = os.path.join(ROOT, "data", "new_v3_tasks.json")
NEWB = os.path.join(ROOT, "data", "new_banking_v4.json")
NEWF3 = os.path.join(ROOT, "data", "new_f3_v4.json")
MOUNTS_PATH = os.path.join(ROOT, "data", "f2_mounts_v4.json")
F1_WAIT_PATH = os.path.join(ROOT, "data", "f1_wait_config_v4.json")
OUT = os.path.join(ROOT, "data", "manifest_v4.json")

# F3 pool gate: MC numerical family REJECTED by user (F3 = RE-Bench long-wait
# tier only, per heartbeat/DISCUSSION_STATE "F3 只有 RE-Bench（长等待档）").
INCLUDE_F3 = False

# v3 的 7 个 banking F2 挂载被替换（v3 报告：绑定弱、内容难度淹没等待效应）
REPLACED_BANKING_F2 = [
    "tau2-banking-084", "tau2-banking-081", "tau2-banking-091",
    "tau2-banking-086", "tau2-banking-041", "tau2-banking-082",
    "tau2-banking-039",
]
NEW_BANKING_F2 = [
    "tau2-banking-037", "tau2-banking-038", "tau2-banking-083",
    "tau2-banking-085", "tau2-banking-073", "tau2-banking-089",
    "tau2-banking-074",
]


def load_mounts() -> dict:
    with open(MOUNTS_PATH, encoding="utf-8") as f:
        mounts = json.load(f)
    mounts.pop("_comment", None)
    return mounts


def load_f1_waits() -> dict:
    with open(F1_WAIT_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    return cfg


def apply_f1_waits(inst: dict, f1cfg: dict) -> None:
    waits = f1cfg.get(inst.get("based_on") or inst["id"])
    if not waits:
        return
    for t in inst.get("tools", []):
        if t.get("slow") and t.get("name") in waits:
            t.setdefault("wait", {})["declared_minutes"] = float(waits[t["name"]])


def baseline_prompt_note(note) -> str:
    """Same task content, no waits: drop async/slow sentences, keep the rest."""
    if not note:
        return None
    sents = re.split(r"(?<=\.)\s+", note.strip())
    kept = []
    for s in sents:
        if re.search(r"SLOW|slow|asynchronous|job_handle|job completes|"
                     r"poll the handle|slow operation", s):
            break
        kept.append(s)
    cleaned = " ".join(kept).strip()
    return cleaned or None


def make_baseline(inst: dict) -> dict:
    b = copy.deepcopy(inst)
    b.pop("f2_base", None)
    b["id"] = "v4-base-" + inst["id"].replace("tau2-", "")
    b["arm"] = "baseline"
    b["based_on"] = inst["id"]
    b["pair_class"] = "f2" if inst["id"] in NEW_MOUNT_IDS else "f1"
    b["tools"] = [t for t in b.get("tools", []) if not t.get("slow")]
    b.pop("rate_limits", None)
    b.pop("rate_limit_note", None)
    note = baseline_prompt_note(b.get("prompt_note"))
    if note:
        b["prompt_note"] = note
    else:
        b.pop("prompt_note", None)
    return b


def main():
    m = json.load(open(V2, encoding="utf-8"))
    m["instances"].extend(json.load(open(NEW3, encoding="utf-8")))
    m["instances"].extend(json.load(open(NEWB, encoding="utf-8")))
    if INCLUDE_F3 and os.path.exists(NEWF3):
        m["instances"].extend(json.load(open(NEWF3, encoding="utf-8")))
    m["manifest_version"] = "lwb-v4"

    f1cfg = load_f1_waits()
    by_id = {i["id"]: i for i in m["instances"]}
    for inst in m["instances"]:
        apply_f1_waits(inst, f1cfg)

    mounts = load_mounts()
    global NEW_MOUNT_IDS
    NEW_MOUNT_IDS = set(mounts)
    for vid in mounts:
        by_id[vid]["f2_base"] = True
    replaced = set(REPLACED_BANKING_F2)

    # arm labels on base tasks
    for inst in m["instances"]:
        if inst.get("benchmark") is False:
            continue
        if inst.get("source") == "control-f3":
            continue  # F3 pool carries its own arm label
        if inst["id"] in replaced:
            inst["arm"] = "dropped"
            inst["benchmark"] = False
            inst["dropped_reason"] = (
                "v4 banking F2 换题（v3 报告：绑定弱、内容难度淹没等待效应）"
            )
        elif inst["id"] in NEW_MOUNT_IDS:
            inst["arm"] = "f2-base"  # not run directly; runs as v3-f2-*
        else:
            inst["arm"] = "f1"

    # F2 instances (24)
    for vid, rl in mounts.items():
        base = copy.deepcopy(by_id[vid])
        base.pop("f2_base", None)
        base["id"] = "v3-f2-" + vid.replace("tau2-", "")
        base["arm"] = "f2"
        base["f2"] = True
        base["based_on"] = vid
        base["rate_limits"] = {}
        for k, v in rl.items():
            if isinstance(v, dict):
                entry = {"cooldown_minutes": v["cooldown_minutes"]}
                if v.get("note"):
                    entry["note"] = v["note"]
            else:
                entry = {"cooldown_minutes": v}
            base["rate_limits"][k] = entry
        base["rate_limit_note"] = (
            "F2 instance derived from the same task content; the cooldown is "
            "the only added mechanism."
        )
        m["instances"].append(base)

    # baseline arm (48): every F1 task + every F2 base gets a no-wait twin
    for vid in sorted(set(mounts) | {i["id"] for i in m["instances"]
                                     if i.get("arm") == "f1"}):
        m["instances"].append(make_baseline(by_id[vid]))

    # unit-test fixture (NOT benchmark content) — same as v3
    fixture = copy.deepcopy(by_id["tau2-retail-2"])
    fixture.pop("f2_base", None)
    fixture["id"] = "f2-mech-retail"
    fixture["benchmark"] = False
    fixture["arm"] = "fixture"
    fixture["tools"] = [
        {"name": "return_delivered_order_items", "slow": True,
         "wait": {"type": "policy", "declared_minutes": 0.1}},
        {"surface": "full_domain"},
    ]
    fixture["rate_limits"] = {
        "get_product_details": {"cooldown_minutes": 0.2},
        "return_delivered_order_items": {"cooldown_minutes": 0.2},
    }
    fixture["difficulty"]["source_label"] = "f2-mechanism-fixture"
    m["instances"].append(fixture)

    n_f1 = sum(1 for i in m["instances"] if i.get("arm") == "f1")
    n_f2 = sum(1 for i in m["instances"] if i.get("arm") == "f2")
    n_base = sum(1 for i in m["instances"] if i.get("arm") == "baseline")
    n_dropped = sum(1 for i in m["instances"] if i.get("arm") == "dropped")
    n_f3 = sum(1 for i in m["instances"] if i.get("arm") == "f3")
    assert n_f1 == 24 and n_f2 == 24 and n_base == 48 and n_dropped == 7, (
        f"pool invariant broken: F1={n_f1} F2={n_f2} baseline={n_base} "
        f"dropped={n_dropped}")
    if INCLUDE_F3:
        assert n_f3 == 8, f"F3 pool broken: {n_f3}"

    m["v4_meta"] = {
        "pools": {"f1": n_f1, "f2": n_f2, "baseline": n_base,
                  "dropped": n_dropped, "f3": n_f3},
        "runnable": n_f1 + n_f2 + n_base + n_f3,
        # 测试阶段永远只跑一个种子（用户拍板 2026-10-11）：× 1 种子 × 2 harness
        "sessions": (n_f1 + n_f2 + n_base + n_f3) * 1 * 2,
        "banking_f2_replaced": REPLACED_BANKING_F2,
        "banking_f2_new": NEW_BANKING_F2,
        "baseline_note": (
            "no-wait twins (v4-base-*): slow tools synchronous, no rate limits, "
            "async text stripped; within-class diffs F1-baseline / F2-baseline "
            "(no same-task cross-class comparison)"
        ),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT}: {len(m['instances'])} entries "
          f"(F1 {n_f1} + F2 {n_f2} + baseline {n_base} + dropped {n_dropped} "
          f"+ 1 fixture) -> {m['v4_meta']['sessions']} sessions")


if __name__ == "__main__":
    main()
