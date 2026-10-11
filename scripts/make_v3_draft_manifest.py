#!/usr/bin/env python3
"""Build data/manifest_v3_draft.json = v2 24 题 + 新抽 24 题 + F2 实例 + 夹具.

组装规则：
- 题库 = manifest_v2（24）+ data/new_v3_tasks.json（24，scripts/extract_v3.py 产出）；
- F1 池 = 题库中未被 f2_mounts 挂载的 24 题；F2 池 = f2_mounts 挂载的 24 题
  （每挂载键生成一个 `v3-f2-*` 实例：同题面 + rate_limits；`based_on` 仅溯源，
  不跑对照——用户决定；F2 题不再计入 F1）；
- 时长编辑点：data/f1_wait_config.json（F1 慢工具声明分钟）与
  data/f2_mounts.json（F2 冷却分钟），改完重跑本脚本生效；
- banking 按内层可发现工具名键控；另有单测夹具 f2-mech-retail（非 benchmark 内容）。
"""
import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "data", "manifest_v2.json")
OUT = os.path.join(ROOT, "data", "manifest_v3_draft.json")

# instance_id -> {rate-limit key: cooldown_minutes}
# 挂载配置外置在 data/f2_mounts.json（F2 冷却时长的唯一编辑点）；F1 慢工具
# 声明时长外置在 data/f1_wait_config.json（F1 时长的唯一编辑点）。两者改完
# 都重跑本脚本生效。原则（用户拍板）：F2 限频挂**核心操作**（查询类 + 修改类），
# 时长按操作相邻调用的典型间隔定（v2 实测）：查询 2 声明分钟 / 慢修改 5 声明分钟。
# 9 题 = 8 难题 + 1 金丝雀（retail-1）。
MOUNTS_PATH = os.path.join(ROOT, "data", "f2_mounts.json")
F1_WAIT_PATH = os.path.join(ROOT, "data", "f1_wait_config.json")
NEW_TASKS = os.path.join(ROOT, "data", "new_v3_tasks.json")


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
    """Override slow-tool declared_minutes from data/f1_wait_config.json."""
    waits = f1cfg.get(inst.get("based_on") or inst["id"])
    if not waits:
        return
    for t in inst.get("tools", []):
        if t.get("slow") and t.get("name") in waits:
            t.setdefault("wait", {})["declared_minutes"] = float(waits[t["name"]])


def main():
    with open(V2, encoding="utf-8") as f:
        m = json.load(f)
    with open(NEW_TASKS, encoding="utf-8") as f:
        new_tasks = json.load(f)
    m["instances"].extend(new_tasks)

    # apply F1 wait durations (editable via data/f1_wait_config.json)
    f1cfg = load_f1_waits()
    by_id = {i["id"]: i for i in m["instances"]}
    for inst in m["instances"]:
        apply_f1_waits(inst, f1cfg)

    mounts = load_mounts()
    for vid in mounts:
        # the mounted base task belongs to F2 only: it is NOT run directly and
        # does not count in the F1 pool (user decision)
        by_id[vid]["f2_base"] = True
    new_instances = []
    for vid, rl in mounts.items():
        base = copy.deepcopy(by_id[vid])
        base.pop("f2_base", None)  # the F2 instance itself IS runnable
        base["id"] = "v3-f2-" + vid.replace("tau2-", "")
        base["f2"] = True
        base["based_on"] = vid
        # normalize shorthand {tool: minutes} -> {tool: {cooldown_minutes: X}}
        base["rate_limits"] = {
            k: {"cooldown_minutes": v} for k, v in rl.items()
        }
        base["rate_limit_note"] = (
            "F2 instance derived from the same task content; the cooldown is "
            "the only added mechanism."
        )
        new_instances.append(base)
    m["instances"].extend(new_instances)

    # unit-test fixture (NOT benchmark content)
    fixture = copy.deepcopy(by_id["tau2-retail-2"])
    fixture["id"] = "f2-mech-retail"
    fixture["benchmark"] = False  # excluded from batch runs and metrics
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

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT}: {len(m['instances'])} instances "
          f"({len(new_instances)} F2 instances + 1 fixture)")


if __name__ == "__main__":
    main()
