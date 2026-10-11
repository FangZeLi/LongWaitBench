# LongWaitBench

在统一多源环境中测量智能体于长时间等待的工具调用期间行为的基准。

## 仓库结构

- `longwaitbench/` — 实验包：统一环境、作业注册表、MCP 服务、会话驱动、追踪与指标
- `scripts/` — 跑批、校验、分析脚本
- `main.py` — 入口

## 依赖

- Python 3.12+
- numpy、pydantic 等 Python 包
- `dsh` 与 `opencode` 两个命令行工具
- 环境变量 `DEEPSEEK_API_KEY`（运行时从环境读取）

## 运行

```bash
export TAU2_DATA_DIR=$PWD/data/tau2bench/data
export LC1_MANIFEST=$PWD/data/manifest.json
export PYTHONPATH=$PWD/data/pylibs
DEEPSEEK_API_KEY=<密钥> python longwaitbench/run_harness_batch.py \
    --workers 2 --trace results/trace.jsonl
```
