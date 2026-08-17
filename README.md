# DeepScalper 复现（A 股 tick 数据）

在 3 秒级 tick 快照数据上复现 DeepScalper（Sun et al., 2022）日内交易强化学习框架。
设计细节见 [docs/design.md](docs/design.md)。

## 环境

```bash
uv sync          # 安装 torch(CPU) / pandas / pyarrow / lightgbm 等
```

## 数据

`data/<code>/` 下放置月度 tick parquet（202602–202607）；实验默认使用 301308 与 688030。

## 运行

```bash
# 冒烟测试
.venv/Scripts/python scripts/smoke_test.py

# 全量实验（48 个作业，8 进程并行；幂等，可中断后续跑）
.venv/Scripts/python scripts/run_all.py --symbols 301308 688030 --workers 8

# 结果汇总（results/summary.csv）
.venv/Scripts/python scripts/summarize.py
```

## 关键参数

回看 600 tick / hindsight 3600 tick / 波动率窗口 300 tick / 每 20 tick 决策 /
7:1:2 按日切分 / 手续费双边 2.5e-4 / 1 倍杠杆（论文 5 倍）/ 最大持仓 ±50 /
价格档相对对手价 ±5 价位 / w=0.1 / η=1.0 / 5 epochs / 3 种子。
修改见 `deepscalper/config.py`。
