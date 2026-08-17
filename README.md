# DeepScalper 复现（A 股 tick 数据）

在 3 秒级 tick 快照数据上复现 DeepScalper（Sun et al., 2022）日内交易强化学习框架。
设计细节见 [docs/design.md](docs/design.md)。

## 环境

```bash
uv sync          # 安装 torch(CUDA 12.8) / pandas / pyarrow / lightgbm 等
```

训练默认按 `Config.device = "auto"` 选择设备（有 CUDA 则用 CUDA，否则 CPU）。

## 数据

`data/<code>/` 下放置月度 tick parquet（202602–202607）；实验默认使用 301308 与 688030。

## 运行

```bash
# 冒烟测试
.venv/bin/python scripts/smoke_test.py

# 全量实验（48 个作业，多进程并行；幂等，可中断后续跑）
.venv/bin/python scripts/run_all.py --symbols 301308 688030

# hindsight 视野敏感性实验（DS / DS-NA 各展开 4 档）
.venv/bin/python scripts/run_all.py --methods DS DS-NA --hindsight-ticks 300 600 900 1200

# 结果汇总（results/summary.csv，按标的 / 方法 / 视野分组）
.venv/bin/python scripts/summarize.py
```

并行度缺省按设备自适应：CUDA 下取 2（避免多进程争抢显存），CPU 下取
`核数 / Config.num_threads`；可用 `--workers` 覆盖。

## 关键参数

回看 600 tick / hindsight 1200 tick / 波动率窗口 300 tick / 每 20 tick 决策 /
7:1:2 按日切分 / 手续费双边 1.0e-4 / 1 倍杠杆（论文 5 倍）/ 最大持仓 ±50 手 /
价格档为穿价档深 1–5 档（逐档扫单，受盘口深度约束）/ w=0.1 / η=1.0 / 5 epochs / 3 种子。
修改见 `deepscalper/config.py`。

仅使用 A 股连续竞价时段（09:30–11:30、13:00–14:57）；集合竞价与午休行情不适用
逐笔穿价撮合，已在加载阶段剔除。不施加 T+1 约束（多空双向、日终强平）。
