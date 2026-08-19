# GridScalper

A 股 3 秒级 tick 快照上的成交驱动网格交易研究。同一数据与回测底座上运行两个算法：

- **预测算法**（`forecast/`）：LightGBM 由 600-tick 窗口统计预测 5 个前瞻目标，预测值生成门控信号，控制固定 $0.1\times\mathrm{ATR}$ 网格的启停——门控只在净持仓为 0 时生效。
- **强化学习算法**（`control/`）：BDQ 智能体在 SMDP 决策点自主设定网格半宽与成交量（半宽取 0 立即平回底仓，取极大值即停手），预测算法的输出同时作为其状态特征。

网格几何、穿价撮合与交易成本（双边佣金 $10^{-4}$、卖出印花税 $5\times10^{-4}$）统一定义在 `strategy/`；样本统一切分为 7:1:2（训练 / 验证 / 测试，单次时序切分）。设计与取舍见 [docs/design.md](docs/design.md)，特征定义见 [data/features.md](data/features.md)。

## 目录结构

```
data/            原始 tick parquet（data/<code>/）与特征定义 features.md
cache/           统一缓存 <symbol>.npz：逐 tick 的窗口特征、前瞻目标与预测结果（两个算法共用）
data_provider/   数据管线：tick 加载与 ATR（ticks.py）、7:1:2 切分（split.py）、统一缓存（windows.py）与预建入口（cache.py）
strategy/        网格策略与回测：成本、几何、ATR 半宽、报价驱动回放、指标、成本叠加
forecast/        预测算法：模型、训练、门控信号、回测；产物在 forecast/runs/
control/         强化学习算法：环境、BDQ 网络、智能体、训练；产物在 control/runs/
webviz/          决策过程查看器（导出 + 单页前端）
utils/           图形样式与绘图
scripts/         RL 实验矩阵、结果汇总与冒烟测试
tests/           单元测试
docs/            设计文档与网格收益推导
```

依赖方向单向：`strategy`（叶子）$\leftarrow$ `data_provider` $\leftarrow$ `forecast` / `control` $\leftarrow$ `webviz`。

## 环境

```bash
uv sync                # torch(CUDA 12.8) / numpy / pandas / lightgbm / matplotlib / wandb
.venv/bin/wandb login  # 首次使用需登录；离线或不记录见下文 --wandb-mode
```

训练默认按 `control.config.Config.device = "auto"` 选择设备（有 CUDA 则用 CUDA，否则 CPU）。

## 数据

`data/<code>/` 下放置月度 tick parquet（如 `301308.SZ_202602_tick.parquet`）。仅使用连续竞价时段（09:30–11:30、13:00–14:57）；每日 $\operatorname{ATR}_3$ 由此前 3 个完整交易日预计算，当日盘中恒定、无前视。窗口特征（47 维）、前瞻目标与预测结果（各 5 维）逐 tick 同存于 `cache/<symbol>.npz`，源数据或参数变化时自动重建，两个算法共同使用。

## 运行

以下入口的 `--symbols` 不指定时默认 `data/` 下全部标的。

```bash
# 统一缓存预建（幂等，跨标的并行构建 cache/<symbol>.npz，--workers 控制并行数与内存峰值；
# 其余入口也会按需自动构建）
.venv/bin/python -m data_provider.cache

# 预测算法：训练与 val/test 评估（幂等，参数或数据变化才重训）→ 产物在 forecast/runs/
.venv/bin/python -m forecast.train

# 预测算法：门控网格回测（测试段）→ forecast/runs/backtest/
.venv/bin/python -m forecast.backtest

# 冒烟测试（可指定标的，默认 301308）
.venv/bin/python scripts/smoke_test.py

# RL 全量实验：训练与测试段评估（基线 + RL 及消融，多进程并行；幂等，可中断后续跑）
.venv/bin/python scripts/run_all.py --symbols 301308 688030

# 超参梯子：hindsight 权重 w 与风险厌恶 λ
.venv/bin/python scripts/run_all.py --methods GRID \
    --hindsight-weights 0.02 0.05 0.1 0.2 --inventory-lambdas 0 1 3 10 30

# 不联网记录（离线落盘后 wandb sync）或完全关闭
.venv/bin/python scripts/run_all.py --wandb-mode offline
.venv/bin/python scripts/run_all.py --wandb-mode disabled

# 按验证集 SR 锁定 (w, λ)，汇总测试集指标 → control/runs/summary.csv（各段行情状态打印到控制台）
.venv/bin/python scripts/summarize.py

# 单元测试（部分用例依赖 data/ 与 cache/ 中的真实数据）
.venv/bin/python -m unittest discover -s tests
```

## 可视化

```bash
# 预测算法：逐日价格、窗口统计与各门控方案的网格回放
.venv/bin/python -m webviz.export --algorithm forecast --symbols 301308

# 强化学习：从 control/runs/ 的检查点回放测试段贪心决策（需先跑 run_all 训练）
.venv/bin/python -m webviz.export --algorithm control --symbols 301308

# 浏览器查看
.venv/bin/python -m http.server 8000   # 访问 http://localhost:8000/webviz/
```

## 产物

- `forecast/runs/`：模型（`model/`）、预测评估（`metrics.json`、可选 `figures/`）、门控回测（`backtest/`）。
- `control/runs/`：`<symbol>/<method>[_w<权重>][_lam<λ>][_seed<k>].json` 结果与同名 `.pt` 检查点（选模后的最佳权重，供 webviz 回放），`summary.csv` 汇总表。
- 每个 RL 作业同时建一个 wandb run（项目 `gridscalper`）：逐验证点记录训练奖励、Q 损失与验证 TR、SR 及选模判据曲线；训练结束记录测试集指标与逐日表（design 7.5）。

## 关键参数

样本切分 7:1:2 / 网格半宽默认 $0.1\times\operatorname{ATR}_3$（下限为价格的 $10^{-3}$）/ 成本：双边佣金 $10^{-4}$、卖出印花税 $5\times10^{-4}$（`strategy/costs.py`）。RL 侧：底仓 50 手 / 仓位带 $[0,100]$ 手 / 半宽 7 档（$0$ 为平回底仓，$0.05$–$0.25\times\operatorname{ATR}_3$，$100$ 为关闭）$\times$ 数量 1 档 / 超时 100 tick / 每 tick 折扣 0.9995 / 存货惩罚 $\lambda=3$ / $w=0.1$ / 5 epochs（每 epoch 验证 3 次，选模取最近 3 点均值）/ 3 种子。预测侧：回看 600 tick、前瞻 300 tick、缓存逐 tick 一行、门控与训练行每 20 tick 一个、LightGBM 每目标一个回归器（平方损失，输入含 symbol_id 分类特征）、seed 2021。修改见 `control/config.py` 与 `forecast/config.py`。
