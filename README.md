# GridScalper

A 股 3 秒级 tick 快照上的分钟决策网格交易研究。同一数据与回测底座上运行两个算法：

- **预测算法**（`forecast/`）：行情模式识别门控——「低残差、强趋势」的低震荡路径是网格不利模式，其事后标签由残差趋势规则的粘性平滑给出，LightGBM 二分类由 30 分钟窗口统计输出不利模式概率，概率门控控制固定 $0.1\times\mathrm{ATR}_3$ 网格的启停——门控只在净持仓为 0 时按分钟锚点节奏生效（状态切换经连续确认去抖）。
- **强化学习算法**（`control/`）：dueling Q 网络智能体在固定每 10 分钟的决策锚点自主设定网格半宽（取 0 立即平回底仓，取极大值即停手），全部标的池化统一训练，`forecast/` 的前瞻回归输出同时作为其状态特征。

决策、特征与门控节奏统一按墙钟分钟计时（压缩分钟网格 0–236，每分钟末快照为锚点），消除事件稀疏标的上快照计数时间轴的扭曲；撮合与成本仍逐快照回放。

网格几何、穿价撮合与交易成本（双边佣金 $10^{-4}$、卖出印花税 $5\times10^{-4}$）统一定义在 `strategy/`；样本统一切分为 7:1:2（训练 / 验证 / 测试，单次时序切分）。设计与取舍见 [docs/design.md](docs/design.md)，特征定义见 [data/features.md](data/features.md)。

## 目录结构

```
data/            原始 tick parquet（data/<symbol>/）与特征定义 features.md
cache/           统一缓存 <symbol>.npz：分钟锚点行的窗口特征、前瞻目标与预测结果（两个算法共用）
data_provider/   数据管线：tick 加载、ATR 与分钟网格（ticks.py）、7:1:2 切分（split.py）、统一缓存（windows.py）与预建入口（cache.py）
strategy/        网格策略与回测：成本、几何、ATR 半宽、报价驱动回放、指标、统一回测入口；产物在 strategy/runs/
forecast/        预测算法：前瞻回归（RL 状态特征，产物在 forecast/runs/）与行情模式识别 regime/
                 （模式定义 / 识别 / 经济验证，产物在 forecast/regime/runs/）
control/         强化学习算法：环境、分支 Q 网络、智能体、训练、超参探索、结果汇总与冒烟测试；产物在 control/runs/
webviz/          决策过程查看器（导出 + 单页前端）
utils/           图形样式与绘图
tests/           单元测试
docs/            设计文档与网格收益推导
```

依赖方向单向：`strategy` 基元（costs / grid / width / engine / metrics，叶子）$\leftarrow$ `data_provider` $\leftarrow$ `forecast` / `control` $\leftarrow$ `strategy/backtest.py`（回测入口）/ `webviz`。

## 环境

```bash
uv sync                # torch(CUDA 12.8) / numpy / pandas / lightgbm / scikit-learn / matplotlib / wandb
.venv/bin/wandb login  # 首次使用需登录；离线或不记录见下文 --wandb-mode
```

训练默认按 `control.config.Config.device = "auto"` 选择设备（有 CUDA 则用 CUDA，否则 CPU）。

## 数据

`data/<symbol>/` 下放置月度 tick parquet（如 `301308.SZ_202602_tick.parquet`）。仅使用连续竞价时段（09:30–11:30、13:00–14:57）；每日 $\operatorname{ATR}_3$ 由此前 3 个完整交易日预计算，当日盘中恒定、无前视。窗口特征（47 维）、前瞻目标与预测结果（各 5 维）按分钟锚点行同存于 `cache/<symbol>.npz`（回看窗口按墙钟取 30 分钟、tick 数变长），源数据或参数变化时自动重建，两个算法共同使用。

## 运行

各入口（含「可视化」一节）的 `--symbols` 不指定时默认 `data/` 下全部标的。

```bash
# 1. 统一缓存预建（幂等，跨标的并行构建 cache/<symbol>.npz，--workers 控制并行数与内存峰值；
#    其余入口也会按需自动构建）
.venv/bin/python -m data_provider.cache

# 2. 前瞻回归：训练与 val/test 评估（幂等，参数或数据变化才重训；预测缓存是 RL 的状态特征；
#    --plot 输出测试段预测对比图与相关性热力图）→ forecast/runs/
.venv/bin/python -m forecast.train --plot

# 3. 行情模式识别：事后标签、识别器训练与 val/test AUC/AP 评估、门控阈值 τ 标定
#    （幂等，统一回测与 webviz 也会按需自动重训）→ forecast/regime/runs/
.venv/bin/python -m forecast.regime.train

# 模式经济验证：常开回放下按不利模式占比分桶比较日度 g → forecast/regime/runs/economics.json
.venv/bin/python -m forecast.regime.economics

# 4. 强化学习算法：全量实验的训练与测试段评估（基线 + RL 及消融，统一训练——每个
#    (方法, w, λ, 种子) 作业内含全部标的；多进程并行、幂等可中断后续跑；只训练 RL 自身，
#    预测缓存只读）→ control/runs/
.venv/bin/python -m control.train

# 超参梯子：hindsight 权重 w 与风险厌恶 λ
.venv/bin/python -m control.train --methods GRID \
    --hindsight-weights 0.02 0.05 0.1 0.2 --inventory-lambdas 0 1 3 10 30

# 超参数一次一因子探索（网络结构 / 优化 / 折扣 / 奖励塑形 / 目标网络与优先级回放，
# 中心点为默认配置，w/λ 档位同上；幂等，断点续跑）→ control/runs/sweep/
.venv/bin/python -m control.sweep

# 组合确认：只训练中心点与给定组合（值不限于梯子档位），多种子配对对照
.venv/bin/python -m control.sweep --combo gamma=0.995 hindsight_weight=0.1 --seeds 0 1 2

# 不联网记录（离线落盘后 wandb sync）或完全关闭
.venv/bin/python -m control.train --wandb-mode offline
.venv/bin/python -m control.train --wandb-mode disabled

# 按验证集 SR 锁定 (w, λ)，汇总测试集指标 → control/runs/summary.csv（各段行情状态打印到控制台）
.venv/bin/python -m control.summarize

# 5. 统一回测（测试段：常开 / 模式门控（事后标签与识别概率）/ RL 智能体；识别器或预测缓存
#    缺失、过期时自动先重训，RL 检查点缺失时跳过 agent 模式）→ strategy/runs/
.venv/bin/python -m strategy.backtest

# 冒烟测试（可指定标的，默认 301308）
.venv/bin/python -m control.smoke_test

# 单元测试（部分用例依赖 data/ 与 cache/ 中的真实数据）
.venv/bin/python -m unittest discover -s tests
```

## 可视化

```bash
# 预测算法：逐日价格、窗口统计（含识别概率）与 常开/识别门控 的网格回放；
# 同日也导出过 control 时，页面对比栏追加「强化学习决策」第三张卡片
.venv/bin/python -m webviz.export --algorithm forecast --symbols 301308

# 强化学习：从 control/runs/ 的检查点回放测试段贪心决策（需先跑 control.train 训练）
.venv/bin/python -m webviz.export --algorithm control --symbols 301308

# 浏览器查看
.venv/bin/python -m http.server 8000   # 访问 http://localhost:8000/webviz/
```

## 产物

- `forecast/runs/`：前瞻回归模型（`model/`）与预测评估（`metrics.json`、可选 `figures/`）。
- `forecast/regime/runs/`：识别器（`model/`）、`meta.json`（含门控阈值 τ）、识别评估 `metrics.json` 与经济验证 `economics.json`。
- `strategy/runs/`：统一回测（`summary.json` 与各指标热力图 SVG：费用后网格收益 $g$、日均闭环率与成交次数（各含等权与按成交笔数加权两个口径）、网格宽幅与门控占比）。
- `control/runs/`：`<method>[_w<权重>][_lam<λ>][_seed<k>].json` 统一训练结果（测试指标逐标的报告 + 全体等权行）与同名 `.pt` 检查点（选模后的最佳权重与逐标的标准化统计量，供统一回测与 webviz 回放），`summary.csv` 汇总表；`sweep/` 为超参探索的逐作业结果与 `sweep/summary.csv` 梯子对比表。
- 每个 RL 作业同时建一个 wandb run（项目 `gridscalper`）：逐验证点记录训练奖励、Q 损失与验证 TR、SR 及选模判据曲线；训练结束记录测试集指标与逐日表（design 7.5）。

## 关键参数

样本切分 7:1:2 / 分钟口径：回看 30 分钟、前瞻 15 分钟、缓存按分钟锚点一行 / 网格半宽默认 $0.1\times\operatorname{ATR}_3$（下限为前收的 $10^{-3}$）/ 成本：双边佣金 $10^{-4}$、卖出印花税 $5\times10^{-4}$（`strategy/costs.py`）。RL 侧：底仓 50 手 / 仓位带 $[0,100]$ 手 / 半宽 7 档（$0$ 为平回底仓，$0.05$–$0.25\times\operatorname{ATR}_3$，$100$ 为关闭；数量档收缩为 1、无独立分支）/ 决策间隔 10 分钟 / 每分钟折扣 0.99（TD 折扣 $0.99^{10}$）/ 存货惩罚 $\lambda=3$ / $w=0.2$ / 统一训练 3 epochs（每 epoch 验证 3 次，SR 逐标的等权聚合，选模取最近 3 点均值）/ 训练回合起点加 $U[0,10)$ 分钟偏移 / 默认单种子（`control.train --seeds` 可扩展多种子）。预测侧：门控与训练行即分钟锚点行、状态切换连续确认 2 拍、seed 2021；模式定义：残差宽比 < 1.3 且斜率宽比 > 0.9 的粘性平滑（保持概率 0.99、观测噪声 0.3），门控阈值 τ 验证段率配平；前瞻回归 LightGBM 每目标一个回归器（平方损失，输入含 symbol_id 分类特征）。修改见 `control/config.py`、`forecast/config.py` 与 `forecast/regime/config.py`。
