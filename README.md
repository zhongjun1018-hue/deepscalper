# GridScalper

成交驱动的自适应网格交易：在 A 股 3 秒级 tick 快照上，用强化学习动态调节网格的
**半宽、方向倾斜与单笔下注规模**。动作不再直接下单，而是设定一条触发规则，
交易由市场在未来某个时刻被动触发，因而问题是 SMDP 而非 MDP。

框架承自 DeepScalper（Sun et al., 2022）的多模态编码器、分支 Dueling Q 网络与
风险辅助任务；设计与全部取舍理由见 [docs/design.md](docs/design.md)。

## 环境

```bash
uv sync                # torch(CUDA 12.8) / numpy / pandas / pyarrow / wandb
.venv/bin/wandb login  # 首次使用需登录；离线或不记录见下文 --wandb-mode
```

训练默认按 `Config.device = "auto"` 选择设备（有 CUDA 则用 CUDA，否则 CPU）。

## 数据

`data/<code>/` 下放置月度 tick parquet（如 `301308.SZ_202602_tick.parquet`）。
仅使用连续竞价时段（09:30–11:30、13:00–14:57）：集合竞价与午休行情不适用逐笔穿价撮合，
已在加载阶段剔除。每日 ATR(3) 由此前 3 个完整交易日预计算，当日盘中恒定、无前视。

## 运行

```bash
# 冒烟测试（可指定标的，默认 301308）
.venv/bin/python scripts/smoke_test.py

# 全量实验（基线 + RL 及消融，多进程并行；幂等，可中断后续跑）
.venv/bin/python scripts/run_all.py --symbols 301308 688030

# 超参梯子：hindsight 权重 w 与风险厌恶 λ
.venv/bin/python scripts/run_all.py --methods GRID \
    --hindsight-weights 0.02 0.05 0.1 0.2 --inventory-lambdas 0 3 10 30 100

# 不联网记录（离线落盘后 wandb sync）或完全关闭
.venv/bin/python scripts/run_all.py --wandb-mode offline
.venv/bin/python scripts/run_all.py --wandb-mode disabled

# 按验证集 SR 锁定 (w, λ)，汇总锁定配置的测试集指标
.venv/bin/python scripts/summarize.py

# 单元测试
.venv/bin/python -m unittest discover -s tests
```

结果写入 `results/<symbol>/<method>[_w<权重>][_lam<λ>][_seed<k>].json`，
含四指标、逐日超额净值与闭环率、design 7.4 的补充指标。并行度缺省按设备自适应：
CUDA 下取 2（避免争抢显存），CPU 下取 `核数 / Config.num_threads`，可用 `--workers` 覆盖。

每个 RL 作业同时建一个 wandb run（项目 `gridscalper`，`--wandb-project` 可改）：
每个 epoch 内验证 3 次，记录训练奖励、Q 损失、波动率辅助损失与验证 TR、SR 曲线
（横轴为累计梯度更新次数）；训练结束后记录测试集四指标、日均买卖笔数与
日均闭环率 2·min(Ns, Nb)/(Ns + Nb)，以及逐日超额收益与闭环率表（design 7.5）。

每个标的单独训练一个智能体；只有偏好超参 $(w,\lambda)$ 与各档位表跨标的共用，
选参只用验证集（design 7.1）。

## 关键参数

底仓 50 手 / 仓位带 [0, 100] 手 / 半宽 7 档（0.075–0.30 × ATR₃）/ 倾斜 7 档 /
数量 4 档（0, 2, 3, 5 手）/ 超时 60 tick / 每 tick 折扣 0.9995 / 存货惩罚 λ = 30 /
w = 0.1 / η = 1.0 / 6:2:2 按日切分 / 5 epochs（每 epoch 验证 3 次）/ 3 种子。
常开网格参照基线为 h = 0.10 × ATR₃、q = 3 手。
当前实验不计显性费用（佣金与印花税置零，见 design 3.4），执行成本照常记账。
修改见 `gridscalper/config.py`。
