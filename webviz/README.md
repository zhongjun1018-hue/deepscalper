# 双算法决策回放查看器

`index.html` 按算法、标的和交易日展示两个算法的实时决策过程，页面为免框架单文件实现，通过 `fetch` 读取 `webviz/data/`：

- **预测门控网格（forecast）**：逐日价格曲线、滑动窗口统计，以及各 门控 $\times$ scheme 的网格回放事件。门控信号控制策略启停，且只在净持仓为 0 时切换（engine 语义）。
- **强化学习（control）**：从控制器检查点回放贪心决策轨迹——逐日 mid 曲线、各决策点的生效网格（半宽档与上下轨）、成交标记与决策点平仓事件。

## 使用

```bash
python -m webviz.export --algorithm forecast --symbols 301308
python -m webviz.export --algorithm control --symbols 301308 --method GRID --seed 0
python -m http.server 8000
```

在项目根目录执行命令后，访问 <http://localhost:8000/webviz/>。页面不能直接用 `file://` 打开。

control 侧 `--checkpoint` 缺省时按 `control/runs/<symbol>/` 的结果命名规则解析：先拼 `<method>_w<w>_lam<λ>_seed<s>.pt`（$w$/$\lambda$ 缺省取 `control.config.Config` 默认值），未命中时退到 `<method>_*_seed<s>.pt` 的唯一匹配（run_all 的命名规则是不适用的超参不在文件名中，如 GRID-NH 无 w 标签）；多重匹配或文件不存在会给出明确错误提示——先运行 `scripts/run_all.py` 完成训练，或用 `--checkpoint PATH` 显式指定。

两个算法可以分别导出，`index.json` 合并记录各自可用的 (symbol, date)，页面右上角切换算法。生成的 `webviz/data/` 已被 `.gitignore` 忽略。

## 数据语义

### forecast（`data/forecast/<symbol>/<date>.json`）

| 内容 | 口径 |
| --- | --- |
| 价格曲线 | 连续竞价快照的 `LastPx`（`data_provider.ticks.load_days`，不含集合竞价与午休） |
| 窗口 | 前瞻窗口长度 300 tick，按门控决策间隔 20 tick 取样展示 |
| 窗口统计 | 前瞻窗口对数中间价路径的实测统计（`data_provider.windows.path_stats`）；波动、路径、极差、残差和趋势位移换算为 bp，效率比和方向反转率显示为百分比 |
| 预测值 | 统一缓存 `cache/<symbol>.npz`（`data_provider.windows.load_cache`）的 `preds` 中 `abs_slope` 一列：行索引即 tick 索引，tick $t$ 的预测覆盖 $(t,t+H]$，与该 tick 之后的前瞻路径对齐 |
| 网格半宽 | 统一缓存的逐日固定半宽 $W$（`strategy.width.grid_width`，$0.1\times\operatorname{ATR}_3$） |
| 网格回放 | `strategy.engine.run_day(trace=True)`：测试段（7:1:2 切分的样本外）有预测的交易日，自 $t_0$（回看长度与首个可预测 tick 的较晚者）起回放；对手方一档严格越界成交，成交价为触发边界 |
| 门控 | `forecast.signals.build_gate_masks`：逐 tick 的布尔信号；引擎只在净持仓为 0 时读取当拍信号，空仓期间每 20 tick 复判一次，敞口归零即刻重判 |
| 门控对比 | 全天开启（none）/ 预测值过滤（prediction）/ 真实值过滤（oracle，仅事后参照）的残差趋势门控对比 |
| Score / 网格收益 | `strategy.metrics` 口径：$\mathrm{Score}=2\min(N_b,N_s)/N$；去量纲单日收益、收益下界及收益/交易次数 |

### control（`data/control/<symbol>/<date>.json`）

| 内容 | 口径 |
| --- | --- |
| mid 曲线 | 连续竞价快照的一档中间价（复用价格曲线绘制，`price` 字段即 mid） |
| 市场构建 | `control.train.build_markets`（窗口特征 + LightGBM 预测，缺块补零）；标准化统计量仅在训练段拟合（`control.features.fit_feature_stats`） |
| 决策回放 | `control.trace.load_checkpoint` 重建网络与 Config，`control.trace.trace_day` 贪心回放测试段交易日 |
| 决策点 | `decisions`：成交 / 超时 / 日终触发；`width` 为生效半宽档（$\times\operatorname{ATR}_3$，0 表示平仓回底仓、100 为关闭档），`center/upper/lower` 为生效网格中心与上下轨（平仓档为 null） |
| 成交标记 | `fills`：网格成交（含决策点立即成交）与平仓 / 日终扫单补记，`side` $\in\{\texttt{buy},\texttt{sell}\}$，`qty` 以手计 |
| 平仓事件 | 决策点平仓（`width` 为 0）在图上以菱形标记；日终平回底仓计入 `fills` |
| 单日摘要 | `log` 为 `TradingEnv.episode_log`：成交笔数、配对率、决策点数、平仓次数、换手与费用等 |

### index.json

```json
{
  "algorithms": {
    "forecast": {"symbols": [{"symbol", "name", "dates", "replay_dates"}]},
    "control": {"symbols": [{"symbol", "name", "dates", "replay_dates", "checkpoint"}]}
  },
  "minute_labels": ["0930", ..., "1456"],
  "closing_minutes": 0,
  "params": {"model", "lookback_ticks", "pred_ticks", "stride_ticks",
             "atr_mult", "atr_window", "min_width_ratio", "gate_rules"}
}
```

`replay_dates` 为有网格回放的日期（forecast 的测试段、control 的全部导出日期）；`params` 由 forecast 导出写入，页面据此渲染门控选项。

日固定网格半宽记为

$$
W_d=\max(0.1\times \mathrm{ATR}_3,\ 10^{-3}\times P_{\text{preclose}}).
$$

当日成交总数大于零时，买入次数、卖出次数和 Score 的关系为

$$
N_d=N_{b,d}+N_{s,d}>0,\qquad\mathrm{Score}_d=\frac{2\min(N_{b,d},N_{s,d})}{N_d}.
$$

残差趋势门控条件为

$$
\frac{q_{0.9}\!\left(\left|\varepsilon\right|\right)}{W_d/P_0}<1.3\quad\text{且}\quad\frac{|s|}{W_d/P_0}>0.9.
$$

ATR 历史数据不足时不生成网格。测试集日期之外的 forecast 页面仍可查看价格和窗口信息，但不会生成网格动态轨迹。
