# 双算法决策回放查看器

`index.html` 按算法、标的和交易日展示两个算法的实时决策过程，页面为免框架单文件实现，通过 `fetch` 读取 `webviz/data/`：

- **模式门控网格（forecast）**：逐日价格曲线、滑动窗口统计，以及各 scheme 的网格回放事件。行情模式识别控制策略启停，且只在净持仓为 0 时切换（engine 语义）。
- **强化学习（control）**：从控制器检查点回放贪心决策轨迹——逐日 mid 曲线、各决策点的生效网格（半宽档与上下轨）、成交标记与决策点平仓事件。

## 使用

```bash
python -m webviz.export --algorithm forecast --symbols 301308
python -m webviz.export --algorithm control --symbols 301308 --method GRID --seed 0
python -m http.server 8000
```

在项目根目录执行命令后，访问 <http://localhost:8000/webviz/>。页面不能直接用 `file://` 打开。`--symbols` 缺省为 `data/` 下全部标的。

control 侧 `--checkpoint` 缺省时按 `control/runs/<symbol>/` 的结果命名规则解析：先拼 `<method>_w<w>_lam<λ>_seed<s>.pt`（$w$/$\lambda$ 缺省取 `control.config.Config` 默认值），未命中时退到 `<method>_*_seed<s>.pt` 的唯一匹配（control.train 的命名规则是不适用的超参不在文件名中，如 GRID-NH 无 w 标签）；多重匹配或文件不存在会给出明确错误提示——先运行 `python -m control.train` 完成训练，或用 `--checkpoint PATH` 显式指定。

两个算法可以分别导出，`index.json` 合并记录各自可用的 (symbol, date)。有 forecast 数据时页面以其为主视图，同日 control 数据并入单日对比栏的第三张卡片；仅导出 control 时页面退化为纯决策回放（隐藏窗口检查器、门控对比与窗口开关）。生成的 `webviz/data/` 已被 `.gitignore` 忽略。

## 数据语义

### forecast（`data/forecast/<symbol>/<date>.json`）

| 内容 | 口径 |
| --- | --- |
| 价格曲线 | 连续竞价快照的 `LastPx`（`data_provider.ticks.load_days`，不含集合竞价与午休） |
| 窗口 | 前瞻窗口长度 300 tick，按门控决策间隔 20 tick 取样展示 |
| 窗口统计 | 前瞻窗口对数中间价路径的实测统计（`data_provider.windows.path_stats`）；波动、路径、极差、残差和趋势位移换算为 bp，效率比和方向反转率显示为百分比 |
| 识别概率 | `forecast.regime.classify.day_prob`：识别器输出的不利模式概率 $P$，stride 网格上推理并前向填充到全 tick；tick $t$ 的概率覆盖 $(t,t+H]$，与该 tick 之后的前瞻路径对齐 |
| 网格半宽 | 统一缓存的逐日固定半宽 $W$（`strategy.width.grid_width`，$0.1\times\operatorname{ATR}_3$） |
| 网格回放 | `strategy.engine.run_day(trace=True)`：测试段（样本外）标签可判定的交易日，统一自可预测起点 $t_0$（回看窗满，即回看长度 600 tick）起回放；对手方一档严格越界成交，成交价为触发边界 |
| 门控 | 逐 tick 的布尔信号；引擎只在净持仓为 0 时读取当拍信号，空仓期间每 20 tick 复判一次，敞口归零即刻重判 |
| 单日对比 | 全天开启（none）/ 识别概率过滤（prediction，$P>\tau$）两种网格方案，同日导出过 control 数据时追加第三张卡片「强化学习决策」（点击切换到贪心回放路径；卡片指标与网格方案同构，$g$ 与统一回测 agent 模式同口径） |
| Score / 网格收益 | `strategy.metrics` 口径：$\mathrm{Score}=2\min(N_b,N_s)/N$；去量纲单日收益、收益/交易次数及当日平均网格半宽 |

### control（`data/control/<symbol>/<date>.json`）

| 内容 | 口径 |
| --- | --- |
| mid 曲线 | 连续竞价快照的一档中间价（复用价格曲线绘制，`price` 字段即 mid） |
| 市场构建 | `control.train.build_markets`（窗口特征 + LightGBM 预测，缺块补零）；标准化统计量仅在训练段拟合（`control.features.fit_feature_stats`） |
| 决策回放 | `control.trace.load_checkpoint` 重建网络与 Config，`control.trace.trace_day` 贪心回放测试段交易日 |
| 决策点 | `decisions`：成交 / 超时 / 日终触发；`width` 为生效半宽档（$\times\operatorname{ATR}_3$，0 表示平仓回底仓、100 为关闭档），`center/upper/lower` 为生效网格中心与上下轨（平仓档为 null） |
| 成交标记 | `fills`：`TradingEnv` 的全部成交，`kind` $\in\{\texttt{grid},\texttt{immediate},\texttt{flatten},\texttt{liquidate}\}$（区间末触发 / 决策点立即成交 / 决策点平仓 / 日终清仓，后两者按对手方一档价成交），`side` $\in\{\texttt{buy},\texttt{sell}\}$，`qty` 以手计 |
| 平仓事件 | 决策点平仓（`width` 为 0）在图上以菱形标记；平仓成交与网格成交同样以圆点标记，`kind` 在悬浮提示中区分 |
| 单日摘要 | `log` 为 `TradingEnv.episode_log`：日内成交笔数、闭环率、决策点数、平仓次数、换手与费用等；成交口径不含日终清仓（design 7.4）；`grid` 为与 forecast 网格方案同构的卡片摘要（$g=$ 超额收益 $\times B/W_d$、时间加权生效半宽、平仓/关闭档的停用时长） |

### index.json

```json
{
  "algorithms": {
    "forecast": {"symbols": [{"symbol", "name", "dates", "replay_dates"}]},
    "control": {"symbols": [{"symbol", "name", "dates", "replay_dates", "checkpoint"}]}
  },
  "minute_labels": ["0930", ..., "1456"],
  "params": {"model", "lookback_ticks", "pred_ticks", "stride_ticks",
             "atr_mult", "atr_window", "min_width_ratio", "pattern"}
}
```

`replay_dates` 为有网格回放的日期（forecast 的测试段、control 的全部导出日期）；`params` 由 forecast 导出写入，`pattern` 含模式阈值、平滑参数与门控概率阈值 $\tau$，页面据此渲染规则说明。条目的 `name` 为证券简称，由 `export.py` 的 `SYMBOL_NAMES` 按代码查表（行情快照无名称字段），未知标的回退为代码本身。

日固定网格半宽记为

$$
W_d=\max(0.1\times \mathrm{ATR}_3,\ 10^{-3}\times P_{\text{preclose}}).
$$

当日成交总数大于零时，买入次数、卖出次数和 Score 的关系为

$$
N_d=N_{b,d}+N_{s,d}>0,\qquad\mathrm{Score}_d=\frac{2\min(N_{b,d},N_{s,d})}{N_d}.
$$

残差趋势模式的逐 tick 规则观测为

$$
\frac{q_{0.9}\!\left(\left|\varepsilon\right|\right)}{w}<1.3\quad\text{且}\quad\frac{|s|}{w}>0.9,
$$

其中 $w=W_d/p_t$ 为当日相对半宽（$p_t$ 为判定 tick 的中间价）。事后标签为其粘性平滑（design 8.2），门控概率阈值 $\tau$ 在验证段率配平。

ATR 历史数据不足时不生成网格。测试集日期之外的 forecast 页面仍可查看价格和窗口信息，但不会生成网格动态轨迹。
