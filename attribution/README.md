# 门控归因（attribution）

给 forecast 的模式门控套一层解释：对引擎每一个**实际停网段**（门控点），从「趋势 vs 震荡」的模式识别角度说明为什么在这里停网、停网避开了什么、不停网会接到什么，并在展示页与常开 / 门控两套网格的回放和日度指标并列。本目录只读 `webviz/` 导出与统一缓存，不训练、不回测；展示页与 `webviz/index.html` 同为免框架单文件，绘图风格一致。

```
attribution/
  points.py     门控点：按引擎状态机重放实际停网段，算常开 / 门控两侧的段内对照
  payload.py    门控点负载：停网段及前后上下文 → 撰写解释所依据的结构化素材
  features.py   47 维特征字典（含义 / 单位 / 分组），与 data/features.md §3 一致
  export.py     页面数据导出：逐日门控点 + 已写好的解释 → runs/<symbol>_<date>.json、runs/index.json
  index.html    只读展示页：常开 / 门控两张网格回放并列，点击图中门控点看当日指标与解释
  runs/         解释 <symbol>_<date>_p<k>.md 与页面数据（不入库）
```

## 运行

```bash
.venv/bin/python -m webviz.export --algorithm forecast --symbols 301308       # 数据源
.venv/bin/python -m attribution.export --symbol 301308 --dates 20260717 20260701   # 页面数据
python -m http.server 8000     # 项目根目录；访问 http://localhost:8000/attribution/
```

`--dates` 缺省为全部测试段回放日。导出每日写 `runs/<symbol>_<date>.json`（全部门控点 + 已有解释）并登记到 `runs/index.json`，页面只读这两类文件与 `webviz/data/`，标的 / 日期下拉即已导出的交易日；新写或改写解释后重新导出即可。

## 门控点

webviz 的 `excluded` 是识别信号（P>τ）区段，引擎真正停网还要经「零敞口锚点读信号 + 连续 `confirm_n` 拍确认」。`points.replay_enabled` 在分钟网格上按 `strategy.engine.run_day` 的状态机重放（敞口取门控网格事件流；锚点是分钟末快照，分钟内成交先于判定），False 连续段即停网段，**每一段是一个门控点**。

页面在两张图的价格线上把每个门控点画成停网触发时刻的菱形标记并编号 `#k`，点击（或 ←/→）选中；门控侧网格区域在停网段断开。下方左表列当日两套网格的买入 / 卖出 / 成交 / 日终敞口 / 闭环率 / 网格收益 / 收益每笔（webviz 导出口径），副标题给该点段内价格位移与常开侧的成交、敞口、盯市，右侧为解释。对照口径：`mtm_W` 为段内新成交按段末价格盯市的浮动盈亏、`drift_W` 为段内价格位移，均以当日半宽为单位；常开在停网起点可能已带仓（`exposure_in`），解释要把这一点讲清楚。

展示日 2026-07-17：常开 18 笔、网格收益 −23.6，门控 12 笔、+2.9；6 个门控点，#1 09:59–10:21 为开盘阶梯下跌（常开接 2 买、浮亏 0.9 个半宽），#2–#5 是 10:52 V 形反弹后概率贴线（0.43–0.50）的一串碎片停网，#6 14:14–14:27 为放量拉升后反转下行（常开再接 1 买）。2026-07-01：上午整段停网 94 分钟，带大幅回抽的下行趋势，常开 16 买 10 卖、净累 6 手、浮亏 5.7 个半宽。

## 负载（`payload.build_payload`）

撰写解释所依据的素材，所有数字都应出自这里。

| 块 | 字段 | 含义 |
| --- | --- | --- |
| `meta` | `symbol / name / date`，`context{start_time, end_time}`，`params{lookback_min, pred_min, confirm_n, probability_threshold, pattern_rule}`，`units` | 上下文区间为停网段前 10 / 后 5 分钟；参数来自 index.json |
| `point` | `id`，`stopped_from / stopped_to / stopped_minutes`，`trigger_minute`，`drift_W`，`always_on / gated{fills, buys, sells, exposure_in, exposure_out, mtm_W}` | 停网段与两侧的段内对照（`points.gating_points`） |
| `day` | `grid_half_width`，`always_on / gated{trades, buys, sells, score, grid_profit}` | 当日半宽与两套网格的日度指标 |
| `minutes[]` | `minute / time / last_px`，`p_adverse`，`adverse_signal`（P>τ），`engine_stopped`（重放的引擎状态），`lookback_features`（47 维，每 3 分钟一次，触发分钟及前两拍必带），`forward_realized`（8 维前瞻实测），`rule_ratios{resid_q90_over_w, abs_slope_over_w}` | 判定与执行分开给出；前瞻实测与规则比值为事后量，只用于核对判定对错 |
| `events` | `always_on / gated[]{time, minute, kind, fill, exposure}` | 区间内两套网格的事件流 |

47 维特征的含义与单位见 [features.py](features.py)（bp = 对数量 ×1e4，pct = ×100，raw 原值）。

## 解释写法

每个门控点一篇 `runs/<symbol>_<date>_p<k>.md`，沿「趋势 vs 震荡」两条轴展开：网格靠震荡往返盈利（去趋势振荡相对半宽越大、反转越频繁，往返越多），被趋势单边吃掉（位移相对半宽越大，同向连续成交越多、敞口越堆越大）。回看侧对比趋势强度 / 效率系数 / 收益自相关 / 残余敞口与去趋势振荡 / 反转率 / 路径里程，事后侧看规则两个比值（斜率位移 / 半宽、90% 分位残差 / 半宽），段内看价格位移与买卖是否配对。

四个固定三级标题，每节 2–4 条要点，全文 500 字以内，中文表述、不写公式与字段名，数字带单位、以半宽为单位的量写作「x 个半宽」：

1. `### 为什么在这里停网`：触发拍及前两拍的回看特征里趋势侧与震荡侧各是什么量级、为何判为趋势压过震荡；概率如何越过 τ 并通过连续确认（日初自举不经确认；带仓期间不读信号）。
2. `### 门控（实际）`：停网时长、重开由哪两拍确认；用前瞻实测与规则比值核对事后是否确为趋势主导、震荡空间是否不足以往返。
3. `### 不门控（反事实）`：常开在同一段内的成交是否同向堆积、敞口从多少到多少、位移与盯市；起点带仓时说明此前已被堆仓。
4. `### 结论`：一句避开了什么 / 付出了什么；一句不确定性。不引入负载之外的信息。

页面按标题顺序渲染，`- ` 开头的行为要点。
