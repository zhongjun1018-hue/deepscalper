# 数据与特征

本文统一描述 forecast 与 control 两个任务所需的全部特征。两个任务的回看窗口相同：任一 tick 都可作为决策点，其特征由截至该 tick 的 $L=600$ tick 构成，缓存因而逐 tick 存一行。forecast 在空仓期间每 $s=20$ tick 判一次门控，预测随后 $H=300$ tick 的路径统计量；control 的决策点由成交 / 超时 / 日终事件触发（见 [../docs/design.md](../docs/design.md) 4.1），落在哪个 tick 就取哪一行。

主体分三部分：§3 两任务共用的公共特征，§4 forecast 的输入与预测，§5 control 的状态特征；§1 与 §2 是两者共同的数据口径与记号约定。

## 1. 数据与窗口

每个标的目录包含一个或多个 Parquet 快照文件，文件可按日或按月组织，交易日统一读取 `MDDate` 字段。特征计算使用以下字段组：

| 字段组 | 主要字段 | 用途 |
| --- | --- | --- |
| 时间 | `MDDate`、`MDTime` | 排序、连续竞价筛选和时段判定 |
| 成交 | `LastPx`、`NumTrades`、`TotalVolumeTrade`、`TotalValueTrade` | 成交增量、方向、笔数与最新价 |
| 一档与十档盘口 | `Buy/Sell<1..10>Price`、`Buy/Sell<1..10>OrderQty` | 中间价、深度、价差和不平衡 |
| 盘口汇总 | `TotalBidQty`、`TotalOfferQty`、`WeightedAvgBidPx`、`WeightedAvgOfferPx` | 全盘口压力 |
| 委托笔数 | `NumBidOrders`、`NumOfferOrders`、`Buy/Sell<1..10>NumOrders` | 一档订单数与各口径笔数失衡（§3.4、§5.1） |
| 撤单 | `WithdrawBuyAmount`、`WithdrawSellAmount` | 撤单失衡（§5.1） |
| 一档订单明细 | `Buy1OrderDetail`、`Sell1OrderDetail` | 大单占比 |
| 日内状态 | `PreClosePx`、`OpenPx`、`HighPx`、`LowPx`、`MaxPx`、`MinPx` | ATR 网格宽度、日内位置与涨跌停距离 |

只保留连续竞价时段（09:30–11:30 含 11:30、13:00–14:57 不含 14:57）且 `LastPx`、`Buy1Price`、`Sell1Price` 均为正的快照（剔除集合竞价、午休与盘口单边为空的快照）；同一标的、日期和 `MDTime` 重复时，保留排序后的最后一条。设当日快照数为 $T$，缓存的行索引即 tick 索引：

- 行 $t$ 的输入窗口为第 $t-L+1$ 至 $t$ 条快照，仅 $t\ge L-1$ 且窗口内中间价路径与累计成交字段均有效时计算特征；
- 目标路径以 $t$ 为锚点、连接随后 $H$ 条快照，仅 $t\ge L-1$（与特征行对齐）、$t+H\le T-1$ 且前瞻路径上同样字段均有限时有目标。

缓存按标的单日最大快照数对齐，窗口不完整、前瞻越界与尾部对齐的行一律记 $\mathrm{NaN}$；缓存不填零、不增加缺失指示列，LightGBM 原生处理缺失值（control 读取时按 §5.2 口径置零）。

forecast 的训练与评估只取每 $s=20$ tick 一行——相邻 tick 的窗口重叠 $(L-1)/L$，全量入模只是重复样本；5 个目标全部有限且至少 1 个输入特征有限的行才进入训练集。推理则覆盖全部 tick，使门控与 control 在任一决策 tick 都能读到当拍预测。

## 2. 统一记号与退化规则

任一窗口包含 $n+1$ 条快照，索引为 $i=0,\ldots,n$。定义一档买价 $B_i$、一档卖价 $S_i$、中间价 $P_i$、对数中间价 $p_i$ 和相邻收益 $r_i$：

$$
P_i=\frac{B_i+S_i}{2},\qquad p_i=\log P_i,\qquad r_i=p_i-p_{i-1},\quad i=1,\ldots,n.
$$

累计成交量、成交额和成交笔数分别记为 $V_i$、$A_i$ 和 $N_i$，窗口增量为

$$
\Delta V=V_n-V_0,\qquad\Delta A=A_n-A_0,\qquad\Delta N=N_n-N_0.
$$

相邻快照增量记为 $v_i=V_i-V_{i-1}$、$a_i=A_i-A_{i-1}$ 和 $\Delta N_i=N_i-N_{i-1}$。符号 $\langle z_i\rangle$ 表示窗口内有限值的算术平均，$\operatorname{std}(z_i)$ 表示相同有限值的总体标准差。除特别说明的退化值外，分母为零或输入无效时结果为 $\mathrm{NaN}$。缓存直接保存以下公式的原始结果，不再额外取对数或缩放。

网格宽度以真实波幅的历史均值定义。对有效交易日 $d$：

$$
\mathrm{TR}_d=\max\left\{\mathrm{Hi}_d-\mathrm{Lo}_d,\lvert \mathrm{Hi}_d-C_d\rvert,\lvert \mathrm{Lo}_d-C_d\rvert\right\},
$$

$$
\mathrm{ATR}_d=\frac{1}{A}\sum_{j=1}^{A}\mathrm{TR}_{d-j},\qquad W_d=\max\left\{\eta\,\mathrm{ATR}_d,\ \varepsilon C_d\right\},
$$

其中 $C_d$ 为当日 `PreClosePx`，$A$ 为 `atr_window`，$\eta$ 为 `atr_mult`，$\varepsilon$ 为 `min_width_ratio`（默认 $A=3$、$\eta=0.1$、$\varepsilon=10^{-3}$）。ATR 只读取此前有效交易日；历史不足 $A$ 日时当日半宽缺失，依赖半宽的网格成交特征（§3.6）记 $\mathrm{NaN}$。forecast 的网格成交特征与门控回测使用当日固定半宽 $W_d$；control 环境的网格触发线同样以 $\mathrm{ATR}$ 为尺（见 `docs/design.md` 3.1）。

## 3. 公共特征

公共特征是经过窗口计算后 forecast 与 control 共同使用的特征：47 维窗口统计（§3.1–§3.6，逐 tick 一行）是 forecast 的模型输入，其中不与微观 / 私有序列重复的 24 维同时进入 control 的宏观向量（§5.2）；标的标识 `symbol_id`（§3.7）在两侧分别作为分类输入与 embedding 索引。

### 3.1 价格形态（4）

| 名称 | 定义 | 含义 |
| --- | --- | --- |
| `oc_ret` | $p_n-p_0$ | 窗口首尾净漂移 |
| `upper` | $\max_{0\le i\le n}p_i-\max(p_0,p_n)$ | 上影：向上试探未守住的幅度 |
| `lower` | $\min(p_0,p_n)-\min_{0\le i\le n}p_i$ | 下影：向下试探未守住的幅度 |
| `vwap_rel` | $\log(\Delta A/\Delta V)-p_n$ | 窗口成交均价相对现价的偏离 |

### 3.2 路径、趋势与波动（12）

令

$$
\bar p=\frac{1}{n+1}\sum_{i=0}^{n}p_i,\qquad\bar r=\frac{1}{n}\sum_{i=1}^{n}r_i,\qquad\bar i=\frac{n}{2}.
$$

对 $p_i$ 关于快照序号做带截距 OLS。斜率和残差为

$$
\beta=\frac{\sum_{i=0}^{n}(i-\bar i)(p_i-\bar p)}{\sum_{i=0}^{n}(i-\bar i)^2},\qquad e_i=p_i-\bar p-\beta(i-\bar i).
$$

当 $n\ge2$ 时，有限样本修正后的已实现双幂变差为

$$
\mathrm{BV}=\frac{\pi}{2}\frac{n}{n-1}\sum_{i=2}^{n}\lvert r_i r_{i-1}\rvert.
$$

| 名称 | 定义 | 含义 |
| --- | --- | --- |
| `rv` | $\mathrm{RV}=\sqrt{\sum_{i=1}^{n}r_i^2}$ | 已实现波动 |
| `path_len` | $\mathrm{PL}=\sum_{i=1}^{n}\lvert r_i\rvert$ | 路径总里程：往返活跃度 |
| `range_rel` | $\mathrm{Rg}=\max_{0\le i\le n}p_i-\min_{0\le i\le n}p_i$ | 窗口高低区间宽度 |
| `resid_abs_mean` | $\mathrm{RM}=\dfrac{1}{n+1}\sum_{i=0}^{n}\lvert e_i\rvert$ | 去趋势后的平均振荡幅度 |
| `resid_abs_q90` | $Q_{0.9}\!\left(\{\lvert e_i\rvert\}_{i=0}^{n}\right)$ | 去趋势振荡的尾部幅度 |
| `abs_slope` | $n\lvert\beta\rvert$ | 趋势强度：窗口线性漂移幅度 |
| `ret2` | $(p_n-p_0)^2$ | 净位移平方：漂移的方差贡献 |
| `er` | $\lvert p_n-p_0\rvert/\mathrm{PL}$ | 效率系数：位移占里程之比，区分趋势与震荡 |
| `rev_rate` | $\dfrac{\#\{i\in\{2,\ldots,n\}:r_i r_{i-1}<0\}}{\#\{i\in\{2,\ldots,n\}:r_i r_{i-1}\ne0\}}$ | 相邻收益反转频率：均值回复强度 |
| `ac1` | $\dfrac{\sum_{i=2}^{n}(r_i-\bar r)(r_{i-1}-\bar r)}{\sum_{i=1}^{n}(r_i-\bar r)^2}$ | 收益一阶自相关：正为动量、负为回复 |
| `semivar_asym` | $\dfrac{\sum_{i=1}^{n}r_i^2\operatorname{sign}(r_i)}{\sum_{i=1}^{n}r_i^2}$ | 上下行波动的不对称 |
| `jump` | $\max(\mathrm{RV}^2-\mathrm{BV},0)/\mathrm{RV}^2$ | 跳跃方差占比 |

当路径为常数时，`er`、`ac1`、`semivar_asym` 和 `jump` 均为 $\mathrm{NaN}$。`rev_rate` 在没有非零相邻收益乘积时为 $\mathrm{NaN}$；$n<2$ 时 `jump` 也为 $\mathrm{NaN}$。

### 3.3 成交与价格冲击（6）

对 $v_i>0$ 的快照区间定义成交均价、方向和有向成交量：

$$
\bar P_i=\frac{a_i}{v_i},\qquad s_i=\operatorname{sign}(\log\bar P_i-p_{i-1}),\qquad x_i=s_i v_i.
$$

当 $v_i\le0$ 时令 $x_i=0$。

| 名称 | 定义 | 含义 |
| --- | --- | --- |
| `amount` | $\Delta A$ | 窗口成交额：活跃度规模 |
| `trade_size` | $\Delta A/\Delta N$ | 平均单笔成交额：大小单结构 |
| `ofi` | $\sum_{i=1}^{n}x_i/\Delta V$ | 有向成交量占比：主动买卖净方向 |
| `kyle` | $\dfrac{\sum_{i=1}^{n}r_i x_i}{\sum_{i=1}^{n}x_i^2}$ | Kyle $\lambda$：单位有向成交量的价格冲击 |
| `idle_share` | $\dfrac{1}{n}\sum_{i=1}^{n}\mathbf{1}_{\{v_i=0\}}$ | 无成交快照占比：交投清淡程度 |
| `trade_conc` | $n\sum_{i=1}^{n}(v_i/\Delta V)^2$ | 成交量时间集中度：1 为均匀，越大越脉冲 |

### 3.4 盘口统计（14）

对档位 $j=1,\ldots,10$，记买卖价格为 $B_{j,i}$、$S_{j,i}$，买卖挂单量为 $Q^b_{j,i}$、$Q^a_{j,i}$，并定义

$$
b_{j,i}=\log B_{j,i},\qquad a_{j,i}=\log S_{j,i},
$$

$$
D_i=\sum_{j=1}^{10}(Q^b_{j,i}+Q^a_{j,i}),\qquad I_i^{(k)}=\frac{\sum_{j=1}^{k}(Q^b_{j,i}-Q^a_{j,i})}{\sum_{j=1}^{k}(Q^b_{j,i}+Q^a_{j,i})},\quad k\in\{1,10\}.
$$

记交易所汇总买卖挂单量为 $Q_i^{\mathrm{TB}}$、$Q_i^{\mathrm{TO}}$，加权平均买卖价为 $P_i^{\mathrm{WB}}$、$P_i^{\mathrm{WO}}$。有效相邻一档报价集合为 $\mathcal V$，其中报价发生变化的子集为 $\mathcal C$。对买卖侧 $c\in\{b,a\}$，令 $P_i^c$ 和 $Q_i^c$ 分别表示一档价格和挂单量，逐快照队列变化率及其有效区间为

$$
\chi_i^c=\frac{\lvert Q_i^c-Q_{i-1}^c\rvert}{Q_{i-1}^c},\qquad
\mathcal H_c=\left\{i:P_i^c=P_{i-1}^c,\ Q_{i-1}^c>0,\ P_i^c,Q_i^c,Q_{i-1}^c\in\mathbb R\right\}.
$$

| 名称 | 定义 | 含义 |
| --- | --- | --- |
| `spread` | $\langle a_{1,i}-b_{1,i}\rangle$ | 平均相对价差：即时交易成本 |
| `spread_cv` | $\operatorname{std}(a_{1,i}-b_{1,i})/\langle a_{1,i}-b_{1,i}\rangle$ | 价差变异系数：价差稳定性 |
| `qi1` | $\langle I_i^{(1)}\rangle$ | 一档量失衡：表层买卖压力 |
| `qi_gap` | $\langle I_i^{(10)}-I_i^{(1)}\rangle$ | 深层与表层失衡的背离 |
| `tqi` | $\left\langle\dfrac{Q_i^{\mathrm{TB}}-Q_i^{\mathrm{TO}}}{Q_i^{\mathrm{TB}}+Q_i^{\mathrm{TO}}}\right\rangle$ | 全盘口委托量失衡：整体买卖压力 |
| `depth_rel` | $\langle D_i\rangle/\Delta V$ | 深度相对成交量：盘口吸收能力 |
| `depth_cv` | $\operatorname{std}(D_i)/\langle D_i\rangle$ | 深度变异系数：深度稳定性 |
| `width` | $\dfrac{1}{2}\left\langle(b_{1,i}-b_{10,i})+(a_{10,i}-a_{1,i})\right\rangle$ | 十档价格覆盖半宽：盘口松紧 |
| `width_asym` | $\langle(a_{10,i}-a_{1,i})-(b_{1,i}-b_{10,i})\rangle$ | 买卖侧价格覆盖的不对称 |
| `far_press` | $\left\langle\dfrac{\log P_i^{\mathrm{WB}}+\log P_i^{\mathrm{WO}}}{2}-p_i\right\rangle$ | 全盘口挂单重心相对中间价的偏离 |
| `quote_rate` | $\lvert\mathcal C\rvert/\lvert\mathcal V\rvert$ | 一档报价变动频率：报价活跃度 |
| `queue_churn` | $\left\langle\dfrac{1}{\lvert\mathcal H_c\rvert}\sum_{i\in\mathcal H_c}\chi_i^c\right\rangle_{c:\lvert\mathcal H_c\rvert>0}$ | 一档队列换手率：挂撤单强度 |
| `l1_count` | $\langle N^b_{1,i}+N^a_{1,i}\rangle$ | 一档委托笔数：队列拥挤度 |
| `l1_top` | $\dfrac{1}{2}\left\langle\dfrac{M_i^b}{Q^b_{1,i}}+\dfrac{M_i^a}{Q^a_{1,i}}\right\rangle$ | 一档最大单占比：大单主导程度 |

其中 $N^b_{j,i}$、$N^a_{j,i}$ 为第 $j$ 档买卖委托笔数，$M_i^b$ 和 $M_i^a$ 分别为买卖一档已披露订单中的最大挂单量。

### 3.5 日内状态与 bar 边界（8）

这些状态只使用行 $t$ 及其之前的数据。当日首条快照的累计成交量记为 $V_{0}$，行末 $b$ 条快照构成当前 bar（区间 $(t-b,t]$，$b$ 为 `bar_ticks`），其成交量与当日已进行的平均每 bar 成交量为

$$
V_{\mathrm{bar}}=V_t-V_{t-b},\qquad \bar V_{\mathrm{bar}}=\frac{V_t-V_{0}}{(t+1)/b}.
$$

记快照 $i$ 的开盘价、当日最高价、当日最低价、涨停价、跌停价和最新成交价分别为 $O_i$、$\mathrm{Hi}_i$、$\mathrm{Lo}_i$、$\mathrm{Up}_i$、$\mathrm{Dn}_i$ 和 $P^{\mathrm{last}}_i$。

| 名称 | 定义 | 含义 |
| --- | --- | --- |
| `rel_day_open` | $p_t-\log O_t$ | 现价相对当日开盘的累计涨跌 |
| `vol_rel` | $V_{\mathrm{bar}}/\bar V_{\mathrm{bar}}$ | 当前 bar 量能相对当日平均：量能异动 |
| `dist_up` | $\log \mathrm{Up}_t-p_t$ | 距涨停的对数距离：上行空间 |
| `dist_dn` | $p_t-\log \mathrm{Dn}_t$ | 距跌停的对数距离：下行空间 |
| `range_pos` | $\operatorname{clip}\!\left(\dfrac{P^{\mathrm{last}}_t-\mathrm{Lo}_t}{\mathrm{Hi}_t-\mathrm{Lo}_t},0,1\right)$ | 现价在当日高低区间中的位置 |
| `day_pos` | $t/(T-1)$ | 日内时间进度 |
| `session` | $\mathbf{1}_{\{t\text{ 处在下午时段}\}}$ | 上 / 下午时段标记 |
| `gap` | $p_{t-b+1}-p_{t-b}$ | 当前 bar 边界处的跳空 |

当 $\mathrm{Hi}_t=\mathrm{Lo}_t$ 时，`range_pos` 取 $0.5$。

### 3.6 网格成交状态（3）

使用与 `strategy/engine.py` 相同的固定宽度成交规则，在输入窗口第一条快照的中间价建立新网格，半宽取当日固定半宽 $W_d$（§2）。对手方一档严格穿越边界时成交一笔（买一上穿上边界则卖出、卖一下穿下边界则买入，严格不等号含浮点容差），成交后把触发边界设为新中心。建网快照自身不判成交；每条快照最多成交一笔，同时穿越时先判卖出侧；日终不强制平仓，残余敞口即 `abs_exposure` 的来源。

| 名称 | 定义 | 含义 |
| --- | --- | --- |
| `buy_count` | $N_b$ | 窗口内网格买入成交次数 |
| `sell_count` | $N_s$ | 窗口内网格卖出成交次数 |
| `abs_exposure` | $\lvert N_b-N_s\rvert$ | 网格残余敞口：单边趋势程度 |

网格成交状态只作为输入特征刻画窗口内的网格可成交性，两个任务都不预测它。

### 3.7 标的标识

`symbol_id` 为标的在排序后运行标的集合中的索引，两个任务共用同一标识：forecast 把它作为分类输入列拼在 47 维窗口特征之后（LightGBM categorical feature）；control 把它经 embedding 拼接进市场嵌入（`control/model.py`）。单标的运行中它为常数、不影响学习；其作用是支持多标的合并训练时按标的区分行为，且标的数变化不改变窗口特征与观测维度。

## 4. forecast 输入与预测

预测算法的输入为 §3 的公共特征全集：47 维窗口特征加 `symbol_id` 分类列，共 48 维。输出为 5 个前瞻目标的逐 tick 预测。

### 4.1 预测目标

5 个目标均为 §3.2 的路径统计量在前瞻路径（以行 $t$ 为锚点、连接随后 $H$ 条快照）上的取值：

| 目标 | 定义 | 含义 |
| --- | --- | --- |
| `path_len` | $\mathrm{PL}$ | 前瞻路径总里程：网格可成交的往返量 |
| `rv` | $\mathrm{RV}$ | 前瞻已实现波动 |
| `range_rel` | $\mathrm{Rg}$ | 前瞻高低区间宽度 |
| `resid_abs_q90` | $Q_{0.9}(\lvert e_i\rvert)$ | 前瞻去趋势振荡的尾部幅度 |
| `abs_slope` | $n\lvert\beta\rvert$ | 前瞻趋势强度 |

### 4.2 训练与评估

LightGBM 为每个目标分别训练一个回归器，全部目标预测条件均值并采用平方损失，预测结果保持目标原始尺度：

$$
\mathcal L(y,\hat y)=(y-\hat y)^2.
$$

逐目标报告 $\mathrm{MAE}$、$\mathrm{RMSE}$ 及相对训练集均值基线的 skill。

## 5. control 状态特征

强化学习算法（`control/`）的观测由微观序列、宏观向量与私有序列三部分组成，另附公共标识 `symbol_id`（§3.7）。微观与宏观特征只用训练集拟合 z-score（截断 $\pm10$），验证集和测试集复用同一统计量。

### 5.1 微观序列（$30\times66$）

回看窗口每 20 tick 抽样一次的 30 步序列，末帧恰为决策点所在快照。前 40 维为十档相对价量：各档价格相对买一价取相对偏离，各档数量相对一档量取 $\log(1+\cdot)$。后 26 维为逐快照微观结构量——§3 的公共特征中凡为「逐快照量的窗口聚合」者，其逐快照原语均以同一定义纳入本序列：

| 维度 | 定义（快照 $i$） | 对应公共特征 |
| --- | --- | --- |
| 相对价差 | $a_{1,i}-b_{1,i}$ | `spread` |
| 一档失衡 | $I_i^{(1)}$ | `qi1` |
| 档位失衡差 | $I_i^{(10)}-I_i^{(1)}$ | `qi_gap` |
| 总量失衡 | $\dfrac{Q_i^{\mathrm{TB}}-Q_i^{\mathrm{TO}}}{Q_i^{\mathrm{TB}}+Q_i^{\mathrm{TO}}}$ | `tqi` |
| 十档深度 | $\log(1+D_i)$ | `depth_rel` |
| 盘口宽度 | $\dfrac{1}{2}\left[(b_{1,i}-b_{10,i})+(a_{10,i}-a_{1,i})\right]$ | `width` |
| 宽度不对称 | $(a_{10,i}-a_{1,i})-(b_{1,i}-b_{10,i})$ | `width_asym` |
| 买加权均价偏离 | $\log P_i^{\mathrm{WB}}-p_i$ | `far_press` |
| 卖加权均价偏离 | $\log P_i^{\mathrm{WO}}-p_i$ | `far_press` |
| 报价变化 | $\mathbf{1}_{\{B_{1,i}\ne B_{1,i-1}\ \text{或}\ S_{1,i}\ne S_{1,i-1}\}}$ | `quote_rate` |
| 一档队列变化率 | $\dfrac{1}{2}\left(\chi_i^b+\chi_i^a\right)$，$i\notin\mathcal H_c$ 的一侧记 0 | `queue_churn` |
| 一档笔数 | $\log(1+N^b_{1,i}+N^a_{1,i})$ | `l1_count` |
| 一档大单占比 | $\dfrac{1}{2}\left(\dfrac{M_i^b}{Q^b_{1,i}}+\dfrac{M_i^a}{Q^a_{1,i}}\right)$ | `l1_top` |
| 委托笔数失衡 | $\dfrac{N_i^{\mathrm{BO}}-N_i^{\mathrm{OO}}}{N_i^{\mathrm{BO}}+N_i^{\mathrm{OO}}}$（`NumBidOrders`、`NumOfferOrders`） | — |
| 十档笔数失衡 | $\dfrac{\sum_{j}(N^b_{j,i}-N^a_{j,i})}{\sum_{j}(N^b_{j,i}+N^a_{j,i})}$ | — |
| 撤单失衡 | $\dfrac{W_i^b-W_i^a}{W_i^b+W_i^a}$（`WithdrawBuy/SellAmount`） | — |
| 成交量增量 | $\log(1+v_i)$ | `idle_share` |
| 成交笔数增量 | $\log(1+\Delta N_i)$ | — |
| 成交均价偏离 | $\log\bar P_i-p_{i-1}$（$v_i\le0$ 或 $a_i\le0$ 时记 0） | `vwap_rel` |
| 有向成交量 | $s_i\log(1+v_i)$，$s_i$ 同 §3.3 | `ofi` |
| 中间价对数收益 | $r_i$ | `oc_ret`、`gap` |
| 日内开盘偏离 | $p_i-\log O_i$ | `rel_day_open` |
| 距涨停 | $\log \mathrm{Up}_i-p_i$ | `dist_up` |
| 距跌停 | $p_i-\log \mathrm{Dn}_i$ | `dist_dn` |
| 日内区间位置 | $\operatorname{clip}\!\left(\dfrac{P^{\mathrm{last}}_i-\mathrm{Lo}_i}{\mathrm{Hi}_i-\mathrm{Lo}_i},0,1\right)$ | `range_pos` |
| 时段 | $\mathbf{1}_{\{i\text{ 处在下午时段}\}}$ | `session` |

成交量、成交额与成交笔数为累计字段的相邻差分，累计量回退时负增量记 0；全部对数量以自然对数为单位，不再另行缩放。

其余公共特征没有可独立纳入的逐快照原语，它们构成宏观向量的窗口统计部分（§5.2）：路径统计量（§3.1 的 `upper`、`lower` 与 §3.2 全部 12 维）、窗口离散度（`spread_cv`、`depth_cv`、`trade_conc`）、窗口回归系数（`kyle`）与网格回放计数（§3.6）都要整段窗口才有定义；`vol_rel` 是 bar 级量能，由宏观向量的相对成交量承担（§5.2）；`amount` 与 `trade_size` 虽由成交量增量、成交笔数增量和成交均价偏离共同确定，但整段窗口的累计量不能由 30 个抽样步恢复，仍保留在宏观向量；`day_pos` 与私有序列的剩余时间等价（§5.3），两处都不重复纳入。

### 5.2 宏观向量（$40$）

$11+24+5$ 维：前 11 维由回看窗口聚合为 $L/b=30$ 根 $b$-tick bar，取当前 bar 的开 / 高 / 低相对其收盘价、收盘价环比、5/10/15/20/25/30-bar 均价偏离和相对成交量；中 24 维为 §3 公共特征的子集；末 5 维为 LightGBM 对 §4.1 五个前瞻目标的预测。后 29 维直接取自统一缓存 `cache/<symbol>.npz`。

中 24 维只保留整段窗口才有定义的量——凡在 §5.1 已有逐快照原语的公共特征都不再进宏观通道，`day_pos` 也因与私有序列的剩余时间等价而排除，避免同一信息在两条通道重复：

| 组 | 进宏观向量 |
| --- | --- |
| 价格形态（§3.1） | `upper`、`lower` |
| 路径、趋势与波动（§3.2） | 全部 12 维 |
| 成交与价格冲击（§3.3） | `amount`、`trade_size`、`kyle`、`trade_conc` |
| 盘口统计（§3.4） | `spread_cv`、`depth_cv` |
| 日内状态（§3.5） | `vol_rel` |
| 网格成交状态（§3.6） | 全部 3 维 |

窗口统计与预测直接取第 $t$ 行：该行的窗口恰好收于 $t$，不含未完结数据。$t<L-1$ 时窗口不完整，该行与缺失、退化的窗口一律记零。这 29 维是整窗口的聚合量与预测值（相邻行重叠 $(L-1)/L$），不是逐时刻原语，故进宏观 MLP 而非 LSTM。

### 5.3 私有序列（$30\times7$）

与微观序列同一时间索引抽样，每步 7 维。记底仓为 $Q_0$、初始权益为 $E_0$、当前网格中心为 $c$，半宽档与数量档的梯子上界为 $h_{\max}$ 和 $q_{\max}$：

| 维度 | 定义 |
| --- | --- |
| 仓位 | $\operatorname{pos}/(2Q_0)$ |
| 现金 | $\operatorname{cash}/E_0$ |
| 剩余时间 | $(T-1-t)/T$ |
| 中心偏离 | $(P_t-c)/\mathrm{ATR}$ |
| 生效半宽 | $h/h_{\max}$ |
| 生效数量 | $q/q_{\max}$ |
| 距上次成交 | $(t-t_{\mathrm{last\_fill}})/T$ |

中心偏离和当前网格参数使超时决策点能够恢复两条边界相对现价的位置。
