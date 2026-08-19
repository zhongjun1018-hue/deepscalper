# GridScalper 设计

本文定义 GridScalper 的交易规则、两个算法的决策机制、学习目标与评估协议。实现位于 `data_provider/`、`strategy/`、`forecast/`、`control/`，实验入口位于 `scripts/`。

## 1. 目标与适用范围

项目在同一数据与回测底座上研究两个算法：

- **预测算法**（`forecast/`）：LightGBM 由 600-tick 窗口统计预测 5 个前瞻目标，预测值生成门控信号，控制固定 $0.1\times\mathrm{ATR}$ 网格的启停；门控只在净持仓为 0 时生效。算法本身不下单，交易由 `strategy/` 的报价驱动回放执行。
- **强化学习算法**（`control/`）：智能体不直接决定某一时刻买卖，而是在每个决策点自主设定整体半宽 $h$（$h=0$ 立即平回底仓，$h$ 极大等同停手）与单次成交量 $q$。参数持续生效，直到成交、超时或日终产生下一决策点，问题按 SMDP 建模。

两个算法共享：同一份原始数据与缓存（`data/`、`cache/`、`data_provider/`），同一套网格几何、撮合与成本口径（`strategy/`），同一次 7:1:2 时序切分（`data_provider/split.py`）。预测算法的输出同时是强化学习的状态特征（5.4）。

RL 框架承自 DeepScalper（Sun et al., 2022）的多模态编码器与分支 Dueling Q 网络；前瞻预测由 LightGBM 承担，网络不含预测辅助任务。本设计是快照级回测模型，不模拟交易所队列、订单撤改单消息或本策略对历史盘口的市场冲击。

## 2. 数据与市场口径

### 2.1 数据范围与加载

月度 parquet 文件位于 `data/<code>/`，由 `data_provider/ticks.py` 统一加载：使用连续竞价时段 09:30–11:30（含 11:30:00 快照）和 13:00–14:57，要求最新价、买一和卖一为正，同一（日期，时刻）重复记录保留最后一条。集合竞价与午休行情不适用逐笔穿价撮合，已在加载阶段剔除。中间价为

$$
p_t=\frac{\operatorname{bid}_{1,t}+\operatorname{ask}_{1,t}}{2}.
$$

当前两个实验标的的量级如下；每日快照数是过滤后的正常观测数量，不是数据截断下限。

| 量 | 301308（创业板） | 688030（科创板） |
|---|---:|---:|
| 交易日 | 119 | 119 |
| 价格中位数 | 380 元 | 16.3 元 |
| $\operatorname{ATR}_3$ / 前收中位数（10%–90%） | 7.2%（4.5%–11.0%） | 4.4%（2.7%–7.0%） |
| 一档价差中位数 | 1.8 bp | 10.9 bp |
| 买一 / 卖一深度中位数 | 3 / 4 手 | 13.3 / 12 手 |
| 每日有效快照数中位数 | 4741 | 3081 |
| 最小变动价位 | 0.01 元 | 0.01 元 |

每日真实波幅为

$$
\operatorname{TR}_D=\max\{\operatorname{Hi}_D-\operatorname{Lo}_D,\ |\operatorname{Hi}_D-C_{D-1}|,\ |\operatorname{Lo}_D-C_{D-1}|\},
$$

当日使用此前 $A=3$ 个完整交易日的均值

$$
\operatorname{ATR}_D=\frac1A\sum_{j=1}^{A}\operatorname{TR}_{D-j}.
$$

ATR 在当日盘中恒定且不含前视。最初 3 日没有足够历史，不生成回合。TR/ATR 的唯一定义在 `strategy/width.py`，数据加载与网格宽度共用。

### 2.2 样本切分

全项目使用同一次按时间的 7:1:2 切分（`data_provider/split.py`）：训练 70%、验证 10%、测试 20%。119 个交易日的边界为：训练 1–83，验证 84–94，测试 95–119。预测模型与 RL 智能体在同一日期切分上训练、验证与测试，样本外口径一致。单一切分下三段可能整段落在不同的市场状态，此时测试指标衡量的是跨状态外推而非泛化；每份结果同时记录三段的行情状态（7.4），作为解读测试指标的前提。

### 2.3 交易规则假设

- 价格最小变动单位为 0.01 元。
- 1 手按 100 股表示。网格每笔成交手数由数量档给出，不模拟科创板 200 股的最小申报量；该简化与队列、市场冲击等近似同属快照级回测口径，落地前需单独评估。
- 不施加 T+1。底仓提供可卖库存，但模型仍可能在累计卖出超过日初可用股份后卖出当日买入；因而结果表示 T+0 仿真，不等同于可直接执行的 A 股现货策略。
- 显性费率按线性模型计入：双边佣金与卖出印花税（3.3）。

卖出证券交易印花税的现行参考税率为 0.05%，来源见[财政部、税务总局公告](https://xj.mof.gov.cn/zcfagui/202311/t20231108_3915476.htm)。

## 3. 网格、撮合与成本

网格几何、穿价判定与费率的唯一定义在 `strategy/`（`grid.py`、`engine.py`、`costs.py`），两个算法共用。

### 3.1 网格几何

中心为 $c$，ATR 倍数半宽为 $h$。生效半宽先取千分之一下限

$$
\bar h=\max\!\left(h\operatorname{ATR}_3,\ \varepsilon c\right),\qquad \varepsilon=10^{-3},
$$

下限 $\varepsilon c$ 保证单格价差不低于往返成本与滑点的量级：ATR 过小时，过密网格的每次配对必亏，该约束直接禁止这种形态。下限的参考价按执行路径取值：RL 环境内为当前网格中心 $c$（`strategy/grid.py` 的 `half_width`）；逐日固定半宽（`strategy/width.py`，预测算法与窗口特征使用）以当日前收为基准（features.md §2）。

卖出边界向上取整到合法价位，买入边界向下取整，两侧对称：

$$
u=\left\lceil\frac{c+\bar h}{\Delta p}\right\rceil\Delta p,\qquad d=\left\lfloor\frac{c-\bar h}{\Delta p}\right\rfloor\Delta p.
$$

预测算法使用的固定半宽为 $h=0.1$（`strategy/width.py` 的 `atr_mult` 默认值）。

### 3.2 被动成交

RL 环境中，每次决策后视为在 $u$ 和 $d$ 各维护一张限价单。只有同时满足以下条件才判定成交：

1. 快照盘口有效且未交叉；
2. NumTrades 相对历史有效计数增加；
3. 对手方一档严格穿过边界：卖出要求 $\operatorname{bid}_1>u$，买入要求 $\operatorname{ask}_1<d$。

成交判断定义在可执行的一档报价上，并要求 NumTrades 增加，避免没有成交发生的纯报价更新。仅触及边界不能确定队列中的本单已成交，因此不计。

被动成交价格取本方限价 $u$ 或 $d$，跳价后的价格改进不归本策略。每条快照最多成交一笔，成交后中心重置为成交价。若买入记 $s=+1$、卖出记 $s=-1$，则

$$
\operatorname{pos}_{k}=\operatorname{pos}_{k-1}+s_k q_k,\qquad c_k=p^{\mathrm{fill}}_k.
$$

快照无法还原队列位置和本策略加入后的成交量。严格穿价避免“触价即成交”的乐观假设，但仍是近似撮合；liquidity_ratio 用成交量相对成交快照对手一档深度监控成交规模。条件 1–2 只作用于 RL 环境；报价驱动回放引擎与窗口网格特征（3.4）仅以对手方一档严格穿价（条件 3）判定成交。

### 3.3 成本与权益

费率在 `strategy/costs.py` 单点定义：双边佣金 $10^{-4}$、卖出印花税 $5\times10^{-4}$，两个算法的回测与记账统一引用。买入为 $s=+1$、卖出为 $s=-1$，成交 $q$ 手、价格 $f$ 时，省略所有账户量共同的 100 股因子：

$$
\operatorname{notional}=qf,
$$

$$
\operatorname{fee}=\operatorname{notional}\left(r_{\mathrm{commission}}+\mathbb{1}_{s=-1}r_{\mathrm{stamp}}\right),
$$

$$
\operatorname{execution\_cost}=s\,q(f-p_t).
$$

现金变化为 $-s\,qf-\operatorname{fee}$。因此成交前后按中间价计量的权益损失恰为 $\operatorname{fee}+\operatorname{execution\_cost}$；两者分别统计。被动成交的执行成本包含成交后价格已经穿过挂单价所反映的逆向选择，立即成交和日终扫单则主要反映价差与档位滑点。

实际券商可能有单笔最低佣金，当前线性模型不覆盖该非线性成本；低价、小单策略在落地前必须单独评估。

### 3.4 报价驱动回放引擎

`strategy/engine.py` 是预测算法的执行载体：逐日回放连续竞价报价，空仓且门控放行时在首个有效中间价建网，严格穿价成交一笔并把触发边界设为新中心；**只在净持仓为 0 时读取门控**，持仓期间继续执行当前网格；收盘停止回放，不强制平仓，未归零敞口按日终敞口记录。引擎成交价为未取整的边界价（取整只发生在 RL 环境的撮合口径中）。

引擎返回无量纲、不含费用的形态评分

$$
\operatorname{grid\_profit}=\frac{(N_b+N_s)+z^2}{2}+z\,\frac{y-x}{W},\qquad\operatorname{grid\_profit\_lower}=\frac{(N_b+N_s)-z^2}{2}-|z|,
$$

其中 $N_b$、$N_s$ 为买卖成交笔数、$z=N_b-N_s$ 为净持仓、$x$ 为入场价、$y$ 为日终价、$W$ 为当日半宽（推导中的格距 $d$）。完整推导见 [grid_profit.md](grid_profit.md)。该评分同时是窗口特征 `buy_count`/`sell_count`/`abs_exposure` 的标签口径；成本由 `strategy/backtest.py` 按 3.3 的费率叠加到成交流上，得到费用后净利润。

### 3.5 账户、仓位与日终（RL 环境）

RL 环境在 `strategy/` 的撮合与成本之上维护账户：底仓为 $Q_0=50$ 手，仓位范围为

$$
0\le\operatorname{pos}\le2Q_0.
$$

首个决策点价格为 $p_0$，初始持仓、现金、权益和奖励基准分别为

$$
\operatorname{pos}_0=Q_0,\qquad\operatorname{cash}_0=Q_0p_0,\qquad E_0=2Q_0p_0,\qquad B=Q_0p_0.
$$

买入量受仓位上界与现金约束；卖出量受当前持仓约束。触及边界时相应一侧关闭。

决策点若新动作设置的边界已被当前对手报价穿过，该限价单按对手价立即成交；$h=0$ 时不建网格，而是按十档盘口逐档扫单，把超额敞口 $\operatorname{pos}-Q_0$ 一次性平回底仓，与日终平仓共用同一撮合路径；盘口深度不足的残余留待后续决策点或日终处理。立即成交与平仓都不产生 $\tau=0$ 的新转移，其成本计入新动作开启区间的 $C$。立即成交后中心按实际成交价重置；平仓不重置中心，网格中心只在网格成交时移动。

日终只平掉超额敞口 $\operatorname{pos}-Q_0$，盘口深度不足留下的终端残余按最后中间价估值。每日回合随后独立重置，不携带隔夜敞口。

## 4. SMDP

### 4.1 决策点与时长

下一决策点是以下事件中最早发生者：

1. 被动成交；
2. 距当前决策点满 $K=100$ 个快照；
3. 当日最后一个快照。

区间时长 $\tau$ 以过滤后的快照索引差计量，满足 $1\le\tau\le K$。快照约每 3 秒一条，因此 $K$ 通常约为 5 分钟，但 tick 时间不是严格墙钟时间。超时机制保证 $h=0$ 与关闭档不会形成吸收态。

### 4.2 折扣

$\gamma$ 使用每 tick 口径：

$$
\gamma_{\mathrm{tick}}=0.99^{1/20}\approx0.9995.
$$

TD 目标的延续价值按 $\gamma_{\mathrm{tick}}^\tau$ 折扣。区间奖励定义为该转移内的累计权益变化，不再除以 $\tau$；这是事件级 SMDP 奖励口径。它避免使用固定“每次决策折扣”时窄网格因决策更频繁而受到系统性更重的折扣。

## 5. 动作与状态

### 5.1 动作空间

两个离散分支为

$$
h\in\{0,\ 0.05,\ 0.1,\ 0.15,\ 0.2,\ 0.25,\ 100\},\qquad q\in\{1\},
$$

联合动作有 $7$ 种，网络输出数为 $7+1=8$。$h$ 以 $\operatorname{ATR}_3$ 为单位，跨标的、跨波动状态可比：$h=0$ 在决策点扫单平回底仓；$h=100$ 的边界在日内无法触及，效果等同关闭网格——停手因此不是特权动作，无需单独的开关维度。$q$ 是风险规模接口，当前收缩为单档 $\{1\}$，扩展为 $\{1,2,3\}$ 即恢复数量维度。

$h\in\{0,100\}$ 时网格不触发，$q$ 对执行无意义：训练只更新半宽分支，下一状态若选择这两档，BDQ 延续价值也只取半宽分支；平仓档 $h=0$ 不建网格，环境把生效数量记为 0。这样避免等价的不触发组合向数量分支注入伪监督。

该因子化仍是假设：两个分支通过共享状态价值与编码器协调，不能完整表达任意动作交互。固定分支消融用于检验这种分解是否带来增量价值。

### 5.2 状态特征

私有状态、市场状态（微观序列与宏观向量）与标的标识的完整定义见 [../data/features.md](../data/features.md) §5 与 §3.7。其中宏观向量的中 24 维为窗口统计（5.3），末 5 维为 LightGBM 前瞻预测（5.4）；微观与宏观特征只用训练集拟合 z-score，验证集和测试集复用同一统计量。

### 5.3 窗口统计

`data_provider/windows.py` 逐 tick 预计算 600 tick 回看窗口的 47 维窗口统计（定义见 features.md §3），缓存于 `cache/<symbol>.npz`，行索引即 tick 索引，因而任一决策点都能取到恰好收于该 tick 的窗口；源数据或参数变化时缓存自动重建。宏观向量只取其中 24 维——微观序列已有逐快照原语的统计量不再重复进入（features.md §5.2）。特征中的网格成交状态由 `strategy/engine.py` 的回放给出（3.4），半宽取当日 $0.1\times\mathrm{ATR}$。

### 5.4 LightGBM 前瞻预测

宏观向量末 5 维是 LightGBM 对五个前瞻目标的预测：path_len、rv、range_rel、resid_abs_q90 与 abs_slope，均为平方损失的均值回归；输入为 47 维窗口统计拼接 symbol_id 分类特征（features.md §3.7）。预测作为状态特征在决策时可见，预测表征的学习压力从网络移至线下模型。

模型在训练段跨标的池化训练一次（各标的训练日期的并集），早停用验证日期，随后对全部交易日逐 tick 预测并回写进统一缓存 `cache/<symbol>.npz` 的 `preds`（8.2）。训练日的预测在样本内、带拟合乐观性，但不向验证/测试段泄漏；预测值只作为状态特征，不进入奖励或 TD 目标。

## 6. 奖励与学习

### 6.1 超额权益奖励

动作在 $t$ 处可能先立即执行：$h>0$ 且边界被穿越时立即成交，$h=0$ 时扫单平仓。记立即执行后的区间持仓为 $\widetilde{\operatorname{pos}}_t$，区间末为 $t+\tau$，该动作引起的立即执行、末端被动成交和日终平仓成本总和为 $C$：

$$
r_t=\frac{(p_{t+\tau}-p_t)\,(\widetilde{\operatorname{pos}}_t-Q_0)-C_{[t,t+\tau]}}{B}.
$$

该奖励扣除了固定底仓的被动收益，度量网格相对 $Q_0$ 底仓的增量结果。逐区间求和满足

$$
\sum_t r_t=\frac{(E_T-E_0)-Q_0(p_T-p_0)}{B},
$$

与决策点划分无关。底仓扣除在未贴现总收益下也可视为控制变量；在 $\gamma^\tau<1$ 的训练目标中，它同时明确了“优化超额收益”这一目标，不宣称严格保持原始贴现问题的最优策略不变。

### 6.2 训练塑形

训练奖励为

$$
r_t^{\mathrm{train}}=\frac1{\sigma_d}\left[r_t+w\frac{\tau}{K}\cdot\frac{(p_{t+H_{\mathrm{hs}}}-p_t)(\widetilde{\operatorname{pos}}_t-Q_0)}{B}\right]-\lambda\left(\frac{(\widetilde{\operatorname{pos}}_t-Q_0)p_t}{B}\right)^2\frac{\tau}{T},
$$

其中 $H_{\mathrm{hs}}=600$ tick 为 hindsight 视野，$T$ 为当日过滤后的快照数，

$$
\sigma_d=\frac{\operatorname{ATR}_3}{\sqrt{8/\pi}\,\operatorname{PreClose}}.
$$

主奖励与 hindsight 项都正比于当日价格波动，按 $\sigma_d$ 归一后各交易日的奖励尺度一致。不归一时，单个决策区间的奖励量级仅 $10^{-5}$，且高波动日的样本在回放中支配梯度，Q 网络最省事的解是把输出压向常数。归一也消去了存货项中的 $\sigma_d^2$：归一后区间收益的方差即 $\left((\widetilde{\operatorname{pos}}_t-Q_0)p_t/B\right)^2\tau/T$，$\lambda$ 因而成为跨交易日、跨标的可比的无量纲风险偏好。归一只作用于训练奖励，回测盈亏与财务指标仍用 $r_t$ 的原始口径。

hindsight 项用未来方向为当前超额敞口提供稠密信用信号，并乘 $\tau/K$，避免通过制造更多成交决策点重复领取塑形奖励。它只用于训练，不进入回测盈亏。

存货项是基于区间方差的局部二次风险代理。它与 Avellaneda–Stoikov 的存货风险思想一致，但在真实收益相关、自适应仓位和非恒定波动下不等于完整策略收益方差；$\lambda$ 应解释为无量纲风险偏好，而不是可由数据唯一估计的参数。候选档为 $\{0,1,3,10,30\}$。

### 6.3 BDQ 目标

两个分支共享状态价值：

$$
Q_d(s,a_d)=V(s)+A_d(s,a_d)-\frac1{|\mathcal A_d|}\sum_{a'_d}A_d(s,a'_d).
$$

在线网络选择下一动作，目标网络估值。网格开启时，联合延续价值为两个分支值的均值：

$$
\bar Q^-(s')=\frac12\sum_{d=1}^{2}Q_d^-(s',\arg\max_{a_d}Q_d(s',a_d)).
$$

若半宽分支选择不触发档（$h\in\{0,100\}$），只使用半宽分支的值。固定分支消融在动作选择和 TD 目标中都使用指定档位。所有有效分支共享同一目标

$$
y=r^{\mathrm{train}}+\mathbb 1_{\mathrm{nonterminal}}\cdot\gamma_{\mathrm{tick}}^\tau\bar Q^-(s').
$$

这保持了 BDQ 对一个联合动作价值的共同估计；各分支使用不同 TD 目标会使共享的 $V(s)$ 接收相互冲突的监督。

经验回放使用 proportional PER，各有效分支的平均绝对 TD 误差作为优先级。

行为策略为 $\varepsilon$-greedy，探索率在前 60% 的训练轮次内由 1.0 线性退火至 0.1，评估与测试一律贪心。每 2 个决策步做一次批量更新，目标网络每 500 次更新同步一次；PER 的重要性采样指数 $\beta$ 按 $1.2\times10^{4}$ 次更新的日程从 0.4 线性升至 1.0，该日程与训练规模（80 个训练日 $\times$ 5 epochs，约 $1.1\times10^{4}$ 次更新）对齐，训练末期偏差修正基本走完。

## 7. 强化学习：训练、基线与评估

### 7.1 切分、选模与选参

样本切分见 2.2（7:1:2 单次时序切分）。每个标的独立训练，默认 5 epochs、3 个随机种子。每个 epoch 内把训练日均分为 3 段，每段末在验证集上贪心回放一次。验证窗口仅 11 日，单点 SR 的噪声因而不可忽略；选模判据取最近 3 个评估点验证 SR 的均值而非单点最大值：单点取 max 是在验证噪声上挑选，评估点越多正偏越大（15 个独立评估点的最大值期望比均值高约 1.7 个标准差），窗口未满的评估点不参与选优。TD 损失受目标网络同步、PER 优先级与探索退火影响，不能作为收敛判据，验证曲线的点数因而决定了「是否还在改进」的可判性。$w$ 和 $\lambda$ 在全部标的与种子上聚合验证集 SR 后按方法选取同一档；测试指标不参与选模或选参，汇总表只展示锁定配置。固定参数扫描的格点预先定义，验证集选择格点，测试集报告选中格点及完整预设曲面。

### 7.2 基线与消融

| 方法 | 定义 |
|---|---|
| HOLD | 持有 $Q_0$ 底仓，报告被主奖励扣除的 beta |
| OPEN | $h=0.1$ 的常开对称网格 |
| SCAN | 预设半宽的单维扫描曲面，验证集 SR 选点 |
| GRID-NH | 去掉 hindsight |
| GRID-FW | 固定半宽分支（$h=0.1$），检验自适应选档的增量价值 |
| GRID | 完整模型 |

基本判据是 GRID 的测试超额收益优于 OPEN，同时比较 inventory_load，防止把单纯降杠杆误判为择时能力。超额收益为正表示同时优于不交易（$h$ 恒取不触发档时超额收益为 0）。

### 7.3 财务指标

逐日超额收益为 $r_d=\operatorname{net\_value}_d-1$。沿用 DeepScalper 的非年化定义（`strategy/metrics.py`）：

$$
\mathrm{TR}=\prod_d(1+r_d)-1,\qquad\mathrm{SR}=\frac{\mathbb E[r_d]}{\operatorname{std}(r_d)},
$$

$$
\mathrm{CR}=\frac{\mathbb E[r_d]}{\mathrm{MDD}},\qquad\mathrm{SoR}=\frac{\mathbb E[r_d]}{\sqrt{\mathbb E[\min(r_d,0)^2]}}.
$$

MDD 的峰值序列包含初始净值 1。分母为零时相应指标记 0。

### 7.4 诊断指标与行情状态

逐日记录并跨日汇总：

- 决策数、成交数、立即成交数、买卖笔数、闭环率及 $\tau$ 分布；
- 平均成交手数、平均中心移动 bp、成交量 / 可见流动性中位比；
- 最大绝对超额仓位、时间加权绝对仓位、inventory_load 与边界停留比例；
- $h$ 与 $q$ 的时间加权动作分布，决策点平仓次数与手数；
- 换手率、显性费用、执行成本和日终平仓手数。

其中

$$
\mathrm{inventory\_load}=\sum_i\frac{\tau_i}{\sum_j\tau_j}\left(\frac{(\operatorname{pos}_i-Q_0)p_i}{B}\right)^2.
$$

闭环率刻画当日买卖笔数的配对程度：

$$
\mathrm{closure\_rate}=\frac{2\min(N_b,N_s)}{N_b+N_s}.
$$

取值 1 表示买卖笔数相等（网格完整往返），0 表示当日只有单边成交或无成交，因而它区分「靠网格往返赚价差」与「靠单边累积敞口赚方向」。

平均中心移动只描述网格中心的移动幅度，不等同于配对后的已实现利润。

每份结果同时记录训练、验证、测试三段的行情状态：日内漂移的均值与标准差、上涨日占比、$\operatorname{ATR}_3/\operatorname{PreClose}$，作为解读测试指标的前提（2.2）。

### 7.5 实验跟踪

RL 作业逐个对应一个 wandb run，run 名与结果文件同名，按标的分组、按方法分类，config 记录 Config 全字段与作业标识 (symbol, method, seed, w, lambda)。记录内容：

- 逐验证点曲线：train/reward（$\varepsilon$-greedy 回放的日均超额收益）、train/q_loss、val/TR、val/SR 与选模判据 val/SR_window，横轴为累计梯度更新次数，epoch 一并记录为可选横轴；
- 训练后 summary：test/TR、SR、CR、SoR，以及测试集日均的 test/n_buys、n_sells、closure_rate；
- 训练后 test/daily 表：测试集逐日超额收益与闭环率。

奖励与 Q 损失都按相邻验证点之间的样本取均值。

结果写入 `control/runs/<symbol>/<method>[_w<权重>][_lam<λ>][_seed<k>].json`；RL 作业同时把选模后的最佳网络权重与 Config 存为同名 `.pt` 检查点，供 webviz 回放决策过程（9）。汇总表写入 `control/runs/summary.csv`。Config.wandb_mode 取 "online" / "offline" / "disabled"，后者关闭全部记录。

## 8. 预测算法：门控网格

### 8.1 决策机制

预测算法不在盘中择时下单。LightGBM 逐 tick 给出 5 个前瞻目标的预测，`forecast/signals.py` 把预测翻译成逐 tick 布尔门控信号，`strategy/engine.py` 执行固定半宽 $0.1\times\mathrm{ATR}$ 的对称网格，并只在净持仓为 0 时读取当拍信号——命中即不开新网，已持仓期间继续执行当前网格直至敞口归零。

与强化学习的分工差异：

| | 预测算法 | 强化学习 |
|---|---|---|
| 决策内容 | 网格启停（二值门控） | 半宽 $h$ 与数量 $q$ 的档位 |
| 决策依据 | LightGBM 前瞻预测 | 智能体 Q 值（状态含预测特征） |
| 决策时点 | 空仓期间每 20 tick 复判一次，敞口归零即刻重判 | 成交/超时/日终触发的 SMDP 决策点 |
| 网格几何 | 固定 $0.1\times\mathrm{ATR}$ | 动作空间 7 档（5.1） |
| 持仓处理 | 不强制平仓，日终记录敞口 | $h=0$ 主动扫单平回底仓，日终只平超额敞口 |

### 8.2 训练与预测缓存

`python -m forecast.train`：以 47 维窗口统计拼接 symbol_id 分类特征为输入、5 个前瞻目标为输出，训练集为全部标的训练日期的并集（按标的切分后取并集，避免异日历标的的训练日落入他人测试段），验证集早停 50 轮，全部目标用平方损失。训练与评估只取每 20 tick 一行（相邻 tick 的回看窗口重叠 599/600，全量入模只是重复样本）。模型与评估写入 `forecast/runs/`（`model/`、`metrics.json`、可选 `figures/`）；推理覆盖全部 tick，预测回写进统一缓存 `cache/<symbol>.npz` 的 `preds`，与窗口特征同文件、逐行对齐，同时供门控信号（8.3）与 RL 状态（5.4）使用。产物按数据签名与参数哈希校验，任一变化自动重训重建（`ensure_predictions` 幂等）。

### 8.3 门控信号

残差趋势门控（`forecast/signals.py`，阈值可在回测 CLI 覆盖）：`resid_abs_q90/w < 1.3` 且 `abs_slope/w > 0.9` 时排除——低残差、强趋势的路径上网格会被单边穿越。其中 $w$ 为当日相对半宽：当日固定半宽除以判定 tick 的中间价。

信号逐 tick 判定，只用截至该 tick 的 600 tick 回看窗口。`strategy/engine.py` 只在净持仓为 0 时读取当拍信号，判定节奏为：

1. 空仓期间每 20 tick 判一次——判定为关则等下一次，判定为开但 20 tick 内未触发买卖也等下一次；
2. 判定为开且触发买卖后，持仓途中的信号变化一律不生效，直到敞口归零才立即重判，并从该 tick 重新计时。

训练按 20 tick 抽样只是为避免重复样本；使用时任一 tick 都能判定，两者的输入口径一致。

三种取数方案：`none`（常开基线）、`oracle`（用缓存的目标真值，门控上限参照）、`prediction`（用 LightGBM 预测，可落地方案）。

### 8.4 回测

`python -m forecast.backtest`：对测试段内有预测的 (标的, 日)，从 $t_0$（回看长度或当日首个预测生效行的次一 tick，取较晚者）起回放 baseline 与各 门控 $\times$ 方案 组合；成交由 `strategy/engine.py` 给出，费用后净利润由 `strategy/backtest.py` 按统一费率叠加，逐日形态指标与汇总用 `strategy/metrics.py`（Score、满轮日占比、grid_profit 等）。结果与热力图写入 `forecast/runs/backtest/`。

## 9. 可视化

`webviz/` 是免构建的浏览器查看器，展示两个算法的实时决策过程：

- `python -m webviz.export --algorithm forecast --symbols ...`：逐日价格曲线、滑动窗口统计与各 门控 $\times$ 方案 的网格回放事件；
- `python -m webviz.export --algorithm control --symbols ...`：从 `control/runs/` 的检查点重建智能体，在测试日上贪心回放，导出决策点（生效半宽与上下轨）、成交与平仓事件。

导出写入 `webviz/data/`（已 gitignore），`python -m http.server 8000` 后访问 `/webviz/`。数据语义见 `webviz/README.md`。

## 10. 代码结构

| 路径 | 职责 |
|---|---|
| `data/` | 原始 tick parquet 与特征定义（features.md） |
| `cache/` | 统一缓存（`<symbol>.npz`：逐 tick 的窗口特征、前瞻目标与预测结果，两个算法共用） |
| `data_provider/` | ticks.py：连续竞价加载与 ATR；split.py：7:1:2 切分；windows.py：统一缓存（窗口特征、目标与预测块） |
| `strategy/` | costs.py：费率；grid.py：网格几何与穿价；width.py：ATR 半宽；engine.py：报价驱动回放；metrics.py：财务与网格指标；backtest.py：成本叠加 |
| `forecast/` | config.py / model.py（LightGBM）/ train.py / signals.py（门控）/ backtest.py / figures.py；产物在 `forecast/runs/` |
| `control/` | config.py / features.py（状态特征与标准化）/ env.py（SMDP、账户、奖励）/ model.py（BDQ）/ agent.py / buffer.py / train.py / baselines.py / tracking.py（wandb）/ trace.py（决策轨迹）；产物在 `control/runs/` |
| `webviz/` | 决策过程查看器（export.py + index.html） |
| `utils/` | 图形样式与绘图 |
| `scripts/` | run_all.py（RL 实验矩阵）、summarize.py（汇总）、smoke_test.py（冒烟） |
| `tests/` | 单元测试 |
| `docs/` | 设计文档与网格收益推导 |

依赖方向单向：`strategy`（叶子）$\leftarrow$ `data_provider` $\leftarrow$ `forecast` / `control` $\leftarrow$ `webviz`；`data_provider` 依赖 `strategy` 是因为窗口标签含网格回放口径（5.3）。

## 11. 已知边界

- 历史快照不能确定真实排队位置；对手方一档严格穿价加成交笔数增加是本回测的成交近似。
- 被动成交不会反事实地修改历史盘口和成交量，因而忽略策略自身的市场冲击。
- T+1、涨跌停期间单边无对手盘、最低佣金和部分交易规费未纳入默认实验。
- 网格每笔成交手数当前固定为 1 手，未模拟科创板 200 股的最小申报量（2.3）。
- tick 折扣按快照数而非真实秒数计量；快照缺失或间隔变化会改变对应的墙钟时长。
- 单一切分下测试段可能与训练段处于不同市场状态，测试结论应同时报告多种子离散度、诊断指标与三段行情状态（2.2、7.4）。

## 12. 参考文献

1. Sun et al. (2022), [DeepScalper](https://arxiv.org/abs/2201.09058).
2. Tavakoli, Pardo, Kormushev (2018), [Action Branching Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1711.08946).
3. Avellaneda and Stoikov (2008), [High-frequency trading in a limit order book](https://doi.org/10.1080/14697680701381228).
