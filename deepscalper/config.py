"""全局配置：论文超参数与 tick 数据适配参数。

窗口结构适配 tick 数据：回看 600 tick，hindsight 视野 1200 tick（60 决策步；
A 股连续竞价仅 4 小时，论文的 180 决策步会使绝大多数决策点截断至日尾，
故按 {300, 600, 900, 1200} 档位调参），波动率标签窗口 300 tick，
每 20 tick 一个决策步，窗口不跨交易日。
交易设定遵循论文 Section 5.4 框架，并适配 A 股：数量以手计（1 手 = 100 股），
最大持仓 50 手；费率取 1.0e-4（双边）；杠杆 1 倍（论文为 5 倍）。
持仓、现金与成交额按同一单位计量并统一以 cash0 归一，故单位换算不影响任何收益指标。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- 路径 ----
    data_dir: str = "data"
    result_dir: str = "results"

    # ---- tick 窗口结构 ----
    lookback_ticks: int = 600       # 回看窗口（tick 数）
    horizon_ticks: int = 300        # 波动率预测标签窗口（tick 数）
    hindsight_ticks: int = 1200     # hindsight 视野：60 决策步 ≈ 1 小时
    step_ticks: int = 20            # 决策间隔（tick 数）
    micro_stride: int = 20          # 微观序列抽样间隔：600/20 = 30 步
    bar_ticks: int = 20             # 宏观 OHLCV bar 长度：600/20 = 30 根

    # ---- 交易设定（论文 5.4 框架，费率按 A 股现实调整）----
    fee_rate: float = 1.0e-4        # 手续费率 δ（买卖双边）
    leverage: float = 1.0           # 杠杆倍数（论文为 5）
    max_position: int = 50          # 最大持仓（手）
    lot_size: int = 100             # 1 手 = 100 股，用于将盘口挂单量折算为手

    # ---- 动作空间（BDQ 两分支）----
    # 价格分支为穿价档深 k：买单依次吃卖 1..k 档、卖单依次吃买 1..k 档，各档成交量受
    # 该档挂单量限制，故 k 越深成交越确定、成交均价越差。以档位而非最小价位计量，是因为
    # 档间距随标的差异悬殊（301308 卖二档中位数距卖一 3 个价位，688030 仅 1 个），
    # 固定价位偏移在两标的上的含义不可比。
    price_levels: tuple[int, ...] = (1, 2, 3, 4, 5)                      # 穿价档深
    quantities: tuple[int, ...] = (-50, -25, -10, -5, 0, 5, 10, 25, 50)  # 委托数量（手，符号=方向，0=不下单）

    # ---- 奖励 / 辅助任务 ----
    hindsight_weight: float = 0.1   # w（论文网格搜索结果最优值）
    vol_loss_weight: float = 1.0    # η

    # ---- 网络结构 ----
    hidden_size: int = 64
    macro_hidden: int = 64
    trunk_hidden: int = 128

    # ---- 训练 ----
    normalize: bool = True          # 基于训练集的 z-score 特征标准化
    norm_clip: float = 10.0         # 标准化后的截断阈值
    gamma: float = 0.99
    lr: float = 1e-4
    grad_clip: float = 10.0         # 梯度全局范数裁剪（DQN 基线按其原始设定不裁剪）
    batch_size: int = 64
    update_every: int = 2           # 每多少个决策步更新一次网络
    target_sync: int = 500          # 目标网络同步间隔（更新次数）
    buffer_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 50_000    # β 由 per_beta_start 线性升至 1.0 所需的更新次数
    epochs: int = 5
    eps_start: float = 1.0
    eps_end: float = 0.1

    # ---- 数据切分 ----
    train_ratio: float = 0.7
    val_ratio: float = 0.1

    # ---- 运行控制 ----
    device: str = "auto"            # torch 设备："auto"（有 CUDA 则用）/ "cpu" / "cuda"
    num_threads: int = 2            # 单训练进程 torch 线程数（配合多进程并行）

    @property
    def n_price(self) -> int:
        return len(self.price_levels)

    @property
    def n_quantity(self) -> int:
        return len(self.quantities)

    @property
    def micro_steps(self) -> int:
        return self.lookback_ticks // self.micro_stride

    @property
    def n_bars(self) -> int:
        return self.lookback_ticks // self.bar_ticks

    def epsilon_at(self, epoch: int, progress: float) -> float:
        """epoch 内进度 progress∈[0,1) 时的探索率：前 60% 训练轮次线性退火至 eps_end。"""
        span = max(1, int(self.epochs * 0.6))
        ratio = min(1.0, (epoch - 1 + progress) / span)
        return self.eps_start + (self.eps_end - self.eps_start) * ratio
