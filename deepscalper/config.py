"""全局配置：论文超参数与 tick 数据适配参数。

窗口结构适配 tick 数据：回看 600 tick，hindsight 视野 3600 tick（180 决策步，
对齐论文最优 h=180 分钟），波动率标签窗口 300 tick，每 20 tick 一个决策步，
窗口不跨交易日、可跨午休。
交易设定遵循论文 Section 5.4 框架：最大持仓 50；费率按 A 股现实取 1.0e-4
（双边，论文为期货费率 2.3e-5）；杠杆 1 倍（论文为 5 倍）。
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
    hindsight_ticks: int = 3600     # hindsight 视野：180 决策步 ≈ 3 小时（论文最优 h=180 分钟）
    step_ticks: int = 20            # 决策间隔（tick 数）
    micro_stride: int = 20          # 微观序列抽样间隔：600/20 = 30 步
    bar_ticks: int = 20             # 宏观 OHLCV bar 长度：600/20 = 30 根

    # ---- 交易设定（论文 5.4 框架，费率按 A 股现实调整）----
    fee_rate: float = 1.0e-4        # 手续费率 δ（买卖双边）
    leverage: float = 1.0           # 杠杆倍数（论文为 5）
    max_position: int = 50          # 最大持仓（股）
    price_tick: float = 0.01        # A 股最小价位（元）

    # ---- 动作空间（BDQ 两分支）----
    # 价位偏移相对对手价（买参考卖一价、卖参考买一价）：买向 off>=0 / 卖向 off<=0 穿价成交，
    # 反向偏移为被动限价（即时撮合下不成交）。相对中间价的固定档位在宽价差标的
    # （如 301308 价差中位数达 11 个价位）上可能永远无法穿价，故取对手价基准。
    price_offsets: tuple = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)  # 价位偏移（×price_tick）
    quantities: tuple = (-50, -25, -10, -5, 0, 5, 10, 25, 50)      # 目标数量（符号=方向，0=不交易）

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
    batch_size: int = 64
    update_every: int = 2           # 每多少个决策步更新一次网络
    target_sync: int = 500          # 目标网络同步间隔（更新次数）
    buffer_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    epochs: int = 5
    eps_start: float = 1.0
    eps_end: float = 0.1

    # ---- 数据切分 ----
    train_ratio: float = 0.7
    val_ratio: float = 0.1

    # ---- 运行控制 ----
    num_threads: int = 2            # 单训练进程 torch 线程数（配合多进程并行）

    @property
    def n_price(self) -> int:
        return len(self.price_offsets)

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
