"""全局配置：网格规则参数、三分支动作档位与训练超参数（design.md）。

窗口结构沿用基线框架（回看 600 tick、hindsight 1200 tick、波动率标签 300 tick），
但决策点改为「成交 / 超时 / 日终」混合触发（design 4.1）。
持仓、现金、成交额统一以「手 × 每股价格」计量，并以 B = Q0·p0 归一（design 3.5）。
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
    hindsight_ticks: int = 1200     # hindsight 视野 H ≈ 1 小时
    micro_stride: int = 20          # 微观序列抽样间隔：600/20 = 30 步
    bar_ticks: int = 20             # 宏观 OHLCV bar 长度：600/20 = 30 根
    timeout_ticks: int = 60         # K：超时触发间隔（≈ 3 分钟）

    # ---- 市场规则（A 股）----
    tick_size: float = 0.01         # 最小变动价位（元）
    lot_size: int = 100             # 1 手 = 100 股，用于将盘口挂单量折算为手
    min_order_lots: int = 2         # 单笔最小申报 200 股；持仓不足时仅允许一次性卖出
    # 当前实验不计显性费用（见 design 3.4）；成本模型保留，置回 1.0e-4 / 5.0e-4 即恢复
    commission_rate: float = 0.0        # 佣金（双边）
    stamp_duty_rate: float = 0.0        # 印花税（仅卖出）
    atr_days: int = 3               # ATR 回溯的完整交易日数 A

    # ---- 账户 ----
    base_position: int = 50         # 底仓 Q0（手）；仓位带为 [0, 2Q0]，底仓居中

    # ---- 动作空间（BDQ 三分支）----
    # 半宽以 ATR3 为单位（跨标的可比），按公比 1.26 等比展开；倾斜端点表示单侧关闭；
    # 数量最小非零档为 2 手，以满足科创板 200 股的最小申报量。
    half_widths: tuple[float, ...] = (0.075, 0.095, 0.12, 0.15, 0.19, 0.24, 0.30)
    min_half_width_ratio: float = 1e-3  # ε：生效半宽下限（相对中心价，千1），防止网格过密
    tilts: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    sizes: tuple[int, ...] = (0, 2, 3, 5)
    tilt_ratio: float = 1.26        # 倾斜梯子公比 k，k^3 ≈ 2

    # ---- 奖励 / 辅助任务 ----
    # w 与 λ 是偏好参数，此处为缺省档，最终由验证集 SR 在梯子上选优（design 6.2 / 7.1）
    hindsight_weight: float = 0.1   # w（按 τ/K 加权，见 design 6.2）
    vol_loss_weight: float = 1.0    # η
    inventory_lambda: float = 30.0  # 存货惩罚 λ（无量纲；梯子 {0, 3, 10, 30, 100}）

    # ---- 网络结构 ----
    hidden_size: int = 64
    macro_hidden: int = 64
    trunk_hidden: int = 128

    # ---- 训练 ----
    normalize: bool = True          # 基于训练集的 z-score 特征标准化
    norm_clip: float = 10.0         # 标准化后的截断阈值
    gamma: float = 0.9995           # 每 tick 折扣 ≈ 0.99^(1/20)；TD 目标用 gamma^τ
    lr: float = 1e-4
    grad_clip: float = 10.0
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

    # ---- 数据切分（6 : 2 : 2，见 design 7.1）----
    train_ratio: float = 0.6
    val_ratio: float = 0.2

    # ---- 运行控制 ----
    device: str = "auto"            # torch 设备："auto"（有 CUDA 则用）/ "cpu" / "cuda"
    num_threads: int = 2            # 单训练进程 torch 线程数（配合多进程并行）

    @property
    def max_position(self) -> int:
        """仓位带上界 2Q0：底仓居中，上下各留 Q0 的超额敞口空间。"""
        return 2 * self.base_position

    @property
    def n_width(self) -> int:
        return len(self.half_widths)

    @property
    def n_tilt(self) -> int:
        return len(self.tilts)

    @property
    def n_size(self) -> int:
        return len(self.sizes)

    @property
    def max_tilt(self) -> int:
        return self.tilts[-1]

    @property
    def micro_steps(self) -> int:
        return self.lookback_ticks // self.micro_stride

    @property
    def n_bars(self) -> int:
        return self.lookback_ticks // self.bar_ticks

    def fee_rate(self, side: float) -> float:
        """显性费率：买入仅佣金，卖出另加印花税（design 3.4）。"""
        return self.commission_rate + (self.stamp_duty_rate if side < 0 else 0.0)

    def epsilon_at(self, epoch: int, progress: float) -> float:
        """epoch 内进度 progress∈[0,1) 时的探索率：前 60% 训练轮次线性退火至 eps_end。"""
        span = max(1, int(self.epochs * 0.6))
        ratio = min(1.0, (epoch - 1 + progress) / span)
        return self.eps_start + (self.eps_end - self.eps_start) * ratio
