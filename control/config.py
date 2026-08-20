"""全局配置：网格规则参数、两分支动作档位与训练超参数（design.md）。

决策点由「成交 / 超时 / 日终」混合触发（design 4.1），可落在任一 tick 上；
每个决策点回看 600 tick，hindsight 视野同为 600 tick。
持仓、现金、成交额统一以「手 × 每股价格」计量，并以 B = Q0·p0 归一（design 3.5）。
"""

from dataclasses import dataclass, field

from data_provider.windows import WindowSpec


@dataclass(frozen=True)
class Config:
    # ---- 路径 ----
    data_dir: str = "data"
    runs_dir: str = "control/runs"
    cache_dir: str = "cache"        # 统一缓存目录（cache/<symbol>.npz）

    # ---- tick 窗口结构（回看与 bar 长度取自 window 规格，避免双份参数）----
    hindsight_ticks: int = 600      # hindsight 视野 H_hs（design 6.2）
    micro_stride: int = 20          # 微观序列抽样间隔：600/20 = 30 步
    timeout_ticks: int = 200        # K：超时触发间隔

    # ---- 市场规则（A 股）----
    tick_size: float = 0.01         # 最小变动价位（元）
    lot_size: int = 100             # 1 手 = 100 股，用于将盘口挂单量折算为手
    # 显性费率统一定义在 strategy/costs.py（design 3.3：双边佣金 1e-4，卖出印花税 5e-4）

    # ---- 账户 ----
    base_position: int = 50         # 底仓 Q0（手）；仓位带为 [0, 2Q0]，底仓居中

    # ---- 动作空间（两分支：半宽 × 数量）----
    # 半宽以 ATR3 为单位（跨标的可比）：0 表示在决策点按对手方一档价平回底仓
    #（仅净持仓非零时可选），100 的半宽日内无法触发、效果等同关闭网格；两者都是
    # 网格不触发档，数量分支在这两档下对执行无意义。数量档是风险规模接口，暂时收缩为 {1}。
    widths: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 100.0)
    sizes: tuple[int, ...] = (1,)

    # ---- 奖励 ----
    # w 与 λ 是偏好参数，此处为缺省档，最终由验证集 SR 在梯子上选优（design 6.2 / 7.1）
    hindsight_weight: float = 0.2   # w（按 τ/K 加权，见 design 6.2）
    inventory_lambda: float = 3.0   # 存货惩罚 λ（无量纲；梯子 {0, 1, 3, 10, 30}）

    # ---- 缓存规格（data_provider.windows 的统一口径；回看、bar、ATR 与半宽下限由此出）----
    window: WindowSpec = field(default_factory=WindowSpec)

    # ---- 状态特征 ----
    use_predictions: bool = True    # 宏观向量是否含 LightGBM 前瞻预测（GRID-NA 消融置 False）

    # ---- 标的标识 ----
    symbols: tuple[str, ...] = ()   # 本次运行的标的集合，决定 symbol_id 与 embedding 规模
    symbol_embed_dim: int = 8

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
    per_beta_steps: int = 12_000    # β 线性升至 1.0 所需的更新次数（design 6.3）
    epochs: int = 5
    val_evals_per_epoch: int = 3    # epoch 内均分的验证评估次数（末次落在 epoch 末）
    val_select_window: int = 3      # 选模所用的验证评估点滑动窗口长度（见 design 7.1）
    eps_start: float = 1.0
    eps_end: float = 0.1

    # ---- 实验跟踪（wandb，见 design 7.5）----
    wandb_project: str = "gridscalper"
    wandb_mode: str = "online"      # "online" / "offline"（离线落盘后补传）/ "disabled"

    # ---- 运行控制 ----
    device: str = "auto"            # torch 设备："auto"（有 CUDA 则用）/ "cpu" / "cuda"
    num_threads: int = 2            # 单训练进程 torch 线程数（配合多进程并行）

    @property
    def lookback_ticks(self) -> int:
        """回看窗口长度：与缓存口径同源。"""
        return self.window.lookback_ticks

    @property
    def bar_ticks(self) -> int:
        """宏观 OHLCV bar 长度：与缓存的 bar 聚合口径同源。"""
        return self.window.bar_ticks

    @property
    def n_symbols(self) -> int:
        return max(1, len(self.symbols))

    @property
    def max_position(self) -> int:
        """仓位带上界 2Q0：底仓居中，上下各留 Q0 的超额敞口空间。"""
        return 2 * self.base_position

    @property
    def n_width(self) -> int:
        return len(self.widths)

    @property
    def n_size(self) -> int:
        return len(self.sizes)

    @property
    def inactive_gears(self) -> tuple[int, ...]:
        """网格不触发的半宽档（平仓 0 与关闭 100，即梯子两端）：此时数量分支无意义。"""
        return (0, len(self.widths) - 1)

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
