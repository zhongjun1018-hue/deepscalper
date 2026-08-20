"""全局配置：网格规则参数、动作档位与训练超参数（design.md）。

决策点为固定每 decision_interval_min 分钟的锚点加日终（design 4.1），成交不触发
决策；每个决策点回看 lookback_min 分钟，hindsight 视野与前瞻预测同为 pred_min 分钟。
持仓、现金、成交额统一以「手 × 每股价格」计量，并以 B = Q0·p0 归一（design 3.5）。
训练为全部标的池化的统一训练（design 7.1），日程按池化规模标定。
"""

from dataclasses import dataclass, field

from data_provider.windows import WindowSpec


@dataclass(frozen=True)
class Config:
    # ---- 路径 ----
    data_dir: str = "data"
    runs_dir: str = "control/runs"
    cache_dir: str = "cache"        # 统一缓存目录（cache/<symbol>.npz）

    # ---- 决策节奏（回看与 bar 长度取自 window 规格，避免双份参数）----
    decision_interval_min: int = 10  # 定长决策间隔（分钟；备选 5，池化后按验证证据启用）

    # ---- 市场规则（A 股）----
    tick_size: float = 0.01         # 最小变动价位（元）
    lot_size: int = 100             # 1 手 = 100 股，用于将盘口挂单量折算为手
    # 显性费率统一定义在 strategy/costs.py（design 3.3：双边佣金 1e-4，卖出印花税 5e-4）

    # ---- 账户 ----
    base_position: int = 50         # 底仓 Q0（手）；仓位带为 [0, 2Q0]，底仓居中

    # ---- 动作空间 ----
    # 半宽以 ATR3 为单位（跨标的可比）：0 表示在决策点按对手方一档价平回底仓
    #（仅净持仓非零时可选），100 的半宽日内无法触发、效果等同关闭网格；两者都是
    # 网格不触发档。数量档是风险规模接口，当前收缩为 {1}：单档时网络退化为单分支
    # dueling（无数量分支），恢复 {1,2,3} 仍只改配置。
    widths: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 100.0)
    sizes: tuple[int, ...] = (1,)

    # ---- 奖励 ----
    # w 与 λ 是偏好参数，此处为缺省档，最终由验证集 SR 在梯子上选优（design 6.2 / 7.1）
    hindsight_weight: float = 0.2   # w（hindsight 视野 = window.pred_min，见 design 6.2）
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

    # ---- 训练（池化日程：每 epoch 约 22 标的 × 80 日 × 24 决策 ≈ 4.4 万条转移、
    # 约 0.7 万次更新；日程与该规模绑定，经验证曲线微调）----
    normalize: bool = True          # 逐标的基于训练段的 z-score 特征标准化
    norm_clip: float = 10.0         # 标准化后的截断阈值
    gamma: float = 0.99             # 每分钟折扣；TD 折扣恒为 gamma^decision_interval_min
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 64
    update_every: int = 6           # 每多少个决策步更新一次网络
    target_sync: int = 2000         # 目标网络同步间隔（更新次数）
    buffer_capacity: int = 300_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 60_000    # β 线性升至 1.0 所需的更新次数（design 6.3）
    epochs: int = 3
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
    def lookback_min(self) -> int:
        """回看窗口长度（分钟）：与缓存口径同源。"""
        return self.window.lookback_min

    @property
    def td_discount(self) -> float:
        """定长区间的 TD 折扣：gamma^decision_interval_min（design 4.2）。"""
        return self.gamma ** self.decision_interval_min

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
        """微观 / 私有序列步数：每分钟一步，即回看分钟数。"""
        return self.window.lookback_min

    @property
    def n_bars(self) -> int:
        return self.window.lookback_min // self.window.bar_min

    def epsilon_at(self, epoch: int, progress: float) -> float:
        """epoch 内进度 progress∈[0,1) 时的探索率：前 60% 训练轮次线性退火至 eps_end。"""
        ratio = min(1.0, (epoch - 1 + progress) / (self.epochs * 0.6))
        return self.eps_start + (self.eps_end - self.eps_start) * ratio
