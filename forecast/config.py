"""前瞻预测配置：数据 / 缓存 / 产物路径、缓存规格、决策间隔与 LightGBM 超参。"""

from dataclasses import dataclass, field

from data_provider.windows import WindowSpec
from forecast.model import DEFAULT_MODEL_KWARGS


@dataclass(frozen=True)
class Config:
    data_dir: str = "data"
    cache_dir: str = "cache"
    runs_dir: str = "forecast/runs"   # 产物：model/ metrics.json figures/ backtest/
    symbols: tuple = ()
    window: WindowSpec = field(default_factory=WindowSpec)
    # 决策间隔：门控每 stride_ticks 个 tick 重判一次，训练行也按此抽样
    # （相邻 tick 的回看窗口重叠 599/600，全量入模只是重复样本）
    stride_ticks: int = 20
    seed: int = 2021
    # LightGBM 覆盖参数
    model_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_MODEL_KWARGS))
