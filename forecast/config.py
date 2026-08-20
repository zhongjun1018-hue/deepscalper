"""前瞻回归配置与两条产线共享的数据口径基类。"""

from dataclasses import dataclass, field

from data_provider.windows import WindowSpec
from forecast.model import DEFAULT_MODEL_KWARGS


@dataclass(frozen=True)
class PipelineConfig:
    """回归与模式识别产线共享的数据 / 采样口径：单一定义，避免两条产线漂移。

    训练与推理行即统一缓存的分钟锚点行（data_provider/windows.py），节奏由分钟
    网格给出，无独立采样参数。
    """
    data_dir: str = "data"
    cache_dir: str = "cache"
    symbols: tuple = ()
    window: WindowSpec = field(default_factory=WindowSpec)
    seed: int = 2021


@dataclass(frozen=True)
class Config(PipelineConfig):
    """前瞻回归产线（RL 状态特征）：产物 model/ metrics.json figures/。"""
    runs_dir: str = "forecast/runs"
    # LightGBM 覆盖参数
    model_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_MODEL_KWARGS))
