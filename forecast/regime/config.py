"""行情模式识别配置：产物路径、模式定义参数与识别器超参。

数据与采样口径继承 forecast.config.PipelineConfig，与回归产线共用单一定义。
"""

from dataclasses import dataclass, field

from forecast.config import PipelineConfig
from forecast.model import DEFAULT_MODEL_KWARGS


@dataclass(frozen=True)
class RegimeConfig(PipelineConfig):
    runs_dir: str = "forecast/regime/runs"   # 产物：model/ meta.json metrics.json economics.json

    # 模式定义：残差趋势规则（低残差、强趋势 → 网格不利），比值以当日相对半宽为单位
    residual_ratio_threshold: float = 1.3
    slope_ratio_threshold: float = 0.9
    # 规则命中视作潜在二元模式的带噪观测：粘性前向-后向平滑成事后标签。
    # 0.99/0.3 相比 0.95/0.2：经济区分度不变、test AUC 0.652→0.700、
    # oracle/prediction 门控倒挂消除（22 标的全流水线对比选定）。
    # 平滑拍即分钟锚点：对角 0.99/拍的隐含状态期望持续时长约 100 分钟
    sticky_stay: float = 0.99
    emission_noise: float = 0.3

    # 门控执行的连续确认去抖：状态切换（开↔关）需连续 confirm_n 个锚点判定一致
    confirm_n: int = 2

    # 识别器 LightGBM 二分类超参（早停轮数含在内），与回归产线共用同一组默认值
    model_kwargs: dict = field(
        default_factory=lambda: dict(DEFAULT_MODEL_KWARGS))
