"""强化学习：定长决策自适应网格的环境与分支 Q 智能体（设计见 docs/design.md）。

数据与切分取自 data_provider（7:1:2 单次时序切分），成本定义取自
strategy/costs.py，窗口特征与前瞻预测取自 data_provider 的统一缓存（分钟锚点行）；
训练为全部标的池化的统一训练，产物写入 control/runs/。
"""
