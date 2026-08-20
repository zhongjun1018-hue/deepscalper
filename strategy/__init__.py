"""网格策略与回测的唯一定义：几何、宽度、回放引擎、指标、成本与统一回测入口。

costs / grid / width / engine / metrics 为仅依赖 numpy/pandas 的策略基元（叶子）；
backtest.py 是回测任务入口，依赖 data_provider、forecast 的门控信号与 control 的
检查点回放，产物写 strategy/runs/。
"""
