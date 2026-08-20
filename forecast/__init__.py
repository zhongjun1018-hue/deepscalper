"""预测算法：行情模式识别门控网格。

regime/ 是方法核心：识别「低残差、强趋势」的网格不利模式（事后标签 + LightGBM
二分类），概率门控控制固定半宽网格的启停；train.py / model.py 是 5 个前瞻目标的
回归产线，预测回写统一缓存的 preds 块，作为 RL 状态特征（control）使用。
回测由 strategy/backtest.py 的统一回测承担，产物分别在 forecast/runs/ 与
forecast/regime/runs/。
"""
