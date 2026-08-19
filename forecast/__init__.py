"""前瞻预测：LightGBM 预测 5 个前瞻目标，预测值经门控信号控制固定半宽网格的启停。

窗口特征与预测块同存于 data_provider 的统一缓存；产物写入 forecast/runs/。
"""
