"""基线方法：传统金融（BAH/MV/TSM）、预测式（MLP/GRU/LGBM）、强化学习（DQN）。

传统与预测式基线共用同一决策网格、撮合假设、费率、杠杆与日终强平规则，
与 DeepScalper 的评估口径完全一致。预测式方法以未来 300 tick 收益为监督目标，
按预测符号生成目标仓位（单标的下等价于论文的 top-k 策略生成器）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .env import DayMarket
from .features import MACRO_DIM

RULE_LOOKBACK = 240  # MV/TSM 规则的均值 / 动量窗口（tick）


def simulate_targets(
    markets: list[DayMarket], cfg: Config, target_fn
) -> np.ndarray:
    """按目标仓位序列模拟交易，返回逐日收益（杠杆后，权益损失以初始现金为下限）。"""
    daily = []
    for m in markets:
        cash, pos = m.cash0, 0.0
        for t in m.decision_points:
            diff = int(np.clip(target_fn(m, t), -cfg.max_position, cfg.max_position) - pos)
            if diff > 0:
                cash -= diff * m.ask1[t] * (1 + cfg.fee_rate)
                pos += diff
            elif diff < 0:
                cash += -diff * m.bid1[t] * (1 - cfg.fee_rate)
                pos += diff
        if pos != 0:
            cash += pos * m.mid[-1] * (1 - cfg.fee_rate * np.sign(pos))
            pos = 0.0
        daily.append(max(-1.0, cfg.leverage * (cash - m.cash0) / m.cash0))
    return np.asarray(daily)


def run_bah(markets: list[DayMarket], cfg: Config) -> np.ndarray:
    """Buy & Hold：期初满仓持有至测试期末（逐日记杠杆净值收益）。

    杠杆净值触及 0 即视为破产，之后日收益记 0（吸收态），避免净值变负
    导致收益率公式爆炸。
    """
    px0 = markets[0].mid[markets[0].decision_points[0]]
    base = markets[0].cash0
    eq = []
    for m in markets:
        raw = cfg.max_position * (m.mid[-1] - px0) - cfg.max_position * px0 * cfg.fee_rate
        eq.append(max(0.0, 1.0 + cfg.leverage * raw / base))
    eq = np.asarray(eq)
    prev = np.concatenate([[1.0], eq[:-1]])
    return np.where(prev > 0, eq / np.maximum(prev, 1e-12) - 1.0, 0.0)


def run_mv(markets: list[DayMarket], cfg: Config) -> np.ndarray:
    """Mean Reversion：价格低于均线做多，高于均线做空。"""

    def rule(m: DayMarket, t: int) -> float:
        ma = m.mid[max(0, t - RULE_LOOKBACK) : t + 1].mean()
        return cfg.max_position if m.mid[t] < ma else -cfg.max_position

    return simulate_targets(markets, cfg, rule)


def run_tsm(markets: list[DayMarket], cfg: Config) -> np.ndarray:
    """Time Series Momentum：过去窗口收益为正做多，为负做空。"""

    def rule(m: DayMarket, t: int) -> float:
        past = m.mid[max(0, t - RULE_LOOKBACK)]
        return cfg.max_position if m.mid[t] > past else -cfg.max_position

    return simulate_targets(markets, cfg, rule)


# ---------- 预测式基线 ----------


def _supervised_data(markets: list[DayMarket], cfg: Config):
    """构建监督学习样本：X 为宏观+末帧微观+虚拟私有位，y 为未来 300 tick 收益。"""
    X, Y, X_seq = [], [], []
    for m in markets:
        for t in m.decision_points:
            flat, seq, _ = _sample_at(m, cfg, t)
            X.append(flat)
            X_seq.append(seq)
            t2 = min(t + cfg.horizon_ticks, m.n - 1)
            Y.append((m.mid[t2] - m.mid[t]) / m.mid[t])
    return np.asarray(X, np.float32), np.asarray(X_seq, np.float32), np.asarray(Y, np.float64)


class _MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, in_dim: int, macro_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, 64, num_layers=2, batch_first=True, dropout=0.0)
        self.head = nn.Linear(64 + macro_dim, 1)

    def forward(self, seq, macro):
        h = self.gru(seq)[1][-1]
        return self.head(torch.cat([h, macro], dim=-1)).squeeze(-1)


def _fit_torch(model, tensors, epochs: int, lr: float = 1e-3, batch: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = [torch.as_tensor(t) for t in tensors]
    n = len(data[0])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            pred = model(*[t[idx] for t in data[:-1]])
            loss = nn.functional.mse_loss(pred, data[-1][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _sample_at(m: DayMarket, cfg: Config, t: int):
    """单个决策点的监督样本（平铺向量 / 微观序列 / 宏观向量）。"""
    lo = t - cfg.lookback_ticks + 1
    macro = m.macro_at(t)
    seq = m.micro[lo : t + 1 : cfg.micro_stride]
    flat = np.concatenate([macro, m.micro[t], np.zeros(3, np.float32)]).astype(np.float32)
    return flat, seq, macro


def _predict_positions(markets, cfg, predict_fn) -> np.ndarray:
    """按预测收益符号生成满仓多 / 空目标仓位并模拟。"""
    daily = []
    for m in markets:
        targets = iter(cfg.max_position if predict_fn(m, t) > 0 else -cfg.max_position for t in m.decision_points)
        daily.append(simulate_targets([m], cfg, lambda mm, t: next(targets))[0])
    return np.asarray(daily)


def run_predictor(
    name: str,
    train_markets: list[DayMarket],
    test_markets: list[DayMarket],
    cfg: Config,
    seed: int = 0,
) -> np.ndarray:
    """训练预测模型并在测试集上按符号策略交易。name ∈ {MLP, GRU, LGBM}。"""
    torch.set_num_threads(cfg.num_threads)
    Xtr, Xtr_seq, ytr = _supervised_data(train_markets, cfg)

    if name == "LGBM":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
        model.fit(Xtr, ytr)

        def predict(m, t):
            return float(model.predict(_sample_at(m, cfg, t)[0][None, :])[0])

    elif name == "MLP":
        model = _fit_torch(_MLP(Xtr.shape[1]), (Xtr, ytr.astype(np.float32)), epochs=5, seed=seed).eval()

        def predict(m, t):
            x = torch.as_tensor(_sample_at(m, cfg, t)[0][None, :])
            with torch.no_grad():
                return float(model(x).item())

    elif name == "GRU":
        macro_tr = Xtr[:, :MACRO_DIM]
        model = _fit_torch(_GRU(Xtr_seq.shape[2], MACRO_DIM), (Xtr_seq, macro_tr, ytr.astype(np.float32)), epochs=5, seed=seed).eval()

        def predict(m, t):
            _, seq, macro = _sample_at(m, cfg, t)
            with torch.no_grad():
                return float(model(
                    torch.as_tensor(seq[None]), torch.as_tensor(macro[None])
                ).item())

    else:
        raise ValueError(f"未知预测基线：{name}")

    return _predict_positions(test_markets, cfg, predict)
