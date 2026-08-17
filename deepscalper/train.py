"""训练与评估流程。

训练遵循论文 5.4 的迭代式方案：每个 epoch 顺序遍历全部训练交易日，
ε-greedy 与环境交互并将转移存入 PER，周期性批量更新；每个 epoch 结束后
在验证集上以贪心策略评估 TR，保存验证最优模型。
"""

from __future__ import annotations

import numpy as np
import torch

from .agent import BDQAgent
from .buffer import Transition
from .config import Config
from .data import DayData
from .env import DayMarket, TradingEnv
from .metrics import financial_metrics


def build_markets(days: list[DayData], cfg: Config) -> list[DayMarket]:
    return [m for m in (DayMarket(d, cfg) for d in days) if m.tradable]


def evaluate_greedy(agent: BDQAgent, markets: list[DayMarket], cfg: Config) -> dict:
    """贪心策略在指定交易日集合上评估，返回四指标、逐日净值与日均成交笔数。"""
    daily_returns, fills = [], []
    for m in markets:
        env = TradingEnv(m, cfg, hindsight=False)
        obs = env.observation()
        while True:
            action = agent.greedy(obs)
            res = env.step(action)
            if res.done:
                break
            obs = res.obs
        daily_returns.append(env.net_value() - 1.0)
        fills.append(env.n_fills)
    metrics = financial_metrics(np.array(daily_returns))
    metrics["daily_returns"] = [float(x) for x in daily_returns]
    metrics["avg_fills_per_day"] = float(np.mean(fills))
    return metrics


def train_agent(
    cfg: Config,
    markets: list[DayMarket],
    val_markets: list[DayMarket],
    seed: int,
    hindsight: bool = True,
    aux_task: bool = True,
    log_prefix: str = "",
) -> tuple[BDQAgent, dict]:
    """训练一个 BDQ 智能体，返回（验证最优智能体, 训练日志）。

    markets / val_markets 需已挂载标准化统计量（见 fit_feature_stats）。
    """
    torch.set_num_threads(cfg.num_threads)
    agent = BDQAgent(cfg, seed=seed, aux_task=aux_task)

    best_val_tr, best_state, history = -np.inf, None, []
    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for day_id, m in enumerate(markets):
            env = TradingEnv(m, cfg, hindsight=hindsight)
            obs = env.observation()
            eps = cfg.epsilon_at(epoch, day_id / max(1, len(markets)))
            while True:
                action = agent.act(obs, eps)
                t_idx = env.step_idx
                priv_hist = env.priv_hist.copy()
                res = env.step(action)
                agent.push(
                    Transition(
                        day_id=day_id,
                        t=t_idx,
                        action_p=action[0],
                        action_q=action[1],
                        reward=res.train_reward,
                        next_t=res.t if not res.done else -1,
                        done=res.done,
                        priv_hist=priv_hist,
                        next_priv_hist=res.priv_hist,
                        vol_label=res.vol_label,
                    )
                )
                if env.step_idx % cfg.update_every == 0:
                    beta = min(1.0, cfg.per_beta_start + (1.0 - cfg.per_beta_start)
                               * agent.updates / cfg.per_beta_steps)
                    loss = agent.update(markets, beta)
                    if loss is not None:
                        losses.append(loss)
                if res.done:
                    break
                obs = res.obs

        val = evaluate_greedy(agent, val_markets, cfg)
        history.append(
            {"epoch": epoch, "val_TR": val["TR"], "val_SR": val["SR"],
             "mean_loss": float(np.mean(losses))}
        )
        print(
            f"{log_prefix} epoch {epoch}/{cfg.epochs} "
            f"loss={history[-1]['mean_loss']:.3e} "
            f"val_TR={val['TR']:.4f} val_SR={val['SR']:.3f}",
            flush=True,
        )
        if val["TR"] > best_val_tr:
            best_val_tr = val["TR"]
            best_state = {k: v.clone() for k, v in agent.online.state_dict().items()}

    if best_state is not None:
        agent.online.load_state_dict(best_state)
        agent.target.load_state_dict(best_state)
    return agent, {"history": history, "best_val_TR": float(best_val_tr)}
