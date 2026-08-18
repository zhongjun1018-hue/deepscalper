"""训练与评估流程。

每个 epoch 顺序遍历全部训练交易日，ε-greedy 与环境事件驱动地交互并将转移存入 PER，
周期性批量更新；每个 epoch 结束后在验证集上以贪心策略评估，按 SR 保存最优模型。
"""

from __future__ import annotations

import numpy as np
import torch

from .agent import BDQAgent
from .buffer import Transition
from .config import Config
from .data import DayData
from .env import DayMarket, TradingEnv, action_params
from .metrics import aggregate_diagnostics, financial_metrics


def build_markets(days: list[DayData], cfg: Config) -> list[DayMarket]:
    return [m for m in (DayMarket(d, cfg) for d in days) if m.tradable]


def evaluate(markets: list[DayMarket], policy) -> dict:
    """按 policy(obs) → 规则层参数逐日回放，返回四指标、逐日超额净值与 8.2 的补充指标。"""
    daily_returns, logs = [], []
    for m in markets:
        env = TradingEnv(m, hindsight=False)
        obs = env.observation()
        while True:
            res = env.step(policy(obs))
            if res.done:
                break
            obs = res.obs
        daily_returns.append(env.net_value() - 1.0)
        logs.append(env.episode_log())
    metrics = financial_metrics(np.array(daily_returns))
    metrics["daily_returns"] = [float(x) for x in daily_returns]
    metrics["diagnostics"] = aggregate_diagnostics(logs)
    return metrics


def train_agent(
    cfg: Config,
    markets: list[DayMarket],
    val_markets: list[DayMarket],
    seed: int,
    hindsight: bool = True,
    aux_task: bool = True,
    fixed_gears: tuple[int | None, int | None, int | None] = (None, None, None),
    log_prefix: str = "",
) -> tuple[BDQAgent, dict]:
    """训练一个 BDQ 智能体，返回（验证最优智能体, 训练日志）。

    markets / val_markets 需已挂载标准化统计量（见 fit_feature_stats）。
    """
    torch.set_num_threads(cfg.num_threads)
    agent = BDQAgent(cfg, seed=seed, aux_task=aux_task, fixed_gears=fixed_gears)

    # 选模用验证集 SR：训练目标是风险调整后的（存货惩罚），以 TR 选模会系统性偏向高杠杆
    best_val_sr, best_state, history = -np.inf, None, []
    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for day_id, m in enumerate(markets):
            env = TradingEnv(m, hindsight=hindsight)
            obs = env.observation()
            eps = cfg.epsilon_at(epoch, day_id / max(1, len(markets)))
            while True:
                action = agent.act(obs, eps)
                t, priv_hist = env.t, env.priv_window(env.t)
                res = env.step(action_params(cfg, action))
                agent.push(
                    Transition(
                        day_id=day_id,
                        t=t,
                        action=action,
                        reward=res.train_reward,
                        tau=res.tau,
                        next_t=res.t if not res.done else -1,
                        done=res.done,
                        priv_hist=priv_hist,
                        next_priv_hist=res.priv_hist,
                        vol_label=res.vol_label,
                    )
                )
                if env.n_steps % cfg.update_every == 0:
                    beta = min(1.0, cfg.per_beta_start + (1.0 - cfg.per_beta_start)
                               * agent.updates / cfg.per_beta_steps)
                    loss = agent.update(markets, beta)
                    if loss is not None:
                        losses.append(loss)
                if res.done:
                    break
                obs = res.obs

        val = evaluate(val_markets, lambda obs: action_params(cfg, agent.greedy(obs)))
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
        if val["SR"] > best_val_sr:
            best_val_sr = val["SR"]
            best_state = {k: v.clone() for k, v in agent.online.state_dict().items()}

    if best_state is not None:
        agent.online.load_state_dict(best_state)
        agent.target.load_state_dict(best_state)
    return agent, {"history": history, "best_val_SR": float(best_val_sr)}
