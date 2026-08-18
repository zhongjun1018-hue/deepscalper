"""训练与评估流程。

每个 epoch 顺序遍历全部训练交易日，ε-greedy 与环境事件驱动地交互并将转移存入 PER，
周期性批量更新；epoch 内均分若干评估点，在验证集上贪心评估，并按滑动窗口内验证 SR
的均值保存最优模型。
两段损失、训练奖励与验证指标同步写入 wandb（design 7.5）。
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .agent import BDQAgent
from .buffer import Transition
from .config import Config
from .data import DayData
from .env import DayMarket, TradingEnv, action_params
from .metrics import aggregate_diagnostics, financial_metrics
from .tracking import Tracker


def build_markets(days: list[DayData], cfg: Config) -> list[DayMarket]:
    return [m for m in (DayMarket(d, cfg) for d in days) if m.tradable]


def regime_stats(markets: list[DayMarket]) -> dict:
    """交易日集合的行情状态：日内漂移分布、上涨日占比与相对波动。

    训练 / 验证 / 测试段的状态差异是解读测试指标的前提（design 7.1）。
    """
    drift = np.array([m.mid[m.n - 1] / m.p0 - 1.0 for m in markets])
    return {
        "n_days": len(markets), "start": markets[0].date, "end": markets[-1].date,
        "drift_mean": float(drift.mean()), "drift_std": float(drift.std()),
        "up_ratio": float((drift > 0).mean()),
        "rel_atr": float(np.mean([m.atr / m.pre_close for m in markets])),
    }


def eval_days(n_days: int, n_evals: int) -> set[int]:
    """epoch 内的评估点：把交易日均分为 n_evals 段，取每段最后一日的索引。

    评估点多于交易日时退化为逐日评估。
    """
    return {max(n_days * (k + 1) // n_evals - 1, 0) for k in range(n_evals)}


def evaluate(markets: list[DayMarket], policy) -> dict:
    """按 policy(obs) → 规则层参数逐日回放，返回四指标、逐日超额净值与 7.4 的补充指标。"""
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
    metrics["daily_closure_rate"] = [log["closure_rate"] for log in logs]
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
    tracker: Tracker | None = None,
) -> tuple[BDQAgent, dict]:
    """训练一个 BDQ 智能体，返回（验证最优智能体, 训练日志）。

    markets / val_markets 需已挂载标准化统计量（见 fit_feature_stats）。
    """
    torch.set_num_threads(cfg.num_threads)
    agent = BDQAgent(cfg, seed=seed, aux_task=aux_task, fixed_gears=fixed_gears)

    # 选模用验证集 SR：训练目标是风险调整后的（存货惩罚），以 TR 选模会系统性偏向高杠杆
    best_val_sr, best_state, history = -np.inf, None, []
    eval_points = eval_days(len(markets), cfg.val_evals_per_epoch)
    val_window = deque(maxlen=min(cfg.val_select_window, cfg.epochs * len(eval_points)))
    q_losses, vol_losses, day_returns = [], [], []
    for epoch in range(1, cfg.epochs + 1):
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
                    losses = agent.update(markets, beta)
                    if losses is not None:
                        q_losses.append(losses[0])
                        vol_losses.append(losses[1])
                if res.done:
                    break
                obs = res.obs
            day_returns.append(env.net_value() - 1.0)   # ε-greedy 回放的当日超额收益
            if day_id not in eval_points:
                continue

            # 评估点之间为一段：奖励与两段损失都按该段的样本取均值
            val = evaluate(val_markets, lambda obs: action_params(cfg, agent.greedy(obs)))
            val_window.append(val["SR"])
            record = {
                "epoch": epoch, "day": day_id + 1, "updates": agent.updates,
                "train_reward": float(np.mean(day_returns)),
                "q_loss": float(np.mean(q_losses)), "vol_loss": float(np.mean(vol_losses)),
                "val_TR": val["TR"], "val_SR": val["SR"],
                "val_SR_window": float(np.mean(val_window)),
            }
            history.append(record)
            q_losses, vol_losses, day_returns = [], [], []
            if tracker is not None:
                tracker.log_eval(record)
            print(
                f"{log_prefix} epoch {epoch}/{cfg.epochs} day {record['day']}/{len(markets)} "
                f"reward={record['train_reward']:.4f} "
                f"q_loss={record['q_loss']:.3e} vol_loss={record['vol_loss']:.3e} "
                f"val_TR={val['TR']:.4f} val_SR={val['SR']:.3f} "
                f"val_SR_win={record['val_SR_window']:.3f}",
                flush=True,
            )
            # 选模用窗口均值而非单点：单点取 max 是在验证噪声上挑选，评估点越多偏差越大；
            # 窗口未满时样本更少、噪声更大，不参与选优
            if len(val_window) == val_window.maxlen and record["val_SR_window"] > best_val_sr:
                best_val_sr = record["val_SR_window"]
                best_state = {k: v.clone() for k, v in agent.online.state_dict().items()}

    if best_state is not None:
        agent.online.load_state_dict(best_state)
        agent.target.load_state_dict(best_state)
    return agent, {"history": history, "best_val_SR": float(best_val_sr)}
