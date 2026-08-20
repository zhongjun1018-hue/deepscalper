"""训练与评估流程，以及 RL 实验的批量入口（`python -m control.train`）。

训练循环：每个 epoch 顺序遍历全部训练交易日，ε-greedy 与环境事件驱动地交互并将转移
存入 PER，周期性批量更新；epoch 内均分若干评估点，在验证集上贪心评估，并按滑动窗口内
验证 SR 的均值保存最优模型。Q 损失、训练奖励与验证指标同步写入 wandb（design 7.5）。

批量入口对指定标的执行网格 RL 的全部变体与基线：每个 (标的, 方法, 种子, w, λ) 为一个
独立作业，结果写入 control/runs/<symbol>/<method>[_w<权重>][_lam<λ>][_seed<k>].json，
RL 作业同时把验证最优的 online 网络写入同名 .pt 检查点（webviz 回放用，见
control/trace.py）；已存在的作业自动跳过，因此可安全重复执行（断点续跑）。w 与 λ 给
多个值即展开为超参梯子，由 control/summarize.py 按验证集 SR 选优（design 6.2 / 7.1）。
本入口只训练 RL 自身：预测块（RL 状态特征之一）不在此重训，缺失时按零特征读取并提示
先跑 `python -m forecast.train`。

作业进程以 spawn 方式启动：CUDA 无法在 fork 出的子进程中初始化，而 spawn 的
子进程为全新解释器，与父进程是否已查询过 CUDA 无关。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import asdict, replace

import numpy as np
import torch

from data_provider.split import chronological_split
from data_provider.ticks import DayData, list_symbols, load_days
from data_provider.windows import load_cache
from strategy.metrics import financial_metrics

from .agent import BDQAgent
from .buffer import Transition
from .config import Config
from .env import DayMarket, TradingEnv, action_params
from .features import MACRO_FEATURE_COLUMNS, PRED_DIM, WINDOW_DIM, fit_feature_stats
from .model import resolve_device
from .tracking import Tracker


def build_markets(
    days: list[DayData],
    cfg: Config,
    cache: dict | None = None,
    symbol_id: int = 0,
) -> list[DayMarket]:
    """由交易日构建 DayMarket；cache 为该标的的统一缓存（windows.load_cache）。

    逐日取窗口特征（只保留进宏观通道的列）与预测的当日行块，横向拼接为 R×(24+5)；
    缓存缺当日行时补零块，cfg.use_predictions=False（GRID-NA 消融）时预测块置零。
    """
    def window_block(date: str):
        if cache is None:
            return None
        hit = np.flatnonzero(cache["dates"] == date)
        if not hit.size:
            return np.zeros((cache["features"].shape[1], WINDOW_DIM + PRED_DIM),
                            dtype=np.float32)
        preds = (cache["preds"][hit[0]] if cfg.use_predictions
                 else np.zeros_like(cache["preds"][hit[0]]))
        return np.concatenate(
            [cache["features"][hit[0]][:, MACRO_FEATURE_COLUMNS], preds], axis=1)

    markets = (DayMarket(d, cfg, window_block(d.date), symbol_id) for d in days)
    return [m for m in markets if m.tradable]


def aggregate_diagnostics(logs: list[dict]) -> dict:
    """将 TradingEnv.episode_log 的逐日指标按日平均（分布类字段逐档平均）。

    NaN 表示当日无定义（如无网格触发时间的 width_rel），按有值日取均值。
    """
    if not logs:
        return {}
    out = {}
    for key, sample in logs[0].items():
        values = [log[key] for log in logs]
        out[key] = (np.mean(values, axis=0).tolist() if isinstance(sample, list)
                    else float(np.nanmean(values)))
    return out


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


def replay_day(market: DayMarket, policy) -> tuple[float, dict]:
    """按 policy(obs) → 规则层参数回放一个交易日，返回（超额收益, episode_log）。"""
    env = TradingEnv(market, hindsight=False)
    obs = env.observation()
    while True:
        res = env.step(policy(obs))
        if res.done:
            break
        obs = res.obs
    return env.net_value() - 1.0, env.episode_log()


def evaluate(markets: list[DayMarket], policy) -> dict:
    """按 policy(obs) → 规则层参数逐日回放，返回四指标、逐日超额净值与 7.4 的补充指标。"""
    daily_returns, logs = [], []
    for m in markets:
        ret, log = replay_day(m, policy)
        daily_returns.append(ret)
        logs.append(log)
    metrics = financial_metrics(np.array(daily_returns))
    metrics["daily_returns"] = [float(x) for x in daily_returns]
    metrics["daily_closure_rate"] = [log["closure_rate"] for log in logs]
    metrics["diagnostics"] = aggregate_diagnostics(logs)
    return metrics


def save_checkpoint(agent: BDQAgent, cfg: Config, path: str) -> None:
    """保存（验证最优的）online 网络权重、配置与消融的固定档位，供回放重建同一贪心策略
    （加载见 control/trace.py）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": agent.online.state_dict(), "config": asdict(cfg),
                "fixed_gears": agent.fixed_gears}, path)


def train_agent(
    cfg: Config,
    markets: list[DayMarket],
    val_markets: list[DayMarket],
    seed: int,
    hindsight: bool = True,
    fixed_gears: tuple[int | None, int | None] = (None, None),
    log_prefix: str = "",
    tracker: Tracker | None = None,
) -> tuple[BDQAgent, dict]:
    """训练一个 BDQ 智能体，返回（验证最优智能体, 训练日志）。

    markets / val_markets 需已挂载标准化统计量（见 fit_feature_stats）。
    """
    torch.set_num_threads(cfg.num_threads)
    agent = BDQAgent(cfg, seed=seed, fixed_gears=fixed_gears)

    # 选模用验证集 SR：训练目标是风险调整后的（存货惩罚），以 TR 选模会系统性偏向高杠杆
    best_val_sr, best_state, history = -np.inf, None, []
    eval_points = eval_days(len(markets), cfg.val_evals_per_epoch)
    val_window = deque(maxlen=min(cfg.val_select_window, cfg.epochs * len(eval_points)))
    q_losses, day_returns = [], []
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
                    )
                )
                if env.n_steps % cfg.update_every == 0:
                    beta = min(1.0, cfg.per_beta_start + (1.0 - cfg.per_beta_start)
                               * agent.updates / cfg.per_beta_steps)
                    q_loss = agent.update(markets, beta)
                    if q_loss is not None:
                        q_losses.append(q_loss)
                if res.done:
                    break
                obs = res.obs
            day_returns.append(env.net_value() - 1.0)   # ε-greedy 回放的当日超额收益
            if day_id not in eval_points:
                continue

            # 评估点之间为一段：奖励与损失都按该段的样本取均值
            val = evaluate(val_markets, lambda obs: action_params(cfg, agent.greedy(obs)))
            val_window.append(val["SR"])
            record = {
                "epoch": epoch, "day": day_id + 1, "updates": agent.updates,
                "train_reward": float(np.mean(day_returns)),
                "q_loss": float(np.mean(q_losses)),
                "val_TR": val["TR"], "val_SR": val["SR"],
                "val_SR_window": float(np.mean(val_window)),
            }
            history.append(record)
            q_losses, day_returns = [], []
            if tracker is not None:
                tracker.log_eval(record)
            print(
                f"{log_prefix} epoch {epoch}/{cfg.epochs} day {record['day']}/{len(markets)} "
                f"reward={record['train_reward']:.4f} "
                f"q_loss={record['q_loss']:.3e} "
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


# --------------------------------------------------------------------------------------
# 批量实验入口（python -m control.train）：作业矩阵展开、并行调度与结果落盘

# 消融：hindsight（NH）、固定半宽档（FW，其余分支照常学习）、状态不含预测特征（NA）
RL_VARIANTS = {
    "GRID": {},
    "GRID-NH": {"hindsight": False},
    "GRID-FW": {"fixed_width": 0.1},
    "GRID-NA": {"use_predictions": False},
}
RULE_METHODS = ("HOLD", "OPEN", "SCAN")   # 无需训练、无随机种子

GPU_WORKERS = 2  # 单卡下的并行作业数：作业瓶颈在 CPU 侧观测重建，少量并发即可打满


def _variant_kwargs(method: str, cfg: Config) -> dict:
    """将变体说明翻译为 train_agent 的参数（消融分支转为档位索引）。

    use_predictions 属于 Config（影响市场构建而非训练循环），在 run_job 中作为
    配置覆盖处理，此处剔除。
    """
    spec = dict(RL_VARIANTS[method])
    spec.pop("use_predictions", None)
    gears = (
        cfg.widths.index(spec.pop("fixed_width")) if "fixed_width" in spec else None,
        None,   # 数量分支不做固定消融
    )
    return {**spec, "fixed_gears": gears}


def _uses_hindsight(method: str) -> bool:
    return method in RL_VARIANTS and RL_VARIANTS[method].get("hindsight", True)


def _result_path(cfg: Config, job: dict) -> str:
    """结果文件名：方法 [_w权重] [_lamλ] [_seed种子]；不适用的超参不出现在文件名中。"""
    parts = [job["method"]]
    for key, tag in (("hindsight_weight", "w"), ("inventory_lambda", "lam"), ("seed", "seed")):
        if job[key] is not None:
            parts.append(f"{tag}{job[key]:g}")
    return os.path.join(cfg.result_dir, job["symbol"], "_".join(parts) + ".json")


def save_result(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def load_symbol_cache(symbol: str, cfg: Config) -> dict:
    """按统一窗口规格加载标的缓存（主进程预建后此处为只读命中）。"""
    return load_cache(
        symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir, spec=cfg.window,
    )


def build_split_markets(symbol: str, cfg: Config) -> dict[str, list[DayMarket]]:
    """按 7:1:2 切分构建一个标的的 train / val / test DayMarket 并挂载标准化统计量。

    symbol_id 与 forecast 同口径：排序后标的集合中的索引；标准化统计量仅用训练集
    拟合，val/test 复用（无前视泄漏）。
    """
    days = load_days(symbol, cfg.data_dir, cfg.window.atr_window)
    split = chronological_split([d.date for d in days])
    cache = load_symbol_cache(symbol, cfg)   # 窗口特征与预测同一文件，主进程已预建
    symbol_id = sorted(cfg.symbols).index(symbol) if symbol in cfg.symbols else 0
    markets = {
        name: build_markets([d for d in days if d.date in set(dates)], cfg, cache, symbol_id)
        for name, dates in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    stats = fit_feature_stats(markets["train"], cfg) if cfg.normalize else None
    for m in markets["train"] + markets["val"] + markets["test"]:
        m.set_stats(stats)
    return markets


def run_job(job: dict, cfg: Config) -> str:
    # baselines 顶层依赖本模块的 evaluate，推迟到调用时导入以断开循环
    from .baselines import run_grid_scan, run_hold_base, run_open_grid

    symbol, method, seed = job["symbol"], job["method"], job["seed"]
    out = _result_path(cfg, job)
    run_name = os.path.splitext(os.path.basename(out))[0]
    label = f"{symbol}/{run_name}"
    if os.path.exists(out):
        return f"skip {label}"
    overrides = {k: job[k] for k in ("hindsight_weight", "inventory_lambda")
                 if job[k] is not None}
    if "use_predictions" in RL_VARIANTS.get(method, {}):
        overrides["use_predictions"] = RL_VARIANTS[method]["use_predictions"]
    cfg = replace(cfg, **overrides)
    t0 = time.time()
    markets = build_split_markets(symbol, cfg)
    train_m, val_m, test_m = markets["train"], markets["val"], markets["test"]

    if method == "HOLD":
        payload = run_hold_base(test_m)
    elif method == "OPEN":
        payload = run_open_grid(test_m)
    elif method == "SCAN":
        payload = run_grid_scan(val_m, test_m)
    else:
        # 只有 RL 作业有训练曲线；规则基线的测试指标由结果文件与 summarize.py 覆盖
        tracker = Tracker(cfg, run_name, job)
        try:
            agent, log = train_agent(cfg, train_m, val_m, seed=seed, log_prefix=f"[{label}]",
                                     tracker=tracker, **_variant_kwargs(method, cfg))
            payload = {**evaluate(test_m, lambda obs: action_params(cfg, agent.greedy(obs))),
                       "train_log": log}
            tracker.log_test(payload)
            save_checkpoint(agent, cfg, os.path.splitext(out)[0] + ".pt")
        finally:
            tracker.finish()  # 失败路径也要收尾，否则进程池复用本进程时下个作业会并入未关闭的 run

    payload.update({"symbol": symbol,
                    "method": method, "seed": seed,
                    "hindsight_weight": job["hindsight_weight"],
                    "inventory_lambda": job["inventory_lambda"],
                    "splits": {name: regime_stats(ms) for name, ms in
                               (("train", train_m), ("val", val_m), ("test", test_m))},
                    "elapsed_sec": round(time.time() - t0, 1)})
    save_result(out, payload)
    return f"done {label} TR={payload['TR']:.4f} ({payload['elapsed_sec']}s)"


def make_jobs(
    symbols: list[str], seeds: tuple[int, ...], methods: list[str],
    weights: tuple[float, ...], lambdas: tuple[float, ...],
) -> list[dict]:
    """展开 (标的 × 方法 × 种子 × w × λ) 作业矩阵。

    基线无需训练、与两个超参均无关；`GRID-NH` 关闭 hindsight bonus，故不随 w 展开。
    不适用的超参记为 None，作业沿用 Config 的默认值。
    """
    jobs = []
    for symbol in symbols:
        for method in methods:
            rl = method in RL_VARIANTS
            ws = weights if _uses_hindsight(method) else (None,)
            lams = lambdas if rl else (None,)
            ss = seeds if rl else (None,)
            jobs += [{"symbol": symbol, "method": method, "seed": s,
                      "hindsight_weight": w, "inventory_lambda": lam}
                     for w in ws for lam in lams for s in ss]
    return sorted(jobs, key=lambda j: j["method"] in RL_VARIANTS)  # 基线先跑


def default_workers(cfg: Config) -> int:
    """并行作业数：GPU 下限制并发以免争抢显存，CPU 下按单进程线程预算切分核数。"""
    if resolve_device(cfg).type == "cuda":
        return GPU_WORKERS
    return max(1, (os.cpu_count() or 1) // cfg.num_threads)


def main() -> None:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=None,
                   help="标的代码，缺省为 data 目录下全部标的")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--methods", nargs="+", choices=[*RULE_METHODS, *RL_VARIANTS],
                   default=[*RULE_METHODS, *RL_VARIANTS])
    p.add_argument("--hindsight-weights", nargs="+", type=float, default=[cfg.hindsight_weight],
                   help="hindsight 权重 w 的档位，给多个值即展开超参梯子（design 6.2）")
    p.add_argument("--inventory-lambdas", nargs="+", type=float, default=[cfg.inventory_lambda],
                   help="存货惩罚 λ 的档位，给多个值即展开超参梯子（design 6.2）")
    p.add_argument("--workers", type=int, default=None, help="并行作业数，缺省按设备自适应")
    p.add_argument("--wandb-project", default=cfg.wandb_project, help="wandb 项目名")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default=cfg.wandb_mode, help="wandb 记录模式，disabled 即不记录")
    args = p.parse_args()

    symbols = args.symbols or list_symbols(cfg.data_dir)
    cfg = replace(cfg, symbols=tuple(symbols),
                  wandb_project=args.wandb_project, wandb_mode=args.wandb_mode)
    device = resolve_device(cfg)
    workers = args.workers if args.workers is not None else default_workers(cfg)
    jobs = make_jobs(symbols, tuple(args.seeds), args.methods,
                     tuple(args.hindsight_weights), tuple(args.inventory_lambdas))
    pending = [j for j in jobs if not os.path.exists(_result_path(cfg, j))]
    print(f"jobs: {len(jobs)} total, {len(pending)} pending; device={device} workers={workers}", flush=True)

    # 统一缓存串行预建：进程池内并发重建同一文件会写坏缓存。
    # 预测块只读不重训（RL 训练只训练自身）：全缺失时按零特征进入状态，提示先跑 forecast。
    for symbol in symbols:
        t0 = time.time()
        cache = load_symbol_cache(symbol, cfg)
        print(f"window cache {symbol}: {time.time()-t0:.1f}s", flush=True)
        if not cache["preds"].any():   # zero_nan 读取下未回写的预测块全为零
            print(f"warn: {symbol} 预测块为空，状态中的前瞻预测按零读取"
                  "（先运行 python -m forecast.train 可回写预测）", flush=True)

    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, j, cfg): j for j in pending}
        for fut in cf.as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # 单个作业失败不阻塞其它作业
                print(f"FAIL {_result_path(cfg, futures[fut])}: {exc}", flush=True)


if __name__ == "__main__":
    main()
