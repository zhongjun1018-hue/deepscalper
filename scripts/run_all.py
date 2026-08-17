"""并行实验运行器：对指定标的执行全部方法（DeepScalper 及消融 / DQN / 预测式 / 传统）。

每个 (标的, 方法, 种子) 为一个独立作业，结果写入 results/<symbol>/<method>_seed<k>.json；
已存在的作业自动跳过，因此脚本可安全重复执行（断点续跑）。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepscalper.baselines import run_bah, run_mv, run_predictor, run_tsm
from deepscalper.config import Config
from deepscalper.data import load_days, split_days
from deepscalper.dqn import evaluate_dqn, train_dqn
from deepscalper.features import fit_feature_stats
from deepscalper.metrics import financial_metrics
from deepscalper.train import build_markets, evaluate_greedy, train_agent

RL_VARIANTS = {
    "DS": {"hindsight": True, "aux_task": True},
    "DS-NH": {"hindsight": False, "aux_task": True},
    "DS-NA": {"hindsight": True, "aux_task": False},
}


def _result_path(cfg: Config, symbol: str, method: str, seed: int | None) -> str:
    name = method if seed is None else f"{method}_seed{seed}"
    return os.path.join(cfg.result_dir, symbol, f"{name}.json")


def _save(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def run_job(job: dict, cfg: Config) -> str:
    symbol, method, seed = job["symbol"], job["method"], job["seed"]
    out = _result_path(cfg, symbol, method, seed)
    if os.path.exists(out):
        return f"skip {symbol}/{method}/{seed}"
    t0 = time.time()
    days = load_days(symbol, cfg.data_dir)
    train_d, val_d, test_d = split_days(days, cfg.train_ratio, cfg.val_ratio)
    train_m = build_markets(train_d, cfg)
    val_m = build_markets(val_d, cfg)
    test_m = build_markets(test_d, cfg)
    del days, train_d, val_d, test_d  # 释放原始 DataFrame

    # 仅用训练集拟合标准化统计量，val/test 复用（无前视泄漏）
    stats = fit_feature_stats(train_m, cfg) if cfg.normalize else None
    for m in train_m + val_m + test_m:
        m.set_stats(stats)

    split_info = {"train": len(train_m), "val": len(val_m), "test": len(test_m)}
    prefix = f"[{symbol}/{method}/seed{seed}]"

    if method in ("BAH", "MV", "TSM"):
        fn = {"BAH": run_bah, "MV": run_mv, "TSM": run_tsm}[method]
        daily = fn(test_m, cfg)
        payload = {**financial_metrics(daily), "daily_returns": daily.tolist()}
    elif method in ("MLP", "GRU", "LGBM"):
        daily = run_predictor(method, train_m, test_m, cfg, seed=seed)
        payload = {**financial_metrics(daily), "daily_returns": daily.tolist()}
    elif method == "DQN":
        net, log = train_dqn(cfg, train_m, val_m, seed=seed, log_prefix=prefix)
        payload = {**evaluate_dqn(net, test_m, cfg), "train_log": log}
    else:
        agent, log = train_agent(cfg, train_m, val_m, seed=seed, log_prefix=prefix, **RL_VARIANTS[method])
        payload = {**evaluate_greedy(agent, test_m, cfg), "train_log": log}

    payload.update({"symbol": symbol, "method": method, "seed": seed,
                    "split_days": split_info, "elapsed_sec": round(time.time() - t0, 1)})
    _save(out, payload)
    return f"done {symbol}/{method}/{seed} TR={payload['TR']:.4f} ({payload['elapsed_sec']}s)"


def make_jobs(symbols: list[str], seeds: tuple[int, ...], methods: list[str]) -> list[dict]:
    jobs = []
    for symbol in symbols:
        for method in methods:
            if method in ("BAH", "MV", "TSM"):
                jobs.append({"symbol": symbol, "method": method, "seed": None})
            else:
                jobs += [{"symbol": symbol, "method": method, "seed": s} for s in seeds]
    order = {"BAH": 0, "MV": 0, "TSM": 0, "LGBM": 1, "MLP": 2, "GRU": 3, "DQN": 4}
    return sorted(jobs, key=lambda j: order.get(j["method"], 5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["301308", "688030"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+",
                   default=["BAH", "MV", "TSM", "LGBM", "MLP", "GRU", "DQN", "DS-NA", "DS-NH", "DS"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    cfg = Config()
    jobs = make_jobs(args.symbols, tuple(args.seeds), args.methods)
    pending = [j for j in jobs if not os.path.exists(_result_path(cfg, j["symbol"], j["method"], j["seed"]))]
    print(f"jobs: {len(jobs)} total, {len(pending)} pending", flush=True)

    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j, cfg): j for j in pending}
        for fut in cf.as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # 单个作业失败不阻塞其它作业
                j = futures[fut]
                print(f"FAIL {j['symbol']}/{j['method']}/{j['seed']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
