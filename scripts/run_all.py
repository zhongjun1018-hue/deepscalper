"""并行实验运行器：对指定标的执行全部方法（DeepScalper 及消融 / DQN / 预测式 / 传统）。

每个 (标的, 方法, 种子, hindsight 视野) 为一个独立作业，结果写入
results/<symbol>/<method>[_h<视野>][_seed<k>].json；已存在的作业自动跳过，
因此脚本可安全重复执行（断点续跑）。

作业进程以 spawn 方式启动：CUDA 无法在 fork 出的子进程中初始化，而 spawn 的
子进程为全新解释器，与父进程是否已查询过 CUDA 无关。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepscalper.baselines import run_bah, run_mv, run_predictor, run_tsm
from deepscalper.config import Config
from deepscalper.data import load_days, split_days
from deepscalper.dqn import evaluate_dqn, train_dqn
from deepscalper.features import fit_feature_stats
from deepscalper.metrics import financial_metrics
from deepscalper.model import resolve_device
from deepscalper.train import build_markets, evaluate_greedy, train_agent

RL_VARIANTS = {
    "DS": {"hindsight": True, "aux_task": True},
    "DS-NH": {"hindsight": False, "aux_task": True},
    "DS-NA": {"hindsight": True, "aux_task": False},
}
RULE_METHODS = ("BAH", "MV", "TSM")          # 无需训练、无随机种子
PREDICTOR_METHODS = ("LGBM", "MLP", "GRU")

GPU_WORKERS = 2  # 单卡下的并行作业数：作业瓶颈在 CPU 侧观测重建，少量并发即可打满


def _result_path(cfg: Config, job: dict) -> str:
    """结果文件名：方法 [_h视野] [_seed种子]；仅含 hindsight bonus 的方法带视野后缀。"""
    parts = [job["method"]]
    if job["hindsight_ticks"] is not None:
        parts.append(f"h{job['hindsight_ticks']}")
    if job["seed"] is not None:
        parts.append(f"seed{job['seed']}")
    return os.path.join(cfg.result_dir, job["symbol"], "_".join(parts) + ".json")


def _untrained_payload(daily: np.ndarray, fills: list[int]) -> dict:
    """传统 / 预测式方法的结果载荷，字段与 RL 方法（train_log 除外）保持一致。"""
    return {**financial_metrics(daily), "daily_returns": daily.tolist(),
            "avg_fills_per_day": float(np.mean(fills))}


def _save(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def run_job(job: dict, cfg: Config) -> str:
    symbol, method, seed = job["symbol"], job["method"], job["seed"]
    out = _result_path(cfg, job)
    label = f"{symbol}/{os.path.splitext(os.path.basename(out))[0]}"
    if os.path.exists(out):
        return f"skip {label}"
    if job["hindsight_ticks"] is not None:
        cfg = dataclasses.replace(cfg, hindsight_ticks=job["hindsight_ticks"])
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
    prefix = f"[{label}]"

    if method in RULE_METHODS:
        fn = {"BAH": run_bah, "MV": run_mv, "TSM": run_tsm}[method]
        payload = _untrained_payload(*fn(test_m, cfg))
    elif method in PREDICTOR_METHODS:
        payload = _untrained_payload(*run_predictor(method, train_m, test_m, cfg, seed=seed))
    elif method == "DQN":
        net, log = train_dqn(cfg, train_m, val_m, seed=seed, log_prefix=prefix)
        payload = {**evaluate_dqn(net, test_m, cfg), "train_log": log}
    else:
        agent, log = train_agent(cfg, train_m, val_m, seed=seed, log_prefix=prefix, **RL_VARIANTS[method])
        payload = {**evaluate_greedy(agent, test_m, cfg), "train_log": log}

    payload.update({"symbol": symbol, "method": method, "seed": seed,
                    "hindsight_ticks": job["hindsight_ticks"],
                    "split_days": split_info, "elapsed_sec": round(time.time() - t0, 1)})
    _save(out, payload)
    return f"done {label} TR={payload['TR']:.4f} ({payload['elapsed_sec']}s)"


def make_jobs(
    symbols: list[str], seeds: tuple[int, ...], methods: list[str], horizons: tuple[int, ...]
) -> list[dict]:
    """展开 (标的 × 方法 × 种子 × hindsight 视野) 作业矩阵。

    仅 hindsight bonus 生效的方法随视野展开，其余方法与该超参无关（视野记为 None）。
    """
    jobs = []
    for symbol in symbols:
        for method in methods:
            hs = horizons if RL_VARIANTS.get(method, {}).get("hindsight") else (None,)
            ss = (None,) if method in RULE_METHODS else seeds
            jobs += [{"symbol": symbol, "method": method, "seed": s, "hindsight_ticks": h}
                     for h in hs for s in ss]
    order = {"BAH": 0, "MV": 0, "TSM": 0, "LGBM": 1, "MLP": 2, "GRU": 3, "DQN": 4}
    return sorted(jobs, key=lambda j: order.get(j["method"], 5))


def default_workers(cfg: Config) -> int:
    """并行作业数：GPU 下限制并发以免争抢显存，CPU 下按单进程线程预算切分核数。"""
    if resolve_device(cfg).type == "cuda":
        return GPU_WORKERS
    return max(1, (os.cpu_count() or 1) // cfg.num_threads)


def main() -> None:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["301308", "688030"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+",
                   choices=[*RULE_METHODS, *PREDICTOR_METHODS, "DQN", *RL_VARIANTS],
                   default=["BAH", "MV", "TSM", "LGBM", "MLP", "GRU", "DQN", "DS-NA", "DS-NH", "DS"])
    p.add_argument("--hindsight-ticks", nargs="+", type=int, default=[cfg.hindsight_ticks],
                   help="hindsight 视野档位（tick），给多个值即做敏感性实验，如 300 600 900 1200")
    p.add_argument("--workers", type=int, default=None, help="并行作业数，缺省按设备自适应")
    args = p.parse_args()

    device = resolve_device(cfg)
    workers = args.workers if args.workers is not None else default_workers(cfg)
    jobs = make_jobs(args.symbols, tuple(args.seeds), args.methods, tuple(args.hindsight_ticks))
    pending = [j for j in jobs if not os.path.exists(_result_path(cfg, j))]
    print(f"jobs: {len(jobs)} total, {len(pending)} pending; device={device} workers={workers}", flush=True)

    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, j, cfg): j for j in pending}
        for fut in cf.as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # 单个作业失败不阻塞其它作业
                print(f"FAIL {_result_path(cfg, futures[fut])}: {exc}", flush=True)


if __name__ == "__main__":
    main()
