"""并行实验运行器：对指定标的执行网格 RL 的全部变体与基线。

每个 (标的, 方法, 种子, w, λ) 为一个独立作业，结果写入
control/runs/<symbol>/<method>[_w<权重>][_lam<λ>][_seed<k>].json，RL 作业同时把验证最优的
online 网络写入同名 .pt 检查点（webviz 回放用，见 control/trace.py）；
已存在的作业自动跳过，因此脚本可安全重复执行（断点续跑）。w 与 λ 给多个值即展开为超参梯子，
由 summarize.py 按验证集 SR 选优（design 6.2 / 7.1）。

数据按 data_provider.split.chronological_split 单次时序切分（7:1:2）；
统一缓存 cache/<symbol>.npz（窗口特征 + 预测）在主进程串行预建，作业进程内只读。

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.split import chronological_split
from data_provider.ticks import list_symbols, load_days
from data_provider.windows import load_cache

from control.baselines import run_grid_scan, run_hold_base, run_open_grid
from control.config import Config
from control.env import action_params
from control.features import fit_feature_stats
from control.model import resolve_device
from control.tracking import Tracker
from control.train import build_markets, evaluate, regime_stats, save_checkpoint, train_agent

# 固定半宽档的消融（其余分支照常学习），另有 hindsight 消融
RL_VARIANTS = {
    "GRID": {},
    "GRID-NH": {"hindsight": False},
    "GRID-FW": {"fixed_width": 0.1},
}
RULE_METHODS = ("HOLD", "OPEN", "SCAN")   # 无需训练、无随机种子

GPU_WORKERS = 2  # 单卡下的并行作业数：作业瓶颈在 CPU 侧观测重建，少量并发即可打满


def _variant_kwargs(method: str, cfg: Config) -> dict:
    """将变体说明翻译为 train_agent 的参数（消融分支转为档位索引）。"""
    spec = dict(RL_VARIANTS[method])
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


def _save(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def _load_cache(symbol: str, cfg: Config) -> dict:
    """按统一窗口规格加载标的缓存（主进程预建后此处为只读命中）。"""
    return load_cache(
        symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir, spec=cfg.window,
    )


def run_job(job: dict, cfg: Config) -> str:
    symbol, method, seed = job["symbol"], job["method"], job["seed"]
    out = _result_path(cfg, job)
    run_name = os.path.splitext(os.path.basename(out))[0]
    label = f"{symbol}/{run_name}"
    if os.path.exists(out):
        return f"skip {label}"
    overrides = {k: job[k] for k in ("hindsight_weight", "inventory_lambda")
                 if job[k] is not None}
    cfg = dataclasses.replace(cfg, **overrides)
    t0 = time.time()
    days = load_days(symbol, cfg.data_dir, cfg.window.atr_window)
    split = chronological_split([d.date for d in days])
    train_d, val_d, test_d = (
        [d for d in days if d.date in dates]
        for dates in (set(split.train), set(split.val), set(split.test))
    )
    cache = _load_cache(symbol, cfg)   # 窗口特征与预测同一文件，主进程已预建
    # symbol_id 与 forecast 同口径：排序后标的集合中的索引
    symbol_id = sorted(cfg.symbols).index(symbol) if symbol in cfg.symbols else 0
    train_m = build_markets(train_d, cfg, cache, symbol_id)
    val_m = build_markets(val_d, cfg, cache, symbol_id)
    test_m = build_markets(test_d, cfg, cache, symbol_id)
    del days, train_d, val_d, test_d  # 释放原始 DataFrame

    # 仅用训练集拟合标准化统计量，val/test 复用（无前视泄漏）
    stats = fit_feature_stats(train_m, cfg) if cfg.normalize else None
    for m in train_m + val_m + test_m:
        m.set_stats(stats)

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
    _save(out, payload)
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
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
    cfg = dataclasses.replace(cfg, symbols=tuple(symbols),
                              wandb_project=args.wandb_project, wandb_mode=args.wandb_mode)
    device = resolve_device(cfg)
    workers = args.workers if args.workers is not None else default_workers(cfg)
    jobs = make_jobs(symbols, tuple(args.seeds), args.methods,
                     tuple(args.hindsight_weights), tuple(args.inventory_lambdas))
    pending = [j for j in jobs if not os.path.exists(_result_path(cfg, j))]
    print(f"jobs: {len(jobs)} total, {len(pending)} pending; device={device} workers={workers}", flush=True)

    # 统一缓存串行预建：进程池内并发重建同一文件会写坏缓存
    for symbol in symbols:
        t0 = time.time()
        _load_cache(symbol, cfg)
        print(f"window cache {symbol}: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    from forecast.config import Config as PredictionConfig   # 惰性导入：仅预建预测时需要
    from forecast.train import ensure_predictions
    prediction_cfg = PredictionConfig(
        data_dir=cfg.data_dir, cache_dir=cfg.cache_dir, symbols=cfg.symbols,
        window=cfg.window,
    )
    ensure_predictions(cfg.symbols, prediction_cfg)
    print(f"predictions: {time.time()-t0:.1f}s", flush=True)

    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, j, cfg): j for j in pending}
        for fut in cf.as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # 单个作业失败不阻塞其它作业
                print(f"FAIL {_result_path(cfg, futures[fut])}: {exc}", flush=True)


if __name__ == "__main__":
    main()
