"""control 超参数一次一因子探索（python -m control.sweep）。

以 Config 默认值为中心点，SWEEP_LADDERS 的每个参数沿梯子单独变动、其余保持默认，
覆盖网络结构、优化、折扣、奖励塑形（w、λ）与目标网络、优先级回放。方法固定为
完整 GRID，作业为统一训练（内含全部标的），切分、训练与选模协议与 control.train
一致；比较判据为验证集 SR（逐标的等权聚合），测试指标仅随表报告、不参与选值。
一次一因子忽略参数间交互，定位是围绕默认配置的敏感性分析而非全局寻优；
w、λ 的正式选参仍走 design 7.1 的梯子协议。

作业为 (参数, 值, 种子)，全部梯子共用同一中心点作业；结果写
control/runs/sweep/<参数>_<值>_seed<k>.json（中心点为 default_seed<k>.json），
已存在的作业自动跳过（断点续跑），每个作业对应一个 wandb run（job_type "sweep"，
config 含 param/value）。作业完成后按验证集 SR 汇总各梯子，写 sweep/summary.csv。

--combo PARAM=VALUE ... 为组合确认模式：只训练中心点与给定组合（多参数同改，值不必
落在梯子上），配合 --seeds 按种子配对对照——验证一次一因子结论叠加后是否仍成立，
或探查梯子之外的单点。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import multiprocessing as mp
import os
import time
from dataclasses import replace

import pandas as pd

from data_provider.ticks import list_symbols

from .config import Config
from .env import action_params
from .model import resolve_device
from .tracking import Tracker
from .train import (METRICS, build_split_markets, default_workers,
                    evaluate_pooled, load_symbol_cache, save_result, train_agent)

# 每个参数一条梯子（含 Config 默认值即中心点），一次只变动一个参数
SWEEP_LADDERS: dict[str, tuple] = {
    # 网络结构
    "hidden_size": (32, 64, 128),
    "macro_hidden": (32, 64, 128),
    "trunk_hidden": (64, 128, 256),
    # 优化
    "lr": (3e-5, 1e-4, 3e-4),
    "batch_size": (32, 64, 128),
    # 每分钟折扣（TD 折扣为 gamma^decision_interval_min）
    "gamma": (0.98, 0.99, 0.995),
    # 奖励塑形（design 6.2 的偏好参数，档位同 control.train 的正式梯子）
    "hindsight_weight": (0.02, 0.05, 0.1, 0.2),
    "inventory_lambda": (0.0, 1.0, 3.0, 10.0, 30.0),
    # 目标网络与优先级回放
    "target_sync": (1000, 2000, 4000),
    "per_alpha": (0.4, 0.6, 0.8),
}


def job_stem(job: dict) -> str:
    """结果文件与 wandb run 的基名：<参数>_<值>_seed<k>；中心点为 default_seed<k>，
    组合为 combo_<参数><值>+..._seed<k>（参数按名排序，见 parse_combo）。"""
    if job["param"] is None:
        return f"default_seed{job['seed']}"
    if job["param"] == "combo":
        tag = "+".join(f"{param}{value:g}" for param, value in job["value"].items())
        return f"combo_{tag}_seed{job['seed']}"
    return f"{job['param']}_{job['value']:g}_seed{job['seed']}"


def _result_path(cfg: Config, job: dict) -> str:
    return os.path.join(cfg.runs_dir, "sweep", job_stem(job) + ".json")


def make_jobs(seeds: tuple[int, ...], params: list[str]) -> list[dict]:
    """展开 (参数 × 非默认档 × 种子) 作业矩阵，外加共享的中心点作业。"""
    defaults = Config()
    jobs = [{"param": None, "value": None, "seed": k} for k in seeds]
    jobs += [{"param": param, "value": value, "seed": k}
             for param in params
             for value in SWEEP_LADDERS[param] if value != getattr(defaults, param)
             for k in seeds]
    return jobs


def parse_combo(items: list[str]) -> dict:
    """把 PARAM=VALUE 列表解析为配置覆盖：参数限于 SWEEP_LADDERS，类型随 Config 字段，
    按参数名排序（组合的身份与书写顺序无关）。"""
    defaults = Config()
    overrides = {}
    for item in items:
        param, _, text = item.partition("=")
        if param not in SWEEP_LADDERS or not text:
            raise ValueError(f"组合项须为 PARAM=VALUE 且参数在梯子集合中：{item}")
        overrides[param] = type(getattr(defaults, param))(text)
    return dict(sorted(overrides.items()))


def make_combo_jobs(seeds: tuple[int, ...], overrides: dict) -> list[dict]:
    """组合确认的作业矩阵：每个种子一个中心点作业加一个组合作业。"""
    return [{"param": param, "value": value, "seed": k}
            for k in seeds
            for param, value in ((None, None), ("combo", overrides))]


def run_job(job: dict, cfg: Config) -> str:
    seed = job["seed"]
    out = _result_path(cfg, job)
    label = job_stem(job)
    if os.path.exists(out):
        return f"skip {label}"
    if job["param"] == "combo":
        cfg = replace(cfg, **job["value"])
    elif job["param"] is not None:
        cfg = replace(cfg, **{job["param"]: job["value"]})
    t0 = time.time()
    markets, _ = build_split_markets(cfg)
    tracker = Tracker(cfg, label, {"method": "sweep", "param": job["param"],
                                   "value": job["value"], "seed": seed})
    try:
        agent, log = train_agent(cfg, markets, seed=seed, log_prefix=f"[{label}]",
                                 tracker=tracker)
        test = evaluate_pooled(markets["test"],
                               lambda obs: action_params(cfg, agent.greedy(obs)))
        payload = {**test["pooled"], "per_symbol": test["per_symbol"], "train_log": log}
        tracker.log_test(payload)
    finally:
        tracker.finish()  # 失败路径也要收尾，否则进程池复用本进程时下个作业会并入未关闭的 run
    payload.update({"param": job["param"], "value": job["value"],
                    "seed": seed, "elapsed_sec": round(time.time() - t0, 1)})
    save_result(out, payload)
    return (f"done {label} val_SR={log['best_val_SR']:.3f} "
            f"TR={payload['TR']:.4f} ({payload['elapsed_sec']}s)")


def load_rows(sweep_dir: str) -> pd.DataFrame:
    """逐作业结果行；中心点的 param/value 记缺失，val_SR 为训练日志的选模最优。"""
    rows = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({"param": r["param"], "value": r["value"],
                     "seed": r["seed"], "val_SR": r["train_log"]["best_val_SR"],
                     **{k: r[k] for k in METRICS}})
    return pd.DataFrame(rows)


def sweep_table(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """各梯子的档位对比：中心点并入每个参数的默认档（值后缀 *），跨种子取均值。"""
    defaults = Config()
    center = df[df["param"].isna()]
    blocks = []
    for param in params:
        default_value = getattr(defaults, param)
        block = pd.concat([df[df["param"] == param],
                           center.assign(param=param, value=default_value)])
        agg = block.groupby("value", sort=True)[["val_SR", *METRICS]].mean()
        agg["n_runs"] = block.groupby("value").size()
        agg = agg.reset_index()
        agg.insert(0, "param", param)
        agg["value"] = agg["value"].map(
            lambda v: f"{v:g}" + ("*" if v == default_value else ""))
        blocks.append(agg)
    return pd.concat(blocks, ignore_index=True).round(4)


def combo_table(df: pd.DataFrame, overrides: dict) -> tuple[pd.DataFrame, int, int]:
    """组合与中心点按种子配对的对照，返回（对照表, 配对数, 组合占优数）。

    末行为配对差（组合 − 中心点）的均值；对照只计两侧都已完成的配对。
    """
    center = df[df["param"].isna()].set_index("seed")
    combo = df[df["param"] == "combo"]
    combo = combo[combo["value"].apply(lambda v: v == overrides)].set_index("seed")
    shared = center.index.intersection(combo.index)
    center, combo = center.loc[shared], combo.loc[shared]
    columns = ["val_SR", *METRICS]
    table = pd.DataFrame([
        {"config": "默认", **center[columns].mean()},
        {"config": "组合", **combo[columns].mean()},
        {"config": "配对差", **(combo[columns] - center[columns]).mean()},
    ]).round(4)
    return table, len(shared), int((combo["val_SR"] > center["val_SR"]).sum())


def main() -> None:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="control 超参数一次一因子探索（中心点为 Config 默认值）")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="标的代码，缺省为 data 目录下全部标的")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--params", nargs="+", choices=list(SWEEP_LADDERS),
                       default=list(SWEEP_LADDERS), help="参与探索的参数梯子，缺省为全部")
    scope.add_argument("--combo", nargs="+", metavar="PARAM=VALUE", default=None,
                       help="组合确认：只训练中心点与该组合（多参数同改，值不限于梯子档位）")
    p.add_argument("--workers", type=int, default=None, help="并行作业数，缺省按设备自适应")
    p.add_argument("--wandb-project", default=cfg.wandb_project, help="wandb 项目名")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default=cfg.wandb_mode, help="wandb 记录模式，disabled 即不记录")
    args = p.parse_args()

    symbols = args.symbols or list_symbols(cfg.data_dir)
    cfg = replace(cfg, symbols=tuple(symbols),
                  wandb_project=args.wandb_project, wandb_mode=args.wandb_mode)
    workers = args.workers if args.workers is not None else default_workers(cfg)
    try:
        combo = parse_combo(args.combo) if args.combo else None
    except ValueError as exc:
        p.error(str(exc))
    jobs = (make_combo_jobs(tuple(args.seeds), combo) if combo
            else make_jobs(tuple(args.seeds), args.params))
    pending = [j for j in jobs if not os.path.exists(_result_path(cfg, j))]
    print(f"sweep jobs: {len(jobs)} total, {len(pending)} pending; "
          f"device={resolve_device(cfg)} workers={workers}", flush=True)

    # 统一缓存串行预建：进程池内并发重建同一文件会写坏缓存（与 control.train 同理）
    for symbol in symbols:
        load_symbol_cache(symbol, cfg)

    with cf.ProcessPoolExecutor(max_workers=workers,
                                mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, j, cfg): j for j in pending}
        for fut in cf.as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # 单个作业失败不阻塞其它作业
                print(f"FAIL {_result_path(cfg, futures[fut])}: {exc}", flush=True)

    sweep_dir = os.path.join(cfg.runs_dir, "sweep")
    df = load_rows(sweep_dir)
    if df.empty:
        print(f"未找到探索结果：{sweep_dir}")
        return
    if combo:
        table, pairs, wins = combo_table(df, combo)
        print(f"\n组合 {combo} 与默认配置的配对对照"
              f"（验证集 SR 为判据；{pairs} 组配对，组合占优 {wins} 组）：")
        print(table.to_string(index=False))
        return
    table = sweep_table(df, args.params)
    out_path = os.path.join(sweep_dir, "summary.csv")
    table.to_csv(out_path, index=False)
    print("\n各梯子档位对比（验证集 SR 为判据，* 为默认档；测试指标仅供报告）：")
    print(table.to_string(index=False))
    print(f"\n已保存：{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
