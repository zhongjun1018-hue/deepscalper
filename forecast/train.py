"""LightGBM 前瞻回归的训练与预测产线（RL 状态特征的数据源）。

各标的缓存按 data_provider.split.chronological_split 逐标的单次时序切分，
训练集跨标的池化 fit、验证集早停；训练与评估只取每 stride_ticks 一行（相邻 tick 的
回看窗口重叠 599/600），推理则覆盖全部 tick 并回写统一缓存 cache/<symbol>.npz 的
preds 块（控制器状态特征，5.4），最后在 val/test 上评估写 runs_dir/metrics.json。
ensure_predictions 为幂等预建入口，strategy/backtest.py 在回放前无条件调用；
RL 训练（control.train）只读预测缓存、不在此重训。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data_provider.split import SPLIT_RATIOS, chronological_split
from data_provider.ticks import list_symbols, symbol_source_signature
from data_provider.windows import (CACHE_SCHEMA, FEATURE_NAMES, TARGET_NAMES,
                                   load_cache, predictions_current,
                                   write_predictions)
from forecast.config import Config
from forecast.model import Model


def _with_symbol(features, index: int):
    """把 symbol_id 分类列拼到窗口特征之后，得到模型输入 (n, 48) f32。"""
    return np.column_stack(
        [features, np.full(len(features), index, dtype=np.float32)]).astype(np.float32)


def symbol_rows(entry: dict, dates, stride: int) -> tuple:
    """单标的的 (窗口特征 (n,47), 目标 (n,5)) 有效行，逐日每 stride 个 tick 取一行。

    行过滤口径：5 目标全有限 ∧ 任一窗口特征有限；窗口特征保留 NaN（LightGBM 原生处理）。
    """
    take = np.isin(entry["dates"], list(dates))
    feats = entry["features"][take][:, ::stride].reshape(-1, len(FEATURE_NAMES))
    targs = entry["targets"][take][:, ::stride].reshape(-1, len(TARGET_NAMES))
    valid = np.isfinite(targs).all(axis=1) & np.isfinite(feats).any(axis=1)
    return feats[valid], targs[valid]


def training_rows(banks: dict, date_set: dict, stride: int) -> tuple:
    """跨标的池化组装 (x (n,48) f32, y (n,5) f32)：symbol_id 为排序后标的集合中的索引。

    banks: {标的: load_cache(zero_nan=False) 的返回}；
    date_set: {标的: 该段日期集合}（逐标的切分，异日历标的不互相泄漏）。
    """
    xs, ys = [], []
    for index, symbol in enumerate(sorted(banks)):
        feats, targs = symbol_rows(banks[symbol], date_set[symbol], stride)
        xs.append(_with_symbol(feats, index))
        ys.append(targs)
    return np.concatenate(xs), np.concatenate(ys).astype(np.float32)


def data_hash(symbols, cfg: Config) -> str:
    """产线身份标识：schema / 标的 / 源文件签名 / 窗口规格 / 切分比 / seed / 超参 / 特征目标名。"""
    payload = {
        "schema": CACHE_SCHEMA,
        "symbols": sorted(symbols),
        "source": {symbol: symbol_source_signature(cfg.data_dir, symbol)
                   for symbol in sorted(symbols)},
        "window": dataclasses.asdict(cfg.window),
        "stride_ticks": cfg.stride_ticks,
        "split_ratios": SPLIT_RATIOS,
        "seed": cfg.seed,
        "model_kwargs": cfg.model_kwargs,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def forecast_metrics(prediction, target, baseline, label_names) -> dict:
    """逐目标 MAE/RMSE 与相对训练集均值基线的 skill。

    baseline: {目标名: 训练集均值}。
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    labels = {}
    for column, name in enumerate(label_names):
        error = prediction[:, column] - target[:, column]
        baseline_error = baseline[name] - target[:, column]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        baseline_mae = float(np.mean(np.abs(baseline_error)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
        labels[name] = {
            "mean_prediction": float(prediction[:, column].mean()),
            "mean_true": float(target[:, column].mean()),
            "objective": "regression",
            "mae": mae,
            "rmse": rmse,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "mae_skill": float(1.0 - mae / baseline_mae) if baseline_mae > 0 else 0.0,
            "rmse_skill": float(1.0 - rmse / baseline_rmse) if baseline_rmse > 0 else 0.0,
        }
    return {
        "samples": int(len(target)),
        "labels": labels,
        "mean_mae": float(np.mean([v["mae"] for v in labels.values()])),
        "mean_rmse": float(np.mean([v["rmse"] for v in labels.values()])),
    }


def _load_banks(cfg: Config) -> dict:
    return {symbol: load_cache(
        symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir,
        spec=cfg.window, zero_nan=False) for symbol in sorted(cfg.symbols)}


def _date_sets(splits: dict, flag: str) -> dict:
    return {symbol: set(getattr(split, flag)) for symbol, split in splits.items()}


def _plot_test_rows(banks, splits, symbols, model, cfg: Config) -> None:
    """测试段预测对比图与相关性热力图（utils/plots.py）→ runs_dir/figures/。"""
    from utils.plots import (clear_result_figures, plot_correlation_heatmap,
                             plot_pred_vs_true)

    xs, ys, ids = [], [], []
    for index, symbol in enumerate(symbols):
        feats, targs = symbol_rows(banks[symbol], set(splits[symbol].test),
                                   cfg.stride_ticks)
        xs.append(_with_symbol(feats, index))
        ys.append(targs)
        ids.append(np.full(len(feats), index, dtype=np.int64))
    prediction = model.predict(np.concatenate(xs))
    figure_dir = str(Path(cfg.runs_dir) / "figures")
    clear_result_figures(figure_dir)
    plot_pred_vs_true(prediction, np.concatenate(ys), np.concatenate(ids), symbols,
                      TARGET_NAMES, figure_dir)
    plot_correlation_heatmap(prediction, np.concatenate(ys), np.concatenate(ids), symbols,
                             TARGET_NAMES, figure_dir)
    print(f"figures: {figure_dir}")


def train(cfg: Config, plot: bool = False) -> Path:
    """池化训练、重建全部标的预测缓存并评估；返回模型目录 runs_dir/model/。"""
    symbols = sorted(cfg.symbols)
    if not symbols:
        raise ValueError("cfg.symbols 为空")
    banks = _load_banks(cfg)
    splits = {symbol: chronological_split(list(banks[symbol]["dates"]))
              for symbol in symbols}
    digest = data_hash(symbols, cfg)

    train_x, train_y = training_rows(banks, _date_sets(splits, "train"), cfg.stride_ticks)
    val_x, val_y = training_rows(banks, _date_sets(splits, "val"), cfg.stride_ticks)
    model = Model(seed=cfg.seed, data_hash=digest, model_kwargs=cfg.model_kwargs)
    print(f"train: {train_x.shape} | validation: {val_x.shape}", flush=True)
    started = time.time()
    model.fit(train_x, train_y, eval_set=(val_x, val_y))
    print(f"fit: {time.time() - started:.2f}s", flush=True)

    baseline = {name: float(train_y[:, column].mean())
                for column, name in enumerate(TARGET_NAMES)}

    model_dir = Path(cfg.runs_dir) / "model"
    model.save(str(model_dir))
    with open(model_dir / "meta.json", "w") as file:
        json.dump({
            "data_hash": digest,
            "symbols": symbols,
            "seed": cfg.seed,
            "window": dataclasses.asdict(cfg.window),
            "stride_ticks": cfg.stride_ticks,
            "split_ratios": SPLIT_RATIOS,
            "model_kwargs": cfg.model_kwargs,
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "splits": {symbol: {"train": splits[symbol].train,
                                "val": splits[symbol].val,
                                "test": splits[symbol].test}
                       for symbol in symbols},
            "baseline": baseline,
        }, file, indent=1)

    # 推理覆盖全部 tick：控制器在任一 tick 决策时都能读到当拍预测
    for index, symbol in enumerate(symbols):
        entry = banks[symbol]
        days, ticks = entry["features"].shape[:2]
        feats = entry["features"].reshape(-1, len(FEATURE_NAMES))
        preds = model.predict(_with_symbol(feats, index)).reshape(days, ticks, -1)
        # 回看窗口未完整的行没有特征，不留预测（NaN；控制器侧按 0 读）
        preds[~np.isfinite(feats).any(axis=1).reshape(days, ticks)] = np.nan
        write_predictions(symbol, preds, digest, cache_dir=cfg.cache_dir)

    metrics = {"data_hash": digest, "symbols": symbols, "baseline": baseline}
    for flag in ("val", "test"):
        x, y = training_rows(banks, _date_sets(splits, flag), cfg.stride_ticks)
        metrics[flag] = forecast_metrics(model.predict(x), y, baseline, TARGET_NAMES)
        print(f"{flag}: {metrics[flag]['samples']} 行 | "
              f"MAE {metrics[flag]['mean_mae']:.6g} | "
              f"RMSE {metrics[flag]['mean_rmse']:.6g}", flush=True)
    with open(Path(cfg.runs_dir) / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=1)

    if plot:
        _plot_test_rows(banks, splits, symbols, model, cfg)
    return model_dir


def plot_test(cfg: Config) -> None:
    """用磁盘上的模型重出测试段图表（训练被幂等跳过时的 --plot 路径）。"""
    digest = data_hash(sorted(cfg.symbols), cfg)
    model = Model(seed=cfg.seed, data_hash=digest, model_kwargs=cfg.model_kwargs)
    model.load(str(Path(cfg.runs_dir) / "model"))
    banks = _load_banks(cfg)
    splits = {symbol: chronological_split(list(banks[symbol]["dates"]))
              for symbol in sorted(cfg.symbols)}
    _plot_test_rows(banks, splits, sorted(cfg.symbols), model, cfg)


def _up_to_date(symbols, cfg: Config) -> bool:
    """全部标的统一缓存中的预测块与当前 data_hash 一致（缓存为唯一校验来源）。"""
    digest = data_hash(symbols, cfg)
    return all(predictions_current(symbol, digest, cfg.cache_dir) for symbol in symbols)


def ensure_predictions(symbols, cfg: Config) -> None:
    """幂等预建：全部标的预测块与当前 data_hash 一致则跳过，否则重训回写统一缓存。"""
    if _up_to_date(symbols, cfg):
        print(f"forecast: 预测缓存与 data_hash 一致，跳过训练 {sorted(symbols)}",
              flush=True)
        return
    train(dataclasses.replace(cfg, symbols=tuple(symbols)))


def main():
    parser = argparse.ArgumentParser(
        description="LightGBM 前瞻预测：池化训练、预测缓存与 val/test 评估")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--runs-dir", default="forecast/runs")
    parser.add_argument("--force", action="store_true", help="忽略现有产物强制重训")
    parser.add_argument("--plot", action="store_true",
                        help="输出测试段预测对比图与相关性热力图到 runs_dir/figures/")
    args = parser.parse_args()
    symbols = tuple(args.symbols or list_symbols(args.data_dir))
    cfg = Config(data_dir=args.data_dir, cache_dir=args.cache_dir,
                 runs_dir=args.runs_dir, symbols=symbols)
    if args.force:
        train(cfg, plot=args.plot)
    elif _up_to_date(symbols, cfg):
        print("forecast: 产物已是最新，跳过训练（--force 强制重训）")
        if args.plot:
            plot_test(cfg)
    else:
        train(cfg, plot=args.plot)


if __name__ == "__main__":
    main()
