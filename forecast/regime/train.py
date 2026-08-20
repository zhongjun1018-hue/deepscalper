"""模式识别产线：事后标签 → 分类器训练 / 评估 / 阈值标定（识别层唯一训练入口）。

各标的按 data_provider.split.chronological_split 逐标的单次时序切分，训练集跨
标的池化 fit、验证集 AUC 早停；评估报告 val/test 的 AUC 与 AP。门控概率阈值 τ
在验证段按真值不利占比率配平——两个取数方案（oracle / prediction）的判正率先天
可比，且不含测试段与盈亏信息。

产物写 runs_dir：model/（booster + 元数据）、meta.json（τ、切分、参数）、
metrics.json。按数据签名与参数哈希校验，任一变化自动重训（ensure_classifier
幂等，供 strategy/backtest.py 与 webviz 在使用前调用）。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data_provider.split import SPLIT_RATIOS
from data_provider.ticks import list_symbols, symbol_source_signature
from data_provider.windows import CACHE_SCHEMA, FEATURE_NAMES

from forecast.regime import classify, labels
from forecast.regime.config import RegimeConfig
from forecast.regime.data import load_banks


def data_hash(symbols, cfg: RegimeConfig) -> str:
    """产线身份标识：schema / 标的 / 源文件签名 / 窗口规格 / 切分比 / 模式与模型参数。"""
    payload = {
        "schema": CACHE_SCHEMA,
        "symbols": sorted(symbols),
        "source": {symbol: symbol_source_signature(cfg.data_dir, symbol)
                   for symbol in sorted(symbols)},
        "window": dataclasses.asdict(cfg.window),
        "stride_ticks": cfg.stride_ticks,
        "split_ratios": SPLIT_RATIOS,
        "seed": cfg.seed,
        "pattern": [cfg.residual_ratio_threshold, cfg.slope_ratio_threshold,
                    cfg.sticky_stay, cfg.emission_noise],
        "model_kwargs": cfg.model_kwargs,
        "feature_names": FEATURE_NAMES,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def recognition_metrics(probabilities, targets) -> dict:
    """二分类识别指标：AUC / AP / 真值不利占比。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {"samples": int(len(targets)),
            "base_rate": float(np.mean(targets)),
            "auc": float(roc_auc_score(targets, probabilities)),
            "average_precision": float(average_precision_score(targets,
                                                               probabilities))}


def train(cfg: RegimeConfig) -> Path:
    """池化训练、率配平 τ 并评估；返回模型目录 runs_dir/model/。"""
    symbols = sorted(cfg.symbols)
    if not symbols:
        raise ValueError("cfg.symbols 为空")
    banks = load_banks(cfg)
    pattern = {symbol: labels.pattern_labels(bank, cfg)
               for symbol, bank in banks.items()}
    digest = data_hash(symbols, cfg)

    train_x, train_y = classify.pooled_rows(banks, pattern, "train", cfg)
    val_x, val_y = classify.pooled_rows(banks, pattern, "val", cfg)
    classifier = classify.Classifier(seed=cfg.seed, data_hash=digest,
                                     model_kwargs=cfg.model_kwargs)
    print(f"train: {train_x.shape} | validation: {val_x.shape}", flush=True)
    started = time.time()
    classifier.fit(train_x, train_y, eval_set=(val_x, val_y))
    print(f"fit: {time.time() - started:.2f}s", flush=True)

    val_prob = classifier.predict(val_x)
    threshold = classify.calibrate_threshold(val_prob, val_y)

    out_dir = Path(cfg.runs_dir)
    model_dir = out_dir / "model"
    classifier.save(str(model_dir))
    with open(out_dir / "meta.json", "w") as file:
        json.dump({
            "data_hash": digest,
            "symbols": symbols,
            "seed": cfg.seed,
            "window": dataclasses.asdict(cfg.window),
            "stride_ticks": cfg.stride_ticks,
            "split_ratios": SPLIT_RATIOS,
            "pattern": {"residual_ratio_threshold": cfg.residual_ratio_threshold,
                        "slope_ratio_threshold": cfg.slope_ratio_threshold,
                        "sticky_stay": cfg.sticky_stay,
                        "emission_noise": cfg.emission_noise},
            "model_kwargs": cfg.model_kwargs,
            "threshold": threshold,
        }, file, indent=1)

    metrics = {"data_hash": digest, "symbols": symbols, "threshold": threshold,
               "val": recognition_metrics(val_prob, val_y)}
    test_x, test_y = classify.pooled_rows(banks, pattern, "test", cfg)
    metrics["test"] = recognition_metrics(classifier.predict(test_x), test_y)
    for flag in ("val", "test"):
        print(f"{flag}: {metrics[flag]['samples']} 行 | "
              f"AUC {metrics[flag]['auc']:.3f} | "
              f"AP {metrics[flag]['average_precision']:.3f}", flush=True)
    with open(out_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=1)
    return model_dir


def load_classifier(symbols, cfg: RegimeConfig) -> tuple:
    """加载与当前 data_hash 一致的 (Classifier, τ)；不一致抛 ValueError。"""
    digest = data_hash(sorted(symbols), cfg)
    classifier = classify.Classifier(seed=cfg.seed, data_hash=digest,
                                     model_kwargs=cfg.model_kwargs)
    classifier.load(str(Path(cfg.runs_dir) / "model"))
    with open(Path(cfg.runs_dir) / "meta.json") as file:
        return classifier, float(json.load(file)["threshold"])


def ensure_classifier(symbols, cfg: RegimeConfig) -> tuple:
    """幂等入口：产物与当前 data_hash 一致则直接加载，否则重训后加载。"""
    cfg = dataclasses.replace(cfg, symbols=tuple(symbols))
    try:
        return load_classifier(symbols, cfg)
    except (FileNotFoundError, ValueError):
        train(cfg)
        return load_classifier(symbols, cfg)


def main():
    parser = argparse.ArgumentParser(
        description="模式识别训练：事后标签、分类器与阈值标定，val/test 评估")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--runs-dir", default="forecast/regime/runs")
    parser.add_argument("--force", action="store_true", help="忽略现有产物强制重训")
    args = parser.parse_args()
    symbols = tuple(sorted(args.symbols or list_symbols(args.data_dir)))
    cfg = RegimeConfig(data_dir=args.data_dir, cache_dir=args.cache_dir,
                       runs_dir=args.runs_dir, symbols=symbols)
    if args.force:
        train(cfg)
        return
    try:
        load_classifier(symbols, cfg)
        print("regime: 产物已是最新，跳过训练（--force 强制重训）")
    except (FileNotFoundError, ValueError):
        train(cfg)


if __name__ == "__main__":
    main()
