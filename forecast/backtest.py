"""门控网格回测：7:1:2 切分的测试段内 baseline（常开）与各 门控×scheme 的逐日对比。

对测试段内有预测的 (symbol, day)，从 t0=lookback_ticks（或当日首个有预测的 tick，
取较晚者）起用 strategy/engine.run_day 回放（每笔成交 1 手、省略 100 股因子；门控只在
净持仓为 0 时按 stride_ticks 的决策节奏读取当拍信号）；报价数据用 data_provider/ticks.load_days，成本由
strategy/backtest.py 按 strategy/costs.py 的统一费率（双边佣金 1e-4、卖出印花税 5e-4）
叠加，逐日指标与汇总用 strategy/metrics.py 的 day_frame/summarize。结果写
runs_dir/backtest/（summary.json + 各指标热力图 SVG）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_provider.split import chronological_split
from data_provider.ticks import list_symbols, load_days
from data_provider.windows import load_cache
from forecast import figures
from forecast.config import Config
from forecast.signals import (DEFAULT_RESIDUAL_RATIO_THRESHOLD,
                                DEFAULT_SLOPE_RATIO_THRESHOLD, GATES, SCHEMES,
                                build_gate_masks, gate_rules)
from forecast.train import ensure_predictions
from strategy import backtest as cost_backtest
from strategy import engine
from strategy import metrics as strategy_metrics


def _day_arrays(day, n_ticks):
    """单日连续竞价序列：(bid1, ask1, mid, 逐 tick 锚点)。

    锚点是门控相对宽度 w = 半宽/锚点 的分母，即该 tick 的 mid；缓存按标的最大快照数
    对齐，本日之外的尾部记 NaN。
    """
    frame = day.frame
    bid1 = frame["Buy1Price"].to_numpy(np.float64)
    ask1 = frame["Sell1Price"].to_numpy(np.float64)
    mid = 0.5 * (bid1 + ask1)
    anchors = np.full(n_ticks, np.nan)
    anchors[:len(mid)] = mid
    return bid1, ask1, mid, anchors


def _summary(records, nets) -> dict:
    """strategy/metrics 形态汇总 + strategy/backtest 费用后净利润汇总。"""
    summary = strategy_metrics.summarize(strategy_metrics.day_frame(records))
    net = cost_backtest.backtest(nets)
    return {**summary, "total_net_profit": net["total"],
            "mean_net_profit": net["mean"], "std_net_profit": net["std"]}


def gate_line(prefix, gate, counts) -> str:
    """一行门控触发占比：oracle / prediction 各占可判定窗口数的比例。"""
    total = counts["total"]
    shares = ", ".join(
        f"{scheme} {100.0 * counts[scheme] / total:.2f}%" if total else f"{scheme} -"
        for scheme in SCHEMES[1:])
    return f"{prefix} {gate} 门控窗口: {shares}（共 {total}）"


def run_symbol(symbol, cfg: Config, t0_tick, gates, rules):
    """回放一个标的测试段内有预测的交易日，返回 (records, nets, gated)。

    records[gate][scheme] 为逐日 run_day 结果（scheme "none" 为共享的 baseline）；
    nets[gate][scheme] 为逐日 (fills, close_mid)，供 strategy/backtest.py 计费。
    """
    cache = load_cache(symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir,
                       spec=cfg.window, zero_nan=False)
    dates = cache["dates"]
    test = set(chronological_split(list(dates)).test)

    days = {day.date: day for day in load_days(symbol, data_dir=cfg.data_dir,
                                               atr_days=cfg.window.atr_window)}
    index = [i for i, date in enumerate(dates)
             if date in test and date in days and np.isfinite(cache["width"][i])]
    records = {gate: {scheme: [] for scheme in SCHEMES} for gate in gates}
    nets = {gate: {scheme: [] for scheme in SCHEMES} for gate in gates}
    gated = {gate: {"total": 0, "oracle": 0, "prediction": 0} for gate in gates}
    if not index:
        return records, nets, gated

    n_ticks = cache["targets"].shape[1]
    arrays = {i: _day_arrays(days[dates[i]], n_ticks) for i in index}
    masks, gated = build_gate_masks(
        {"oracle": cache["targets"][index], "prediction": cache["preds"][index]},
        dates[index].tolist(), cache["width"][index],
        np.stack([arrays[i][3] for i in index]), gates, rules)

    for i in index:
        date = str(dates[i])
        bid1, ask1, mid, _ = arrays[i]
        # 当日首个有预测的 tick（oracle 真值与预测值同一判定点集），回放自其次一 tick 起
        usable = (np.isfinite(cache["targets"][i]).all(axis=1) &
                  np.isfinite(cache["preds"][i]).any(axis=1))
        if not usable.any():
            continue
        start = max(t0_tick, int(np.flatnonzero(usable).min()) + 1)
        take = np.arange(start, len(mid))
        if len(take) == 0:
            continue
        width = float(cache["width"][i])
        close_mid = float(mid[take][-1])
        baseline = engine.run_day(bid1[take], ask1[take], mid[take], hard_exclude=None,
                                  width=width, trace=True)
        for gate in gates:
            for scheme in SCHEMES:
                mask = masks.get((gate, scheme, date))
                result = baseline if scheme == "none" else engine.run_day(
                    bid1[take], ask1[take], mid[take],
                    hard_exclude=None if mask is None else mask[take],
                    width=width, decide_interval=cfg.stride_ticks, trace=True)
                records[gate][scheme].append(result)
                nets[gate][scheme].append(
                    (cost_backtest.fills_from_events(result["events"]), close_mid))
    return records, nets, gated


def main():
    parser = argparse.ArgumentParser(description="预测门控网格回测（7:1:2 切分的测试段）")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--runs-dir", default="forecast/runs")
    parser.add_argument("--t0-tick", type=int, default=None,
                        help="最早起步 tick，缺省取 lookback_ticks")
    parser.add_argument("--gate-thresholds", nargs=2, type=float,
                        metavar=("RESID_RATIO", "SLOPE_RATIO"),
                        default=(DEFAULT_RESIDUAL_RATIO_THRESHOLD,
                                 DEFAULT_SLOPE_RATIO_THRESHOLD),
                        help="门控阈值：残差宽比 / 斜率宽比")
    args = parser.parse_args()
    if args.t0_tick is not None and args.t0_tick < 0:
        parser.error("--t0-tick 须非负")
    symbols = sorted(args.symbols or list_symbols(args.data_dir))
    cfg = Config(data_dir=args.data_dir, cache_dir=args.cache_dir,
                 runs_dir=args.runs_dir, symbols=tuple(symbols))
    t0_tick = args.t0_tick if args.t0_tick is not None else cfg.window.lookback_ticks
    rules = gate_rules(*args.gate_thresholds)

    ensure_predictions(symbols, cfg)  # 预测缺失或过期时先重训重建

    records = {gate: {scheme: [] for scheme in SCHEMES} for gate in GATES}
    nets = {gate: {scheme: [] for scheme in SCHEMES} for gate in GATES}
    gated = {gate: {"total": 0, "oracle": 0, "prediction": 0} for gate in GATES}
    symbol_summaries = {}
    symbol_gated = {}
    for symbol in symbols:
        symbol_records, symbol_nets, counts = run_symbol(
            symbol, cfg, t0_tick, GATES, rules)
        for gate in GATES:
            for scheme in SCHEMES:
                records[gate][scheme] += symbol_records[gate][scheme]
                nets[gate][scheme] += symbol_nets[gate][scheme]
            for key in gated[gate]:
                gated[gate][key] += counts[gate][key]
            print(gate_line(f"[{symbol}]", gate, counts[gate]), flush=True)
        symbol_summaries[symbol] = {
            gate: {scheme: _summary(symbol_records[gate][scheme],
                                    symbol_nets[gate][scheme])
                   for scheme in SCHEMES}
            for gate in GATES}
        symbol_gated[symbol] = counts

    if not records[GATES[0]]["none"]:
        raise ValueError("测试段没有可回放的交易日（ATR 热身或无预测）")
    summaries = {}
    for gate in GATES:
        print(gate_line("全标的", gate, gated[gate]), flush=True)
        summaries[gate] = {scheme: _summary(records[gate][scheme], nets[gate][scheme])
                           for scheme in SCHEMES}

    out_dir = Path(cfg.runs_dir) / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as file:
        json.dump({
            "symbols": symbols,
            "t0_tick": t0_tick,
            "gate_thresholds": {"residual_ratio": args.gate_thresholds[0],
                                "slope_ratio": args.gate_thresholds[1]},
            "gated": gated,
            "summaries": summaries,
            "per_symbol": {"summaries": symbol_summaries, "gated": symbol_gated},
        }, file, ensure_ascii=False, indent=1)
    for path in figures.save_charts(summaries, symbol_summaries, gated, symbol_gated,
                                    str(out_dir)):
        print(f"backtest chart saved to {path}", flush=True)


if __name__ == "__main__":
    main()
