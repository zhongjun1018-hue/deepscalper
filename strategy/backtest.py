"""统一网格回测：测试段上 常开 / 模式门控（真值、预测）/ RL 智能体 的逐日对比。

四种模式共用同一指标口径（strategy/metrics.summarize）：以当日基准格距
W_d = max(0.1·ATR3, ε·前收) 归一的费用后网格收益 g（docs/grid_profit.md §七），
加日均闭环率与日均买卖笔数（均为总和除以该指标非零日的日数）、有成交日平均的
日终敞口 |N_b − N_s| 与满闭环率（闭环率为 1 的日占比）以及相对网格宽幅
（生效半宽 / 当日开盘价）。前三种模式由 strategy/engine.run_day 回放固定半宽 W_d
网格（每笔成交 1 手、省略 100 股因子；门控只在净持仓为 0 时按分钟锚点节奏读取
当拍信号，状态切换经 confirm_n 连续确认），g = 费用后净利润 / W_d（费率见
strategy/costs.py）。agent 模式从 control/runs 的统一训练检查点重建智能体，在各
标的测试日上按定长决策贪心回放 TradingEnv（与 control 评估同一路径），以同一 W_d
归一其相对底仓的超额净利；检查点缺失时跳过该模式。

门控信号来自 forecast/regime：oracle 用事后模式标签（labels.pattern_labels），
prediction 用识别器概率过验证段率配平的阈值 τ，逐分钟信号按锚点前向填充到 tick
（regime.data.expand_minutes）；识别器缺失或过期时经 ensure_classifier 先重训。
agent 状态依赖的预测缓存经 forecast/train.ensure_predictions 预建。结果写
runs_dir（summary.json + 指标热力图 SVG）；「全体」行为各标的指标的等权均值。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from data_provider.ticks import list_symbols
from forecast.regime.classify import day_prob
from forecast.regime.config import RegimeConfig
from forecast.regime.data import expand_minutes, load_bank, replay_grid_day
from forecast.regime.labels import pattern_labels
from forecast.regime.train import ensure_classifier
from forecast.config import Config as ForecastConfig
from forecast.train import ensure_predictions
from strategy import figures, metrics
from strategy.grid import half_width

# 模式即汇总表的列：前三种走报价驱动回放，agent 走 control 检查点的贪心回放
ENGINE_MODES = ("open", "oracle", "prediction")
MODES = (*ENGINE_MODES, "agent")


def gate_line(prefix, counts) -> str:
    """一行门控触发占比：oracle / prediction 各占可判定锚点数的比例。"""
    total = counts["total"]
    shares = ", ".join(
        f"{scheme} {100.0 * counts[scheme] / total:.2f}%" if total else f"{scheme} -"
        for scheme in ("oracle", "prediction"))
    return f"{prefix} 模式门控: {shares}（共 {total}）"


def run_engine_modes(bank, classifier, threshold, symbol_id, cfg: RegimeConfig,
                     start_minute) -> tuple:
    """报价驱动回放一个标的的测试段，返回 (records, gated)。

    records[mode] 为逐日记录（metrics.summarize 的输入口径）；"open" 为常开
    baseline，"oracle" / "prediction" 分别用事后模式标签与识别概率 > τ 判定门控，
    逐分钟信号按锚点前向填充到 tick 后交给 engine。
    """
    pattern = pattern_labels(bank, cfg)
    records = {mode: [] for mode in ENGINE_MODES}
    gated = {"total": 0, "oracle": 0, "prediction": 0}
    for i in bank.day_indices("test"):
        n_ticks = len(bank.quotes[i][2])
        prob = day_prob(classifier, bank, i, symbol_id)
        anchors = bank.anchors[i]
        with np.errstate(invalid="ignore"):
            oracle_min = pattern[i] == 1
            prediction_min = prob > threshold
            masks = {
                "open": None,
                "oracle": expand_minutes(oracle_min, anchors, n_ticks) > 0.5,
                "prediction": expand_minutes(prediction_min, anchors, n_ticks) > 0.5,
            }
        judgeable = bank.judgeable[i]
        gated["total"] += int(judgeable.sum())
        gated["oracle"] += int((oracle_min & judgeable).sum())
        gated["prediction"] += int((prediction_min & judgeable).sum())

        for mode in ENGINE_MODES:
            record = replay_grid_day(bank, i, masks[mode], cfg, start_minute)
            if record:
                records[mode].append(record)
    return records, gated


def load_agent(args) -> tuple | None:
    """加载 control 统一训练检查点，返回 (net, cfg, stats, device, 检查点文件名)。

    检查点缺失或不唯一时打印原因并返回 None（agent 列整体留空）。
    """
    from control.trace import load_checkpoint, resolve_checkpoint
    try:
        path = resolve_checkpoint(args.method, args.seed, args.w, args.lam,
                                  args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        print(f"agent 模式跳过：{exc}", flush=True)
        return None

    from control.config import Config as ControlConfig
    from control.model import resolve_device

    device = resolve_device(ControlConfig())
    net, cfg, stats = load_checkpoint(path, device)
    return net, cfg, stats, device, os.path.basename(path)


def run_agent_mode(symbol, agent, args) -> list:
    """按统一检查点贪心回放该标的的测试日，返回逐日记录。"""
    from control.env import action_params
    from control.trace import greedy_policy, prepare_test_markets
    from control.train import replay_day

    net, cfg, stats, device, _ = agent
    test_markets = prepare_test_markets(symbol, cfg, stats,
                                        args.data_dir, args.cache_dir)
    gears = greedy_policy(net, device)

    def policy(obs):
        return action_params(cfg, gears(obs))

    records = []
    for market in test_markets:
        ret, log = replay_day(market, policy)
        # 相对底仓的超额净利以当日基准格距归一：g 与引擎模式同分母
        base_width = half_width(cfg.window.atr_mult, market.atr, market.pre_close,
                                cfg.window.min_width_ratio)
        records.append({"g": ret * market.base_value / base_width,
                        "n_buys": log["n_buys"], "n_sells": log["n_sells"],
                        "closure_rate": log["closure_rate"],
                        "width_rel": log["width_rel"]})
    return records


def pool_summaries(symbol_summaries: dict, mode: str) -> dict | None:
    """「全体」行：各标的该模式指标的等权均值（n_days 取总和，NaN 按有值标的计）；
    无标的有数据记 None。"""
    rows = [entry[mode] for entry in symbol_summaries.values()
            if entry.get(mode) is not None]
    if not rows:
        return None

    def pool(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    pooled = {key: pool(key) for key in rows[0]}
    pooled["n_days"] = int(sum(row["n_days"] for row in rows))
    return pooled


def main():
    parser = argparse.ArgumentParser(
        description="统一网格回测（测试段）：常开 / 模式门控 / RL 智能体对比")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--runs-dir", default="strategy/runs")
    parser.add_argument("--start-minute", type=int, default=None,
                        help="回放起点分钟，缺省取 lookback_min−1（可预测起点，回看窗满）")
    parser.add_argument("--checkpoint", default=None,
                        help="agent：检查点路径（缺省按 control/runs 命名规则解析）")
    parser.add_argument("--method", default="GRID", help="agent：方法名（与 control.train 文件名一致）")
    parser.add_argument("--seed", type=int, default=0, help="agent：随机种子")
    parser.add_argument("--w", type=float, default=None,
                        help="agent：hindsight 权重 w（缺省取 control Config 默认值）")
    parser.add_argument("--lam", type=float, default=None,
                        help="agent：存货惩罚 λ（缺省取 control Config 默认值）")
    args = parser.parse_args()
    if args.start_minute is not None and args.start_minute < 0:
        parser.error("--start-minute 须非负")
    symbols = sorted(args.symbols or list_symbols(args.data_dir))
    # 识别器与预测缓存身份取 data 目录全部标的（symbol_id 与训练映射一致，避免子集
    # 回测覆写共享产物），--symbols 只选回测子集
    model_symbols = sorted(list_symbols(args.data_dir))
    cfg = RegimeConfig(data_dir=args.data_dir, cache_dir=args.cache_dir,
                       symbols=tuple(model_symbols))
    start_minute = (args.start_minute if args.start_minute is not None
                    else cfg.window.lookback_min - 1)

    # agent 状态依赖预测缓存；识别器缺失或过期时先重训
    ensure_predictions(model_symbols, ForecastConfig(
        data_dir=args.data_dir, cache_dir=args.cache_dir,
        symbols=tuple(model_symbols)))
    classifier, threshold = ensure_classifier(model_symbols, cfg)
    agent = load_agent(args)

    gated = {"total": 0, "oracle": 0, "prediction": 0}
    symbol_summaries, symbol_gated = {}, {}
    for symbol in symbols:
        bank = load_bank(symbol, cfg)
        records, counts = run_engine_modes(bank, classifier, threshold,
                                           model_symbols.index(symbol),
                                           cfg, start_minute)
        if agent is not None:
            records["agent"] = run_agent_mode(symbol, agent, args)
        symbol_summaries[symbol] = {
            mode: metrics.summarize(records[mode]) if records.get(mode) else None
            for mode in MODES}
        symbol_gated[symbol] = counts
        for key in gated:
            gated[key] += counts[key]
        print(gate_line(f"[{symbol}]", counts), flush=True)

    if all(entry["open"] is None for entry in symbol_summaries.values()):
        raise ValueError("测试段没有可回放的交易日（ATR 热身或标签不可判定）")
    print(gate_line("全标的", gated), flush=True)
    summaries = {mode: pool_summaries(symbol_summaries, mode) for mode in MODES}

    out_dir = Path(args.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as file:
        json.dump({
            "symbols": symbols,
            "start_minute": start_minute,
            "probability_threshold": threshold,
            "checkpoint": agent[-1] if agent is not None else None,
            "gated": gated,
            "summaries": summaries,
            "per_symbol": {"summaries": symbol_summaries, "gated": symbol_gated},
        }, file, ensure_ascii=False, indent=1)
    for path in figures.save_charts(summaries, symbol_summaries, gated, symbol_gated,
                                    str(out_dir)):
        print(f"backtest chart saved to {path}", flush=True)


if __name__ == "__main__":
    main()
