"""统一网格回测：测试段上 常开 / 模式门控（真值、预测）/ RL 智能体 的逐日对比。

四种模式共用同一指标口径（strategy/metrics.summarize）：以当日基准格距
W_d = max(0.1·ATR3, ε·前收) 归一的费用后网格收益 g（docs/grid_profit.md §七），
加日均闭环率与日均买卖笔数（各报逐日等权与按成交笔数加权两个口径，加权分母
剔除零交易日）以及相对网格宽幅（生效半宽 / 当日开盘价）。前三种模式由 strategy/engine.run_day
回放固定半宽 W_d 网格（每笔成交 1 手、省略 100 股因子；门控只在净持仓为 0 时按
stride_ticks 的决策节奏读取当拍信号），g = 费用后净利润 / W_d（费率见
strategy/costs.py）。agent 模式从 control/runs 检查点重建智能体，在测试日上贪心
回放 TradingEnv（与 control 评估同一路径），以同一 W_d 归一其相对底仓的超额净利；
检查点缺失时跳过该模式。

门控信号来自 forecast/regime：oracle 用事后模式标签（labels.pattern_labels），
prediction 用识别器概率过验证段率配平的阈值 τ；识别器缺失或过期时经
ensure_classifier 先重训。agent 状态依赖的预测缓存经 forecast/train.
ensure_predictions 预建。结果写 runs_dir（summary.json + 指标热力图 SVG）；
「全体」行为各标的指标的等权均值。
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
from forecast.regime.data import load_bank, replay_grid_day
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
    """一行门控触发占比：oracle / prediction 各占可判定 tick 数的比例。"""
    total = counts["total"]
    shares = ", ".join(
        f"{scheme} {100.0 * counts[scheme] / total:.2f}%" if total else f"{scheme} -"
        for scheme in ("oracle", "prediction"))
    return f"{prefix} 模式门控: {shares}（共 {total}）"


def run_engine_modes(bank, classifier, threshold, symbol_id, cfg: RegimeConfig,
                     t0_tick) -> tuple:
    """报价驱动回放一个标的的测试段，返回 (records, gated)。

    records[mode] 为逐日记录（metrics.summarize 的输入口径）；"open" 为常开
    baseline，"oracle" / "prediction" 分别用事后模式标签与识别概率 > τ 判定门控。
    """
    pattern = pattern_labels(bank, cfg)
    records = {mode: [] for mode in ENGINE_MODES}
    gated = {"total": 0, "oracle": 0, "prediction": 0}
    for i in bank.day_indices("test"):
        mid = bank.quotes[i][2]
        if len(mid) <= t0_tick:
            continue
        prob = day_prob(classifier, bank, i, symbol_id, cfg)
        with np.errstate(invalid="ignore"):
            masks = {"open": None,
                     "oracle": pattern[i] == 1,
                     "prediction": prob > threshold}
        judgeable = bank.judgeable[i]
        gated["total"] += int(judgeable.sum())
        for scheme in ("oracle", "prediction"):
            gated[scheme] += int((masks[scheme] & judgeable).sum())

        for mode in ENGINE_MODES:
            # 掩码逐 tick 对齐整日序列（缓存 R 行），回放内部自 t0 切片
            records[mode].append(replay_grid_day(bank, i, masks[mode], cfg, t0_tick))
    return records, gated


def run_agent_mode(symbol, args) -> tuple[list, str] | None:
    """从 control 检查点贪心回放该标的的测试日，返回 (逐日记录, 检查点名)。

    检查点缺失或不唯一时打印原因并返回 None（该标的的 agent 列留空）。
    """
    from control.trace import resolve_checkpoint
    try:
        path = resolve_checkpoint(symbol, args.method, args.seed, args.w, args.lam,
                                  args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[{symbol}] agent 模式跳过：{exc}", flush=True)
        return None

    from control.config import Config as ControlConfig
    from control.env import action_params
    from control.model import resolve_device
    from control.trace import greedy_policy, load_checkpoint, prepare_test_markets
    from control.train import replay_day

    device = resolve_device(ControlConfig())
    net, cfg, fixed_gears = load_checkpoint(path, device)
    _, test_markets = prepare_test_markets(symbol, cfg, args.data_dir, args.cache_dir)
    gears = greedy_policy(net, device, fixed_gears)

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
    return records, os.path.basename(path)


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
    parser.add_argument("--t0-tick", type=int, default=None,
                        help="回放起点 tick，缺省取 lookback_ticks（可预测起点，回看窗满）")
    parser.add_argument("--checkpoint", default=None,
                        help="agent：检查点路径（缺省按 control/runs 命名规则解析，仅单标的适用）")
    parser.add_argument("--method", default="GRID", help="agent：方法名（与 control.train 文件名一致）")
    parser.add_argument("--seed", type=int, default=0, help="agent：随机种子")
    parser.add_argument("--w", type=float, default=None,
                        help="agent：hindsight 权重 w（缺省取 control Config 默认值）")
    parser.add_argument("--lam", type=float, default=None,
                        help="agent：存货惩罚 λ（缺省取 control Config 默认值）")
    args = parser.parse_args()
    if args.t0_tick is not None and args.t0_tick < 0:
        parser.error("--t0-tick 须非负")
    symbols = sorted(args.symbols or list_symbols(args.data_dir))
    if args.checkpoint and len(symbols) > 1:
        parser.error("--checkpoint 只适用于单标的回测")
    cfg = RegimeConfig(data_dir=args.data_dir, cache_dir=args.cache_dir,
                       symbols=tuple(symbols))
    t0_tick = args.t0_tick if args.t0_tick is not None else cfg.window.lookback_ticks

    # agent 状态依赖预测缓存；识别器缺失或过期时先重训
    ensure_predictions(symbols, ForecastConfig(
        data_dir=args.data_dir, cache_dir=args.cache_dir, symbols=tuple(symbols)))
    classifier, threshold = ensure_classifier(symbols, cfg)

    gated = {"total": 0, "oracle": 0, "prediction": 0}
    symbol_summaries, symbol_gated, checkpoints = {}, {}, {}
    for symbol_id, symbol in enumerate(symbols):
        bank = load_bank(symbol, cfg)
        records, counts = run_engine_modes(bank, classifier, threshold, symbol_id,
                                           cfg, t0_tick)
        agent = run_agent_mode(symbol, args)
        if agent is not None:
            records["agent"], checkpoints[symbol] = agent
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
            "t0_tick": t0_tick,
            "probability_threshold": threshold,
            "checkpoints": checkpoints,
            "gated": gated,
            "summaries": summaries,
            "per_symbol": {"summaries": symbol_summaries, "gated": symbol_gated},
        }, file, ensure_ascii=False, indent=1)
    for path in figures.save_charts(summaries, symbol_summaries, gated, symbol_gated,
                                    str(out_dir)):
        print(f"backtest chart saved to {path}", flush=True)


if __name__ == "__main__":
    main()
