"""webviz 数据导出：forecast（模式门控网格）与 control（强化学习）在同一页面的逐日决策回放。

导出布局（index.html 通过 fetch 读取，需 `python -m http.server` 伺服）：
  webviz/data/forecast/<symbol>/<date>.json   逐日价格曲线、滑动窗口统计、常开与识别门控网格回放
  webviz/data/control/<symbol>/<date>.json    贪心决策点（逐段生效网格）、成交标记与单日摘要
  webviz/data/index.json                        两侧可用的 (symbol, date) 目录与展示参数

页面以 forecast 目录为主索引，同日的 control 数据充当单日对比栏的第三张卡片
（强化学习决策）及其回放路径。forecast 数据源：报价 data_provider.ticks.load_days
（另用 path_stats 复算展示统计）；窗口统计、半宽与切分统一取自 forecast.regime.data
的 bank，窗口与门控节奏均为分钟锚点；门控为识别概率 > τ（prediction，逐分钟信号
按锚点前向填充到 tick），识别器缺失或过期时经 ensure_classifier 先重训；网格事件
strategy.engine.run_day(trace=True)（锚点门控 + 连续确认，只在净持仓为 0 时生效）。
网格回放只覆盖测试段（样本外），其余日期仅导出价格与窗口。

control 数据源：control.trace.load_checkpoint 重建统一训练的网络、Config 与逐标的
标准化统计量，greedy_policy 构建贪心策略，prepare_test_markets 构建测试段市场；逐测试日 control.trace.trace_day 贪心回放（定长决策，区间内网格随成交逐段
重建），单日摘要与门控卡片同指标口径（g = 超额收益 × B / W_d，与 strategy.backtest 的 agent 模式一致）。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from data_provider.ticks import list_symbols, load_days, minute_index, minute_labels
from data_provider.windows import path_stats
from forecast.regime.classify import day_prob
from forecast.regime.config import RegimeConfig
from forecast.regime.data import expand_minutes, load_bank
from forecast.regime.train import ensure_classifier
from strategy import costs, engine, metrics
from strategy.grid import half_width

DISPLAY_PATH_NAMES = ["rv", "path_len", "range_rel", "resid_abs_mean", "resid_abs_q90",
                      "abs_slope", "er", "rev_rate"]

# 标的代码 → 证券简称（行情源无名称字段，展示层按代码查表，未知标的回退代码）
SYMBOL_NAMES = {
    "000096": "广聚能源", "000560": "我爱我家", "000566": "海南海药",
    "002111": "威海广泰", "002134": "天津普林", "002370": "亚太药业",
    "002387": "维信诺", "002673": "西部证券", "300497": "富祥股份",
    "300765": "石药创新", "301308": "江波龙", "600571": "信雅达",
    "600712": "南宁百货", "600835": "上海机电", "600847": "万里股份",
    "600996": "贵广网络", "601288": "农业银行", "603897": "长城科技",
    "688030": "山石网科", "688061": "灿瑞科技", "688592": "司南导航",
    "688772": "珠海冠宇",
}


def symbol_entry(symbol: str, dates: list, replay_dates: list, **extra) -> dict:
    """index.json 的标的条目：名称查 SYMBOL_NAMES，未知标的回退代码。"""
    return {"symbol": symbol, "name": SYMBOL_NAMES.get(symbol, symbol),
            "dates": dates, "replay_dates": replay_dates, **extra}


def day_table(day) -> dict:
    """单日连续竞价快照的展示数组：minute / x（分钟+秒小数）/ bid1 / ask1 / mid / lastpx。"""
    frame = day.frame
    minute = minute_index(frame["MDTime"]).astype(np.int64)
    text = frame["MDTime"].astype(str).str.zfill(9)
    second = text.str[4:6].astype(np.int64) + text.str[6:9].astype(np.int64) / 1000.0
    bid1 = frame["Buy1Price"].to_numpy(np.float64)
    ask1 = frame["Sell1Price"].to_numpy(np.float64)
    return {
        "minute": minute,
        "x": minute + second.to_numpy() / 60.0,
        "bid1": bid1,
        "ask1": ask1,
        "mid": 0.5 * (bid1 + ask1),
        "lastpx": frame["LastPx"].to_numpy(np.float64),
        "preclose": day.pre_close,
    }


def path_slope(path) -> float:
    """带符号的 OLS 斜率（每快照）；path_stats 的 abs_slope 只保留绝对值。"""
    position = np.arange(len(path), dtype=np.float64)
    position -= position.mean()
    return float((path - path.mean()) @ position / (position ** 2).sum())


def finite(value, digits):
    """有限值按位舍入；非有限落为 None（JSON 不允许 NaN）。"""
    return round(float(value), digits) if np.isfinite(value) else None


def build_windows(table: dict, prob: np.ndarray, width: float | None,
                  anchors: np.ndarray, cfg: RegimeConfig) -> list:
    """滑动窗口展示负载：趋势拟合、网格上下界、前瞻路径的实测统计与识别概率。

    按分钟锚点取样：分钟 m 锚点的前瞻路径覆盖未来 pred_min 分钟，第 m 拍概率
    覆盖同一区间。
    """
    log_mid = np.log(table["mid"])
    minute = table["minute"]
    total = len(anchors)
    windows = []
    for m in range(cfg.window.lookback_min - 1, total - cfg.window.pred_min):
        anchor = anchors[m]
        end_candidates = anchors[m + 1: m + cfg.window.pred_min + 1]
        end_candidates = end_candidates[end_candidates >= 0]
        if anchor < 0 or not len(end_candidates):
            continue
        end = int(end_candidates[-1])
        window = {"start": int(minute[anchor]), "end": int(minute[end]),
                  "minutes": int(minute[end] - minute[anchor])}
        path = log_mid[anchor:end + 1]
        if np.isfinite(path).all():
            # 前瞻路径以行末快照（锚点）开盘，网格也以该价格为中心
            open_price = float(np.exp(path[0]))
            half = path_slope(path) * (len(path) - 1) / 2.0
            fit_start = float(np.exp(path.mean() - half))
            fit_end = float(np.exp(path.mean() + half))
            window.update({
                "fit_start": round(fit_start, 4),
                "fit_end": round(fit_end, 4),
                "slope_price": round((fit_end - fit_start) / max(window["minutes"], 1), 6),
                "open": round(open_price, 4),
                "close": round(float(np.exp(path[-1])), 4),
            })
            if width is not None:
                window["upper"] = round(open_price + width, 4)
                window["lower"] = round(open_price - width, 4)
            stats = path_stats(path)
            window["features"] = {name: finite(stats[name][0], 8)
                                  for name in DISPLAY_PATH_NAMES}
        if np.isfinite(prob[m]):
            window["prediction"] = {"probability": round(float(prob[m]), 4)}
        windows.append(window)
    return windows


def excluded_runs(mask, table: dict, t0: int) -> list:
    """门控排除的连续 tick 段，换算为 x 轴（压缩分钟）区间列表；t0 前的段裁掉。"""
    if mask is None:
        return []
    x = table["x"]
    values = np.asarray(mask, dtype=bool)[:len(x)]
    segments = []
    tick = 0
    while tick < len(values):
        if not values[tick]:
            tick += 1
            continue
        start = tick
        while tick < len(values) and values[tick]:
            tick += 1
        if tick - 1 < t0:
            continue
        segments.append([round(float(x[max(start, t0)]), 3), round(float(x[tick - 1]), 3)])
    return segments


def build_grid(table: dict, width: float, t0: int, hard_exclude,
               anchors: np.ndarray, confirm_n: int) -> dict | None:
    """自 t0 tick 起回放固定半宽网格：engine 事件流 + 形态指标 + 门控覆盖。"""
    n = len(table["mid"])
    if t0 >= n:
        return None
    take = np.arange(t0, n)   # 门控掩码逐 tick 对齐，随报价序列一并切片
    anchors = anchors[anchors >= t0] - t0
    result = engine.run_day(table["bid1"][take], table["ask1"][take], table["mid"][take],
                            hard_exclude=None if hard_exclude is None else hard_exclude[take],
                            width=width, anchors=anchors, confirm_n=confirm_n, trace=True)
    x = table["x"][take]
    minute = table["minute"][take]
    events = []
    for event in result["events"]:
        item = {key: event[key]
                for key in ("kind", "center", "upper", "lower", "exposure")}
        item["minute"] = int(minute[event["index"]])
        item["x"] = round(float(x[event["index"]]), 3)
        if "fill" in event:
            item["fill"] = round(float(event["fill"]), 4)
        for key in ("center", "upper", "lower"):
            item[key] = round(float(item[key]), 4)
        events.append(item)
    trades = result["buys"] + result["sells"]
    excluded = excluded_runs(hard_exclude, table, t0)
    # 卡片指标与统一回测同口径（regime.data.replay_grid_day）：g = 费用后净利润 / W_d
    g = costs.daily_net(costs.fills_from_events(result["events"]),
                        float(table["mid"][-1])) / width
    return {
        "width": round(float(width), 4),
        "buys": int(result["buys"]),
        "sells": int(result["sells"]),
        "trades": int(trades),
        "score": (round(metrics.closure_rate(result["buys"], result["sells"]), 6)
                  if trades else None),
        "grid_profit": round(g, 6),
        "profit_per_trade": round(g / trades, 6) if trades else None,
        "gated_minutes": int(round(sum(end - start for start, end in excluded))),
        "evaluated_minutes": int(round(float(table["x"][-1] - table["x"][t0]))),
        "excluded": excluded,
        "events": events,
    }


def write_json(path: str, payload: dict) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, separators=(",", ":"))


def export_forecast_symbol(symbol: str, args, cfg: RegimeConfig, classifier,
                           threshold: float, symbol_id: int) -> dict:
    """导出一个标的的 forecast 算法回放数据，返回 index.json 的符号条目。"""
    bank = load_bank(symbol, cfg)
    days = {d.date: d for d in load_days(symbol, data_dir=args.data_dir,
                                         atr_days=cfg.window.atr_window)}
    tables = {date: day_table(days[date]) for date in bank.dates}
    del days

    # 网格回放只在测试段（样本外）且标签可判定的交易日上构建
    candidate_set = set(bank.day_indices("test"))
    out_dir = os.path.join(args.out_dir, "forecast", symbol)
    os.makedirs(out_dir, exist_ok=True)
    replay_dates = []
    for i, date in enumerate(bank.dates):
        table = tables[date]
        anchors = bank.anchors[i]
        width = float(bank.width[i])
        if not np.isfinite(width):
            width = None
        prob = day_prob(classifier, bank, i, symbol_id)
        payload = {
            "symbol": symbol,
            "date": str(date),
            "t0": -1,
            "width": round(width, 4) if width is not None else None,
            "preclose": round(float(table["preclose"]), 4),
            "x": [round(float(v), 3) for v in table["x"]],
            "price": [round(float(v), 4) for v in table["lastpx"]],
            "windows": build_windows(table, prob, width, anchors, cfg),
        }
        if i in candidate_set:
            # 统一自可预测起点（回看窗满首分钟的锚点）起回放
            t0 = bank.replay_start(i, cfg.window.lookback_min - 1)
            with np.errstate(invalid="ignore"):
                mask = expand_minutes(prob > threshold, anchors,
                                      len(table["mid"])) > 0.5
            grids = {"none": build_grid(table, width, t0, None, anchors,
                                        cfg.confirm_n),
                     "prediction": build_grid(table, width, t0, mask, anchors,
                                              cfg.confirm_n)}
            if grids["none"] is not None:
                payload["t0"] = int(table["minute"][t0])
                payload["grids"] = grids
                replay_dates.append(str(date))
        write_json(os.path.join(out_dir, f"{date}.json"), payload)
    return symbol_entry(symbol, [str(date) for date in bank.dates], replay_dates)


def day_summary(result: dict, market, cfg, table: dict) -> dict:
    """trace_day 结果的单日摘要，与 build_grid 的卡片指标同构（单日对比栏的 RL 卡片）。

    g 与 strategy.backtest 的 agent 模式同口径（超额收益 × B / W_d）；「平均半宽」为
    时间加权生效半宽；gated_minutes 记网格停用时长（平仓档 0 与关闭档，对应门控卡片
    的命中时长）。
    """
    log = result["log"]
    base_width = half_width(cfg.window.atr_mult, market.atr, market.pre_close,
                            cfg.window.min_width_ratio)
    g = result["ret"] * market.base_value / base_width
    trades = log["n_fills"]
    exposure = sum(f["qty"] if f["side"] == "buy" else -f["qty"]
                   for f in result["fills"] if f["kind"] != "liquidate")
    idle_share = sum(share for gear, share in zip(cfg.widths, log["width_time"])
                     if gear in (0.0, cfg.widths[-1]))
    evaluated = float(table["x"][-1] - table["x"][market.start])
    return {
        "width": finite(log["width_rel"] * market.open_px, 4),
        "buys": log["n_buys"],
        "sells": log["n_sells"],
        "trades": trades,
        "score": round(log["closure_rate"], 6) if trades else None,
        "exposure": round(float(exposure), 2),
        "grid_profit": round(float(g), 6),
        "profit_per_trade": round(float(g) / trades, 6) if trades else None,
        "gated_minutes": int(round(idle_share * evaluated)),
        "evaluated_minutes": int(round(evaluated)),
    }


def export_control_symbol(symbol: str, checkpoint: tuple, args) -> dict:
    """按统一训练检查点回放该标的测试段的贪心决策轨迹并导出，返回 index.json 的符号条目。"""
    from data_provider.ticks import load_days as load_symbol_days
    from control.trace import greedy_policy, prepare_test_markets, trace_day

    net, cfg, stats, device, path = checkpoint
    test_m = prepare_test_markets(symbol, cfg, stats, args.data_dir, args.cache_dir)
    day_lookup = {d.date: d for d in load_symbol_days(symbol, data_dir=args.data_dir,
                                                      atr_days=cfg.window.atr_window)}
    tables = {m.date: day_table(day_lookup[m.date]) for m in test_m}
    del day_lookup
    policy = greedy_policy(net, device)

    out_dir = os.path.join(args.out_dir, "control", symbol)
    os.makedirs(out_dir, exist_ok=True)
    exported = []
    for market in test_m:
        table = tables[market.date]
        result = trace_day(market, policy)
        decisions = [{
            "t": d["t"],
            "x": round(float(table["x"][d["t"]]), 3),
            "minute": int(table["minute"][d["t"]]),
            "width": float(d["width"]),
            "size": int(d["size"]),
            "grids": [{
                "t": g["t"],
                "x": round(float(table["x"][g["t"]]), 3),
                "center": round(float(g["center"]), 4),
                "upper": round(float(g["upper"]), 4),
                "lower": round(float(g["lower"]), 4),
            } for g in d["grids"]],
        } for d in result["decisions"]]
        fills = [{
            "t": f["t"],
            "x": round(float(table["x"][f["t"]]), 3),
            "side": f["side"],
            "price": round(float(f["price"]), 4),
            "qty": float(f["qty"]),
            "kind": f["kind"],
        } for f in result["fills"]]
        write_json(os.path.join(out_dir, f"{market.date}.json"), {
            "symbol": symbol,
            "date": market.date,
            "preclose": round(float(market.pre_close), 4),
            "x": [round(float(v), 3) for v in table["x"]],
            "price": [round(float(v), 4) for v in market.mid],
            "decisions": decisions,
            "fills": fills,
            "grid": day_summary(result, market, cfg, table),
            # width_rel 在全天不可触发时为 NaN，逐键过 finite 保证 JSON 可被前端解析
            "log": {key: finite(value, 6) if isinstance(value, float) else value
                    for key, value in result["log"].items()},
        })
        exported.append(market.date)
    return symbol_entry(symbol, exported, exported,
                        checkpoint=os.path.basename(path))


def load_control_checkpoint(args, parser) -> tuple:
    """加载统一训练检查点，返回 (net, cfg, stats, device, path)。"""
    from control.config import Config as ControlConfig
    from control.model import resolve_device
    from control.trace import load_checkpoint, resolve_checkpoint

    try:
        path = resolve_checkpoint(args.method, args.seed, args.w, args.lam,
                                  args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    device = resolve_device(ControlConfig())
    net, cfg, stats = load_checkpoint(path, device)
    return net, cfg, stats, device, path


def forecast_params(cfg: RegimeConfig, threshold: float) -> dict:
    """index.json 的展示参数（forecast 语义：分钟口径窗口参数、模式与门控参数）。"""
    return {
        "model": "LightGBM",
        "lookback_min": cfg.window.lookback_min,
        "pred_min": cfg.window.pred_min,
        "confirm_n": cfg.confirm_n,
        "atr_mult": cfg.window.atr_mult,
        "atr_window": cfg.window.atr_window,
        "min_width_ratio": cfg.window.min_width_ratio,
        "pattern": {
            "residual_ratio_threshold": cfg.residual_ratio_threshold,
            "slope_ratio_threshold": cfg.slope_ratio_threshold,
            "sticky_stay": cfg.sticky_stay,
            "emission_noise": cfg.emission_noise,
            "probability_threshold": round(threshold, 4),
        },
    }


def update_index(out_dir: str, algorithm: str, entries: list, params: dict | None = None) -> None:
    """合并写 index.json：本次导出的算法段落整体替换，另一算法的既有目录与 params 保留。"""
    path = os.path.join(out_dir, "index.json")
    algorithms = {}
    if os.path.exists(path):
        with open(path) as file:
            stored = json.load(file)
        algorithms = stored.get("algorithms", {})
        params = params if params is not None else stored.get("params")
    algorithms[algorithm] = {"symbols": entries}
    index = {"algorithms": algorithms,
             "minute_labels": [str(label) for label in minute_labels()]}
    if params is not None:
        index["params"] = params
    write_json(path, index)


def main() -> None:
    parser = argparse.ArgumentParser(description="webviz 数据导出（forecast / control 决策回放）")
    parser.add_argument("--algorithm", choices=["forecast", "control"], required=True)
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--checkpoint", default=None,
                        help="control：检查点路径（缺省按 control/runs 命名规则解析）")
    parser.add_argument("--method", default="GRID", help="control：方法名（与 control.train 文件名一致）")
    parser.add_argument("--seed", type=int, default=0, help="control：随机种子")
    parser.add_argument("--w", type=float, default=None,
                        help="control：hindsight 权重 w（缺省取 control Config 默认值）")
    parser.add_argument("--lam", type=float, default=None,
                        help="control：存货惩罚 λ（缺省取 control Config 默认值）")
    args = parser.parse_args()

    symbols = sorted(args.symbols or list_symbols(args.data_dir))
    if args.algorithm == "forecast":
        # 识别器身份取 data 目录全部标的（symbol_id 与训练映射一致），--symbols 只选导出子集
        model_symbols = sorted(list_symbols(args.data_dir))
        cfg = RegimeConfig(data_dir=args.data_dir, cache_dir=args.cache_dir,
                           symbols=tuple(model_symbols))
        classifier, threshold = ensure_classifier(model_symbols, cfg)
        entries = [export_forecast_symbol(symbol, args, cfg, classifier, threshold,
                                          model_symbols.index(symbol))
                   for symbol in symbols]
        update_index(args.out_dir, "forecast", entries,
                     forecast_params(cfg, threshold))
    else:
        checkpoint = load_control_checkpoint(args, parser)
        entries = [export_control_symbol(symbol, checkpoint, args)
                   for symbol in symbols]
        update_index(args.out_dir, "control", entries)
    for entry in entries:
        print(f"{args.algorithm} {entry['symbol']}: {len(entry['dates'])} 个交易日"
              f"（可回放 {len(entry['replay_dates'])}）", flush=True)


if __name__ == "__main__":
    main()
