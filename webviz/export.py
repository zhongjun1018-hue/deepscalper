"""webviz 数据导出：forecast（预测门控网格）与 control（强化学习）两个算法的逐日决策回放。

导出布局（index.html 通过 fetch 读取，需 `python -m http.server` 伺服）：
  webviz/data/forecast/<symbol>/<date>.json   逐日价格曲线、滑动窗口统计、各 门控×scheme 网格回放
  webviz/data/control/<symbol>/<date>.json    逐日 mid 曲线、贪心决策点（生效网格）与成交标记
  webviz/data/index.json                        两算法可用的 (symbol, date) 目录与展示参数

forecast 数据源：报价 data_provider.ticks.load_days；窗口统计、目标、预测与逐日半宽
统一取自 data_provider.windows.load_cache（另用 path_stats 复算展示统计）；门控掩码
forecast.signals.build_gate_masks；网格事件 strategy.engine.run_day(trace=True)
（门控只在净持仓为 0 时生效）。网格回放只覆盖 7:1:2 切分的测试段（样本外），其余日期
仅导出价格与窗口。

control 数据源：control.trace.load_checkpoint 重建网络与 Config；市场按 control/train.py build_markets
构建、control.features.fit_feature_stats 仅在训练段拟合（切分 data_provider.split.chronological_split）；
逐测试日 control.trace.trace_day 贪心回放。
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from data_provider.split import chronological_split
from data_provider.ticks import load_days, minute_index, minute_labels
from data_provider.windows import TARGET_NAMES, load_cache, path_stats
from forecast.config import Config as ForecastConfig
from forecast.signals import GATES, GATE_RULES, SCHEMES, build_gate_masks
from strategy import engine

DISPLAY_PATH_NAMES = ["rv", "path_len", "range_rel", "resid_abs_mean", "resid_abs_q90",
                      "abs_slope", "er", "rev_rate"]
SLOPE_TARGET_INDEX = TARGET_NAMES.index("abs_slope")


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


def tick_anchors(mid: np.ndarray, n_ticks: int) -> np.ndarray:
    """逐 tick 锚点 mid（门控相对宽度 w = 半宽/锚点 的分母）；本日之外的尾部记 NaN。"""
    anchors = np.full(n_ticks, np.nan)
    anchors[:len(mid)] = mid
    return anchors


def path_slope(path) -> float:
    """带符号的 OLS 斜率（每快照）；path_stats 的 abs_slope 只保留绝对值。"""
    position = np.arange(len(path), dtype=np.float64)
    position -= position.mean()
    return float((path - path.mean()) @ position / (position ** 2).sum())


def finite(value, digits):
    """有限值按位舍入；非有限落为 None（JSON 不允许 NaN）。"""
    return round(float(value), digits) if np.isfinite(value) else None


def build_windows(table: dict, preds: np.ndarray, width: float | None,
                  cfg: ForecastConfig) -> list:
    """滑动窗口展示负载：趋势拟合、网格上下界、前瞻路径的实测统计与预测趋势位移。

    按门控的决策间隔取样：锚点 tick 的前瞻路径为 (t, t+H]，第 t 行预测覆盖同一区间。
    """
    log_mid = np.log(table["mid"])
    minute = table["minute"]
    n = len(log_mid)
    windows = []
    for anchor in range(cfg.window.lookback_ticks - 1, n, cfg.stride_ticks):
        end = anchor + cfg.window.pred_ticks
        if end > n - 1:
            break
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
        if np.isfinite(preds[anchor, SLOPE_TARGET_INDEX]):
            window["prediction"] = {
                "abs_slope": round(float(preds[anchor, SLOPE_TARGET_INDEX]), 8),
            }
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
               decide_interval: int) -> dict | None:
    """自 t0 tick 起回放固定半宽网格：engine 事件流 + 形态指标 + 门控覆盖。"""
    n = len(table["mid"])
    if t0 >= n:
        return None
    take = np.arange(t0, n)   # 门控掩码逐 tick 对齐，随报价序列一并切片
    result = engine.run_day(table["bid1"][take], table["ask1"][take], table["mid"][take],
                            hard_exclude=None if hard_exclude is None else hard_exclude[take],
                            width=width, decide_interval=decide_interval, trace=True)
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
    return {
        "width": round(float(width), 4),
        "buys": int(result["buys"]),
        "sells": int(result["sells"]),
        "trades": int(trades),
        "score": (round(float(2 * min(result["buys"], result["sells"]) / trades), 6)
                  if trades else None),
        "grid_profit": round(result["grid_profit"], 6),
        "grid_profit_lower": round(result["grid_profit_lower"], 6),
        "profit_per_trade": round(result["grid_profit"] / trades, 6) if trades else None,
        "gated_minutes": int(round(sum(end - start for start, end in excluded))),
        "evaluated_minutes": int(round(float(table["x"][-1] - table["x"][t0]))),
        "excluded": excluded,
        "events": events,
    }


def write_json(path: str, payload: dict) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, separators=(",", ":"))


def export_forecast_symbol(symbol: str, args, cfg: ForecastConfig) -> dict:
    """导出一个标的的 forecast 算法回放数据，返回 index.json 的符号条目。"""
    cache = load_cache(symbol, data_dir=args.data_dir, cache_dir=args.cache_dir,
                       spec=cfg.window, zero_nan=False)
    dates = cache["dates"]

    days = {d.date: d for d in load_days(symbol, data_dir=args.data_dir)}
    tables = {date: day_table(days[date]) for date in dates}
    del days

    # 门控掩码只在测试段（样本外）且半宽有效的交易日上构建
    test = set(chronological_split(list(dates)).test)
    candidates = [i for i, date in enumerate(dates)
                  if date in test and np.isfinite(cache["width"][i])]
    n_ticks = cache["targets"].shape[1]
    masks, _ = build_gate_masks(
        {"oracle": cache["targets"][candidates],
         "prediction": cache["preds"][candidates]},
        dates[candidates], cache["width"][candidates],
        np.stack([tick_anchors(tables[dates[i]]["mid"], n_ticks) for i in candidates]))
    candidate_set = set(candidates)

    out_dir = os.path.join(args.out_dir, "forecast", symbol)
    os.makedirs(out_dir, exist_ok=True)
    replay_dates = []
    for i, date in enumerate(dates):
        table = tables[date]
        width = float(cache["width"][i])
        if not np.isfinite(width):
            width = None
        payload = {
            "symbol": symbol,
            "date": str(date),
            "t0": -1,
            "width": round(width, 4) if width is not None else None,
            "preclose": round(float(table["preclose"]), 4),
            "x": [round(float(v), 3) for v in table["x"]],
            "price": [round(float(v), 4) for v in table["lastpx"]],
            "closing_x": [],
            "closing_price": [],
            "windows": build_windows(table, cache["preds"][i], width, cfg),
        }
        if i in candidate_set and width is not None:
            # 当日首个有预测的样本行（oracle 真值与预测值同一行集），回放自其次一 tick 起
            usable = (np.isfinite(cache["targets"][i]).all(axis=1)
                      & np.isfinite(cache["preds"][i]).any(axis=1))
            if usable.any():
                t0 = int(np.flatnonzero(usable).min()) + 1
                grids = {"none": build_grid(table, width, t0, None, cfg.stride_ticks)}
                for gate in GATES:
                    grids[gate] = {
                        scheme: build_grid(table, width, t0,
                                           masks.get((gate, scheme, date)),
                                           cfg.stride_ticks)
                        for scheme in SCHEMES[1:]
                    }
                if grids["none"] is not None:
                    payload["t0"] = int(table["minute"][t0])
                    payload["grids"] = grids
                    replay_dates.append(str(date))
        write_json(os.path.join(out_dir, f"{date}.json"), payload)
    return {"symbol": symbol, "name": symbol,
            "dates": [str(date) for date in dates], "replay_dates": replay_dates}


def resolve_checkpoint(symbol: str, args, parser) -> str:
    """控制器检查点路径：--checkpoint 显式给定，否则按 control/runs 的结果命名规则解析。

    完整名为 <method>_w<w>_lam<λ>_seed<s>.pt（w/λ 缺省取 control Config 默认值）；run_all 的
    命名规则是不适用的超参不在文件名中（如 GRID-NH 无 w 标签），完整名未命中时退到
    <method>*_seed<s>.pt 的唯一匹配。
    """
    if args.checkpoint:
        path = args.checkpoint
    else:
        from control.config import Config as ControlConfig
        defaults = ControlConfig()
        w = defaults.hindsight_weight if args.w is None else args.w
        lam = defaults.inventory_lambda if args.lam is None else args.lam
        folder = os.path.join(defaults.result_dir, symbol)
        path = os.path.join(folder, f"{args.method}_w{w:g}_lam{lam:g}_seed{args.seed}.pt")
        if not os.path.exists(path):
            matches = sorted(glob.glob(os.path.join(
                folder, f"{args.method}_*_seed{args.seed}.pt")))
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                parser.error(f"{symbol} 的 {args.method} 检查点不唯一："
                             + "、".join(os.path.basename(m) for m in matches)
                             + "；请用 --w/--lam 或 --checkpoint 明确指定。")
    if not os.path.exists(path):
        parser.error(f"未找到 {symbol} 的 RL 检查点：{path}。"
                     "请先运行 scripts/run_all.py 完成训练（断点续跑会复用已有产物），"
                     "或用 --checkpoint 显式指定检查点路径。")
    return path


def export_control_symbol(symbol: str, args, parser) -> dict:
    """从控制器检查点回放测试段的贪心决策轨迹并导出，返回 index.json 的符号条目。"""
    import torch

    from control.config import Config as ControlConfig
    from control.features import fit_feature_stats
    from control.model import resolve_device, to_batch
    from control.trace import load_checkpoint, trace_day
    from control.train import build_markets

    path = resolve_checkpoint(symbol, args, parser)
    device = resolve_device(ControlConfig())
    net, cfg = load_checkpoint(path, device)

    days = load_days(symbol, args.data_dir, cfg.window.atr_window)
    split = chronological_split([d.date for d in days])
    train_days = [d for d in days if d.date in set(split.train)]
    test_days = [d for d in days if d.date in set(split.test)]
    cache = load_cache(symbol, data_dir=args.data_dir, cache_dir=args.cache_dir,
                       spec=cfg.window, zero_nan=True)
    # symbol_id 与 forecast 同口径：排序后标的集合中的索引
    symbol_id = sorted(cfg.symbols).index(symbol) if symbol in cfg.symbols else 0
    train_m = build_markets(train_days, cfg, cache, symbol_id)
    test_m = build_markets(test_days, cfg, cache, symbol_id)
    # 仅用训练集拟合标准化统计量，测试段复用（无前视泄漏，与 run_all 同一口径）
    stats = fit_feature_stats(train_m, cfg) if cfg.normalize else None
    for market in train_m + test_m:
        market.set_stats(stats)
    tables = {d.date: day_table(d) for d in test_days}
    del days, train_days, train_m

    def policy(obs):
        with torch.no_grad():
            q = net(*to_batch([obs], device))
        return int(q[0].argmax(-1).item()), int(q[1].argmax(-1).item())

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
            "center": round(float(d["center"]), 4),
            "upper": round(float(d["upper"]), 4) if d["upper"] is not None else None,
            "lower": round(float(d["lower"]), 4) if d["lower"] is not None else None,
        } for d in result["decisions"]]
        fills = [{
            "t": f["t"],
            "x": round(float(table["x"][f["t"]]), 3),
            "side": f["side"],
            "price": round(float(f["price"]), 4),
            "qty": float(f["qty"]),
        } for f in result["fills"]]
        write_json(os.path.join(out_dir, f"{market.date}.json"), {
            "symbol": symbol,
            "date": market.date,
            "preclose": round(float(market.pre_close), 4),
            "x": [round(float(v), 3) for v in table["x"]],
            "price": [round(float(v), 4) for v in market.mid],
            "closing_x": [],
            "closing_price": [],
            "decisions": decisions,
            "fills": fills,
            "log": result["log"],
        })
        exported.append(market.date)
    return {"symbol": symbol, "name": symbol, "dates": exported,
            "replay_dates": exported, "checkpoint": os.path.basename(path)}


def forecast_params(cfg: ForecastConfig) -> dict:
    """index.json 的展示参数（forecast 语义：tick 口径窗口参数、门控规则、宽度参数）。"""
    return {
        "model": "LightGBM",
        "lookback_ticks": cfg.window.lookback_ticks,
        "pred_ticks": cfg.window.pred_ticks,
        "stride_ticks": cfg.stride_ticks,
        "atr_mult": cfg.window.atr_mult,
        "atr_window": cfg.window.atr_window,
        "min_width_ratio": cfg.window.min_width_ratio,
        "gate_rules": {
            gate: [{"signal": signal, "operator": operator, "threshold": threshold}
                   for signal, operator, threshold in GATE_RULES[gate]]
            for gate in GATES
        },
    }


def update_index(out_dir: str, algorithm: str, entries: list, params: dict | None = None) -> None:
    """合并写 index.json：本次导出的算法段落整体替换，另一算法的既有目录保留。"""
    path = os.path.join(out_dir, "index.json")
    index = {"algorithms": {}}
    if os.path.exists(path):
        with open(path) as file:
            index = json.load(file)
        index.setdefault("algorithms", {})
    index["minute_labels"] = [str(label) for label in minute_labels()]
    index["closing_minutes"] = 0   # 新数据源只含连续竞价，无收盘集合竞价段
    index["algorithms"][algorithm] = {"symbols": entries}
    if params is not None:
        index["params"] = params
    write_json(path, index)


def main() -> None:
    parser = argparse.ArgumentParser(description="webviz 数据导出（forecast / control 决策回放）")
    parser.add_argument("--algorithm", choices=["forecast", "control"], required=True)
    parser.add_argument("--symbols", nargs="+", required=True, help="标的代码")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--checkpoint", default=None,
                        help="control：检查点路径（缺省按 control/runs 命名规则解析）")
    parser.add_argument("--method", default="GRID", help="control：方法名（与 run_all 文件名一致）")
    parser.add_argument("--seed", type=int, default=0, help="control：随机种子")
    parser.add_argument("--w", type=float, default=None,
                        help="control：hindsight 权重 w（缺省取 control Config 默认值）")
    parser.add_argument("--lam", type=float, default=None,
                        help="control：存货惩罚 λ（缺省取 control Config 默认值）")
    args = parser.parse_args()

    symbols = list(dict.fromkeys(args.symbols))
    if args.algorithm == "forecast":
        cfg = ForecastConfig(data_dir=args.data_dir, cache_dir=args.cache_dir)
        entries = [export_forecast_symbol(symbol, args, cfg) for symbol in symbols]
        update_index(args.out_dir, "forecast", entries, forecast_params(cfg))
    else:
        entries = [export_control_symbol(symbol, args, parser) for symbol in symbols]
        update_index(args.out_dir, "control", entries)
    for entry in entries:
        print(f"{args.algorithm} {entry['symbol']}: {len(entry['dates'])} 个交易日"
              f"（可回放 {len(entry['replay_dates'])}）", flush=True)


if __name__ == "__main__":
    main()
