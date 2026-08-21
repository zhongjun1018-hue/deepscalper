"""门控点负载：把一个停网段及其前后上下文装配成撰写解释所依据的结构化素材。

数据源只读：webviz/data/forecast/<symbol>/<date>.json（价格、逐锚点识别概率、前瞻实测统计、
常开 / 门控两套网格回放）、webviz/data/index.json（τ 与窗口、模式参数）、cache/<symbol>.npz
（分钟锚点行的 47 维回看窗口特征）。

负载结构（README「负载」）：
  meta     标的 / 日期 / 参数 / 上下文区间
  point    停网段：起止、触发分钟、常开 / 门控两侧的段内对照（points.gating_points）
  day      当日：半宽 W_d、两套网格的日度指标
  minutes  区间逐分钟：价格、识别概率、判定 / 停网状态、47 维特征（抽样）、前瞻实测与规则比值
  events   区间内两套网格的事件流
"""

from __future__ import annotations

import json
import os

import numpy as np

from data_provider.windows import FEATURE_NAMES, load_cache
from forecast.regime.config import RegimeConfig

from attribution.features import FEATURE_DICT, REALIZED_DICT, display_value
from attribution.points import M, price_at, replay_enabled

WEBVIZ_DATA = "webviz/data"
CONTEXT_BEFORE, CONTEXT_AFTER = 10, 5   # 停网段前后携带的上下文分钟数
FEATURE_STRIDE = 3                      # 47 维特征抽样步长；触发分钟及前两拍始终携带


def load_index(data_dir: str = WEBVIZ_DATA) -> dict:
    with open(os.path.join(data_dir, "index.json")) as file:
        return json.load(file)


def load_day(symbol: str, date: str, data_dir: str = WEBVIZ_DATA) -> dict:
    with open(os.path.join(data_dir, "forecast", symbol, f"{date}.json")) as file:
        return json.load(file)


def replay_dates(symbol: str, data_dir: str = WEBVIZ_DATA) -> list[str]:
    """有网格回放（测试段）的日期。"""
    for entry in load_index(data_dir)["algorithms"]["forecast"]["symbols"]:
        if entry["symbol"] == symbol:
            return list(entry["replay_dates"])
    return []


def day_features(symbol: str, date: str, cfg: RegimeConfig = RegimeConfig()) -> np.ndarray:
    """(M, 47) 当日分钟锚点行的回看窗口特征（统一缓存，命中时不重建）。"""
    cache = load_cache(symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir,
                       spec=cfg.window, zero_nan=False)
    return cache["features"][list(cache["dates"]).index(date)]


def build_payload(symbol: str, date: str, point: dict, data_dir: str = WEBVIZ_DATA) -> dict:
    index = load_index(data_dir)
    labels = index["minute_labels"]
    params = index["params"]
    threshold = params["pattern"]["probability_threshold"]
    day = load_day(symbol, date, data_dir)
    name = next(e["name"] for e in index["algorithms"]["forecast"]["symbols"]
                if e["symbol"] == symbol)
    feats = day_features(symbol, date)
    enabled = replay_enabled(day, params["confirm_n"])
    windows = {w["start"]: w for w in day["windows"]}
    width = day["width"]

    start = max(point["start"] - CONTEXT_BEFORE, 0)
    end = min(point["end"] + CONTEXT_AFTER, M - 1)
    trigger = point["start"]
    with_features = {trigger, trigger - 1, trigger - 2}

    minutes = []
    for k, m in enumerate(range(start, end + 1)):
        window = windows.get(m, {})
        prob = window.get("prediction", {}).get("probability")
        row = {
            "minute": m,
            "time": labels[m],
            "last_px": price_at(day, m),
            "p_adverse": prob,
            "adverse_signal": None if prob is None else prob > threshold,
            "engine_stopped": None if m < day["t0"] else not bool(enabled[m]),
        }
        if k % FEATURE_STRIDE == 0 or m in with_features:
            row["lookback_features"] = {
                nm: display_value(feats[m, i], FEATURE_DICT[nm][1])
                for i, nm in enumerate(FEATURE_NAMES)}
        realized = window.get("features")
        if realized:
            row["forward_realized"] = {
                nm: display_value(realized[nm], REALIZED_DICT[nm][1]) for nm in REALIZED_DICT}
            rel_w = width / window["open"]   # 当日相对半宽 w = W_d / 锚点价
            row["rule_ratios"] = {
                "resid_q90_over_w": display_value(realized["resid_abs_q90"] / rel_w, "raw"),
                "abs_slope_over_w": display_value(realized["abs_slope"] / rel_w, "raw")}
        minutes.append(row)

    def events_in(key: str) -> list:
        return [{"time": labels[e["minute"]], "minute": e["minute"], "kind": e["kind"],
                 "fill": e.get("fill"), "exposure": e["exposure"]}
                for e in day["grids"][key]["events"] if start <= e["minute"] <= end]

    def card(key: str) -> dict:
        grid = day["grids"][key]
        return {k: grid[k] for k in ("trades", "buys", "sells", "score", "grid_profit")}

    return {
        "meta": {
            "symbol": symbol, "name": name, "date": date,
            "context": {"start_time": labels[start], "end_time": labels[end]},
            "params": {"lookback_min": params["lookback_min"], "pred_min": params["pred_min"],
                       "confirm_n": params["confirm_n"], "probability_threshold": threshold,
                       "pattern_rule": params["pattern"]},
            "units": {"bp": "万分之一（对数量 ×1e4）", "pct": "百分比", "raw": "原值",
                      "W": "当日网格半宽 W_d"},
        },
        "point": {
            "id": point["id"], "stopped_from": labels[point["start"]],
            "stopped_to": labels[point["end"]], "stopped_minutes": point["minutes"],
            "trigger_minute": trigger, "drift_W": point["drift_W"],
            "always_on": point["always_on"], "gated": point["gated"],
        },
        "day": {"grid_half_width": width,
                "always_on": card("none"), "gated": card("prediction")},
        "minutes": minutes,
        "events": {"always_on": events_in("none"), "gated": events_in("prediction")},
    }
