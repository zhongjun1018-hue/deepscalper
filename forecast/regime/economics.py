"""经济验证：常开回放（门控不介入）下模式占比与网格盈亏的关系。

负期望基线下，门控回测的 mean_g 与成交次数强相关，「少交易」会被误读成
「识别准」。常开模式的成交不受门控影响，因此按当日不利模式占比分桶后
「高占比日的 g 显著更差」才是模式经济含义的直接证据。

    python -m forecast.regime.economics

写 runs_dir/economics.json：逐日 (g, 成交, 模式占比) 明细与分段三分位表。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_provider.ticks import list_symbols

from forecast.regime.config import RegimeConfig
from forecast.regime.data import load_banks, replay_grid_day
from forecast.regime.labels import pattern_labels


def day_table(banks: dict, pattern: dict, cfg: RegimeConfig) -> list[dict]:
    """全部切分段的逐日记录：常开回放 g、成交数与当日不利模式占比。"""
    rows = []
    for symbol, bank in banks.items():
        for flag in ("train", "val", "test"):
            for i in bank.day_indices(flag):
                record = replay_grid_day(bank, i, None, cfg)
                if not record:
                    continue
                day = pattern[symbol][i]
                judged = day >= 0
                rows.append({
                    "symbol": symbol, "date": str(bank.dates[i]), "split": flag,
                    "g": record["g"],
                    "trades": record["n_buys"] + record["n_sells"],
                    "share": (float((day == 1).sum() / judged.sum())
                              if judged.any() else float("nan"))})
        print(f"economics {symbol}: 常开回放完成", flush=True)
    return rows


def share_terciles(rows: list[dict], flags) -> list[dict]:
    """按占比三分位分桶的 (g, 成交) 表。"""
    picked = [r for r in rows if r["split"] in flags]
    share = np.array([r["share"] for r in picked])
    g = np.array([r["g"] for r in picked])
    trades = np.array([r["trades"] for r in picked], dtype=np.float64)
    ok = np.isfinite(share) & np.isfinite(g)
    share, g, trades = share[ok], g[ok], trades[ok]
    if len(g) < 3:
        return []
    buckets = np.array_split(np.argsort(share, kind="stable"), 3)
    return [{"share_range": [float(share[idx].min()), float(share[idx].max())],
             "n_days": int(len(idx)),
             "mean_g": float(g[idx].mean()),
             "se_g": float(g[idx].std(ddof=1) / np.sqrt(len(idx))),
             "mean_trades": float(trades[idx].mean())}
            for idx in buckets]


def main():
    parser = argparse.ArgumentParser(
        description="模式经济验证：常开回放下按不利模式占比分桶比较日度 g")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--runs-dir", default="forecast/regime/runs")
    args = parser.parse_args()
    cfg = RegimeConfig(data_dir=args.data_dir, cache_dir=args.cache_dir,
                       runs_dir=args.runs_dir,
                       symbols=tuple(sorted(args.symbols
                                            or list_symbols(args.data_dir))))

    banks = load_banks(cfg)
    pattern = {symbol: pattern_labels(bank, cfg)
               for symbol, bank in banks.items()}
    rows = day_table(banks, pattern, cfg)
    terciles = {scope: share_terciles(rows, flags)
                for scope, flags in (("train_val", ("train", "val")),
                                     ("test", ("test",)))}

    out_dir = Path(cfg.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "economics.json", "w") as file:
        json.dump({"terciles": terciles, "days": rows},
                  file, ensure_ascii=False, indent=1)
    for scope, rows_ in terciles.items():
        print(scope)
        for r in rows_:
            print(f"  占比 {r['share_range'][0]:.2f}-{r['share_range'][1]:.2f}: "
                  f"g={r['mean_g']:+.2f} ±{r['se_g']:.2f} "
                  f"成交 {r['mean_trades']:.1f}（{r['n_days']} 日）", flush=True)


if __name__ == "__main__":
    main()
