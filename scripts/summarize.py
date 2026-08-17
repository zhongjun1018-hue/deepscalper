"""汇总实验结果：读取 results/<symbol>/*.json，输出均值±标准差汇总表。"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

METHODS = ["BAH", "MV", "TSM", "MLP", "GRU", "LGBM", "DQN", "DS-NH", "DS-NA", "DS"]
METRICS = ["TR", "SR", "CR", "SoR"]


def main(result_dir: str = "results") -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({"symbol": r["symbol"], "method": r["method"], "seed": r.get("seed"),
                     **{k: r[k] for k in METRICS}})
    if not rows:
        print("未找到结果文件")
        return
    df = pd.DataFrame(rows)

    out_rows = []
    for (symbol, method), g in df.groupby(["symbol", "method"]):
        row = {"symbol": symbol, "method": method, "n_runs": len(g)}
        for k in METRICS:
            v = g[k].to_numpy()
            row[k] = f"{v.mean():.4f} ± {v.std():.4f}" if len(g) > 1 else f"{v.mean():.4f}"
            row[f"{k}_mean"] = v.mean()
        out_rows.append(row)
    summary = pd.DataFrame(out_rows)
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    summary = summary.sort_values(["symbol", "method"])

    out_path = os.path.join(result_dir, "summary.csv")
    summary.drop(columns=[f"{k}_mean" for k in METRICS]).to_csv(out_path, index=False)
    cols = ["symbol", "method", "n_runs"] + METRICS
    print(summary[cols].to_string(index=False))
    print(f"\n已保存：{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
