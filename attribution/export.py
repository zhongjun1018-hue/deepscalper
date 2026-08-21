"""页面数据导出：逐日重放门控点，连同 runs/ 里已写好的解释一起写成页面 JSON。

  .venv/bin/python -m attribution.export --symbol 301308 --dates 20260717 20260701

解释由人工按 README「解释写法」撰写为 runs/<symbol>_<date>_p<k>.md；本脚本每日写
runs/<symbol>_<date>.json（全部门控点 + 已有解释）并登记到 runs/index.json，供 index.html 读取。
"""

from __future__ import annotations

import argparse
import json
import os

from attribution.payload import load_day, load_index, replay_dates
from attribution.points import gating_points

RUNS_DIR = "attribution/runs"


def explanation_path(symbol: str, date: str, point_id: int) -> str:
    return os.path.join(RUNS_DIR, f"{symbol}_{date}_p{point_id}.md")


def write_day(symbol: str, date: str, points: list[dict]) -> None:
    explanations = {}
    for point in points:
        path = explanation_path(symbol, date, point["id"])
        if os.path.exists(path):
            with open(path) as file:
                explanations[point["id"]] = file.read().strip()
    with open(os.path.join(RUNS_DIR, f"{symbol}_{date}.json"), "w") as file:
        json.dump({"symbol": symbol, "date": date, "points": points,
                   "explanations": explanations}, file, ensure_ascii=False)
    index_path = os.path.join(RUNS_DIR, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as file:
            index = json.load(file)
    index[symbol] = sorted(set(index.get(symbol, [])) | {date})
    with open(index_path, "w") as file:
        json.dump(index, file)


def main() -> None:
    parser = argparse.ArgumentParser(description="门控归因页面数据导出")
    parser.add_argument("--symbol", default="301308")
    parser.add_argument("--dates", nargs="*", default=None, help="缺省为全部测试段回放日")
    args = parser.parse_args()
    confirm_n = load_index()["params"]["confirm_n"]
    os.makedirs(RUNS_DIR, exist_ok=True)
    for date in args.dates or replay_dates(args.symbol):
        points = gating_points(load_day(args.symbol, date), confirm_n)
        write_day(args.symbol, date, points)
        print(f"{date}: {len(points)} 个门控点，"
              f"{sum(os.path.exists(explanation_path(args.symbol, date, p['id'])) for p in points)} 篇解释")


if __name__ == "__main__":
    main()
