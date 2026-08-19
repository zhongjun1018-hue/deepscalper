"""统一缓存预建入口：python -m data_provider.cache，逐标的串行构建 cache/<symbol>.npz。

load_cache 幂等：缓存与窗口级 metadata 一致即跳过重建，中断后重跑从未完成的标的继续；
串行执行使峰值内存只有单个标的的量。预测块（preds）由 forecast.train 训练后回写，
不属于本入口。
"""

from __future__ import annotations

import argparse
import time

from data_provider.ticks import list_symbols
from data_provider.windows import WindowSpec, load_cache


def build_all(symbols: list[str], data_dir: str = "data", cache_dir: str = "cache",
              spec: WindowSpec = WindowSpec()) -> None:
    """逐标的串行构建（或校验命中）统一缓存。"""
    for index, symbol in enumerate(symbols, 1):
        t0 = time.time()
        load_cache(symbol, data_dir=data_dir, cache_dir=cache_dir, spec=spec,
                   zero_nan=False)
        print(f"[{index}/{len(symbols)}] {symbol}: {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="统一缓存预建（幂等，逐标的串行）")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    build_all(args.symbols or list_symbols(args.data_dir),
              data_dir=args.data_dir, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
