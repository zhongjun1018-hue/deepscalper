"""统一缓存预建入口：python -m data_provider.cache，跨标的并行构建 cache/<symbol>.npz。

load_cache 幂等：缓存与窗口级 metadata 一致即跳过重建，中断后重跑从未完成的标的继续。
不同标的写不同文件，跨标的并行安全（同一标的并发重建才会写坏缓存，见 scripts/run_all.py）；
内存峰值约为单标的构建用量 × workers，内存紧张时用 --workers 1 退回串行。
预测块（preds）由 forecast.train 训练后回写，不属于本入口。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import time

from data_provider.ticks import list_symbols
from data_provider.windows import WindowSpec, load_cache


def _build_one(symbol: str, data_dir: str, cache_dir: str, spec: WindowSpec) -> float:
    """构建（或校验命中）单个标的的统一缓存，返回耗时（秒）。"""
    t0 = time.time()
    load_cache(symbol, data_dir=data_dir, cache_dir=cache_dir, spec=spec, zero_nan=False)
    return time.time() - t0


def build_all(symbols: list[str], data_dir: str = "data", cache_dir: str = "cache",
              spec: WindowSpec = WindowSpec(), workers: int = 1) -> None:
    """构建（或校验命中）统一缓存；workers>1 时跨标的进程池并行。"""
    total = len(symbols)
    if workers <= 1:
        for index, symbol in enumerate(symbols, 1):
            elapsed = _build_one(symbol, data_dir, cache_dir, spec)
            print(f"[{index}/{total}] {symbol}: {elapsed:.1f}s", flush=True)
        return
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, symbol, data_dir, cache_dir, spec): symbol
                   for symbol in symbols}
        for index, future in enumerate(cf.as_completed(futures), 1):
            print(f"[{index}/{total}] {futures[future]}: {future.result():.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="统一缓存预建（幂等，跨标的并行）")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="标的代码，缺省为 data 目录下全部标的")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="并行标的数；内存峰值随之线性增长，1 为串行")
    args = parser.parse_args()
    build_all(args.symbols or list_symbols(args.data_dir),
              data_dir=args.data_dir, cache_dir=args.cache_dir, workers=args.workers)


if __name__ == "__main__":
    main()
