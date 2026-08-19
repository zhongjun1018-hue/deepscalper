"""网格几何：半宽、边界取整与严格穿越判定（engine 与 control/env 共用）。"""

import math

PRICE_EPS = 1e-9  # 价格比较与取整的浮点容差


def half_width(mult: float, atr: float, ref_price: float, min_ratio: float) -> float:
    """生效半宽 = max(mult×ATR, min_ratio×参考价)：ATR 档之外设相对价下限，防止网格过密。"""
    return max(mult * atr, min_ratio * ref_price)


def boundaries(center: float, hw: float, tick_size: float) -> tuple[float, float]:
    """中心 center、半宽 hw 下的（卖出边界, 买入边界）。

    卖出边界向上、买入边界向下取整到最小变动价位（容差 PRICE_EPS），
    保证是合法限价且不缩窄名义半宽。
    """
    upper = math.ceil((center + hw) / tick_size - PRICE_EPS) * tick_size
    lower = math.floor((center - hw) / tick_size + PRICE_EPS) * tick_size
    return upper, lower


def sell_crossed(bid1: float, upper: float) -> bool:
    """买一价严格穿越卖出边界（容差随价位缩放）。"""
    return bid1 > upper + PRICE_EPS * max(abs(upper), 1.0)


def buy_crossed(ask1: float, lower: float) -> bool:
    """卖一价严格穿越买入边界（容差随价位缩放）。"""
    return ask1 < lower - PRICE_EPS * max(abs(lower), 1.0)
