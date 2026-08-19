"""回测结果热力图：每个留存指标一张「标的 × scheme」门控面板图。

版式统一：行为各标的加「全体」，列为三种 scheme（无门控 / Oracle / 预测门控）；
收益类指标用以零为中心的 DIVERGING_COLORMAP，其余指标用从零起的
SEQUENTIAL_COLORMAP（样式令牌见 utils/figure_style.py）。
"""

import math
import os

import numpy as np

from utils.figure_style import (AXIS_COLOR, DIVERGING_COLORMAP, MUTED_COLOR,
                                SEQUENTIAL_COLORMAP, TEXT_COLOR, cell_text_color,
                                font_faces, save_figure)

SCHEME_NAMES = {
    "zh": {"none": "无门控", "oracle": "Oracle", "prediction": "预测门控"},
    "en": {"none": "No gate", "oracle": "Oracle", "prediction": "Predicted gate"},
}
GATE_NAMES = {
    "zh": {"residual": "残差趋势过滤"},
    "en": {"residual": "Residual-trend Filter"},
}

# key 读取 summaries 单元格（strategy/metrics.summarize + 费用后净利润字段）；
# gated=True 改读门控窗口占比表；vmax 固定自然上限，缺省取数据最大值。
HEATMAPS = [
    {
        "stem": "scores_weighted",
        "name": {"zh": "成交加权 Score", "en": "Fill-weighted Score"},
        "key": "weighted_score_mean", "vmax": 1.0, "fmt": "{:.3f}",
        "unit": {"zh": "均值", "en": "Mean"},
    },
    {
        "stem": "scores_equal",
        "name": {"zh": "交易日等权 Score", "en": "Equal-weighted Score"},
        "key": "equal_score_mean", "vmax": 1.0, "fmt": "{:.3f}",
        "unit": {"zh": "均值", "en": "Mean"},
    },
    {
        "stem": "counts_rounds",
        "name": {"zh": "日均满轮次数", "en": "Full Rounds per Day"},
        "key": "mean_rounds", "fmt": "{:.2f}",
        "unit": {"zh": "日均次数", "en": "Count per day"},
    },
    {
        "stem": "counts_buys",
        "name": {"zh": "日均买入次数", "en": "Buys per Day"},
        "key": "mean_buys", "fmt": "{:.2f}",
        "unit": {"zh": "日均次数", "en": "Count per day"},
    },
    {
        "stem": "counts_sells",
        "name": {"zh": "日均卖出次数", "en": "Sells per Day"},
        "key": "mean_sells", "fmt": "{:.2f}",
        "unit": {"zh": "日均次数", "en": "Count per day"},
    },
    {
        "stem": "shares_closed",
        "name": {"zh": "满轮日占比", "en": "Closed-day Share"},
        "key": "closed_day_share", "scale": 100.0, "vmax": 100.0, "fmt": "{:.1f}",
        "unit": {"zh": "占比 (%)", "en": "Share (%)"},
    },
    {
        "stem": "shares_gated",
        "name": {"zh": "门控窗口占比", "en": "Gated-window Share"},
        "gated": True, "fmt": "{:.1f}",
        "unit": {"zh": "占比 (%)", "en": "Share (%)"},
    },
    {
        "stem": "profits_grid",
        "name": {"zh": "网格收益", "en": "Grid profit"},
        "key": "mean_grid_profit", "diverging": True, "fmt": "{:.2f}",
        "unit": {"zh": "日均值", "en": "Daily mean"},
    },
    {
        "stem": "profits_lower",
        "name": {"zh": "收益下界", "en": "Profit lower bound"},
        "key": "mean_grid_profit_lower", "diverging": True, "fmt": "{:.2f}",
        "unit": {"zh": "日均值", "en": "Daily mean"},
    },
    {
        "stem": "profits_per_trade",
        "name": {"zh": "收益 / 交易次数", "en": "Profit / trades"},
        "key": "mean_profit_per_trade", "diverging": True, "fmt": "{:.2f}",
        "unit": {"zh": "日均值", "en": "Daily mean"},
    },
    {
        "stem": "profits_net",
        "name": {"zh": "费用后净利润", "en": "Net profit after fees"},
        "key": "mean_net_profit", "diverging": True, "fmt": "{:.2f}",
        "unit": {"zh": "日均值", "en": "Daily mean"},
    },
]


def save_charts(summaries, symbol_summaries, gated, symbol_gated, output_root):
    """每个留存指标落一张热力图 SVG，返回全部路径。"""
    import matplotlib.pyplot as plt

    english, chinese, face = font_faces()
    language = "zh" if chinese is not None else "en"
    title_font = chinese or english
    gates = list(summaries)
    schemes = list(summaries[gates[0]])
    symbols = list(symbol_summaries)
    labels = [SCHEME_NAMES[language][scheme] for scheme in schemes]

    def pick(zh, en):
        return zh if language == "zh" else en

    def build_matrix(spec, gate):
        if spec.get("gated"):
            def share(gate_counts):
                total = gate_counts["total"]
                return [100.0 * gate_counts[scheme] / total
                        if total and scheme != "none" else float("nan")
                        for scheme in schemes]
            rows = [share(symbol_gated[symbol][gate]) for symbol in symbols]
            rows.append(share(gated[gate]))
            return np.array(rows)
        scale = spec.get("scale", 1.0)
        rows = [symbol_summaries[symbol][gate] for symbol in symbols] + [summaries[gate]]
        return np.array([[scale * row[scheme][spec["key"]] for scheme in schemes]
                         for row in rows])

    def heat_panel(axis, gate, matrix, colormap, vmin, vmax, fmt, metric,
                   show_symbol_labels):
        image = axis.imshow(
            np.ma.masked_invalid(matrix), cmap=colormap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest")
        norm = image.norm
        for row in range(matrix.shape[0]):
            for column in range(len(schemes)):
                value = matrix[row, column]
                if math.isfinite(value):
                    axis.text(
                        column, row, fmt.format(value), ha="center", va="center",
                        fontproperties=english, fontsize=11,
                        color=cell_text_color(colormap, norm(value)))
                else:
                    axis.text(
                        column, row, "—", ha="center", va="center",
                        fontproperties=english, fontsize=11, color=MUTED_COLOR)
        axis.set_xticks(range(len(schemes)))
        axis.set_xticklabels(
            labels, fontproperties=title_font, fontsize=10.5, color=TEXT_COLOR)
        row_labels = symbols + [pick("全体", "All")]
        axis.set_yticks(range(matrix.shape[0]))
        axis.set_yticklabels(
            row_labels if show_symbol_labels else [], fontsize=10.5, color=TEXT_COLOR)
        if show_symbol_labels:
            for tick, name in zip(axis.get_yticklabels(), row_labels):
                tick.set_fontproperties(face(name))
        axis.axhline(len(symbols) - 0.5, color=TEXT_COLOR, linewidth=1.2)
        axis.set_xticks(np.arange(-0.5, len(schemes), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        title = "{} · {}".format(GATE_NAMES[language][gate], metric)
        axis.set_title(
            title, fontproperties=face(title), fontsize=13, color=TEXT_COLOR, pad=10)
        axis.tick_params(which="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        return image

    def attach_colorbar(figure, axes, image, label):
        colorbar = figure.colorbar(
            image, ax=axes.ravel().tolist(), fraction=0.05, pad=0.02)
        colorbar.set_label(
            label, fontproperties=title_font, fontsize=10, color=TEXT_COLOR)
        colorbar.outline.set_edgecolor(AXIS_COLOR)
        colorbar.outline.set_linewidth(0.8)
        colorbar.ax.tick_params(colors=TEXT_COLOR, width=0.8)
        for tick in colorbar.ax.get_yticklabels():
            tick.set_fontproperties(english)

    paths = []
    height = max(6.0, 0.45 * (len(symbols) + 1))
    for spec in HEATMAPS:
        name = spec["name"][language]
        matrices = [build_matrix(spec, gate) for gate in gates]
        finite = [value for matrix in matrices
                  for value in matrix.ravel() if math.isfinite(value)]
        if spec.get("diverging"):
            bound = max([1.0] + [abs(value) for value in finite])
            colormap, vmin, vmax = DIVERGING_COLORMAP, -bound, bound
        else:
            colormap, vmin = SEQUENTIAL_COLORMAP, 0.0
            vmax = max(1.0, spec.get("vmax", max(finite) if finite else 1.0))
        figure, axes = plt.subplots(
            1, len(gates), figsize=(14.0, height), squeeze=False,
            facecolor="white", constrained_layout=True)
        image = None
        for column, gate in enumerate(gates):
            image = heat_panel(
                axes[0, column], gate, matrices[column], colormap, vmin, vmax,
                spec["fmt"], name, show_symbol_labels=column == 0)
        attach_colorbar(figure, axes, image, pick(
            spec["unit"]["zh"], spec["unit"]["en"]))
        path = os.path.join(output_root, "{}.svg".format(spec["stem"]))
        save_figure(figure, path)
        plt.close(figure)
        paths.append(path)
    return paths
