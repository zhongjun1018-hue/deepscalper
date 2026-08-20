"""回测结果热力图：每个留存指标一张「标的 × 模式」面板图。

版式统一：行为各标的加「全体」，列为统一回测的各模式（常开 / 真值门控 / 预测门控 /
RL 智能体，无数据的模式列整体省略）；收益类指标用以零为中心的 DIVERGING_COLORMAP，
其余指标用从零起的 SEQUENTIAL_COLORMAP（样式令牌见 utils/figure_style.py）。
"""

import math
import os

import numpy as np

from utils.figure_style import (AXIS_COLOR, DIVERGING_COLORMAP, MUTED_COLOR,
                                SEQUENTIAL_COLORMAP, TEXT_COLOR, cell_text_color,
                                font_faces, save_figure)

MODE_NAMES = {
    "zh": {"open": "常开", "oracle": "真值门控", "prediction": "预测门控",
           "agent": "RL 智能体"},
    "en": {"open": "Always-on", "oracle": "Oracle gate", "prediction": "Predicted gate",
           "agent": "RL agent"},
}

# key 读取 summaries 单元格（strategy/metrics.summarize 的字段）；
# gated=True 改读门控窗口占比表（仅 oracle / prediction 两列有定义）；
# vmax 固定自然上限，缺省取数据最大值。
HEATMAPS = [
    {
        "stem": "profits_g",
        "name": {"zh": "费用后网格收益 g", "en": "Net grid profit g"},
        "key": "mean_g", "diverging": True, "fmt": "{:.2f}",
        "unit": {"zh": "格距倍数（日均）", "en": "Grid spacings per day"},
    },
    {
        "stem": "closure_rate",
        "name": {"zh": "日均闭环率", "en": "Closure Rate"},
        "key": "mean_closure_rate", "scale": 100.0, "vmax": 100.0, "fmt": "{:.1f}",
        "unit": {"zh": "占比 (%)", "en": "Share (%)"},
    },
    {
        "stem": "counts_trades",
        "name": {"zh": "日均成交次数", "en": "Trades per Day"},
        "key": "mean_trades", "fmt": "{:.1f}",
        "unit": {"zh": "日均次数", "en": "Count per day"},
    },
    {
        "stem": "width_rel",
        "name": {"zh": "日均网格宽幅", "en": "Grid Half-width"},
        "key": "mean_width_rel", "scale": 100.0, "fmt": "{:.2f}",
        "unit": {"zh": "半宽 / 前收 (%)", "en": "Half-width / preclose (%)"},
    },
    {
        "stem": "shares_gated",
        "name": {"zh": "门控窗口占比", "en": "Gated-window Share"},
        "gated": True, "vmax": 100.0, "fmt": "{:.1f}",
        "unit": {"zh": "占比 (%)", "en": "Share (%)"},
    },
]


def save_charts(summaries, symbol_summaries, gated, symbol_gated, output_root):
    """每个留存指标落一张热力图 SVG，返回全部路径。"""
    import matplotlib.pyplot as plt

    english, chinese, face = font_faces()
    language = "zh" if chinese is not None else "en"
    title_font = chinese or english
    symbols = list(symbol_summaries)
    modes = [mode for mode, entry in summaries.items() if entry is not None]
    labels = [MODE_NAMES[language][mode] for mode in modes]
    row_labels = symbols + [("全体" if language == "zh" else "All")]

    def build_matrix(spec):
        if spec.get("gated"):
            def share(counts):
                total = counts["total"]
                return [100.0 * counts[mode] / total
                        if total and mode in counts else float("nan")
                        for mode in modes]
            rows = [share(symbol_gated[symbol]) for symbol in symbols] + [share(gated)]
            return np.array(rows)
        scale = spec.get("scale", 1.0)
        rows = [symbol_summaries[symbol] for symbol in symbols] + [summaries]
        return np.array([[scale * row[mode][spec["key"]]
                          if row.get(mode) is not None else float("nan")
                          for mode in modes] for row in rows])

    def heat_panel(axis, matrix, colormap, vmin, vmax, fmt, metric):
        image = axis.imshow(
            np.ma.masked_invalid(matrix), cmap=colormap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest")
        norm = image.norm
        for row in range(matrix.shape[0]):
            for column in range(len(modes)):
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
        axis.set_xticks(range(len(modes)))
        axis.set_xticklabels(
            labels, fontproperties=title_font, fontsize=10.5, color=TEXT_COLOR)
        axis.set_yticks(range(matrix.shape[0]))
        axis.set_yticklabels(row_labels, fontsize=10.5, color=TEXT_COLOR)
        for tick, name in zip(axis.get_yticklabels(), row_labels):
            tick.set_fontproperties(face(name))
        axis.axhline(len(symbols) - 0.5, color=TEXT_COLOR, linewidth=1.2)
        axis.set_xticks(np.arange(-0.5, len(modes), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.set_title(
            metric, fontproperties=face(metric), fontsize=13, color=TEXT_COLOR, pad=10)
        axis.tick_params(which="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        return image

    def attach_colorbar(figure, axis, image, label):
        colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.02)
        colorbar.set_label(
            label, fontproperties=title_font, fontsize=10, color=TEXT_COLOR)
        colorbar.outline.set_edgecolor(AXIS_COLOR)
        colorbar.outline.set_linewidth(0.8)
        colorbar.ax.tick_params(colors=TEXT_COLOR, width=0.8)
        for tick in colorbar.ax.get_yticklabels():
            tick.set_fontproperties(english)

    paths = []
    height = max(4.5, 0.45 * (len(symbols) + 1))
    for spec in HEATMAPS:
        name = spec["name"][language]
        matrix = build_matrix(spec)
        finite = [value for value in matrix.ravel() if math.isfinite(value)]
        if spec.get("diverging"):
            bound = max([1.0] + [abs(value) for value in finite])
            colormap, vmin, vmax = DIVERGING_COLORMAP, -bound, bound
        else:
            colormap, vmin = SEQUENTIAL_COLORMAP, 0.0
            vmax = max(1.0, spec.get("vmax", max(finite) if finite else 1.0))
        figure, axis = plt.subplots(
            figsize=(9.0, height), facecolor="white", constrained_layout=True)
        image = heat_panel(axis, matrix, colormap, vmin, vmax, spec["fmt"], name)
        attach_colorbar(figure, axis, image, spec["unit"][language])
        path = os.path.join(output_root, "{}.svg".format(spec["stem"]))
        save_figure(figure, path)
        plt.close(figure)
        paths.append(path)
    return paths
