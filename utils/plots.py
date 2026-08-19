import os

import matplotlib.pyplot as plt
import numpy as np

from utils.figure_style import (ACCENT_BLUE, ACCENT_BROWN, DIVERGING_COLORMAP,
                                GRID_COLOR, MUTED_COLOR, PLOT_STYLE, TEXT_COLOR,
                                cell_text_color)


plt.switch_backend('agg')

TRUE_COLOR = ACCENT_BLUE
PREDICTION_COLOR = ACCENT_BROWN


def _rank(values):
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat((starts + ends - 1) / 2, ends - starts)
    return ranks


def _correlation(left, right, method='pearson'):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    if method == 'spearman':
        left, right = _rank(left), _rank(right)
    return float(np.corrcoef(left, right)[0, 1])


def _selected_symbols(symbol_ids, symbols, max_symbols):
    selected = [index for index in range(len(symbols)) if (symbol_ids == index).any()]
    return selected[:max_symbols] if max_symbols else selected


def clear_result_figures(folder):
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        stem, extension = os.path.splitext(name)
        if (extension in ('.png', '.svg') and
                (stem.startswith('pred_vs_true_') or
                 stem == 'prediction_correlation_heatmap')):
            os.remove(os.path.join(folder, name))


def _finish_axis(axis):
    axis.grid(axis='y', color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def plot_pred_vs_true(prediction, target, symbol_ids, symbols, label_names, folder,
                      max_symbols=0, max_points=600):
    """Save one compact predicted-versus-true chart per target."""
    symbol_ids = np.asarray(symbol_ids)
    selected = _selected_symbols(symbol_ids, symbols, max_symbols)
    os.makedirs(folder, exist_ok=True)

    paths = []
    columns = 2 if len(selected) > 6 else 1
    rows_count = int(np.ceil(len(selected) / columns))
    with plt.rc_context(PLOT_STYLE):
        for target_column, label in enumerate(label_names):
            figure, axes = plt.subplots(
                rows_count, columns, squeeze=False,
                figsize=(7.0 * columns, max(2.35 * rows_count, 3.4)))
            for axis, symbol_id in zip(axes.flat, selected):
                rows = np.flatnonzero(symbol_ids == symbol_id)[-max_points:]
                truth = target[rows, target_column]
                forecast = prediction[rows, target_column]
                axis.plot(truth, color=TRUE_COLOR, linewidth=1.25, label='Actual')
                axis.plot(forecast, color=PREDICTION_COLOR, linewidth=1.15,
                          linestyle=(0, (4, 2)), label='Prediction')
                axis.set_title('{}   r = {:.3f}'.format(
                    symbols[symbol_id], _correlation(forecast, truth)), loc='left')
                _finish_axis(axis)
            for axis in axes.flat[len(selected):]:
                axis.set_visible(False)
            for axis in axes[-1]:
                if axis.get_visible():
                    axis.set_xlabel('Test sample')
            figure.suptitle(label.replace('_', ' '), fontsize=15, fontweight='semibold')
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, legend_labels, loc='upper center', ncol=2,
                          bbox_to_anchor=(0.5, 0.975), frameon=False)
            figure.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
            path = os.path.join(folder, 'pred_vs_true_{}.svg'.format(label))
            figure.savefig(path, bbox_inches='tight')
            plt.close(figure)
            paths.append(path)
    return paths


def plot_correlation_heatmap(prediction, target, symbol_ids, symbols, label_names, folder,
                             max_symbols=0):
    """Save Pearson and Spearman correlations by symbol and target."""
    symbol_ids = np.asarray(symbol_ids)
    selected = _selected_symbols(symbol_ids, symbols, max_symbols)
    row_masks = [symbol_ids == symbol_id for symbol_id in selected]
    row_masks.append(np.isin(symbol_ids, selected))

    matrices = {}
    for method in ('pearson', 'spearman'):
        matrix = np.empty((len(row_masks), len(label_names)))
        for row, rows in enumerate(row_masks):
            for column in range(len(label_names)):
                matrix[row, column] = _correlation(
                    prediction[rows, column], target[rows, column], method)
        matrices[method] = matrix

    ylabels = [symbols[index] for index in selected] + ['All']
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1, 2, sharey=True, constrained_layout=True,
            figsize=(max(13, 2.6 * len(label_names)), max(4, 0.42 * len(row_masks))))
        image = None
        for axis, method in zip(axes, ('pearson', 'spearman')):
            matrix = matrices[method]
            image = axis.imshow(matrix, cmap=DIVERGING_COLORMAP, vmin=-1, vmax=1,
                                aspect='auto')
            axis.set_xticks(range(len(label_names)), label_names, rotation=25, ha='right')
            axis.set_yticks(range(len(ylabels)), ylabels)
            axis.set_xlabel('Target')
            axis.set_title('{} correlation'.format(method.title()), fontsize=13,
                           fontweight='semibold')
            axis.axhline(len(selected) - 0.5, color=TEXT_COLOR, linewidth=1)
            for row in range(len(row_masks)):
                for column in range(len(label_names)):
                    value = matrix[row, column]
                    text = '{:.2f}'.format(value) if np.isfinite(value) else '-'
                    color = (cell_text_color(DIVERGING_COLORMAP, (value + 1.0) / 2.0)
                             if np.isfinite(value) else MUTED_COLOR)
                    axis.text(column, row, text, ha='center', va='center',
                              color=color, fontsize=8)
        axes[0].set_ylabel('Symbol')
        colorbar = figure.colorbar(image, ax=axes, pad=0.02)
        colorbar.set_label('Correlation')

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'prediction_correlation_heatmap.svg')
        figure.savefig(path, bbox_inches='tight')
        plt.close(figure)
    return path
