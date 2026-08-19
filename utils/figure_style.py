"""Shared figure tokens for every plotting module.

The palette follows the project reference sheet: color 1 #48B2D9 (blue),
color 3 #AB8341 (brown), color 5 #A39161 (khaki), color 6 #C8C896 (olive),
color 8 #F0E7D7 (cream) and the #990000 highlight red, on white surfaces
with the sheet's gray ramp for axes, grids and muted text.  Sequential
heatmaps run white-cream-olive-khaki-brown; diverging heatmaps run red to
cream to blue (the blue pole deepens over the last stretch so strong gains
stay as saturated as strong losses).
"""

import os

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'path'

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

TEXT_COLOR = '#000000'
MUTED_COLOR = '#787878'
AXIS_COLOR = '#969696'
GRID_COLOR = '#C0C0C0'
ACCENT_BLUE = '#48B2D9'
ACCENT_RED = '#990000'
ACCENT_BROWN = '#AB8341'
ACCENT_KHAKI = '#A39161'
ACCENT_OLIVE = '#C8C896'
ACCENT_CREAM = '#F0E7D7'
BAD_COLOR = '#EDEDED'

SEQUENTIAL_COLORMAP = LinearSegmentedColormap.from_list(
    'sheet_sequential',
    ['#FFFFFF', ACCENT_CREAM, ACCENT_OLIVE, ACCENT_KHAKI, ACCENT_BROWN])
DIVERGING_COLORMAP = LinearSegmentedColormap.from_list(
    'sheet_diverging',
    [(0.0, ACCENT_RED), (0.5, ACCENT_CREAM), (0.85, ACCENT_BLUE), (1.0, '#2477A0')])
SEQUENTIAL_COLORMAP.set_bad(BAD_COLOR)
DIVERGING_COLORMAP.set_bad(BAD_COLOR)

PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 10,
    'axes.edgecolor': AXIS_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'axes.titlecolor': TEXT_COLOR,
    'axes.titlesize': 11,
    'axes.titleweight': 'medium',
    'xtick.color': MUTED_COLOR,
    'ytick.color': MUTED_COLOR,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
}

ENGLISH_FONT_NAMES = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
CHINESE_FONT_NAMES = ['SimSun', '宋体', 'Songti SC', 'STSong']
CJK_FONT_FILES = [
    '/mnt/c/Windows/Fonts/simsun.ttc',
    '/mnt/c/Windows/Fonts/SourceHanSansSC-Regular.ttf',
    '/mnt/c/Windows/Fonts/SourceHanSansCN-Normal.otf',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]


def _named_font(names):
    from matplotlib import font_manager
    for name in names:
        matches = [entry for entry in font_manager.fontManager.ttflist
                   if entry.name == name]
        if matches:
            regular = next((entry for entry in matches
                            if entry.style == 'normal'
                            and entry.weight in ('normal', 400)), matches[0])
            return font_manager.FontProperties(fname=regular.fname)
    return None


def font_faces():
    """Resolve the shared (english, chinese-or-None, face) font triple."""
    from matplotlib import font_manager
    english = _named_font(ENGLISH_FONT_NAMES)
    chinese = _named_font(CHINESE_FONT_NAMES)
    if chinese is None:
        for path in CJK_FONT_FILES:
            if not os.path.exists(path):
                continue
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                continue
            chinese = font_manager.FontProperties(fname=path)
            break
    title_font = chinese or english

    def face(text):
        return title_font if any('一' <= char <= '鿿' for char in text) else english

    return english, chinese, face


def cell_text_color(colormap, normalized):
    """White ink on dark heatmap cells, sheet text ink on light ones."""
    red, green, blue, _ = colormap(normalized)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return 'white' if luminance < 0.55 else TEXT_COLOR


def save_figure(figure, path):
    """Save one figure as SVG with the glyphs outlined for portability."""
    figure.savefig(path, facecolor='white')
