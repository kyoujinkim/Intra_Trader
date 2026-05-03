import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import config


def setup_matplotlib():
    """Configure matplotlib backend and Korean font. Call once at module load."""
    try:
        plt.figure().clear()
        plt.close()
    except Exception:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            matplotlib.use('Agg')
            print("[시스템] 비대화형 백엔드(Agg)가 활성화되었습니다.")

    font_list = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR', 'DejaVu Sans']
    found = False
    for font in font_list:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font
            found = True
            break
    plt.rcParams['axes.unicode_minus'] = False
    if not found:
        print("[경고] 한글 폰트를 찾을 수 없습니다. 시각화 시 한글이 깨질 수 있습니다.")


def safe_show():
    """Call plt.show() only when an interactive backend is active."""
    backend = matplotlib.get_backend().lower()
    interactive = ['tkagg', 'qt5agg', 'macosx', 'nbagg', 'module://ipykernel.pylab.backend_inline']
    if any(b in backend for b in interactive):
        plt.show()


def load_indicator_excel(path=None, start_dt=None):
    """Load all indicator sheets from the Excel file.

    Returns a dict keyed by config.SHEET_NAMES keys (e.g. 'strength', 'resilience', ...).
    """
    path = path or config.INDICATOR_XLSX
    sheets = {}
    for key, sheet_name in config.SHEET_NAMES.items():
        df = pd.read_excel(path, sheet_name=sheet_name, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if start_dt is not None:
            df = df[df.index >= pd.to_datetime(start_dt)]
        sheets[key] = df
    return sheets


def window_features(s, res, d, sd, sdd) -> dict:
    """Return a flat dict of mean/max/min features for one rolling window."""
    return {
        "str_mean":  s.mean(),   "str_max":  s.max(),   "str_min":  s.min(),
        "res_mean":  res.mean(),
        "disp_mean": d.mean(),   "disp_max": d.max(),   "disp_min": d.min(),
        "sd_mean":   sd.mean(),  "sd_max":   sd.max(),  "sd_min":   sd.min(),
        "sdd_mean":  sdd.mean(), "sdd_max":  sdd.max(), "sdd_min":  sdd.min(),
    }
