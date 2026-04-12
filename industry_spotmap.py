import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

def setup():
    """Configure matplotlib backend and Korean font."""
    try:
        plt.figure().clear()
        plt.close()
    except Exception:
        matplotlib.use("Agg")

    plt.rcParams["axes.unicode_minus"] = False
    font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"]
    for font in font_candidates:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams["font.family"] = font
            break

    os.makedirs("./output", exist_ok=True)

def plot_temporal_cluster_map(meta_df: pd.DataFrame, labels: np.ndarray, sector: str):
    """
    Heatmap: rows = tickers, columns = window_end dates, colour = cluster label.
    Shows how each stock's regime evolves over time.
    """
    df = meta_df[["code", "window_end", "cluster"]].copy()

    pivot = df.pivot_table(
        index="code", columns="window_end", values="cluster", aggfunc="first"
    )
    #pivot.columns = pivot.columns.strftime("%Y-%m-%d")

    fig_w = max(14, len(pivot.columns) * 0.35)
    fig_h = max(6,  len(pivot.index)   * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 1
    cmap = sns.color_palette("tab10", n_colors=max(n_clusters, 2))

    sns.heatmap(
        pivot.astype(float),
        cmap=cmap,
        linewidths=0.3,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Cluster", "shrink": 0.5},
        vmin=-0.5,
        vmax=n_clusters - 0.5,
    )
    ax.set_title(f"Temporal Cluster Map — {sector}\n(each cell = cluster label for that ticker × window)")
    ax.set_xlabel("Window End Date")
    ax.set_ylabel("Ticker")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"./output/sector_temporal_map/06_temporal_map_{sector}.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    setup()
    total_stock_cluster = pd.read_csv('total_stock_cluster.csv', index_col=0)
    total_stock_cluster['code'] = total_stock_cluster['ticker'].apply(lambda x: 'A'+x[3:9])
    total_stock_cluster = total_stock_cluster[total_stock_cluster['window_end'] >= '2026-03-02']
    sector = pd.read_csv('sector.csv')
    total_stock_cluster['sector'] = total_stock_cluster['ticker'].map(sector.set_index('Code')['W26Sector'])
    sector_list = total_stock_cluster['sector'].unique().tolist()

    for s in sector_list:
        subset = total_stock_cluster[total_stock_cluster['sector'] == s]
        plot_temporal_cluster_map(subset, subset['cluster'].values, sector=s)