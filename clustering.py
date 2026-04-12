"""
clustering.py
─────────────
Loads price microstructure indicators from price_microstructure_analysis.xlsx,
slices each stock's time series into rolling windows (patches), builds a feature
vector per window, then clusters all (ticker, window) samples together.

Pipeline:
  load_data()
      ↓
  build_windowed_feature_matrix()  ← rolling window patches → one sample per (ticker, date)
      ↓
  find_optimal_k()         ← silhouette across k range
      ↓
  run_all_clusterings()    ← K-Means, Agglomerative, DBSCAN, GMM
      ↓
  profile_clusters()       ← mean indicators, return stats per cluster
  plot_temporal_cluster_map()  ← heatmap: ticker × time, coloured by cluster
  plot_ticker_cluster_dist()   ← how each stock distributes across clusters over time
      ↓
  cluster_results.xlsx     ← labels + feature matrix saved

CLI:
  # global clustering (all sectors together)
  python clustering.py --window 20 --stride 5 --model gmm --k_range 2 10

  # per-sector clustering (recommended)
  python clustering.py --window 20 --stride 5 --model gmm --k_range 2 8 --sector_csv ./sector.csv
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from scipy.stats import skew
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────────────────────

def load_sector(path: str = "./sector.csv") -> dict:
    """
    Load sector.csv and return {ticker_code: sector_name}.
    Expects columns: Code, Sector.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    mapping = dict(zip(df["Code"].str.strip(), df["Sector"].str.strip()))
    sectors = sorted(set(mapping.values()))
    print(f"    Sector map loaded: {len(mapping)} tickers → {len(sectors)} sectors: {sectors}")
    return mapping


def load_data(path: str = "./price_microstructure_analysis.xlsx") -> dict:
    """
    Load all indicator sheets.
    Returns dict of {sheet_name: DataFrame(dates × tickers)}.
    """
    sheet_map = {
        "strength":       "Strength_Score",
        "resilience":     "Resilience",
        "disparity":      "Disparity",
        "signed_disp":    "Signed_Disparity",
        "s_disp_diff":    "Signed_Disp_Diff",
        "coil_release":   "Coil_Release",
        "returns":        "Returns",
    }
    sheets = {}
    print("[1/5] Loading data from Excel...")
    for key, sheet in sheet_map.items():
        df = pd.read_excel(path, sheet_name=sheet, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        sheets[key] = df
        print(f"    {key}: {df.shape[0]} dates × {df.shape[1]} tickers")
    return sheets


# ──────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ────────────────────────────────────────────────────────
def _pct_above(series: pd.Series, pct: float = 0.75) -> float:
    """Fraction of observations above the pct-th percentile of the series."""
    threshold = series.quantile(pct)
    return (series > threshold).mean()


def _neg_run_max(series: pd.Series) -> int:
    """Longest consecutive run of negative values."""
    max_run = cur = 0
    for v in series.dropna():
        if v < 0:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _window_features(s, res, d, sd, sdd) -> dict:
    """
    Extract statistical features from one window slice of all indicators.
    Returns a flat dict. Called once per (ticker, window).
    """
    row = {}

    # ── Strength ───────────────────────────────────────────────
    row["str_mean"]      = s.mean()
    row["str_max"]       = s.max()
    row["str_min"]       = s.min()

    # ── Resilience ─────────────────────────────────────────────
    row["res_mean"]      = res.mean()

    # ── Disparity ──────────────────────────────────────────────
    row["disp_mean"]     = d.mean()
    row["disp_max"]      = d.max()
    row["disp_min"]      = d.min()

    # ── Signed_Disparity ──────────────────────────────────────────────
    row["sd_mean"]       = sd.mean()
    row["sd_max"]        = sd.max()
    row["sd_min"]        = sd.min()

    # ── Signed_Disp_Diff ──────────────────────────────────────────────
    row["sdd_mean"]      = sdd.mean()
    row["sdd_max"]       = sdd.max()
    row["sdd_min"]       = sdd.min()

    return row


def build_windowed_feature_matrix(
    sheets: dict,
    window: int = 20,
    stride: int = 5,
) -> tuple:
    """
    Slice each ticker's indicator time series into rolling patches of `window`
    trading days, stepping `stride` days at a time.

    Each (ticker, window_end_date) becomes one sample.

    Returns:
        feat_df  : DataFrame(samples × features)  — clustering input (X)
        y_df     : DataFrame(samples × 3)          — return stats (outcome, not used in clustering)
        meta_df  : DataFrame(samples × 2)          — ticker + window_end_date for plotting
    """
    print(f"[2/5] Building windowed feature matrix (window={window}, stride={stride})...")

    # Align all sheets on a common date index
    common_dates = sheets["strength"].index
    for key in sheets:
        common_dates = common_dates.intersection(sheets[key].index)
    common_dates = common_dates.sort_values()

    tickers = sheets["strength"].columns.tolist()
    feat_records, y_records, meta_records = [], [], []

    for ticker in tickers:
        # Extract aligned series for this ticker
        cols = {
            "s":   sheets["strength"][ticker].reindex(common_dates),
            "res": sheets["resilience"][ticker].reindex(common_dates),
            "d":   sheets["disparity"][ticker].reindex(common_dates),
            "sd":  sheets["signed_disp"][ticker].reindex(common_dates),
            "sdd": sheets["s_disp_diff"][ticker].reindex(common_dates),
            "cr":  sheets["coil_release"][ticker].reindex(common_dates),
            "ret": sheets["returns"][ticker].reindex(common_dates),
        }

        n = len(common_dates)
        starts = range(0, n - window + 1, stride)

        for start in starts:
            end = start + window
            w = {k: v.iloc[start:end].dropna() for k, v in cols.items()}

            # Skip windows with too much missing data
            if any(len(w[k]) < window // 2 for k in w):
                continue

            row = _window_features(w["s"], w["res"], w["d"], w["sd"], w["sdd"])
            feat_records.append(row)

            y_records.append({
                "ret_mean":   w["ret"].mean(),
                "ret_std":    w["ret"].std(),
                "ret_sharpe": w["ret"].mean() / (w["ret"].std() + 1e-9),
            })

            meta_records.append({
                "ticker":       ticker,
                "window_start": common_dates[start],
                "window_end":   common_dates[end - 1],
            })

    feat_df = pd.DataFrame(feat_records)
    y_df    = pd.DataFrame(y_records)
    meta_df = pd.DataFrame(meta_records)

    # Drop rows with any NaN features
    valid = feat_df.notna().all(axis=1)
    feat_df = feat_df[valid].reset_index(drop=True)
    y_df    = y_df[valid].reset_index(drop=True)
    meta_df = meta_df[valid].reset_index(drop=True)

    print(f"    Windowed samples : {len(feat_df)}  "
          f"({len(tickers)} tickers × ~{len(feat_df)//max(len(tickers),1)} windows each)")
    print(f"    Feature columns  : {feat_df.shape[1]}")
    return feat_df, y_df, meta_df


# ──────────────────────────────────────────────────────────────
# 3. Dimensionality Reduction
# ──────────────────────────────────────────────────────────────

def reduce_dimensions(X_scaled: np.ndarray, n_components: int = 2):
    """PCA for 2D visualization. Returns (X_pca, pca_object)."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"    PCA explained variance: {pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}% ({n_components} components)")
    return X_pca, pca


# ──────────────────────────────────────────────────────────────
# 4. Optimal k
# ──────────────────────────────────────────────────────────────

def _make_model(model_name: str, k: int, X_scaled: np.ndarray):
    """Instantiate a fresh clustering model for given k."""
    if model_name == "gmm":
        return GaussianMixture(n_components=k, random_state=42, n_init=10)
    if model_name == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward")
    if model_name == "dbscan":
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 90)
        return DBSCAN(eps=eps, min_samples=3)
    return KMeans(n_clusters=k, random_state=42, n_init=10)


def find_optimal_k(X_scaled: np.ndarray, k_range_list: list, model_name: str) -> int:
    """Silhouette analysis across k range. Returns suggested k."""
    k_range = range(k_range_list[0], k_range_list[1] + 1)
    print("[3/5] Finding optimal number of clusters...")
    silhouettes = []

    for k in k_range:
        m = _make_model(model_name, k, X_scaled)
        labels = m.fit_predict(X_scaled)
        # DBSCAN may produce noise (-1); exclude from silhouette
        valid = labels != -1
        n_unique = len(set(labels[valid]))
        if n_unique >= 2:
            silhouettes.append(silhouette_score(X_scaled[valid], labels[valid]))
        else:
            silhouettes.append(-1.0)

    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))

    axes.plot(list(k_range), silhouettes, "o-", color="darkorange")
    axes.set_title("Silhouette Score")
    axes.set_xlabel("Number of Clusters (k)")
    axes.set_ylabel("Silhouette Score")
    axes.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./output/01_optimal_k.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"    Best k by silhouette = {best_k}  (score={max(silhouettes):.4f})")
    return best_k


# ──────────────────────────────────────────────────────────────
# 5. Clustering
# ──────────────────────────────────────────────────────────────

def run_all_clusterings(X_scaled: np.ndarray, n_clusters: int, model_name: str = "kmeans") -> dict:
    """
    Run K-Means, Agglomerative, DBSCAN, GMM.
    Returns dict of {method: label_array}.
    """
    print(f"[4/5] Running clustering (k={n_clusters})...")
    results = {}

    # K-Means
    #km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    #results["kmeans"] = km.fit_predict(X_scaled)

    # Agglomerative (Ward linkage)
    #agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    #results["agglomerative"] = agg.fit_predict(X_scaled)

    # GMM (soft assignment, use argmax)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    results["gmm"] = gmm.fit_predict(X_scaled)

    # DBSCAN — auto eps via 5-NN distance
    #from sklearn.neighbors import NearestNeighbors
    #nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
    #distances, _ = nn.kneighbors(X_scaled)
    #eps_auto = np.percentile(distances[:, -1], 90)
    #db = DBSCAN(eps=eps_auto, min_samples=3)
    #results["dbscan"] = db.fit_predict(X_scaled)

    # Evaluate
    print("\n    Method             Silhouette   Davies-Bouldin   n_clusters")
    print("    " + "─" * 58)
    for method, labels in results.items():
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
        valid = labels != -1
        if n_unique >= 2 and valid.sum() >= n_unique:
            sil = silhouette_score(X_scaled[valid], labels[valid])
            db_score = davies_bouldin_score(X_scaled[valid], labels[valid])
            print(f"    {method:<18} {sil:>9.4f}   {db_score:>14.4f}   {n_unique}")
        else:
            print(f"    {method:<18} {'N/A':>9}   {'N/A':>14}   {n_unique}")

    return results


# ──────────────────────────────────────────────────────────────
# 6. Visualization
# ──────────────────────────────────────────────────────────────

def plot_pca_scatter(X_pca: np.ndarray, labels: np.ndarray, tickers: list, title: str, filename: str):
    """2D PCA scatter coloured by cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        label_name = f"Noise" if lab == -1 else f"Cluster {lab}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette[i], label=label_name, s=80, alpha=0.8)
        #for j, ticker in enumerate(np.array(tickers)[mask]):
        #    ax.annotate(ticker, (X_pca[mask, 0][j], X_pca[mask, 1][j]),
        #                fontsize=6, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"./output/{filename}", dpi=150, bbox_inches="tight")
    plt.close()


def plot_dendrogram(X_scaled: np.ndarray, labels: list):
    """
    Ward linkage dendrogram. Only drawn when sample count <= 300 to keep it legible.
    `labels` is either a list of ticker strings (stock-level) or empty strings (window-level).
    """
    if len(X_scaled) > 300:
        print("    [skip] Dendrogram skipped: >300 samples — too dense to read.")
        return
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 0.15), 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=7,
               color_threshold=0.7 * max(Z[:, 2]), ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig("./output/02_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_radar(profiles: pd.DataFrame, filename: str):
    """
    Radar (spider) chart comparing cluster mean indicator profiles.
    profiles: DataFrame(clusters × features) — already normalized for display.
    """
    features = profiles.columns.tolist()
    n_feat = len(features)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", n_colors=len(profiles))

    for i, (cluster_id, row) in enumerate(profiles.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {cluster_id}", color=palette[i])
        ax.fill(angles, values, alpha=0.12, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    ax.set_title("Cluster Indicator Profiles (Z-score normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=150, bbox_inches="tight")
    plt.close()


def plot_return_distributions(y_df: pd.DataFrame, labels: np.ndarray, method: str):
    """Box plot of return metrics by cluster."""
    df = y_df.copy()
    df["cluster"] = labels

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cols = {"ret_mean": "Mean Return", "ret_std": "Return Std", "ret_sharpe": "Sharpe Proxy"}
    for ax, (col, col_label) in zip(axes, cols.items()):
        df.boxplot(column=col, by="cluster", ax=ax, grid=False)
        ax.set_title(col_label)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(col_label)

    plt.suptitle(f"Return Distribution by Cluster ({method})")
    plt.tight_layout()
    plt.savefig(f"./output/05_return_dist_{method}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_heatmap(profiles_raw: pd.DataFrame, filename: str):
    """Heatmap of mean feature values per cluster."""
    fig, ax = plt.subplots(figsize=(16, max(4, len(profiles_raw) * 0.8)))
    sns.heatmap(profiles_raw.T, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title("Cluster Mean Feature Heatmap")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"./output/{filename}", dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_cluster_map(meta_df: pd.DataFrame, labels: np.ndarray, method: str):
    """
    Heatmap: rows = tickers, columns = window_end dates, colour = cluster label.
    Shows how each stock's regime evolves over time.
    """
    df = meta_df[["ticker", "window_end"]].copy()
    df["cluster"] = labels

    pivot = df.pivot_table(
        index="ticker", columns="window_end", values="cluster", aggfunc="first"
    )
    pivot.columns = pivot.columns.strftime("%Y-%m-%d")

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
    ax.set_title(f"Temporal Cluster Map — {method}\n(each cell = cluster label for that ticker × window)")
    ax.set_xlabel("Window End Date")
    ax.set_ylabel("Ticker")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"./output/06_temporal_map_{method}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ticker_cluster_dist(meta_df: pd.DataFrame, labels: np.ndarray, method: str):
    """
    Stacked bar chart: for each ticker, shows what fraction of its windows
    fell in each cluster.  Reveals which stocks are 'regime-stable' vs. 'migrators'.
    """
    df = meta_df[["ticker"]].copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]  # exclude noise

    counts = df.groupby(["ticker", "cluster"]).size().unstack(fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)

    n_clusters = fractions.shape[1]
    palette = sns.color_palette("tab10", n_colors=n_clusters)

    fig, ax = plt.subplots(figsize=(max(14, len(fractions) * 0.2), 6))
    fractions.plot(kind="bar", stacked=True, ax=ax, color=palette, width=0.85)
    ax.set_title(f"Per-Ticker Cluster Distribution — {method}")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Fraction of Windows")
    ax.legend(title="Cluster", loc="upper right", fontsize=8)
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"./output/07_ticker_cluster_dist_{method}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 7. Cross-sector summary
# ──────────────────────────────────────────────────────────────

def plot_cross_sector_summary(sector_profiles: dict, radar_features: list):
    """
    Compare cluster centroids across sectors.

    sector_profiles : {sector_name: profile_df(clusters × features)}
    Produces two plots:
      - Heatmap: each row = (sector, cluster), each col = feature
      - Bar chart: number of windows per sector per cluster
    """
    rows = []
    for sector, profile in sector_profiles.items():
        for cluster_id, row in profile.iterrows():
            if cluster_id == -1:
                continue
            entry = {"sector": sector, "cluster": f"C{cluster_id}",
                     "label": f"{sector}/C{cluster_id}"}
            for feat in radar_features:
                if feat in row.index:
                    entry[feat] = row[feat]
            entry["n_windows"] = row.get("n_windows", np.nan)
            rows.append(entry)

    if not rows:
        return

    summary = pd.DataFrame(rows).set_index("label")

    # ── Heatmap ────────────────────────────────────────────────
    feat_cols = [f for f in radar_features if f in summary.columns]
    heat_data = summary[feat_cols].astype(float)
    heat_norm = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(12, len(feat_cols) * 1.2),
                                    max(5, len(heat_norm) * 0.35)))
    sns.heatmap(heat_norm, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.5})
    ax.set_title("Cross-Sector Cluster Feature Heatmap (Z-score normalized)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sector / Cluster")
    plt.tight_layout()
    plt.savefig("./output/08_cross_sector_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Window count bar chart ─────────────────────────────────
    count_data = summary.groupby("sector")["cluster"].value_counts().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(max(10, len(count_data) * 1.2), 5))
    count_data.plot(kind="bar", stacked=True, ax=ax,
                    colormap="tab10", width=0.7)
    ax.set_title("Windows per Sector per Cluster")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Number of Windows")
    ax.legend(title="Cluster", loc="upper right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("./output/09_cross_sector_window_count.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("    Cross-sector heatmap → ./output/08_cross_sector_heatmap.png")
    print("    Cross-sector window count → ./output/09_cross_sector_window_count.png")


# ──────────────────────────────────────────────────────────────
# 8. Profiling
# ──────────────────────────────────────────────────────────────

def profile_clusters(feat_df: pd.DataFrame, y_df: pd.DataFrame,
                     labels: np.ndarray, method: str) -> pd.DataFrame:
    """
    Compute per-cluster mean of all features + return stats.
    Returns profile DataFrame(clusters × features).
    """
    df = feat_df.copy()
    df["ret_mean"]   = y_df["ret_mean"].values
    df["ret_std"]    = y_df["ret_std"].values
    df["ret_sharpe"] = y_df["ret_sharpe"].values
    df["cluster"] = labels
    profile = df.groupby("cluster").mean()

    print(f"\n    ── Cluster Profiles ({method}) ──")
    for cluster_id, row in profile.iterrows():
        n_windows = int((labels == cluster_id).sum())
        tag = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n    [{tag}]  n_windows={n_windows}")
        print(f"      Strength     : mean={row['str_mean']:.3f}  max={row['str_max']:.3f}  min={row['str_min']:.3f}")
        print(f"      Resilience   : mean={row['res_mean']:.3f}")
        print(f"      Disparity    : mean={row['disp_mean']:.3f}  max={row['disp_max']:.3f}  min={row['disp_min']:.3f}")
        print(f"      Signed_Disp  : mean={row['sd_mean']:.3f}  max={row['sd_max']:.3f}  min={row['sd_min']:.3f}")
        print(f"      S_Disp_Diff  : mean={row['sdd_mean']:.3f}  max={row['sdd_max']:.3f}  min={row['sdd_min']:.3f}")
        print(f"      Returns      : mean={row['ret_mean']:.4f}  std={row['ret_std']:.4f}  "
              f"sharpe={row['ret_sharpe']:.4f}")

    return profile


def print_ticker_distribution(meta_df: pd.DataFrame, labels: np.ndarray, method: str):
    """
    For each cluster, show which tickers appear most frequently.
    More useful than listing individual (ticker, window) pairs.
    """
    df = meta_df[["ticker"]].copy()
    df["cluster"] = labels
    print(f"\n    ── Top Tickers per Cluster ({method}) ──")
    for cluster_id in sorted(df["cluster"].unique()):
        tag = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        sub = df[df["cluster"] == cluster_id]
        top = sub["ticker"].value_counts().head(10)
        members = ", ".join([f"{t}({c})" for t, c in top.items()])
        print(f"    [{tag}]  {len(sub)} windows  —  top: {members}")


# ──────────────────────────────────────────────────────────────
# 8. Save Results
# ──────────────────────────────────────────────────────────────

def save_results(feat_df: pd.DataFrame, y_df: pd.DataFrame,
                 meta_df: pd.DataFrame, cluster_labels: dict,
                 profiles: dict, tag: str = "all"):
    """Save feature matrix, meta, cluster labels, and profiles to Excel."""
    safe_tag = tag.replace(" ", "_").replace("/", "_")
    output_path = f"./cluster_results_{safe_tag}.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        feat_df.to_excel(writer, sheet_name="Features")
        y_df.to_excel(writer, sheet_name="Returns")
        meta_df.to_excel(writer, sheet_name="Meta")

        label_df = pd.DataFrame(cluster_labels)
        label_df.to_excel(writer, sheet_name="Cluster_Labels")

        for method, profile in profiles.items():
            profile.to_excel(writer, sheet_name=f"Profile_{method[:6]}")

    print(f"    Results saved → {output_path}")


# ──────────────────────────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Intra-Trader: Microstructure Clustering Pipeline")
    parser.add_argument("--input", type=str, default="./price_microstructure_analysis.xlsx",
                        help="Path to input Excel file")
    parser.add_argument("--sector_csv", type=str, default="./sector.csv",
                        help="Path to sector.csv (Code, Sector). If provided, clusters within each sector independently.")
    parser.add_argument("--window", type=int, default=10,
                        help="Rolling window size in trading days (default: 20)")
    parser.add_argument("--stride", type=int, default=3,
                        help="Step size between windows in trading days (default: 5)")
    parser.add_argument("--model", type=str,
                        choices=["kmeans", "gmm", "agglomerative", "dbscan"],
                        default="gmm",
                        help="Primary clustering method (default: gmm)")
    parser.add_argument("--k_range", type=int, nargs=2, default=[4, 4],
                        help="Range of k to evaluate e.g. --k_range 2 10")
    parser.add_argument("--min_windows", type=int, default=10,
                        help="Minimum windows required to cluster a sector (default: 15)")
    parser.add_argument("--no_save", action="store_true",
                        help="Skip saving output files")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

RADAR_FEATURES = [
    #"str_mean", "str_max", "str_min",
    "ret_mean",
    "res_mean",
    "disp_mean", "disp_max", "disp_min",
    "sd_mean", "sd_max", "sd_min",
    "sdd_mean", "sdd_max", "sdd_min",
]

TRAIN_FEATURES = [
    #"str_mean", "str_max", "str_min",
    "res_mean",
    "sdd_mean", "sdd_max", "sdd_min",
]


def _run_one_sector(
    sector: str,
    feat_df: pd.DataFrame,
    y_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    X_scaled: np.ndarray,
    X_pca: np.ndarray,
    args,
) -> tuple:
    """
    Run the full clustering pipeline for one sector (or 'All').
    Returns (primary_labels_array, profile_df_for_primary_method).
    primary_labels_array is aligned to the rows of the inputs (already sector-filtered).
    """
    tag = sector.replace(" ", "_")
    os.makedirs(f"./output/{tag}", exist_ok=True)

    n_samples = len(feat_df)
    n_tickers = meta_df["ticker"].nunique()
    print(f"\n  {'─'*54}")
    print(f"  Sector: {sector}  ({n_samples} windows, {n_tickers} tickers)")
    print(f"  {'─'*54}")

    # Optimal k
    best_k = find_optimal_k(X_scaled, k_range_list=args.k_range, model_name=args.model)
    # Save silhouette plot to sector subfolder
    if os.path.exists("./output/01_optimal_k.png"):
        os.replace("./output/01_optimal_k.png", f"./output/{tag}/01_optimal_k.png")

    # Dendrogram (auto-skips if >300 samples)
    plot_dendrogram(X_scaled, labels=[""] * n_samples)
    if os.path.exists("./output/02_dendrogram.png"):
        os.replace("./output/02_dendrogram.png", f"./output/{tag}/02_dendrogram.png")

    # Run all clusterings
    cluster_labels_dict = run_all_clusterings(X_scaled, n_clusters=best_k, model_name=args.model)
    primary_labels = cluster_labels_dict[args.model]

    # Profile
    profiles = {}
    profiles[args.model] = profile_clusters(feat_df, y_df, cluster_labels_dict[args.model], args.model)
    print_ticker_distribution(meta_df, cluster_labels_dict[args.model], args.model)

    profiles = {}
    for method, labels in cluster_labels_dict.items():
        profiles[method] = profile_clusters(feat_df, y_df, labels, method)
        print_ticker_distribution(meta_df, labels, method)

    # PCA scatter
    for method, labels in cluster_labels_dict.items():
        plot_pca_scatter(
            X_pca, labels,
            tickers=meta_df["ticker"].tolist(),
            title=f"PCA — {method} [{sector}] (k={best_k}, w={args.window}d)",
            filename=f"{tag}/03_pca_{method}.png",
        )
    print(f"    PCA scatter → ./output/{tag}/03_pca_*.png")

    # Radar + heatmap
    valid_profile = profiles[args.model][profiles[args.model].index != -1]
    if not valid_profile.empty and len(valid_profile) >= 2:
        radar_data = valid_profile[RADAR_FEATURES]
        radar_norm = (radar_data - radar_data.mean()) / (radar_data.std() + 1e-9)
        plot_cluster_radar(radar_norm, f"{tag}/04_radar_{args.model}.png")
        plot_cluster_heatmap(radar_data, f"{tag}/04b_heatmap_{args.model}.png")
        print(f"    Radar/Heatmap → ./output/{tag}/04_radar_*.png")

    # Return distributions
    plot_return_distributions(y_df, primary_labels, f"{tag}_{args.model}")
    if os.path.exists(f"./output/05_return_dist_{tag}_{args.model}.png"):
        os.replace(
            f"./output/05_return_dist_{tag}_{args.model}.png",
            f"./output/{tag}/05_return_dist.png",
        )
    print(f"    Return dist → ./output/{tag}/05_return_dist.png")

    # Temporal cluster map
    plot_temporal_cluster_map(meta_df, primary_labels, tag)
    src_temporal = f"./output/06_temporal_map_{tag}.png"
    if os.path.exists(src_temporal):
        os.replace(src_temporal, f"./output/{tag}/06_temporal_map.png")
    print(f"    Temporal map → ./output/{tag}/06_temporal_map.png")

    # Per-ticker cluster distribution
    plot_ticker_cluster_dist(meta_df, primary_labels, f"{tag}_{args.model}")
    if os.path.exists(f"./output/07_ticker_cluster_dist_{tag}_{args.model}.png"):
        os.replace(
            f"./output/07_ticker_cluster_dist_{tag}_{args.model}.png",
            f"./output/{tag}/07_ticker_cluster_dist.png",
        )
    print(f"    Ticker dist → ./output/{tag}/07_ticker_cluster_dist.png")

    # Save Excel
    if not args.no_save:
        save_results(feat_df, y_df, meta_df, cluster_labels_dict, profiles, tag=tag)

    return primary_labels, profiles[args.model]


def main():
    args = parse_args()
    setup()

    print("=" * 60)
    print("  Intra-Trader: Microstructure Clustering Pipeline")
    print(f"  window={args.window}  stride={args.stride}  model={args.model}")
    mode = f"sector ({args.sector_csv})" if args.sector_csv else "global"
    print(f"  mode={mode}")
    print("=" * 60)

    # ── 1. Load indicator data ─────────────────────────────────
    sheets = load_data(args.input)

    # ── 2. Build windowed feature matrix ──────────────────────
    feat_df, y_df, meta_df = build_windowed_feature_matrix(
        sheets, window=args.window, stride=args.stride
    )

    # ── 3. Attach sector labels ───────────────────────────────
    if args.sector_csv == 'all':
        meta_df['sector'] = "All"
    elif args.sector_csv:
        print(f"\n[Sector] Loading {args.sector_csv}...")
        sector_map = load_sector(args.sector_csv)
        meta_df["sector"] = meta_df["ticker"].map(sector_map).fillna("Unknown")
    else:
        meta_df["sector"] = "All"

    # ── 4. Scale ALL features once (consistent across sectors) ─
    feat_train_df = feat_df[TRAIN_FEATURES]
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(feat_train_df.values)
    X_pca_all, _ = reduce_dimensions(X_scaled_all, n_components=2)

    # ── 5. Loop over sectors ──────────────────────────────────
    sectors = sorted(meta_df["sector"].unique())
    sector_profiles = {}   # {sector: profile_df} for cross-sector summary

    for sector in sectors:
        mask = (meta_df["sector"] == sector).values
        n_windows = int(mask.sum())

        if n_windows < args.min_windows:
            print(f"\n  [skip] {sector}: {n_windows} windows < min_windows={args.min_windows}")
            continue

        primary_labels, profile = _run_one_sector(
            sector=sector,
            feat_df=feat_df[mask].reset_index(drop=True),
            y_df=y_df[mask].reset_index(drop=True),
            meta_df=meta_df[mask].reset_index(drop=True),
            X_scaled=X_scaled_all[mask],
            X_pca=X_pca_all[mask],
            args=args,
        )
        sector_profiles[sector] = profile

    # ── 6. Cross-sector summary (only when multiple sectors) ──
    if len(sector_profiles) > 1:
        print("\n[Cross-sector summary]")
        plot_cross_sector_summary(sector_profiles, RADAR_FEATURES)

    print("\n" + "=" * 60)
    print(f"  Done. {len(sector_profiles)} sector(s) clustered.")
    print("  Figures → ./output/<sector>/")
    print("  Excel   → ./cluster_results_<sector>.xlsx")
    print("=" * 60)


if __name__ == "__main__":
    main()