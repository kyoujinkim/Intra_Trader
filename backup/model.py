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


def main():
    input = "./price_microstructure_analysis.xlsx"
    window, stride, model, k, sector_csv = 5, 5, "gmm", 4, "all"

    # ── 1. Load indicator data ─────────────────────────────────
    sheets = load_data(input)

    # ── 2. Build windowed feature matrix ──────────────────────
    feat_df, y_df, meta_df = build_windowed_feature_matrix(
        sheets, window=window, stride=stride
    )

    # ── 3. Attach sector labels ───────────────────────────────
    meta_df['sector'] = "All"

    # ── 4. Scale ALL features once (consistent across sectors) ─
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(feat_df.values)

    # ── 5. Fit GMM on ALL data to get consistent clusters across sectors ─
    print(f"[3/6] Clustering with {model.upper()} (k={k})...")
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X_scaled_all)

    # 6. Predict cluster labels for ALL samples using the fitted model ─
    # Get all
    print(f"[4/6] Standardize All feature data ...")
    feat_df_all, y_df, meta_df = build_windowed_feature_matrix(
        sheets, window=window, stride=1
    )
    X_scaled_all = scaler.transform(feat_df_all.values)

    # Predict cluster labels for ALL samples using the fitted model
    print(f"[5/6] Predict cluster of All datas...")
    result = gmm.predict(X_scaled_all)

    print(f"[6/6] Saving results to total_stock_cluster.csv...")
    meta_df['cluster'] = result
    total_data = pd.concat([meta_df, feat_df_all], axis=1)
    total_data.to_csv('total_stock_cluster.csv')

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()