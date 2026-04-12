# Intra-Trader: Price Microstructure Analysis & Equity Clustering

## Project Goal

Identify characteristic behavioral patterns of KOSPI equities by analyzing intraday bid-ask price microstructure. The current direction shifts from a supervised prediction approach (Random Forest) toward unsupervised clustering to discover natural groupings in stock behavior and extract actionable insights from those clusters.

---

## Architecture Overview

```
Refinitiv API (5-min bars)
    └── load_rd.py
            │
            ▼
    data/KR7*.parquet  (BID, ASK, TRDPRC_1 — 100 KOSPI stocks)
            │
            ▼
    calc_indicator.py
            │
            ▼
    price_microstructure_analysis.xlsx
    ┌───────────────────────────────┐
    │  Sheets:                      │
    │  · Strength_Score             │
    │  · Resilience                 │
    │  · Disparity                  │
    │  · Signed_Disparity           │
    │  · Coil_Release               │
    │  · Returns                    │
    │  · Total_Mean                 │
    └───────────────────────────────┘
            │
            ▼
    [NEXT] clustering.py
            │
            ▼
    Cluster profiles + insight
```

---

## Indicators (calc_indicator.py)

Each indicator is computed per stock per trading day using 5-minute intraday bars.

| Indicator | Description |
|---|---|
| `mid_price` | (BID + ASK) / 2 |
| `spread` | ASK − BID |
| `pressure` | Position of last trade within spread: (LAST − BID) / spread |
| `strength_score` | Rolling 12-bar mean of pressure (0 = aggressive selling, 1 = aggressive buying) |
| `resilience` | Rolling std of spread — how stable/unstable the spread is |
| `intra_vol` | Annualized intraday volatility from mid-price returns |
| `signed_vol` | Directional volatility (sign-weighted) |
| `disparity` | DTW distance between stock's intraday path and the U-curve archetype |
| `signed_disparity` | Direction of the deviation from U-curve |
| `coil_release` | `(1 − strength) × disparity × sign(signed_disparity)` — captures low-strength divergence from U-curve |

### U-Curve Concept

KOSPI intraday volume and volatility typically follow a U-shaped pattern (high at open/close, low midday). `disparity` measures how far a stock deviates from this expected intraday rhythm using Dynamic Time Warping (DTW). High disparity + low strength = "coil" state; the hypothesis is that this precedes directional moves.

---

## Why Random Forest Underperformed

- The target (top 30% 5-day forward return) is noisy and weakly related to microstructure features alone.
- RF treats each row independently — it cannot capture the sequential/regime nature of microstructure states.
- Cross-stock heterogeneity: KOSPI 100 spans wildly different sectors and liquidity regimes. A single model cannot handle all of them homogeneously.
- Weak label signal: short-window forward returns are dominated by macro noise, not stock-specific microstructure.

---

## New Direction: Clustering

Instead of predicting returns directly, first **cluster stocks by microstructure behavior** to identify structurally similar groups, then analyze return/risk properties within each cluster.

### Step 1 — Feature Engineering for Clustering

Aggregate indicator time series into stock-level feature vectors. Candidate features:

- Mean, std, skewness of `strength_score` (distribution of buy/sell pressure)
- Mean, std of `resilience` (spread stability profile)
- Mean, std of `disparity` (how often/strongly the stock diverges from U-curve)
- Autocorrelation of `coil_release` (persistence of coil states)
- Frequency of high `coil_release` events (> threshold)
- Intraday return asymmetry: morning vs afternoon `pressure` difference
- Strength mean-reversion speed (AR(1) coefficient of `strength_score`)

### Step 2 — Dimensionality Reduction

- Apply PCA or UMAP on the feature matrix (100 stocks × N features) to reduce noise and visualize structure.
- Use explained variance to select components that capture dominant behavioral axes.

### Step 3 — Clustering Algorithms

Try multiple approaches and compare:

| Method | Rationale |
|---|---|
| K-Means | Baseline; fast; works well if clusters are roughly spherical |
| Agglomerative (Ward) | Hierarchical; reveals multi-scale structure; dendrogram useful for insight |
| DBSCAN | Finds density-based clusters; naturally handles outliers (illiquid stocks) |
| GMM | Soft assignment; useful if stocks have mixed behaviors |

Evaluate with: silhouette score, Davies-Bouldin index, and intra-cluster return variance.

### Step 4 — Cluster Profiling

For each cluster, compute:
- Mean indicator profiles (radar/spider chart)
- Distribution of 5-day and 20-day forward returns
- Sector composition (e.g., tech, financials, industrials)
- Liquidity characteristics (spread level, intra_vol)
- Typical `coil_release` trajectory around signal events

### Step 5 — Insight Extraction

Target questions:
1. Do stocks that frequently enter "coil" states (low strength + high disparity) have higher subsequent return variance?
2. Are there stocks that consistently follow the U-curve vs. persistently deviate?
3. Does cluster membership persist over time, or do stocks migrate between clusters?
4. Which cluster profile is most predictive of directional moves when `coil_release` spikes?

---

## Planned Files

| File | Purpose |
|---|---|
| `load_rd.py` | Fetch 5-min intraday data from Refinitiv → parquet |
| `calc_indicator.py` | Compute microstructure indicators → Excel |
| `clustering.py` | Feature engineering, clustering, cluster profiling |
| `visualize.py` | Radar charts, UMAP scatter, return distribution by cluster |
| `prediction_algo.py` | (legacy) RF model — may be retrained cluster-conditionally |

---

## Data

- **Source:** Refinitiv (LSEG) real-time desktop session
- **Universe:** KOSPI top 100 by market cap / volume (see `backup/top100.csv`)
- **Granularity:** 5-minute bars, Seoul timezone (KST)
- **Period:** March 2025 – ongoing
- **Storage:** `data/KR7{ISIN}.parquet` per stock

---

## Environment

```
python >= 3.10
pandas, numpy
scikit-learn
dtaidistance
umap-learn
matplotlib, seaborn
openpyxl
refinitiv-data  (requires Refinitiv Workspace desktop session)
```

---

## Current Status

| Component | Status |
|---|---|
| Data pipeline (load_rd.py) | Done |
| Indicator calculation (calc_indicator.py) | Done |
| Excel output | Done |
| Random Forest (prediction_algo.py) | Done — weak performance |
| **Clustering pipeline** | **Planned — next priority** |
| Cluster visualization | Planned |
| Insight report | Planned |