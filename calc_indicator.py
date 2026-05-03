import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from dtaidistance import dtw

from config import DATA_DIR, INDICATOR_XLSX, ROLLING_WINDOW, MARKET_HOURS

pd.set_option('mode.chained_assignment', None)  # 경고 off


# ──────────────────────────────────────────────────────────────
# Indicator functions  (unchanged logic)
# ──────────────────────────────────────────────────────────────

def calculate_price_microstructure(df, window=ROLLING_WINDOW):
    """df: ['bid_price', 'ask_price', 'last_price'] 시계열 데이터"""
    df['mid_price']   = (df['ask_price'] + df['bid_price']) / 2
    df['spread']      = df['ask_price'] - df['bid_price']
    df['mid_returns'] = df['mid_price'].pct_change()

    df['pressure'] = (df['last_price'] - df['bid_price']) / (df['spread'] + 1e-9)

    df['strength_score'] = df['pressure'].rolling(window=window).mean()
    df['resilience']     = (df['spread'] / df['mid_price'] * 100).rolling(window=window).std()

    intra_vol  = df['mid_returns'].rolling(window=window).std() * np.sqrt(12 * 9)
    intra_ret = df['mid_price'] / df['mid_price'].shift(window).bfill() - 1
    signed = intra_vol * np.sign(intra_ret + 1e-9)   # 0인 경우 양수로 처리
    signed_vol = signed.rolling(window=window).mean()

    df['intra_vol']  = intra_vol
    df['signed_vol'] = signed_vol

    df = df.between_time(MARKET_HOURS[0], MARKET_HOURS[1])
    return df.dropna()


def get_dtw_correlation(path_a, path_b, window=10):
    a = (path_a - path_a.mean()) / (path_a.std() + 1e-9)
    b = (path_b - path_b.mean()) / (path_b.std() + 1e-9)

    dist_pos = dtw.distance(a, b,   window=window)
    dist_neg = dtw.distance(a, -b,  window=window)

    return (dist_neg - dist_pos) / (dist_neg + dist_pos + 1e-9)


# ──────────────────────────────────────────────────────────────
# Worker  (must be top-level for multiprocessing on Windows)
# ──────────────────────────────────────────────────────────────

def process_ticker_vol(u: str):
    """
    Process one ticker end-to-end.
    Returns (ticker, str_series, res_series, disp_series,
             signed_disp_series, coil_series, ret_series)
    or None if data is missing / an error occurs.
    """
    df = pd.read_parquet(f'{DATA_DIR}/{u}.parquet', engine='pyarrow')
    if len(df) == 0 or len(df.columns) < 3:
        print(f"{u}: 데이터 부족으로 분석 불가")
        return None

    df = (df[['BID', 'ASK', 'TRDPRC_1']]
          .rename(columns={'BID': 'bid_price', 'ASK': 'ask_price', 'TRDPRC_1': 'last_price'}))
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
    df['ymd'] = df.index.date

    # ffill within same day only
    for col in ('bid_price', 'ask_price', 'last_price'):
        df[col] = df.groupby('ymd')[col].ffill()
    df = df.dropna(subset=['bid_price', 'ask_price', 'last_price'])

    df_ind = calculate_price_microstructure(df)

    # U-Curve reference (median intraday pattern)
    df_ind['hm'] = (df_ind.index.hour.astype(str).str.zfill(2) +
                    df_ind.index.minute.astype(str).str.zfill(2))

    df_curve_ymd_hm = df_ind.groupby(['ymd', 'hm'])['signed_vol'].median().unstack()

    return {u:df_curve_ymd_hm}

def process_ticker(u: str):
    """
    Process one ticker end-to-end.
    Returns (ticker, str_series, res_series, disp_series,
             signed_disp_series, coil_series, ret_series)
    or None if data is missing / an error occurs.
    """
    try:
        df = pd.read_parquet(f'{DATA_DIR}/{u}.parquet', engine='pyarrow')
        if len(df) == 0 or len(df.columns) < 3:
            print(f"{u}: 데이터 부족으로 분석 불가")
            return None

        df = (df[['BID', 'ASK', 'TRDPRC_1']]
              .rename(columns={'BID': 'bid_price', 'ASK': 'ask_price', 'TRDPRC_1': 'last_price'}))
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
        df['ymd'] = df.index.date

        # ffill within same day only
        for col in ('bid_price', 'ask_price', 'last_price'):
            df[col] = df.groupby('ymd')[col].ffill()
        df = df.dropna(subset=['bid_price', 'ask_price', 'last_price'])

        df_ind = calculate_price_microstructure(df)

        # U-Curve reference (median intraday pattern)
        df_ind['hm'] = (df_ind.index.hour.astype(str).str.zfill(2) +
                        df_ind.index.minute.astype(str).str.zfill(2))

        last_price  = df_ind.groupby('ymd')['mid_price'].last()
        pct_chg = last_price.pct_change().fillna(0)
        pct_chg_rolling = pct_chg.rolling(window=3).mean().fillna(0).shift(-3)

        pct_chg_shift = pct_chg.shift(1).fillna(0)
        df_ind['day'] = df_ind.index.date
        df_ind['prev_pct_chg'] = df_ind['day'].map(pct_chg_shift)
        df_ind['prev_pct_chg'] = np.where(df_ind['prev_pct_chg'] > 0.01, 1,
                                          np.where(df_ind['prev_pct_chg'] < -0.01, -1, 0))
        df_ucurve_group_down = df_ind[df_ind['prev_pct_chg'] == -1].groupby('hm')['intra_vol'].median()
        df_ucurve_group_neutral = df_ind[df_ind['prev_pct_chg'] == 0].groupby('hm')['intra_vol'].median()
        df_ucurve_group_up = df_ind[df_ind['prev_pct_chg'] == 1].groupby('hm')['intra_vol'].median()
        df_sucurve_group_down = df_ind[df_ind['prev_pct_chg'] == -1].groupby('hm')['signed_vol'].median()
        df_sucurve_group_neutral = df_ind[df_ind['prev_pct_chg'] == 0].groupby('hm')['signed_vol'].median()
        df_sucurve_group_up = df_ind[df_ind['prev_pct_chg'] == 1].groupby('hm')['signed_vol'].median()
        df_ucurves = {
            -1: df_ucurve_group_down,
             0: df_ucurve_group_neutral,
             1: df_ucurve_group_up,
        }
        df_sucurves = {
            -1: df_sucurve_group_down,
             0: df_sucurve_group_neutral,
             1: df_sucurve_group_up,
        }

        disp_result = []
        signed_disp_result = []
        signed_disp_diff_result = []
        for group in [-1, 0, 1]:
            disp = df_ind[df_ind['prev_pct_chg'] == group].groupby('ymd')['intra_vol'].apply(lambda x: dtw.distance(x.values, df_ucurves[group].values))
            signed_disp = df_ind[df_ind['prev_pct_chg'] == group].groupby('ymd')['signed_vol'].apply(lambda x: dtw.distance(x.values, df_sucurves[group].values))
            signed_disp_diff = df_ind[df_ind['prev_pct_chg'] == group].groupby('ymd')['signed_vol'].apply(lambda x: x.values.sum() - df_sucurves[group].sum())
            disp_result.append(disp)
            signed_disp_result.append(signed_disp)
            signed_disp_diff_result.append(signed_disp_diff)

        disp_result_df = pd.concat(disp_result)
        signed_disp_result_df = pd.concat(signed_disp_result)
        signed_disp_diff_result_df = pd.concat(signed_disp_diff_result)

        # Daily aggregation
        df_daily = df_ind.groupby('ymd').agg({
            'strength_score': 'mean',
            'resilience':     'mean',
            'intra_vol':      'mean',
            'signed_vol':     'mean',
        })
        df_daily['disparity']        = disp_result_df
        df_daily['signed_disparity'] = signed_disp_result_df
        df_daily['signed_disp_diff'] = signed_disp_diff_result_df
        df_daily['rolling_return']   = pct_chg_rolling
        df_daily['coil_release']     = (
            (1 - df_daily['strength_score'])
            * df_daily['disparity']
            * np.sign(df_daily['signed_disparity'])
        )

        return (
            u,
            df_daily['strength_score'],
            df_daily['resilience'],
            df_daily['disparity'],
            df_daily['signed_disparity'],
            df_daily['signed_disp_diff'],
            df_daily['coil_release'],
            df_daily['rolling_return'],
            df_daily['intra_vol']
        )

    except Exception as e:
        print(f"{u}: 오류 발생 — {e}")
        return None


def run(universe: list):
    results = []
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_ticker, universe),
            total=len(universe),
            desc='Tickers',
        ):
            if result is not None:
                results.append(result)

    # Restore original universe order
    order = {u: i for i, u in enumerate(universe)}
    results.sort(key=lambda r: order[r[0]])

    codes         = [r[0] for r in results]
    strength      = [r[1] for r in results]
    resilience    = [r[2] for r in results]
    disparity     = [r[3] for r in results]
    signed_disp   = [r[4] for r in results]
    s_disp_dff    = [r[5] for r in results]
    coil_release  = [r[6] for r in results]
    returns       = [r[7] for r in results]
    intra_vol     = [r[8] for r in results]

    def make_df(series_list, codes):
        df = pd.concat(series_list, axis=1)
        df.columns = codes
        return df

    str_df          = make_df(strength,     codes)
    res_df          = make_df(resilience,   codes)
    disp_df         = make_df(disparity,    codes)
    signed_disp_df  = make_df(signed_disp,  codes)
    s_disp_diff_df  = make_df(s_disp_dff, codes)
    coil_release_df = make_df(coil_release, codes)
    ret_df          = make_df(returns,      codes)
    # get relative returns by subtracting mean return of all tickers on the same day
    ret_df = ret_df.subtract(ret_df.mean(axis=1), axis=0)
    intra_vol_df = make_df(intra_vol, codes)

    total_mean_df = pd.DataFrame({
        'strength':         str_df.median(axis=1),
        'resilience':       res_df.median(axis=1),
        'disparity':        disp_df.median(axis=1),
        'signed_disparity': signed_disp_df.median(axis=1),
        'signed_disp_diff': s_disp_diff_df.median(axis=1),
        'coil_release':     coil_release_df.median(axis=1),
        'returns':          ret_df.median(axis=1),
        'intra_vol':        intra_vol_df.median(axis=1),
    })

    print("Saving to Excel...")
    with pd.ExcelWriter(INDICATOR_XLSX) as writer:
        str_df.to_excel(writer,          sheet_name='Strength_Score')
        res_df.to_excel(writer,          sheet_name='Resilience')
        disp_df.to_excel(writer,         sheet_name='Disparity')
        signed_disp_df.to_excel(writer,  sheet_name='Signed_Disparity')
        s_disp_diff_df.to_excel(writer,  sheet_name='Signed_Disp_Diff')
        coil_release_df.to_excel(writer, sheet_name='Coil_Release')
        ret_df.to_excel(writer,          sheet_name='Returns')
        intra_vol_df.to_excel(writer,     sheet_name='Intra_Vol')
        total_mean_df.to_excel(writer,   sheet_name='Total_Mean')

    print(f"Done. {len(codes)}/{len(universe)} tickers processed successfully.")

def run_vol(universe):
    results = []
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_ticker_vol, universe),
            total=len(universe),
            desc='Tickers',
        ):
            if result is not None:
                results.append(result)

    r_dict = {}
    for r in results:
        ticker = list(r.keys())[0]
        r_dict[ticker] = r[ticker]

    start_dt = pd.to_datetime('2026-03-01')
    r_df = []
    for ticker, vol in r_dict.items():
        vol.index = pd.to_datetime(vol.index)
        r_df.append(vol.loc[start_dt:])
    r_df = pd.concat(r_df, axis=0)
    r_df_daily = r_df.groupby(r_df.index.date).median()

    a = 1

# ──────────────────────────────────────────────────────────────
# Main  (Windows requires the if __name__ guard for mp.Pool)
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    universe = pd.read_csv('./top100.csv')['instrument'].tolist()

    # Leave one core free for the main process and OS
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Processing {len(universe)} tickers with {n_workers} workers...")

    run(universe)
    #run_vol(universe)