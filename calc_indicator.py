import argparse
import multiprocessing as mp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from dtaidistance import dtw

from src.config import DATA_DIR, TICK_DIR, OUTPUT_DIR, ROLLING_WINDOW, MARKET_HOURS

pd.set_option('mode.chained_assignment', None)


class Indicator_Processor:
    def __init__(self, country, days=None):
        self.universe = pd.read_csv(f"{DATA_DIR}/{country}/universe.csv", encoding='CP949')['Type'].tolist()
        self.sector = pd.read_csv(f"{DATA_DIR}/{country}/sector.csv")
        self.config = {
            "TZ": MARKET_HOURS[country][0],
            "START_MH": MARKET_HOURS[country][1],
            "END_MH": MARKET_HOURS[country][2],
            "COUNTRY": country,
            "DAYS": days,
        }


    # ──────────────────────────────────────────────────────────────
    # Indicator functions
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_price_microstructure(df, window=ROLLING_WINDOW):
        df['mid_price']   = (df['ask_price'] + df['bid_price']) / 2
        df['spread']      = df['ask_price'] - df['bid_price']
        df['mid_returns'] = df['mid_price'].pct_change()

        df['pressure'] = (df['last_price'] - df['bid_price']) / (df['spread'] + 1e-9)

        df['strength_score'] = df['pressure'].rolling(window=window).mean()
        df['resilience']     = (df['spread'] / df['mid_price'] * 100).rolling(window=window).std()

        intra_vol  = df['mid_returns'].rolling(window=window).std() * np.sqrt(12 * 9)
        intra_ret  = df['mid_price'] / df['mid_price'].shift(window).bfill() - 1
        signed     = intra_vol * np.sign(intra_ret + 1e-9)   # treat zero as positive
        signed_vol = signed.rolling(window=window).mean()

        df['intra_vol']  = intra_vol
        df['signed_vol'] = signed_vol

        return df

    # ──────────────────────────────────────────────────────────────
    # Shared data-loading helper
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _load_ticker(u: str, config:dict):
        """Load parquet, clean, compute microstructure indicators, and add hm column."""
        df = pd.read_parquet(f'{TICK_DIR}/{config["COUNTRY"]}/{u}.parquet', engine='pyarrow')
        if len(df) == 0 or len(df.columns) < 3:
            print(f"{u}: 데이터 부족으로 분석 불가")
            return None

        df = (df[['BID', 'ASK', 'TRDPRC_1']]
              .rename(columns={'BID': 'bid_price', 'ASK': 'ask_price', 'TRDPRC_1': 'last_price'}))
        df.index = df.index.tz_localize('UTC').tz_convert(config['TZ'])
        df['ymd'] = df.index.date

        if config['DAYS'] != -1:
            df = df[df.index > df.index[-1] - relativedelta(days=config['DAYS'])]

        # value 0 in to na
        df = df.replace(0, pd.NA)

        df = df.ffill().dropna(subset=['bid_price', 'ask_price', 'last_price'])
        df = df[~df.index.duplicated(keep='last')]

        df_ind = Indicator_Processor.calculate_price_microstructure(df)

        df_ind = df_ind.between_time(config["START_MH"], config["END_MH"])

        df_ind['hm'] = df_ind.index.strftime('%H%M')
        return df_ind


    # ──────────────────────────────────────────────────────────────
    # Workers  (must be top-level for multiprocessing on Windows)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def process_ticker_vol(u: str, config:dict):
        try:
            df_ind = Indicator_Processor._load_ticker(u, config)
            if df_ind is None:
                return None

            df_curve_ymd_hm = df_ind.groupby(['ymd', 'hm'])['signed_vol'].median()
            return (u, df_curve_ymd_hm)

        except Exception as e:
            print(f"{u}: 오류 발생 — {e}")
            return None

    @staticmethod
    def process_ticker(u: str, config:dict):
        try:
            df_ind = Indicator_Processor._load_ticker(u, config)
            if df_ind is None:
                return None

            last_price       = df_ind.groupby('ymd')['mid_price'].last()
            pct_chg          = last_price.pct_change().fillna(0)
            pct_chg_rolling  = pct_chg.rolling(window=3).mean().fillna(0).shift(-3)

            pct_chg_shift = pct_chg.shift(1).fillna(0)
            df_ind['day'] = df_ind.index.date
            df_ind['prev_pct_chg'] = df_ind['day'].map(pct_chg_shift)
            df_ind['prev_pct_chg'] = np.where(df_ind['prev_pct_chg'] > 0.01, 1,
                                              np.where(df_ind['prev_pct_chg'] < -0.01, -1, 0))

            df_ucurves  = {g: df_ind[df_ind['prev_pct_chg'] == g].groupby('hm')['intra_vol'].median()  for g in (-1, 0, 1)}
            df_sucurves = {g: df_ind[df_ind['prev_pct_chg'] == g].groupby('hm')['signed_vol'].median() for g in (-1, 0, 1)}

            disp_result, signed_disp_result, signed_disp_diff_result = [], [], []
            for group in (-1, 0, 1):
                mask = df_ind['prev_pct_chg'] == group
                disp_result.append(
                    df_ind[mask].groupby('ymd')['intra_vol'].apply(
                        lambda x: dtw.distance(x.values, df_ucurves[group].values)))
                signed_disp_result.append(
                    df_ind[mask].groupby('ymd')['signed_vol'].apply(
                        lambda x: dtw.distance(x.values, df_sucurves[group].values)))
                signed_disp_diff_result.append(
                    df_ind[mask].groupby('ymd')['signed_vol'].apply(
                        lambda x: x.values.sum() - df_sucurves[group].sum()))

            disp_result_df           = pd.concat(disp_result)
            signed_disp_result_df    = pd.concat(signed_disp_result)
            signed_disp_diff_result_df = pd.concat(signed_disp_diff_result)

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

            return (
                u,
                df_daily['strength_score'],
                df_daily['resilience'],
                df_daily['disparity'],
                df_daily['signed_disparity'],
                df_daily['signed_disp_diff'],
                df_daily['rolling_return'],
                df_daily['intra_vol'],
            )

        except Exception as e:
            print(f"{u}: 오류 발생 — {e}")
            return None


    def run(self, n_workers: int = max(1, mp.cpu_count() - 1)):
        results = []
        # create config fixed process_ticker function
        partial_fun = partial(self.process_ticker, config=self.config)
        with mp.Pool(processes=n_workers) as pool:
            for result in tqdm(pool.imap_unordered(partial_fun, self.universe),
                               total=len(self.universe), desc='Tickers'):
                if result is not None:
                    results.append(result)

        order = {u: i for i, u in enumerate(self.universe)}
        results.sort(key=lambda r: order[r[0]])

        codes, strength, resilience, disparity, signed_disp, s_disp_diff, returns, intra_vol = zip(*results)
        codes = list(codes)

        def make_df(series_list):
            df = pd.concat(series_list, axis=1)
            df.columns = codes
            return df

        str_df          = make_df(strength)
        res_df          = make_df(resilience)
        disp_df         = make_df(disparity)
        signed_disp_df  = make_df(signed_disp)
        s_disp_diff_df  = make_df(s_disp_diff)
        ret_df          = make_df(returns)
        ret_df          = ret_df.subtract(ret_df.mean(axis=1), axis=0)
        intra_vol_df    = make_df(intra_vol)

        total_mean_df = pd.DataFrame({
            'strength':         str_df.median(axis=1),
            'resilience':       res_df.median(axis=1),
            'disparity':        disp_df.median(axis=1),
            'signed_disparity': signed_disp_df.median(axis=1),
            'signed_disp_diff': s_disp_diff_df.median(axis=1),
            'returns':          ret_df.median(axis=1),
            'intra_vol':        intra_vol_df.median(axis=1),
        })

        print("Saving to Excel...")
        with pd.ExcelWriter(f"{OUTPUT_DIR}/{self.config['COUNTRY']}/indicators.xlsx") as writer:
            str_df.to_excel(writer,          sheet_name='Strength_Score')
            res_df.to_excel(writer,          sheet_name='Resilience')
            disp_df.to_excel(writer,         sheet_name='Disparity')
            signed_disp_df.to_excel(writer,  sheet_name='Signed_Disparity')
            s_disp_diff_df.to_excel(writer,  sheet_name='Signed_Disp_Diff')
            ret_df.to_excel(writer,          sheet_name='Returns')
            intra_vol_df.to_excel(writer,    sheet_name='Intra_Vol')
            total_mean_df.to_excel(writer,   sheet_name='Total_Mean')

        print(f"Done. {len(codes)}/{len(self.universe)} tickers processed successfully.")

        return True


    def run_vol(self, n_workers: int = max(1, mp.cpu_count() - 1)):
        results = []
        partial_fun = partial(self.process_ticker_vol, config=self.config)
        with mp.Pool(processes=n_workers) as pool:
            for result in tqdm(pool.imap_unordered(partial_fun, self.universe),
                               total=len(self.universe), desc='Tickers'):
                if result is not None:
                    results.append(result)

        r_df = []
        for ticker, vol in results:
            vol.name = ticker
            r_df.append(vol)
        r_df = pd.concat(r_df, axis=1)

        def group_by_sector(df, sector):

            if sector == -1:
                r_df_s = df.median(axis=1)
            else:
                r_df_t = df.T
                r_df_t[sector] = r_df_t.index.map(
                    lambda x: sector + '_' + self.sector[self.sector['Code'] == x][sector].values[0] if x in self.sector[
                        'Code'].values else 'Unknown')

                r_df_s = r_df_t.groupby(sector).median().T

            return r_df_s

        r_df_sector = pd.concat([group_by_sector(r_df,-1), group_by_sector(r_df,'Sector'), group_by_sector(r_df,'IndustryGroup'), group_by_sector(r_df,'Industry')], axis=1)

        print("Saving to Excel...")
        with pd.ExcelWriter(f"{OUTPUT_DIR}/{self.config['COUNTRY']}/vol_curves.xlsx") as writer:
            r_df.sort_index().to_excel(writer,          sheet_name='Equity Vols')
            r_df_sector.sort_index().to_excel(writer,   sheet_name='Sector Vols')

        print(f"Done. {len(self.universe)} tickers processed successfully.")

        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--country',default='us',help='Country Abbreviation', type=str, choices=['kr','us'])
    parser.add_argument('-d','--days', default=-1, help='days to calculate median vol', type=int, )

    return parser.parse_args()

# ──────────────────────────────────────────────────────────────
# Main  (Windows requires the if __name__ guard for mp.Pool)
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    n_workers = max(1, mp.cpu_count() - 1)

    calc = Indicator_Processor(country=args.country, days=args.days)
    calc.run()
    calc.run_vol()
