import argparse
import concurrent
import os
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser

import pandas as pd
import refinitiv.data as rd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import datetime as dt

from src.config import DATA_DIR, TICK_DIR, DATA_FIELD

pd.set_option('future.no_silent_downcasting', True)

def _fetch(t, c, u, r):
    path = f'{TICK_DIR}/{c}/{u}.parquet'
    if os.path.exists(path):
        prev_data = pd.read_parquet(path)
    else:
        prev_data = pd.DataFrame(columns=DATA_FIELD)

    if len(prev_data) > 0:
        s = prev_data.index[-1]
    else:
        s = t - relativedelta(years=1)
        prev_data = pd.DataFrame(columns=DATA_FIELD)

    data = rd.get_history(universe=r, fields=DATA_FIELD, interval='5min', start=s, end=t)
    if len(data) == 0:
        print(f"{u}: 신규 데이터 없음")
        return {u: False}

    total_data = prev_data.combine_first(data).sort_index()
    total_data = total_data[~total_data.index.duplicated(keep='last')]

    total_data.to_parquet(path, engine='pyarrow')

    return {u: True}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--country', type=str, default='kr', choices=['kr', 'us'], help='country code')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    t_date = dt.datetime.today()

    # Load App Key from config file
    config = ConfigParser()
    config.read('D:/config.ini')
    app_key = config['main']['api_key']

    session = rd.session.desktop.Definition(app_key=app_key).get_session()
    rd.session.set_default(session)
    session.open()

    universe_df = pd.read_csv(f'{DATA_DIR}/{args.country}/universe.csv', index_col=0, encoding='CP949')
    universe = universe_df.index
    ref = universe_df['RIC'].tolist()

    with ThreadPoolExecutor(max_workers=4) as executor:
        pool = [
            executor.submit(_fetch, t_date, args.country, u, r) for u, r in zip(universe, ref)
        ]

        results = [p.result() for p in tqdm(concurrent.futures.as_completed(pool), total=len(pool))]

    r_dict = {}
    for r in results:
        r_key = list(r.keys())[0]
        r_dict[r_key] = r[r_key]

    r_df = pd.DataFrame.from_dict(r_dict, orient='index')
    print(r_df.value_counts())
    print(r_df[~r_df])