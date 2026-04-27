import os
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser

import pandas as pd
import refinitiv.data as rd
from tqdm import tqdm
import datetime as dt

pd.set_option('future.no_silent_downcasting', True)

def _fetch(u, r):
    if os.path.exists('./data/{}.parquet'.format(u)):
        #continue
        prev_data = pd.read_parquet('./data/{}.parquet'.format(u))
    else:
        prev_data = pd.DataFrame()

    if len(prev_data) > 0:
        if prev_data.index[-1] > end:
            print(f"{u}: 데이터 최신")
            return {u: False}

    if 'Timestamp' in prev_data.columns:
        prev_data.set_index('Timestamp', inplace=True)
        prev_data.index = pd.to_datetime(prev_data.index)
        prev_data = prev_data.between_time('23:50', '07:00')
    elif prev_data.index.name == 'Timestamp':
        prev_data.index = pd.to_datetime(prev_data.index)
        prev_data = prev_data.between_time('23:50', '07:00')

    if len(prev_data) == 0:
        prev_data = pd.DataFrame(columns=data_fields)

    data = rd.get_history(universe=r, fields=data_fields, interval='5min', start=start, end=end)
    if len(data) == 0:
        print(f"{u}: 신규 데이터 없음")
        return {u: False}

    data = data.between_time('23:50', '07:00')

    total_data = prev_data.combine_first(data).sort_index()
    total_data = total_data[~total_data.index.duplicated(keep='last')]

    total_data.to_parquet('./data/{}.parquet'.format(u), engine='pyarrow')

    return {u: True}


data_fields = ['TRDPRC_1', 'BID', 'ASK']

start = dt.datetime(2026, 4, 12, 23, 50)
end = dt.datetime(2026, 4, 23, 7, 0)

if __name__ == "__main__":

    # Load App Key from config file
    config = ConfigParser()
    config.read('D:/config.ini')
    app_key = config['main']['api_key']

    session = rd.session.desktop.Definition(app_key=app_key).get_session()
    rd.session.set_default(session)
    session.open()

    universe = pd.read_csv('./top100.csv')['instrument'].tolist()
    ref = rd.get_data(universe=universe, fields=['TR.RIC'])['RIC'].tolist()

    with ThreadPoolExecutor(max_workers=4) as executor:
        pool = [
            executor.submit(_fetch, u, r) for u, r in tqdm(zip(universe, ref), total=len(ref))
        ]
        results = [p.result() for p in pool]

    r_dict = {}
    for r in results:
        r_key = list(r.keys())[0]
        r_dict[r_key] = r[r_key]

    r_df = pd.DataFrame.from_dict(r_dict, orient='index')
    print(r_df.value_counts())
    print(r_df[~r_df])