
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Lab. 1                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: Andrea Jiménez IF706970 Github: andreajimenezorozco                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/andreajimenezorozco/Lab-1_MyST_Spring2021                                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import glob
import pandas as pd
import os
import pandas_datareader.data as web
import numpy as np


def multiple_csv(path,file_name):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    file_list = []
    for file in all_files:
        df = pd.read_csv(file, usecols=[0, 3])
        df['Fecha'] = file
        df['Fecha'] = [i.replace(path+str("\\")+file_name,'') for i in df['Fecha']]
        df['Fecha'] = [i.replace('.csv','') for i in df['Fecha']]
        file_list.append(df)
    all_df = pd.concat(file_list, ignore_index=True)
    return all_df


def get_closes(tickers, start_date=None, end_date=None, freq=None):
    closes = pd.DataFrame(columns=tickers, index=web.YahooDailyReader(tickers[0], start=start_date, end=end_date
                                                                      , interval=freq).read().index)
    for ticker in tickers:
        df = web.YahooDailyReader(symbols=ticker, start=start_date, end=end_date, interval=freq).read()
        closes[ticker] = df['Adj Close']
    closes.index_name = 'Date'
    closes = closes.sort_index()
    return closes


def get_open(tickers, start_date=None, end_date=None, freq=None):
    open_ = pd.DataFrame(columns=tickers, index=web.YahooDailyReader(tickers[0], start=start_date, end=end_date
                                                                      , interval=freq).read().index)
    for ticker in tickers:
        df = web.YahooDailyReader(symbols=ticker, start=start_date, end=end_date, interval=freq).read()
        open_[ticker] = df['Open']
    open_.index_name = 'Date'
    open_ = open_.sort_index()
    return open_


def global_dataframe(historical, closes, open_, dates):
    fix_dates = sorted(list(set(closes.index.astype(str).tolist()) & set(dates)))
    fix_prices_closes = closes.iloc[[int(np.where(closes.index.astype(str) == i)[0]) for i in fix_dates]]
    fix_prices_closes = fix_prices_closes.reindex(sorted(fix_prices_closes.columns), axis=1)

    fix_prices_open = open_.iloc[[int(np.where(open_.index.astype(str) == i)[0]) for i in fix_dates]]
    fix_prices_open = fix_prices_open.reindex(sorted(fix_prices_open.columns), axis=1)

    historical_dates = {}
    for date in fix_dates:
        historical_dates[date] = historical[historical.index == date].reset_index().set_index('Ticker').T

    for date in fix_dates:
        historical_dates[date].loc['Close'] = 0
        for ticker in fix_prices_closes.columns:
            historical_dates[date].loc['Close'][ticker] = fix_prices_closes.loc[date][ticker]
            historical_dates[date].columns.sort_values()

    for date in fix_dates:
        historical_dates[date].loc['Open'] = 0
        for ticker in fix_prices_open.columns:
            historical_dates[date].loc['Open'][ticker] = fix_prices_open.loc[date][ticker]
            historical_dates[date].columns.sort_values()

    df_list = [v for k, v in historical_dates.items()]
    df = pd.concat(df_list, axis=1)
    df = df.T
    df = df.reset_index()

    return df



