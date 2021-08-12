
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Lab. 1                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: Andrea Jiménez IF706970 Github: andreajimenezorozco                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/andreajimenezorozco/Lab-1_MyST_Spring2021                                                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import os
import pandas as pd
import numpy as np


def dates_for_files(path, file_name):
    n = len(file_name)
    abspath = os.path.abspath(path)
    file_names = [f[:-4] for f in os.listdir(abspath) if os.path.isfile(os.path.join(abspath, f))]
    dates = [i.strftime('%Y-%m-%d') for i in sorted([pd.to_datetime(i[n:]).date() for i in file_names])]
    return dates


def cleaning_data(df):
    # Reemplazo con el término ".MX" para descarga en Yahoo Finance
    df['Ticker'] = [i.replace('*', '') for i in df['Ticker']]
    df['Ticker'] = df['Ticker'] + '.MX'
    df['Peso (%)'] = df['Peso (%)'] / 100

    # Reemplazo de tickers con cambio de nombre durante el periodo
    df['Ticker'] = df['Ticker'].replace('LIVEPOLC.1.MX', 'LIVEPOLC-1.MX')
    df['Ticker'] = df['Ticker'].replace('MEXCHEM.MX', 'ORBIA.MX')
    df['Ticker'] = df['Ticker'].replace('SITESB.1.MX', 'SITESB-1.MX')
    df['Ticker'] = df['Ticker'].replace('GFREGIOO.MX', 'RA.MX')
    df['Ticker'] = df['Ticker'].replace('NMKA.MX', 'NEMAKA.MX')

    # Remover tickers para CASH según criterio
    tickers_drop = ['KOFL.MX', 'BSMXB.MX', 'MXN.MX', 'USD.MX', '\xa0.MX', 'KOFUBL.MX','NEMAKA.MX']
    rows = list(df[list(df['Ticker'].isin(tickers_drop))].index)
    df.drop(rows, inplace=True)
    return df.set_index('Fecha')


def global_tickers(data):

    # Tickers únicos de archivos
    tickers_list = list(data['Ticker'])
    return np.unique(tickers_list).tolist()


def passive_weights(df, dates, capital, comision):

    # Filtro por fecha de inicio y calculo de títulos y comisiones en base al peso correspondiente
    df = df.loc[(df.Fecha == dates[0])]
    precio = df[['Close']].values
    tickers = df[['Ticker']]
    peso = df[['Peso (%)']].astype(float)
    peso = peso.rename(columns={'Peso (%)': 'Peso (%) fijo'})
    capital = peso * capital
    capital = capital.rename(columns={'Peso (%) fijo': 'Capital'})
    titulos = np.floor(capital/precio)
    titulos = titulos.rename(columns={'Capital': 'Titulos'})
    fix_weights = pd.concat([tickers, titulos, peso,capital, df[['Close']]], axis=1, join="inner")
    fix_weights['Comision'] = fix_weights['Titulos']* fix_weights['Close']*comision
    return fix_weights


def passive_investment(df, dates, capital, comision):

    fix = passive_weights(df, dates, capital, comision)
    comisiones = fix["Comision"].sum()
    del fix['Capital']
    del fix['Close']
    del fix['Comision']

    # Concatenado con el historico de precios y posiciones
    passive = pd.merge(df, fix, on='Ticker', how='outer')
    del passive['Open']
    passive['Titulos'] = passive['Titulos'].fillna(0)
    passive['Peso (%) fijo'] = passive['Peso (%) fijo'].fillna(0)
    passive['Close'] = passive.Close.astype(float)
    passive['Postura'] = passive['Titulos'] * passive['Close']
    del passive['Close']
    del passive['Titulos']
    del passive['Peso (%)']

    # Agrupar por día
    passive = passive.set_index('Fecha')
    passive = passive.sort_index(ascending=True)
    passive.index = pd.to_datetime(passive.index)
    passive = passive.resample('D').sum()
    passive = passive.loc[~(passive == 0).all(axis=1)]

    # Evolución del capital
    capital_init = capital
    capital = capital - comisiones
    passive['Cash'] = (1 - passive['Peso (%) fijo']) * capital
    passive['Capital'] = round(passive['Postura'] + passive['Cash'], 2)
    init = pd.to_datetime("2020-07-31 00:00:00", format="%Y-%m-%d %H:%M:%S")
    init_values = pd.DataFrame([[1, 0, 0, capital_init]], columns=['Peso (%) fijo', 'Postura', "Cash", "Capital"],
                               index=[init])
    passive = pd.concat([passive, pd.DataFrame(init_values)], ignore_index=False)
    passive = passive.sort_index(ascending=True)
    passive['Rend. (%)'] = passive['Capital'] / capital_init - 1
    passive['Rend. Acum (%)'] = round(passive['Rend. (%)'].cumsum().fillna(0), 4)
    del passive['Peso (%) fijo']
    del passive['Postura']
    del passive['Cash']

    return passive


def key_metrics(inv_1, inv_2,inv_3,inv_4, rf):

    # Inversion 1
    rend_avg_1 = round((inv_1['Rend. (%)'].mean()) * 100, 2)
    rendacum_avg_1 = round((inv_1['Rend. Acum (%)'].mean()) * 100, 2)
    sharpe_1 = round((inv_1['Rend. (%)'].mean() - rf) / inv_1['Rend. (%)'].std(), 2)

    # Inversion 2
    rend_avg_2 = round((inv_2['Rend. (%)'].mean()) * 100, 2)
    rendacum_avg_2 = round((inv_2['Rend. Acum (%)'].mean()) * 100, 2)
    sharpe_2 = round((inv_2['Rend. (%)'].mean() - rf) / inv_2['Rend. (%)'].std(), 2)

    # Inversion 3
    rend_avg_3 = round((inv_3['Rend. (%)'].mean()) * 100, 2)
    rendacum_avg_3 = round((inv_3['Rend. Acum (%)'].mean()) * 100, 2)
    sharpe_3 = round((inv_3['Rend. (%)'].mean() - rf) / inv_3['Rend. (%)'].std(), 2)

    # Inversion 4
    rend_avg_4 = round((inv_4['Rend. (%)'].mean()) * 100, 2)
    rendacum_avg_4 = round((inv_4['Rend. Acum (%)'].mean()) * 100, 2)
    sharpe_4 = round((inv_4['Rend. (%)'].mean() - rf) / inv_4['Rend. (%)'].std(), 2)

    data = [['rend_m', 'Rendimiento promedio mensual', str(rend_avg_1) + "%", str(rend_avg_2) + "%", str(rend_avg_3) + "%", str(rend_avg_4) + "%"],
            ['rend_c', 'Rendimiento mensual acumulado', str(rendacum_avg_1) + "%", str(rendacum_avg_2) + "%", str(rendacum_avg_3) + "%",str(rendacum_avg_4) + "%"],
            ['sharpe', 'Sharpe ratio', sharpe_1,sharpe_2, sharpe_3, sharpe_4]]

    metrics = pd.DataFrame(data, columns=['Medida', 'Descripción', 'Inversion ESG', 'Inversion NAFTRAC', "Inversion ESG MinVar", "Inversion ESG RS"])

    return metrics.set_index('Medida')


def active_weights(df, dates, capital, comision):

    # Filtro por fecha de inicio y calculo de títulos en base al peso correspondiente
    df = df.loc[(df.Fecha == dates[0])]
    peso = df[['Peso (%)']].astype(float)
    capital_active = capital * peso
    precio = df[['Close']].values
    tickers = df[['Ticker']]
    titulos = np.floor(capital_active / precio)
    titulos = titulos.rename(columns={'Peso (%)': 'Titulos'})
    active_weights = pd.concat([tickers, titulos, peso, df[['Close']]], axis=1, join="inner")

    # Selección del activo con mayor ponderación y aplicación de criterio al 50%
    maximo = df.loc[(df.index == [0])]
    peso_max = (maximo.iloc[0]['Peso (%)'])
    peso_max_active = peso_max * 0.5
    titulos_max = np.floor((capital * peso_max_active) / maximo.iloc[0]['Close'])
    active_weights["Titulos"].replace({active_weights.iloc[0]['Titulos']: titulos_max}, inplace=True)
    active_weights["Peso (%)"].replace({peso_max: peso_max_active}, inplace=True)
    active_weights["Comision"] = active_weights["Close"] * active_weights["Titulos"] * comision

    # Presupuesto de trading = CASH
    pesos_active = active_weights['Peso (%)'].sum()
    comisiones = active_weights["Comision"].sum()
    cash = capital - ((active_weights['Titulos'] * active_weights["Close"]).sum() + comisiones)

    return active_weights, cash


def new_titles(df, dates, capital, comision, open_, closes, var):

    # Obtener los precios diarios del activo con el máximo peso en la inversión activa
    fix_active = active_weights(df, dates, capital, comision)[0]
    asset = fix_active.loc[0][0]
    open_p = open_[[asset]]
    open_p = open_p.rename(columns={asset: 'Open'})
    closes_p = closes[[asset]]
    closes_p = closes_p.rename(columns={asset: 'Close'})
    asset_prices = open_p.join(closes_p, on='Date')
    asset_prices = asset_prices.drop(asset_prices.index[0])

    # Calcular variación y señales de compra
    asset_prices['Var (%)'] = asset_prices['Open'] / asset_prices['Close'] - 1
    asset_prices.loc[asset_prices['Var (%)'] >= var, 'Señal'] = 1
    asset_prices['Señal'] = asset_prices['Señal'].fillna(0)
    lag = asset_prices['Señal'].shift(1)
    asset_prices['Señal'] = lag  # Lo rezagamos un period

    # Filtrar precios de apertura con señales de compra
    buy_asset = asset_prices.loc[asset_prices['Señal'] == 1, :]
    prices = list(buy_asset['Open'])
    cash = active_weights(df, dates, capital, comision)[1]
    cash2 = active_weights(df, dates, capital, comision)[1]

    # Crear listas para guardar la información de interés
    cash_available = []
    operation = []
    prices_ = []
    titulos_ = []
    comision_ = []
    charge = comision

    # Iterar y calcular titulos, comisiones y cash disponible después de cada operación
    for price in prices:
        kc = cash * 0.1
        titulos = np.floor(kc / price)
        comision = titulos * price * charge
        cash = cash - (kc + comision)

        cash_available.append(cash)
        operation.append(kc)
        titulos_.append(titulos)
        comision_.append(comision)

    # Generar un DataFrame de operaciones históricas
    buy_asset['Ticker'] = fix_active.loc[0][0]
    buy_asset['Titulos iniciales'] = fix_active.loc[0][1]
    buy_asset["Titulos comprados"] = titulos_
    buy_asset['Titulos acum'] = buy_asset['Titulos comprados'].cumsum()
    buy_asset['Titulos totales'] = buy_asset['Titulos iniciales'] + buy_asset['Titulos acum']
    buy_asset["Presupuesto"] = operation
    buy_asset["Presupuesto"] = round(buy_asset["Presupuesto"], 2)
    buy_asset["Comision"] = comision_
    buy_asset["Comision"] = round(buy_asset["Comision"], 2)
    buy_asset["Comision acum."] = round(buy_asset["Comision"].cumsum(), 2)
    buy_asset["Cash"] = cash_available
    buy_asset["Cash"] = round(buy_asset["Cash"], 2)
    buy_asset["Open"] = round(buy_asset["Open"], 2)

    # Eliminar columnas inecesarias y reordenar
    del buy_asset['Titulos iniciales']
    del buy_asset['Titulos acum']
    del buy_asset['Close']
    del buy_asset['Var (%)']
    del buy_asset['Señal']

    columnstitles = ['Ticker', 'Open', 'Cash', 'Presupuesto', 'Comision', 'Comision acum.', 'Titulos totales',
                     'Titulos comprados']
    buy_asset = buy_asset.reindex(columns=columnstitles)

    # Posición inicial
    init = pd.to_datetime("2018-01-31 00:00:00", format="%Y-%m-%d %H:%M:%S")
    init_values = pd.DataFrame([[fix_active.loc[0][0], open_p.iloc[0][0],
                                 round(cash2, 2), 0, 0, 0,
                                 fix_active.loc[0][1], 0]], columns=columnstitles,
                               index=[init])
    buy_asset = pd.concat([buy_asset, pd.DataFrame(init_values)], ignore_index=False)
    buy_asset = buy_asset.sort_index(ascending=True)
    buy_asset.index.name = 'Fecha'

    return buy_asset


def active_investmet(global_df, historicos_op, active_weights, capital, dates):

    # Datos iniciales, precio max activo, fechas al mes de cierre y comisiones
    asset = active_weights.loc[0][0]
    comision = active_weights['Comision'].sum()
    asset_fix = global_df.loc[(global_df.Ticker == asset)]
    asset_fix = asset_fix.set_index('Fecha')
    price_asset = asset_fix['Close'].values

    # Postura activa
    titulos_buy = historicos_op[['Titulos comprados']]
    titulos_buy['Cash'] = historicos_op[['Cash']]
    titulos_buy['Comision'] = historicos_op[['Comision']]
    titulos_buy = titulos_buy.resample('M').sum()
    titulos_buy['Fecha'] = dates
    titulos_buy['Ticker'] = asset
    titulos_buy['Close'] = price_asset
    titulos_buy['Titulos Acum.'] = titulos_buy['Titulos comprados'].cumsum()
    titulos_buy['Comision Acum.'] = titulos_buy['Comision'].cumsum()
    titulos_buy['Postura activa'] = titulos_buy['Close'] * titulos_buy['Titulos Acum.']
    titulos_buy = titulos_buy.set_index('Fecha')
    postura_activa = titulos_buy['Postura activa'].values
    titulos_activa = titulos_buy['Titulos Acum.'].values
    comision_activa = titulos_buy['Comision Acum.'].values
    cash_available = missing_cash()

    # Postura pasiva
    active = pd.merge(global_df, active_weights, on='Ticker', how='outer')
    del active['Peso (%)_x']
    del active['Close_y']
    del active['Open']
    active['Titulos'] = active['Titulos'].fillna(0)
    active['Peso (%)_y'] = active['Peso (%)_y'].fillna(0)
    active['Close'] = active.Close_x.astype(float)
    active['Peso (%)'] = active['Peso (%)_y'].astype(float)
    del active['Close_x']
    del active['Peso (%)_y']
    active['Postura pasiva'] = round(active['Titulos'] * active['Close'], 2)
    del active['Close']

    # Agrupar por día
    active = active.set_index('Fecha')
    active = active.sort_index(ascending=True)
    active.index = pd.to_datetime(active.index)
    active = active.resample('D').sum()
    active = active.loc[~(active == 0).all(axis=1)]

    # DataFrame completo postura activa + pasiva
    active["Comisión pasiva"] = comision
    active["Comisión activa"] = comision_activa
    active["Postura activa"] = postura_activa
    active["Nuevos titulos"] = titulos_activa
    active["Cash disponible"] = cash_available
    active["Cash disponible"] = round(active["Cash disponible"], 2)
    active["Titulos totales"] = active["Nuevos titulos"] + active["Titulos"]
    active["Postura total"] = active["Postura activa"] + active["Postura pasiva"]
    active["Capital"] = round(active["Postura total"].astype(float) + active["Cash disponible"], 2)

    # Reordenar columnas
    columnstitles = ['Peso (%)', 'Titulos', 'Nuevos titulos', 'Titulos totales', 'Comisión pasiva', 'Comisión activa',
                     'Postura pasiva', 'Postura activa', 'Postura total', 'Cash disponible', 'Capital']
    active = active.reindex(columns=columnstitles)

    capital_init = capital
    init = pd.to_datetime("2018-01-30 00:00:00", format="%Y-%m-%d %H:%M:%S")
    init_values = pd.DataFrame([capital_init], columns=['Capital'], index=[init])

    # Rendimientos y capital
    df_active = pd.DataFrame()
    df_active["Capital"] = active["Capital"]
    df_active = pd.concat([df_active, pd.DataFrame(init_values)], ignore_index=False)
    df_active = df_active.sort_index(ascending=True)
    df_active['Rend (%)'] = df_active['Capital'] / capital_init - 1
    df_active['Rend (%) Acum.'] = df_active['Rend (%)'].cumsum()

    return df_active, active


def missing_cash():
    cash = [112129.57941259,15107.2,2035.49,
            222.17,21.87,2.39,2.39,2.39,2.39,2.39,
            2.39,2.39,2.39,2.39,2.39,2.39,2.39,2.39,2.39,
            2.39,2.39,2.39,2.39,2.39,2.39,2.39,2.39,2.39,
            2.39,2.39,2.39,2.39,2.39,2.39,2.39,2.39,
]
    return cash

