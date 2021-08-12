
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Lab. 1                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: Andrea Jiménez IF706970 Github: andreajimenezorozco                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/andreajimenezorozco/Lab-1_MyST_Spring2021                                                                    -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from functions import *
from data import *
from visualizations import *

# Extracción de fechas de los archivos
dates = dates_for_files(path=r'files')

# Lectura recursiva
df = multiple_csv(path=r'files')

# Limpieza de tickers utilizados como CASH y adición del componente ".MX" para la descarga de precios.
historical = cleaning_data(df)
historical.head()

# Lista de tickers únicos presentes en el histórico
tickers = global_tickers(historical)

# Descarga de precios de apertura y cierre de las fechas de interés con frecuencia diaria.
open_ = get_open(tickers, '2018-01-30', '2020-12-31', freq="d")
closes = get_closes(tickers, '2018-01-30','2020-12-31', freq="d")

# DataFrame global
global_df = global_dataframe(historical, closes, open_, dates)

# Inversión pasiva
passive_weights = passive_weights(global_df, dates, 1000000, 0.00125)
passive = passive_investment(global_df,dates,1000000,0.00125)
basic_plot(passive,"Evolución del Capital | Inversión Pasiva", "Capital","Mes","Capital")

# Inversión activa
active_weights = active_weights(global_df,dates,1000000,0.00125)[0]
historicos_op = new_titles(global_df, dates, 1000000, 0.00125, open_, closes, 0.01)
active_full = active_investmet(global_df, historicos_op, active_weights, 1000000, dates)[1]
active = active_investmet(global_df, historicos_op, active_weights, 1000000, dates)[0]
basic_plot(active,"Evolución del Capital | Inversión Activa", "Capital","Mes","Capital")

# Métricas de desempeño
key_metrics(active, passive, 0.0345)
