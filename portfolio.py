import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Monte Carlo portafolio
def port_mc(closes, n_port, tickers_, rf):

    #Rendimientos y volatilidad diaria
    ret = closes.pct_change().dropna()
    mean_ret = ret.mean()
    vol  = ret.std()
    ret_summary = pd.DataFrame({'Mean': mean_ret,'Vol': vol})
    ret_summary = ret_summary.T

    #Portafolio
    n_act = len(tickers_)
    corr = ret.corr()
    W = np.random.dirichlet(np.ones(n_act), n_port)
    Ep = ret_summary.loc['Mean'][:n_act].values.T.dot(W.T)
    cov = pd.DataFrame(data= (np.diag(ret_summary.loc['Vol'][:n_act].values).dot(corr)).dot(np.diag(ret_summary.loc['Vol'][:n_act].values)),
                        columns = ret_summary.columns[:n_act], index = ret_summary.columns[:n_act])
    vol = np.zeros((n_port,))
    for i in range(n_port):
        vol[i] = (W[i, :].T.dot(cov).dot(W[i, :]))**0.5

    #Sharpe Ratio
    rf = rf/360 #Tasa libre de riesgo anualizada
    RS = (Ep - rf) / vol

    #DataFrame Pesos
    portafolios = pd.DataFrame({
                                '$w1$': W[:, 0],
                                '$w2$': W[:, 1],
                                '$w3$': W[:, 2],
                                '$w4$': W[:, 3],
                                '$w5$': W[:, 4],
                                '$w6$': W[:, 5],
                                '$w7$': W[:, 6],
                                '$w8$': W[:, 7],
                                '$w9$': W[:, 8],
                                '$w10$': W[:, 9],
                                '$w11$': W[:, 10],
                                '$w12$': W[:, 11],
                                '$w13$': W[:, 12],
                                '$w14$': W[:, 13],
                                '$w15$': W[:, 14],
                                '$w16$': W[:, 15],
                                '$w17$': W[:, 16],
                                '$w18$': W[:, 17],
                                '$w19$': W[:, 18],
                                '$w20$': W[:, 19],
                                '$w21$': W[:, 20],
                                '$w22$': W[:, 21],
                                '$w23$': W[:, 22],
                                '$E[p]$': Ep,
                                '$\sigma$': vol,
                                'RS': RS
                               })

    # Portafolio mínima varianza
    minvar = portafolios.loc[portafolios['$\sigma$'].idxmin()]
    minvar = np.array(minvar).astype(float)
    EMV = portafolios.loc[portafolios['RS'].idxmax()]
    EMV = np.array(EMV).astype(float)

    MC_weights_minvar = pd.DataFrame({"Tickers": tickers_, "Pesos MC": minvar[0:-3]})
    MC_weights_RS = pd.DataFrame({"Tickers": tickers_, "Pesos MC": EMV[0:-3]})

    return(MC_weights_minvar,MC_weights_RS,portafolios)


def new_weights(passive_weights_ESG, MC_weights, capital, comision):
    passive_new_weights = passive_weights_ESG.join(MC_weights)
    peso = passive_new_weights[['Pesos MC']].astype(float)
    precio = passive_new_weights[['Close']].values
    capital = peso * capital
    titulos = np.floor(capital/precio)

    del passive_new_weights['Capital']
    del passive_new_weights['Peso (%) fijo']
    del passive_new_weights['Comision']
    del passive_new_weights['Titulos']

    passive_new_weights["Titulos"] = titulos
    passive_new_weights["Comision"] = passive_new_weights['Titulos'] * passive_new_weights['Close'] * comision
    passive_new_weights["Capital"] = passive_new_weights['Titulos'] * passive_new_weights['Close']
    passive_new_weights = passive_new_weights.rename(columns={"Pesos MC": "Peso (%) fijo"})

    return (passive_new_weights)


def passive_investment(df, fix , dates, capital, comision):

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
