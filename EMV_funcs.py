# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:46:05 2021

@author: DParis
"""


#Imporatar las librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web

rf=0.0407

# Se definen funciones necesarias

def get_adj_closes(tickers, start_date=None, end_date=None):
    # Fecha inicio por defecto (start_date='2010-01-01') y fecha fin por defecto (end_date=today)
    # Descargamos DataFrame con todos los datos
    closes = web.DataReader(name=tickers, data_source='yahoo', start=start_date, end=end_date)
    # Solo necesitamos los precios ajustados en el cierre
    closes = closes['Adj Close']
    # Se ordenan los índices de manera ascendente
    closes.sort_index(inplace=True)
    return closes

def get_data(tickers,start_date,end_date):
    start_date=start_date
    end_date=end_date
    tickers=tickers.index
    closes = get_adj_closes(tickers=tickers, start_date=start_date, end_date=end_date)
    return closes

#Resumen Anual
def ret_summary(ret):
    cov=ret.cov()
    cor=ret.corr()
    mean_ret=ret.mean()*252
    vol=ret.std()*np.sqrt(252)
    ret_summary=pd.DataFrame({'Rend_Anual':mean_ret,'Vol_Anual':vol})
    return(ret_summary)

#Definicion parametros
def par(ret_summ,ret):
    cor=ret.corr()
    # 1. Sigma: matriz de varianza-covarianza Sigma = S.dot(corr).dot(S)
    S=np.diag(ret_summ.iloc[:,1].values)#las volatilidades y rendimientos tiene que tener el mismo orden
    #que la tabla de correlación
    Sigma=S.dot(cor).dot(S)
    # 2. Eind: rendimientos esperados activos individuales
    Eind=ret_summ.iloc[:,0].values
    return(S,Sigma,Eind)


# Función objetivo
def varianza(w,Sigma):
    return w.T.dot(Sigma).dot(w)



# Pesos, rendimiento y riesgo del portafolio de mínima varianza
def data_pmv(Eind,Sigma):
    # Dato inicial
    n=len(Eind)
    w0=np.ones(n)/n
    # Cotas de las variables
    bnds=((0,1),)*n
    # Restricciones
    cons={'type':'eq','fun':lambda w:w.sum()-1}
    minvar = minimize(fun=varianza,x0=w0,args=(Sigma,),bounds=bnds,constraints=cons)
    w_minvar = minvar.x
    E_minvar = Eind.T.dot(w_minvar)
    s_minvar = (w_minvar.T.dot(Sigma).dot(w_minvar))**0.5
    return(w_minvar,E_minvar,s_minvar)

#MAXIMIZAR RADIO SHARPE 
# Función objetivo
def minus_SR(w,Sigma,Eind,rf):
    sp = (w.T.dot(Sigma).dot(w))**0.5
    Ep =  Eind.T.dot(w)
    SR= (Ep - rf)/sp#vol portafolio
    return -SR

# Portafolio EMV
def portEMV(fun,Eind,Sigma):
    n=len(Eind)
    w0=np.ones(n)/n
    # Cotas de las variables
    bnds=((0.001,1),)*n   
    # Restricciones
    cons={'type':'eq','fun':lambda w:w.sum()-1}
    EMV=minimize(fun=minus_SR,x0=w0,args=(Sigma,Eind,rf),bounds=bnds,constraints=cons)
    w_EMV = EMV.x
    E_EMV =Eind.T.dot(w_EMV)
    s_EMV =(w_EMV.T.dot(Sigma).dot(w_EMV))**0.5
    return(w_EMV , E_EMV, s_EMV)


# Portafolio EMV
def portEMVpond(fun,Eind,Sigma):
    n=len(Eind)
    w0=np.ones(n)/n
    # Cotas de las variables
    #bnds=((0.001,1),)*n
    bnds=[(0,1),]*n
    #OPTIONAL 
    bnds[0]=(0,0.11)
    bnds[1]=(0,0.11)
    bnds[2]=(0,0.21)
    bnds[3]=(0,0.11)
    bnds[4]=(0,0.11)
    
    # Restricciones
    cons={'type':'eq','fun':lambda w:w.sum()-1}
    EMV=minimize(fun=minus_SR,x0=w0,args=(Sigma,Eind,rf),bounds=bnds,constraints=cons)
    w_EMV = EMV.x
    E_EMV =Eind.T.dot(w_EMV)
    s_EMV =(w_EMV.T.dot(Sigma).dot(w_EMV))**0.5
    return(w_EMV , E_EMV, s_EMV)

def RS_EMV(rf,E_EMV,s_EMV):
    RS_EMV=(E_EMV - rf)/ s_EMV
    return(RS_EMV)

def plot_portas(w_EMV,Sigma,w_minvar,s_EMV,s_minvar,E_EMV,E_minvar):
    cov = w_EMV.T.dot(Sigma).dot(w_minvar)
    corr= cov/ (s_EMV *s_minvar)
    W =np.linspace(0,3,100)
    portafolios=pd.DataFrame(data={'w': W,'1-w': 1-W, 'Media':W*E_EMV +(1 -W)*E_minvar,
                        'Vol': ((W*s_EMV)**2 + ((1 - W)*s_minvar)**2 + 2*W* (1-W)*cov)**0.5})
    portafolios['RS']=(portafolios['Media']-rf)/portafolios['Vol']   
    plt.figure(figsize=(6, 4))
    plt.scatter(portafolios['Vol'], portafolios['Media'], c=portafolios['RS'], cmap='RdYlBu')
    plt.plot(s_minvar,E_minvar,'og',ms=7,label='Port.Min.Var')
    plt.plot(s_EMV,E_EMV,'ob',ms=7,label='Port.EMV')
    plt.legend(loc='best')
    plt.colorbar()
    plt.xlabel("Volatilidad $\sigma$")
    plt.ylabel("Rendimiento esperado $E[r]$")
    plt.grid()
    return(plt)