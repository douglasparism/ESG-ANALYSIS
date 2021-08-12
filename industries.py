# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:18:09 2021

@author: DParis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from sklearn import preprocessing
from sklearn import linear_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class stanalysis:
    
    def __init__(self, data: pd.DataFrame, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None):
        
        self.start_date = start_date
        self.end_date = end_date
        self.closes = data.loc[start_date:end_date].iloc[::-1].dropna(axis=1)
        self.empresas = list(self.closes.columns)
        self.annual_ret_summary = None
        self.d_ret = self.closes.pct_change().dropna()
        self.mean_ret = self.d_ret.mean()
        self.annual_ret = pd.DataFrame({'Annual ret':self.mean_ret*12})
        self.vol = self.d_ret.std()
        self.annual_vol = pd.DataFrame({'Annual vol': self.vol*np.sqrt(12)})
        self.annual_ret_summary = pd.DataFrame({'Mean':self.mean_ret*12, 
                                                'Vol' : self.vol*np.sqrt(12)})
        
        
    def metrics(self,rf):
        rendacum_avg = round(self.d_ret.cumsum().fillna(0), 4).mean()*12
        rendacum = round(self.d_ret.cumsum().fillna(0), 4).loc[self.end_date]*12
        rend_avg = self.mean_ret*12
        vol_avg = self.vol*np.sqrt(12)
        sharpe = round((self.mean_ret*12 - rf) / self.vol*np.sqrt(12), 2)
        
        return pd.DataFrame({'Avg Acum Ret':rendacum_avg, 
                             'Rend Acum' : rendacum,
                             'Avg Ret' : rend_avg,
                             'Avg Vol' : vol_avg,
                             'Sharpe' : sharpe})
            
        #return self.annual_ret_summary.append(
        #    pd.DataFrame([self.annual_ret_summary.mean()],
        #                 index=["Promedio de la industria"]))
    
    def graph(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(self.closes)
        df = pd.DataFrame(x_scaled)
        df.columns = self.empresas
        df.index = self.closes.index
        return df.plot(figsize=(10,6));
    
    
    def recta(self,calificaciones):

        X_ls = [[i] for i in self.annual_ret_summary['Vol']]
        Y_ls = np.array(self.annual_ret_summary['Mean'])
        clf = linear_model.LinearRegression()
        clf.fit(X_ls, Y_ls)
        
        # Gráfico rendimiento esperado vs. volatilidad
        # Puntos a graficar
        x_points = self.annual_ret_summary.loc[:,'Vol']
        y_points = self.annual_ret_summary.loc[:,'Mean']
        # Ventana para graficar
        plt.figure(figsize = (10,6))
        
        # Grafico recta ajustada
        x_recta = np.linspace(min(self.annual_ret_summary['Vol'])-0.1,
                              max(self.annual_ret_summary['Vol'])+0.1,100)
        x_recta_ls = [[i] for i in x_recta]
        plt.plot(x_recta,clf.predict(x_recta_ls),'r',lw=3,
                 label='Recta ajustada de la industria')
        
        calificaciones = calificaciones.loc[self.empresas]/100
        calificaciones = [calificaciones.fillna(0).iloc[h] for h in 
                          range(0,len(calificaciones.index))]
        
        # Graficar puntos
        plt.scatter(x_points,y_points,c = calificaciones, s=100)
        
        plt.grid()
        # Etiquetas de los ejes
        plt.xlabel('Volatilidad ($\sigma$)')
        plt.ylabel('Rendimiento esperado ($E[r]$)')
        # Etiqueta de cada instrument
        [plt.text(x_points[i],y_points[i],self.annual_ret_summary.index[i]) 
         for i in range(len(self.annual_ret_summary.index))]
        
        plt.legend()
        plt.show()
        pass
    
    
    
    def esg_vs_noesg(self,calificaciones, graphsize: Optional[tuple] = (10,6), 
                     dotsize: Optional[int] = 100, 
                     opt_method: Optional[str] = "BFGS"):
        
        
        calificaciones = calificaciones.loc[self.empresas]/100
        empresas_esg = calificaciones.gt(0)
        empresas_esg = self.annual_ret_summary[empresas_esg.values]
        
        empresas_no_esg = ~calificaciones.gt(0)
        empresas_no_esg = self.annual_ret_summary[empresas_no_esg.values]
        
        calificaciones = [calificaciones.fillna(0).iloc[h] for h in 
                          range(0,len(calificaciones.index))]
        
        
        X_ls_esg = [[i] for i in empresas_esg['Vol']]
        Y_ls_esg = np.array(empresas_esg['Mean'])
        clf_esg = linear_model.LinearRegression()
        clf_esg.fit(X_ls_esg, Y_ls_esg)
        
        
        X_ls_no_esg = [[i] for i in empresas_no_esg['Vol']]
        Y_ls_no_esg = np.array(empresas_no_esg['Mean'])
        clf_no_esg = linear_model.LinearRegression()
        clf_no_esg.fit(X_ls_no_esg, Y_ls_no_esg)
        
        # Gráfico rendimiento esperado vs. volatilidad
        # Puntos a graficar
        x_points = self.annual_ret_summary.loc[:,'Vol']
        y_points = self.annual_ret_summary.loc[:,'Mean']
        # Ventana para graficar
        plt.figure(figsize = graphsize)
        
        # Grafico recta ajustada
        x_recta = np.linspace(min(self.annual_ret_summary['Vol'])-0.1,
                              max(self.annual_ret_summary['Vol'])+0.1,100)
        
        x_recta_ls = [[i] for i in x_recta]
        
        plt.plot(x_recta_ls,clf_esg.predict(x_recta_ls),'g',lw=3,label='Relación rendimiento-volatilidad de la industria ESG')
        
        plt.plot(x_recta_ls,clf_no_esg.predict(x_recta_ls),'purple',lw=3,label='Relación rendimiento-volatilidad de la industria no ESG')
        
        
        # Graficar puntos
        sc = plt.scatter(x_points,y_points,c = calificaciones, s=dotsize)
        plt.colorbar(sc).set_label('ESG Risk [0 means no Risk Assesment]')
        plt.grid()
        # Etiquetas de los ejes
        plt.xlabel('Volatilidad ($\sigma$)')
        plt.ylabel('Rendimiento esperado ($E[r]$)')
        # Etiqueta de cada instrument
        [plt.text(x_points[i],y_points[i],self.annual_ret_summary.index[i]) 
         for i in range(len(self.annual_ret_summary.index))]
        
        plt.title(self.start_date + " al " + self.end_date)
        plt.legend()
        plt.show()
        m_esg = (clf_esg.predict(x_recta_ls)[10]-clf_esg.predict(x_recta_ls)[0])/(x_recta_ls[10][0]-x_recta_ls[0][0])
        m_no_esg = (clf_no_esg.predict(x_recta_ls)[10]-clf_no_esg.predict(x_recta_ls)[0])/(x_recta_ls[10][0]-x_recta_ls[0][0])
        print("La pendiente de la industria ESG es de ", round(m_esg,2), "mientras que para la industria no ESG es de ",round(m_no_esg,2))
        pass   