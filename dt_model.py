#!/usr/bin/python3

import backtrader.indicators as btind
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm

from dt_help import Helper
from PyEMD import EMD
from scipy.stats import norm

class EMDAnalysis():
    def __init__(
        self,
        data: pd.DataFrame,
        ):
        self.data = data

    def get_vars(self):
        daily_returns = self.data['close'].pct_change().dropna()
        log_returns = np.log(1 + self.data['close'].pct_change().dropna())
        self.close = self.data['close']
        self.daily_returns = daily_returns
        self.log_returns = log_returns

    def get_imfs_hilbert_ts(self):
        df = pd.DataFrame({'close': self.close, 'daily_returns': self.daily_returns,'log_returns': self.log_returns})

        emd = EMD()
        dt = df.index
        ts = df['close'].values
        IMFs = emd(ts)
        N = IMFs.shape[0]+1
        
        plt.subplots(figsize=(32,20),sharex=True)
        for n, imf in enumerate(IMFs):
            plt.subplot(N,1,n+1)
            plt.plot(dt,imf, 'g')
            if(n==0):
                tlt = 'noise'
            elif(n==len(IMFs)-1):
                tlt = 'trend'
            else:
                tlt = "imf "+str(n)
            plt.ylabel(tlt)

            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

        plt.subplot(N,1,N)
        plt.plot(dt,ts, 'r')
        plt.xlabel("dates")
        plt.ylabel("raw ts")
        plt.savefig('data_out/imfs_ts.pdf')

        df_imfs = pd.DataFrame()
        for i, el in enumerate(IMFs):
            df_imfs['imf_'+str(i)] = el
        df_imfs['raw'] = df['close'].values
        df_imfs.columns = ['noise'] + [el for el in df_imfs.columns[1:-2]] + ['trend','raw']
        df_imfs['denoise_detrend'] = df_imfs['raw'] - df_imfs[[el for el in df_imfs.columns if 'imf_' in el]].sum(axis=1)
        self.imfs = df_imfs

        
