#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import timedelta
from dt_help import Helper
from pycaret.regression import *
from PyEMD import EMD

class EmdML():
    def __init__(
        self, 
        data: pd.DataFrame,
        yvar: 'raw',
        mpg: dict,
        num_selected_models: int
    ):
        self.data = data
        self.cols = self.data.columns.values
        self.feat_days = 3
        self.trgt_days = 5
        self.yvar = yvar
        self.to_blacklist = ['ridge','en','lar','llar','omp','br','ard','par','ransac','tr',
                             'huber','kr','et','ada','gbr','mlp','lightgbm','catboost']
        self.mpg = mpg
        self.num_selected_models = num_selected_models
        
    @Helper.timing
    def get_feat_lag(self):
        data_temp = []
        self.data.reset_index(inplace=True)
        cols = self.data.columns.drop(['Dates','volume'])
        for i in self.feat_days:
            data_temp.append(self.data[cols].pct_change(periods=i).add_suffix("_"+str(i)+'D'))
        df = pd.concat([self.data]+data_temp,axis=1)
        df.set_index('Dates',inplace=True)
        df.dropna(inplace=True)
        df.drop(columns=['volume'],inplace=True)
        self.all_feat_lag = df

    def get_imfs_hilbert_ts(self):
        daily_returns = self.data['close'].pct_change().dropna()
        log_returns = np.log(1 + self.data['close'].pct_change().dropna())
        self.close = self.data['close']
        self.daily_returns = daily_returns
        self.log_returns = log_returns

        df = pd.DataFrame({'close': self.close, 'daily_returns': self.daily_returns,'log_returns': self.log_returns})
        df.dropna(inplace=True)
        emd = EMD()
        dt = df.index
        ts = df['daily_returns'].values
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
        df_imfs['raw'] = df['daily_returns'].values
        df_imfs.columns = ['noise'] + [el for el in df_imfs.columns[1:-2]] + ['trend','raw']
        df_imfs['denoise_detrend'] = df_imfs['raw'] - df_imfs[[el for el in df_imfs.columns if 'imf_' in el]].sum(axis=1)
        df_imfs.insert(0,'Dates',df.index)
        self.all_components_emd = df_imfs
        self.all_feat_emd = df_imfs[[el for el in df_imfs.columns if 'denoise_' not in el]]
        
    @Helper.timing
    def get_target(self):
        yvar = self.yvar
        y_trgt = pd.DataFrame()
        y_trgt['Dates'] = self.all_feat_emd['Dates']
        y_trgt[yvar+'_'+str(self.trgt_days)+'D'] = self.data['close'].iloc[1:].pct_change(periods=-self.trgt_days).values
        self.y_trgt = y_trgt.dropna()
        
    @Helper.timing
    def process_all(self):
        yvar = self.yvar
        self.features = self.all_feat_emd[[el for el in self.all_feat_emd.columns if self.yvar not in el]]
        self.feat_trgt = pd.merge(left=self.features,right=self.y_trgt,how='inner',on='Dates',suffixes=(False,False))
        
    @Helper.timing
    def regr_models(self):
        a = setup(self.feat_trgt,
                  target=self.yvar+'_'+str(self.trgt_days)+'D',
                  ignore_features=['Dates'],
                  session_id=11,
                  silent=True,
                  profile=False,
                  remove_outliers=True,
                  )
        self.regrs_models = compare_models(blacklist=self.to_blacklist,turbo=True)

    @Helper.timing
    def get_best_models(self):
        df = self.regrs_models.data
        df.sort_values(by=['R2'],ascending=False,inplace=True)
        models = df['Model'].values[:self.num_selected_models]
        common = set(self.mpg.keys()).intersection(set(models))
        self.selected_models = [self.mpg[el] for el in common]
        self.best_model = self.selected_models[0]
        
    @Helper.timing
    def bagg_tune_best_model(self):
        self.best_model_tuned = tune_model(self.best_model)
        self.best_model_tuned_bagged = ensemble_model(self.best_model_tuned,method='Bagging')
        
    @Helper.timing
    def stacking_model(self):
        models_tuned = []
        for el in [ mdl for mdl in self.selected_models if self.best_model not in mdl ]:
            models_tuned.append(tune_model(el))
        self.model_stack = create_stacknet(estimator_list=[models_tuned,[self.best_model_tuned_bagged]])

    @Helper.timing
    def save_model(self):
        save_model(model=self.model_stack,model_name='model_saved')

    @Helper.timing
    def predict(self):
        regressor = load_model('model_saved')
        predicted_return = predict_model(regressor,data=self.features)
        predicted_return = predicted_return[['Dates','Label']]
        predicted_return.columns = ['Dates','return_'+self.yvar+'_'+str(self.trgt_days)+'D']
        data_pred = self.data.reset_index()
        predicted_values = data_pred[['Dates','close']]
        predicted_values = predicted_values.tail(len(predicted_return))
        predicted_values = pd.merge(left=predicted_values,right=predicted_return,on=['Dates'],how='inner')
        predicted_values['close_T'+str(self.trgt_days)+'D']=(predicted_values['close']*
                                              (1+predicted_values['return_'+self.yvar+'_'+str(self.trgt_days)+'D'])).round(decimals=2)
        predicted_values['Dates_T+'+str(self.trgt_days)+'D'] = predicted_values['Dates']+timedelta(days=self.trgt_days)
        self.predicted_values = predicted_values
        print(self.predicted_values)
